from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window


TILE_SIZE = 256
MAX_TILE_Z = 10


def list_basemaps(basemap_dir: Path | None, cache_dir: Path | None) -> list[dict[str, Any]]:
    if basemap_dir is None or not basemap_dir.exists():
        return []
    maps: list[dict[str, Any]] = []
    for path in sorted([*basemap_dir.rglob("*.tif"), *basemap_dir.rglob("*.tiff")]):
        try:
            with rasterio.open(path) as dataset:
                bounds = dataset.bounds
                maps.append(
                    {
                        "id": _basemap_id(path),
                        "name": path.name,
                        "path": str(path),
                        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                        "width": int(dataset.width),
                        "height": int(dataset.height),
                        "band_count": int(dataset.count),
                        "tile_url_template": f"/offline/basemaps/{_basemap_id(path)}/tiles/{{z}}/{{x}}/{{y}}.png",
                    }
                )
        except Exception as exc:
            maps.append(
                {
                    "id": _basemap_id(path),
                    "name": path.name,
                    "path": str(path),
                    "error": str(exc),
                }
            )
    return maps


def get_basemap_tile_path(
    basemap_dir: Path | None,
    cache_dir: Path | None,
    basemap_id: str,
    z: int,
    x: int,
    y: int,
) -> Path | None:
    if z < 0 or z > MAX_TILE_Z:
        return None
    tiles_per_axis = 2**z
    if x < 0 or y < 0 or x >= tiles_per_axis or y >= tiles_per_axis:
        return None
    source = _find_basemap(basemap_dir, basemap_id)
    if source is None or cache_dir is None:
        return None
    tile = cache_dir / basemap_id / str(z) / str(x) / f"{y}.png"
    meta = tile.with_suffix(".json")
    if tile.exists() and meta.exists():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            if payload.get("source_mtime") == source.stat().st_mtime:
                return tile
        except Exception:
            pass
    _build_tile(source, tile, meta, z=z, x=x, y=y)
    return tile


def tile_bounds_for_basemap(bounds: tuple[float, float, float, float], z: int, x: int, y: int) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bounds
    tiles_per_axis = 2**z
    lon_step = (max_lon - min_lon) / tiles_per_axis
    lat_step = (max_lat - min_lat) / tiles_per_axis
    left = min_lon + x * lon_step
    right = min_lon + (x + 1) * lon_step
    top = max_lat - y * lat_step
    bottom = max_lat - (y + 1) * lat_step
    return left, bottom, right, top


def _build_tile(source: Path, tile: Path, meta: Path, z: int, x: int, y: int) -> None:
    tile.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as dataset:
        bounds = dataset.bounds
        left, bottom, right, top = tile_bounds_for_basemap(
            (bounds.left, bounds.bottom, bounds.right, bounds.top),
            z,
            x,
            y,
        )
        window = dataset.window(left, bottom, right, top).round_offsets().round_lengths()
        full_window = Window(0, 0, dataset.width, dataset.height)
        window = _intersect_window(window, full_window)
        indexes = [1, 2, 3] if dataset.count >= 3 else [1]
        if window.width <= 0 or window.height <= 0:
            data = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        else:
            data = dataset.read(
                indexes,
                window=window,
                out_shape=(len(indexes), TILE_SIZE, TILE_SIZE),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            )
            if len(indexes) == 1:
                data = np.repeat(data, 3, axis=0)
            data = _to_uint8(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(
                tile,
                "w",
                driver="PNG",
                width=TILE_SIZE,
                height=TILE_SIZE,
                count=3,
                dtype="uint8",
            ) as output:
                output.write(data)
        meta.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_mtime": source.stat().st_mtime,
                    "z": z,
                    "x": x,
                    "y": y,
                    "bounds": [left, bottom, right, top],
                    "tile": str(tile),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _to_uint8(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return data
    arr = data.astype(np.float32, copy=False)
    output = np.zeros(arr.shape, dtype=np.uint8)
    for band_index in range(arr.shape[0]):
        band = arr[band_index]
        valid = np.isfinite(band)
        if not np.any(valid):
            continue
        low, high = np.percentile(band[valid], [2, 98])
        if high <= low:
            high = low + 1.0
        output[band_index] = np.clip((band - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return output


def _find_basemap(basemap_dir: Path | None, basemap_id: str) -> Path | None:
    if basemap_dir is None or not basemap_dir.exists():
        return None
    for path in [*basemap_dir.rglob("*.tif"), *basemap_dir.rglob("*.tiff")]:
        if _basemap_id(path) == basemap_id:
            return path
    return None


def _basemap_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _intersect_window(left: Window, right: Window) -> Window:
    col_off = max(left.col_off, right.col_off)
    row_off = max(left.row_off, right.row_off)
    col_end = min(left.col_off + left.width, right.col_off + right.width)
    row_end = min(left.row_off + left.height, right.row_off + right.height)
    return Window(col_off, row_off, max(0, col_end - col_off), max(0, row_end - row_off))
