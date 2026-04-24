from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import transform, transform_bounds

from app.services.vector_utils import normalize_embedding


@dataclass(slots=True)
class TileRecord:
    path: Path
    crs: str
    width: int
    height: int
    count: int
    nodata: float | None
    bounds_wgs84: tuple[float, float, float, float]


class TileCatalog:
    def __init__(self, tiles: List[TileRecord], embedding_dim: int):
        self.tiles = tiles
        self.embedding_dim = embedding_dim

    @classmethod
    def scan(cls, data_dir: Path, expected_band_count: int | None) -> "TileCatalog":
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        tiles: List[TileRecord] = []
        discovered_band_count: int | None = expected_band_count
        for path in sorted(data_dir.glob("*.tif*")):
            with rasterio.open(path) as ds:
                if discovered_band_count is None:
                    discovered_band_count = ds.count
                if ds.count != discovered_band_count:
                    raise ValueError(
                        f"Unexpected band count for {path}: expected {discovered_band_count}, got {ds.count}."
                    )
                bounds_wgs84 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
                tiles.append(
                    TileRecord(
                        path=path,
                        crs=str(ds.crs),
                        width=ds.width,
                        height=ds.height,
                        count=ds.count,
                        nodata=None if ds.nodata is None else float(ds.nodata),
                        bounds_wgs84=tuple(float(value) for value in bounds_wgs84),
                    )
                )
        if not tiles:
            raise ValueError(f"No tif files found in {data_dir}")
        if discovered_band_count is None:
            raise ValueError(f"Unable to determine embedding dimension from {data_dir}")
        return cls(tiles, embedding_dim=discovered_band_count)

    def iter_intersecting_bbox(self, bbox_values: Iterable[float]) -> Iterator[TileRecord]:
        min_lon, min_lat, max_lon, max_lat = bbox_values
        for tile in self.tiles:
            tmin_lon, tmin_lat, tmax_lon, tmax_lat = tile.bounds_wgs84
            if not (tmax_lon < min_lon or tmax_lat < min_lat or tmin_lon > max_lon or tmin_lat > max_lat):
                yield tile

    def locate_tiles(self, lon: float, lat: float) -> Iterator[TileRecord]:
        yield from self.iter_intersecting_bbox([lon, lat, lon, lat])

    def fetch_embedding_for_point(self, tile: TileRecord, lon: float, lat: float) -> dict:
        with rasterio.open(tile.path) as ds:
            xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
            row, col = ds.index(xs[0], ys[0])
            if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
                raise ValueError("Point is outside the tile extent.")
            values = ds.read(window=((row, row + 1), (col, col + 1)))[:, 0, 0].astype(np.float32)
            nodata = ds.nodata
            if nodata is not None and np.any(values == nodata):
                raise ValueError("Point falls on an invalid or nodata pixel.")
            if np.all(values == 0):
                raise ValueError("Point falls on an all-zero embedding pixel.")
            embedding = normalize_embedding(values).tolist()
            return {
                "tile_path": str(tile.path),
                "row": int(row),
                "col": int(col),
                "embedding": embedding,
            }

    def fetch_embedding_for_pixel(self, tile_path: str, row: int, col: int) -> list[float]:
        with rasterio.open(tile_path) as ds:
            values = ds.read(window=((row, row + 1), (col, col + 1)))[:, 0, 0].astype(np.float32)
            nodata = ds.nodata
            if nodata is not None and np.any(values == nodata):
                raise ValueError("Selected result falls on an invalid or nodata pixel.")
            if np.all(values == 0):
                raise ValueError("Selected result falls on an all-zero embedding pixel.")
            return normalize_embedding(values).astype(np.float32).tolist()

    @staticmethod
    def transform_pixel_centers_to_wgs84(ds: rasterio.DatasetReader, rows, cols):
        xs = ds.transform.c + (cols + 0.5) * ds.transform.a + (rows + 0.5) * ds.transform.b
        ys = ds.transform.f + (cols + 0.5) * ds.transform.d + (rows + 0.5) * ds.transform.e
        transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)
        return np.asarray(lons), np.asarray(lats)
