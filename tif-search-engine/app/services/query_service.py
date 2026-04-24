from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform

from app.services.boundary_service import BoundaryService
from app.services.catalog_service import TileCatalog
from app.services.vector_utils import normalize_embedding


def _iter_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row_off in range(0, height, block_size):
        for col_off in range(0, width, block_size):
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=min(block_size, width - col_off),
                height=min(block_size, height - row_off),
            )


def _window_intersection(left: Window, right: Window) -> Window | None:
    row_off = max(int(left.row_off), int(right.row_off))
    col_off = max(int(left.col_off), int(right.col_off))
    row_end = min(int(left.row_off + left.height), int(right.row_off + right.height))
    col_end = min(int(left.col_off + left.width), int(right.col_off + right.width))
    if row_off >= row_end or col_off >= col_end:
        return None
    return Window(col_off=col_off, row_off=row_off, width=col_end - col_off, height=row_end - row_off)


def _expand_window(window: Window) -> Window:
    row_off = math.floor(float(window.row_off))
    col_off = math.floor(float(window.col_off))
    row_end = math.ceil(float(window.row_off + window.height))
    col_end = math.ceil(float(window.col_off + window.width))
    return Window(col_off=col_off, row_off=row_off, width=col_end - col_off, height=row_end - row_off)


def _window_from_lonlat_bbox(ds: rasterio.DatasetReader, bbox: list[float], densify_steps: int = 8) -> Window | None:
    min_lon, min_lat, max_lon, max_lat = bbox
    samples = np.linspace(0.0, 1.0, num=max(densify_steps, 2), dtype=np.float64)

    lons: list[float] = []
    lats: list[float] = []
    for ratio in samples:
        lon = min_lon + (max_lon - min_lon) * float(ratio)
        lat = min_lat + (max_lat - min_lat) * float(ratio)
        lons.extend([lon, lon, min_lon, max_lon])
        lats.extend([min_lat, max_lat, lat, lat])

    xs, ys = transform("EPSG:4326", ds.crs, lons, lats)
    rows: list[int] = []
    cols: list[int] = []
    for x, y in zip(xs, ys):
        row, col = ds.index(x, y)
        rows.append(int(row))
        cols.append(int(col))

    if not rows or not cols:
        return None

    row_off = max(0, min(rows))
    col_off = max(0, min(cols))
    row_end = min(ds.height, max(rows) + 1)
    col_end = min(ds.width, max(cols) + 1)
    if row_off >= row_end or col_off >= col_end:
        return None

    # Expand by one pixel to avoid losing boundary pixels due to rounding.
    row_off = max(0, row_off - 1)
    col_off = max(0, col_off - 1)
    row_end = min(ds.height, row_end + 1)
    col_end = min(ds.width, col_end + 1)
    return Window(col_off=col_off, row_off=row_off, width=col_end - col_off, height=row_end - row_off)


def _haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1 - a, 0.0)))
    return 6_371_000 * c


@dataclass(slots=True)
class QueryService:
    year: int
    boundary: BoundaryService
    catalog: TileCatalog
    search_block_size: int
    max_bbox_area_km2: float
    embedding_dim: int

    def get_embedding_by_point(self, lon: float, lat: float) -> dict:
        if not self.boundary.contains_point(lon, lat):
            raise ValueError("Point is outside Ningbo boundary.")
        last_error: ValueError | None = None
        for tile in self.catalog.locate_tiles(lon, lat):
            try:
                result = self.catalog.fetch_embedding_for_point(tile, lon, lat)
            except ValueError as exc:
                last_error = exc
                continue
            return {"year": self.year, "lon": lon, "lat": lat, **result}
        if last_error is not None:
            raise last_error
        raise ValueError("Point is not covered by any valid tif pixel.")

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int,
        bbox: list[float],
        min_distance_m: float,
        min_score: float = 0.0,
    ) -> dict:
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"embedding must contain exactly {self.embedding_dim} values.")

        effective_bbox = self.boundary.clip_bbox(bbox)
        if effective_bbox is None:
            return {"top_k": top_k, "result_count": 0, "results": []}

        bbox_area_km2 = self.boundary.bbox_area_km2(effective_bbox)
        if bbox_area_km2 > self.max_bbox_area_km2:
            raise ValueError(
                f"bbox area {bbox_area_km2:.2f} km^2 exceeds limit {self.max_bbox_area_km2:.2f} km^2."
            )

        query = normalize_embedding(np.asarray(embedding, dtype=np.float32))
        if min_distance_m > 0:
            matches = self._search_with_distance_filter(query, top_k, effective_bbox, min_score, min_distance_m)
        else:
            matches = self._search_top_k(query, top_k, effective_bbox, min_score)
        return {"top_k": top_k, "result_count": len(matches), "results": matches}

    def _search_top_k(
        self,
        query: np.ndarray,
        top_k: int,
        bbox: list[float],
        min_score: float,
    ) -> list[dict]:
        scored: list[tuple[float, int, float, float, str, int, int]] = []
        for counter, candidate in enumerate(self._iter_bbox_candidates(bbox), start=1):
            score = float(candidate["embedding"] @ query)
            if score < min_score:
                continue
            scored.append(
                (
                    score,
                    counter,
                    float(candidate["lon"]),
                    float(candidate["lat"]),
                    str(candidate["tile_path"]),
                    int(candidate["row"]),
                    int(candidate["col"]),
                )
            )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        results: list[dict] = []
        for rank, (score, _counter, lon, lat, tile_path, row, col) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "lon": lon,
                    "lat": lat,
                    "embedding": self.catalog.fetch_embedding_for_pixel(tile_path, row, col),
                    "tile_path": tile_path,
                    "row": row,
                    "col": col,
                }
            )
        return results

    def _search_with_distance_filter(
        self,
        query: np.ndarray,
        top_k: int,
        bbox: list[float],
        min_score: float,
        min_distance_m: float,
    ) -> list[dict]:
        scored: list[tuple[float, int, float, float, str, int, int]] = []
        for counter, candidate in enumerate(self._iter_bbox_candidates(bbox), start=1):
            score = float(candidate["embedding"] @ query)
            if score < min_score:
                continue
            scored.append(
                (
                    score,
                    counter,
                    float(candidate["lon"]),
                    float(candidate["lat"]),
                    str(candidate["tile_path"]),
                    int(candidate["row"]),
                    int(candidate["col"]),
                )
            )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        results: list[dict] = []
        kept_points: list[tuple[float, float]] = []
        for score, _counter, lon, lat, tile_path, row, col in scored:
            if any(_haversine_distance_m(lon, lat, kept_lon, kept_lat) < min_distance_m for kept_lon, kept_lat in kept_points):
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "lon": lon,
                    "lat": lat,
                    "embedding": self.catalog.fetch_embedding_for_pixel(tile_path, row, col),
                    "tile_path": tile_path,
                    "row": row,
                    "col": col,
                }
            )
            kept_points.append((lon, lat))
            if len(results) >= top_k:
                break
        return results

    def _iter_bbox_candidates(self, bbox: list[float]):
        min_lon, min_lat, max_lon, max_lat = bbox
        for tile in self.catalog.iter_intersecting_bbox(bbox):
            with rasterio.open(tile.path) as ds:
                read_window = _window_from_lonlat_bbox(ds, bbox)
                if read_window is None:
                    continue
                clipped = _window_intersection(
                    Window(col_off=0, row_off=0, width=ds.width, height=ds.height),
                    _expand_window(read_window),
                )
                if clipped is None:
                    continue

                for base_window in _iter_windows(int(clipped.width), int(clipped.height), self.search_block_size):
                    window = Window(
                        col_off=int(clipped.col_off) + int(base_window.col_off),
                        row_off=int(clipped.row_off) + int(base_window.row_off),
                        width=int(base_window.width),
                        height=int(base_window.height),
                    )
                    block = ds.read(window=window).astype(np.float32, copy=False)
                    nodata = ds.nodata
                    valid_mask = np.ones((int(window.height), int(window.width)), dtype=bool)
                    if nodata is not None:
                        valid_mask &= np.all(block != nodata, axis=0)
                    if not np.any(valid_mask):
                        continue

                    local_rows, local_cols = np.where(valid_mask)
                    rows = local_rows + int(window.row_off)
                    cols = local_cols + int(window.col_off)
                    lons, lats = self.catalog.transform_pixel_centers_to_wgs84(ds, rows, cols)
                    bbox_mask = (
                        (lons >= min_lon)
                        & (lons <= max_lon)
                        & (lats >= min_lat)
                        & (lats <= max_lat)
                    )
                    if not np.any(bbox_mask):
                        continue
                    inside_mask = np.asarray(self.boundary.contains_xy(lons[bbox_mask], lats[bbox_mask]), dtype=bool)
                    if not np.any(inside_mask):
                        continue

                    rows = rows[bbox_mask][inside_mask]
                    cols = cols[bbox_mask][inside_mask]
                    lons = lons[bbox_mask][inside_mask]
                    lats = lats[bbox_mask][inside_mask]
                    local_rows = rows - int(window.row_off)
                    local_cols = cols - int(window.col_off)
                    raw_vectors = block[:, local_rows, local_cols].T.astype(np.float32, copy=False)
                    norms = np.linalg.norm(raw_vectors, axis=1)
                    nonzero_mask = norms > 0
                    if not np.any(nonzero_mask):
                        continue

                    rows = rows[nonzero_mask]
                    cols = cols[nonzero_mask]
                    lons = lons[nonzero_mask]
                    lats = lats[nonzero_mask]
                    raw_vectors = raw_vectors[nonzero_mask]
                    norms = norms[nonzero_mask]
                    normalized_vectors = raw_vectors / norms[:, None]

                    for index in range(len(rows)):
                        yield {
                            "lon": float(lons[index]),
                            "lat": float(lats[index]),
                            "tile_path": str(tile.path),
                            "row": int(rows[index]),
                            "col": int(cols[index]),
                            "embedding": normalized_vectors[index],
                        }
