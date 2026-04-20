from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import transform, transform_bounds


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
    def __init__(self, tiles: List[TileRecord]):
        self.tiles = tiles

    @classmethod
    def scan(cls, data_dir: Path) -> "TileCatalog":
        tiles: List[TileRecord] = []
        for path in sorted(data_dir.glob("*.tif*")):
            with rasterio.open(path) as ds:
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
        return cls(tiles)

    def tile_count(self) -> int:
        return len(self.tiles)

    def iter_intersecting_bbox(self, bbox_values: Iterable[float]):
        min_lon, min_lat, max_lon, max_lat = bbox_values
        for tile in self.tiles:
            tmin_lon, tmin_lat, tmax_lon, tmax_lat = tile.bounds_wgs84
            if not (tmax_lon < min_lon or tmax_lat < min_lat or tmin_lon > max_lon or tmin_lat > max_lat):
                yield tile

    def locate_tile(self, lon: float, lat: float) -> Optional[TileRecord]:
        for tile in self.iter_intersecting_bbox([lon, lat, lon, lat]):
            return tile
        return None

    def fetch_embedding_for_point(self, tile: TileRecord, lon: float, lat: float) -> dict:
        with rasterio.open(tile.path) as ds:
            xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
            row, col = ds.index(xs[0], ys[0])
            if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
                raise ValueError("Point is outside the tile extent.")
            values = ds.read(window=((row, row + 1), (col, col + 1)))[:, 0, 0]
            nodata = ds.nodata
            if nodata is not None and np.any(values == nodata):
                raise ValueError("Point falls on an invalid or nodata pixel.")
            return {
                "tile_path": str(tile.path),
                "row": int(row),
                "col": int(col),
                "embedding": values.astype(np.float32).tolist(),
            }

    def transform_pixel_centers_to_wgs84(self, ds: rasterio.DatasetReader, rows, cols):
        xs = ds.transform.c + (cols + 0.5) * ds.transform.a + (rows + 0.5) * ds.transform.b
        ys = ds.transform.f + (cols + 0.5) * ds.transform.d + (rows + 0.5) * ds.transform.e
        transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)
        return np.asarray(lons), np.asarray(lats)
