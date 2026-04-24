from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from shapely import contains_xy
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union


def _extract_text(element: ET.Element, suffix: str) -> str | None:
    for candidate in element.iter():
        if candidate.tag.endswith(suffix):
            return candidate.text
    return None


def _parse_ring(coordinates_text: str) -> List[tuple[float, float]]:
    coordinates: List[tuple[float, float]] = []
    for part in coordinates_text.replace("\n", " ").split():
        lon, lat, *_rest = part.split(",")
        coordinates.append((float(lon), float(lat)))
    if coordinates and coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    return coordinates


@dataclass(slots=True)
class BoundaryService:
    geometry: MultiPolygon
    source_path: Path

    @classmethod
    def from_kml(cls, kml_path: Path) -> "BoundaryService":
        root = ET.fromstring(kml_path.read_text(encoding="gb2312", errors="ignore"))
        polygons: List[Polygon] = []

        for placemark in root.iter():
            if not placemark.tag.endswith("Placemark"):
                continue
            coordinates_text = _extract_text(placemark, "coordinates")
            if not coordinates_text:
                continue
            ring = _parse_ring(coordinates_text)
            if len(ring) < 4:
                continue
            polygon = Polygon(ring)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            if isinstance(polygon, Polygon):
                polygons.append(polygon)
            elif isinstance(polygon, MultiPolygon):
                polygons.extend(list(polygon.geoms))

        if not polygons:
            raise ValueError(f"No polygons found in KML: {kml_path}")

        merged = unary_union(polygons)
        if isinstance(merged, Polygon):
            geometry = MultiPolygon([merged])
        else:
            geometry = MultiPolygon(list(merged.geoms))
        return cls(geometry=geometry, source_path=kml_path)

    @property
    def bbox(self) -> list[float]:
        min_lon, min_lat, max_lon, max_lat = self.geometry.bounds
        return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]

    def contains_point(self, lon: float, lat: float) -> bool:
        point = Point(lon, lat)
        return bool(self.geometry.contains(point) or self.geometry.touches(point))

    def contains_xy(self, lons: Sequence[float], lats: Sequence[float]):
        try:
            return contains_xy(self.geometry, lons, lats)
        except Exception:
            return [self.contains_point(lon, lat) for lon, lat in zip(lons, lats)]

    def clip_bbox(self, bbox_values: Iterable[float]) -> list[float] | None:
        min_lon, min_lat, max_lon, max_lat = bbox_values
        clipped = self.geometry.intersection(box(min_lon, min_lat, max_lon, max_lat))
        if clipped.is_empty:
            return None
        return [float(value) for value in clipped.bounds]

    @staticmethod
    def bbox_area_km2(bbox_values: Iterable[float]) -> float:
        min_lon, min_lat, max_lon, max_lat = bbox_values
        center_lat = (min_lat + max_lat) / 2.0
        lat_m = max(max_lat - min_lat, 0.0) * 111_320.0
        lon_m = max(max_lon - min_lon, 0.0) * 111_320.0 * max(math.cos(math.radians(center_lat)), 1e-9)
        return (lat_m * lon_m) / 1_000_000.0
