from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from shapely import contains_xy
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping, shape
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
    def from_kml(cls, kml_path: Path, cache_path: Path | None = None) -> "BoundaryService":
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return cls(geometry=shape(payload), source_path=kml_path)

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

        service = cls(geometry=geometry, source_path=kml_path)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(mapping(service.geometry)), encoding="utf-8")
        return service

    @property
    def bbox(self) -> list[float]:
        min_lon, min_lat, max_lon, max_lat = self.geometry.bounds
        return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]

    def contains_point(self, lon: float, lat: float) -> bool:
        return bool(self.geometry.contains(Point(lon, lat)) or self.geometry.touches(Point(lon, lat)))

    def contains_xy(self, lons: Sequence[float], lats: Sequence[float]):
        try:
            return contains_xy(self.geometry, lons, lats)
        except Exception:
            return [self.contains_point(lon, lat) for lon, lat in zip(lons, lats)]

    def intersects_bbox(self, bbox_values: Iterable[float]) -> bool:
        min_lon, min_lat, max_lon, max_lat = bbox_values
        return self.geometry.intersects(box(min_lon, min_lat, max_lon, max_lat))

    def clip_bbox(self, bbox_values: Iterable[float]) -> list[float] | None:
        min_lon, min_lat, max_lon, max_lat = bbox_values
        clipped = self.geometry.intersection(box(min_lon, min_lat, max_lon, max_lat))
        if clipped.is_empty:
            return None
        return [float(value) for value in clipped.bounds]
