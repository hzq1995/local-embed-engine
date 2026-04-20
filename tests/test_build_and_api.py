from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from math import atan2, cos, radians, sin, sqrt
from pyproj import Transformer
from rasterio.transform import from_origin

from app.config import Settings
from app.main import create_app
from app.services.build_service import build_index


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad = radians(lon1)
    lat1_rad = radians(lat1)
    lon2_rad = radians(lon2)
    lat2_rad = radians(lat2)
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6_371_000 * c


def create_test_kml(path: Path) -> None:
    coordinates = [
        (121.5400, 29.8720),
        (121.5480, 29.8720),
        (121.5480, 29.8640),
        (121.5400, 29.8640),
        (121.5400, 29.8720),
    ]
    coord_text = " ".join(f"{lon},{lat},0" for lon, lat in coordinates)
    payload = f"""<?xml version="1.0" encoding="gb2312"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coord_text}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
"""
    path.write_text(payload, encoding="gb2312")


def create_test_geotiff(path: Path) -> None:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x0, y0 = transformer.transform(121.5400, 29.8720)
    x1, y1 = transformer.transform(121.5480, 29.8640)
    width = 4
    height = 4
    pixel_size_x = (x1 - x0) / width
    pixel_size_y = (y0 - y1) / height
    transform = from_origin(x0, y0, pixel_size_x, pixel_size_y)

    data = np.zeros((64, height, width), dtype=np.int8)
    for band in range(64):
        data[band, :, :] = band + 1
    data[:, 0, 0] = -128

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=64,
        dtype="int8",
        crs="EPSG:32651",
        transform=transform,
        nodata=-128,
    ) as ds:
        ds.write(data)


class BuildAndApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "tiles"
        self.data_dir.mkdir()
        self.boundary_kml = self.root / "宁波市.kml"
        self.derived_dir = self.root / "derived"
        create_test_kml(self.boundary_kml)
        create_test_geotiff(self.data_dir / "test.tiff")
        self.build_result = build_index(self.data_dir, self.boundary_kml, self.derived_dir, block_size=2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _settings(self) -> Settings:
        return Settings(
            service_name="Test Service",
            host="127.0.0.1",
            port=8010,
            year=2024,
            data_dir=self.data_dir,
            boundary_kml_path=self.boundary_kml,
            derived_dir=self.derived_dir,
            boundary_cache_path=self.derived_dir / "ningbo_boundary.geojson",
            metadata_path=self.derived_dir / "metadata.parquet",
            embeddings_path=self.derived_dir / "embeddings.npy",
            index_path=self.derived_dir / "faiss.index",
            build_info_path=self.derived_dir / "build_info.json",
        )

    def test_build_outputs_expected_artifacts(self) -> None:
        self.assertGreater(self.build_result.vector_count, 0)
        self.assertTrue((self.derived_dir / "metadata.parquet").exists())
        self.assertTrue((self.derived_dir / "embeddings.npy").exists())
        self.assertTrue((self.derived_dir / "build_info.json").exists())
        build_info = json.loads((self.derived_dir / "build_info.json").read_text(encoding="utf-8"))
        self.assertEqual(build_info["vector_count"], self.build_result.vector_count)
        self.assertEqual(build_info["tile_count"], self.build_result.tile_count)

    def test_api_endpoints(self) -> None:
        app = create_app(self._settings())
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["index_loaded"])

            point_response = client.post("/embedding/by-point", json={"lon": 121.546, "lat": 29.868})
            self.assertEqual(point_response.status_code, 200)
            point_payload = point_response.json()
            self.assertEqual(len(point_payload["embedding"]), 64)

            outside_response = client.post("/embedding/by-point", json={"lon": 121.60, "lat": 30.10})
            self.assertEqual(outside_response.status_code, 422)

            search_response = client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"],
                    "top_k": 3,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                },
            )
            self.assertEqual(search_response.status_code, 200)
            payload = search_response.json()
            self.assertGreaterEqual(payload["result_count"], 1)
            self.assertLessEqual(payload["result_count"], 3)
            self.assertEqual(len(payload["results"][0]["embedding"]), 64)

            filtered_response = client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"],
                    "top_k": 10,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                    "min_distance_m": 300,
                },
            )
            self.assertEqual(filtered_response.status_code, 200)
            filtered_payload = filtered_response.json()
            self.assertGreaterEqual(filtered_payload["result_count"], 1)
            self.assertLess(filtered_payload["result_count"], 10)
            filtered_results = filtered_payload["results"]
            for left_index, left in enumerate(filtered_results):
                for right in filtered_results[left_index + 1:]:
                    distance_m = haversine_distance_m(left["lon"], left["lat"], right["lon"], right["lat"])
                    self.assertGreaterEqual(distance_m, 300)

            invalid_embedding = client.post("/search/by-embedding", json={"embedding": [1, 2, 3]})
            self.assertEqual(invalid_embedding.status_code, 422)

            invalid_top_k = client.post(
                "/search/by-embedding",
                json={"embedding": point_payload["embedding"], "top_k": 1001},
            )
            self.assertEqual(invalid_top_k.status_code, 422)


if __name__ == "__main__":
    unittest.main()
