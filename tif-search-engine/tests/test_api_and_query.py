from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.transform import from_origin
from rasterio.warp import transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.main import create_app
from app.services.boundary_service import BoundaryService
from app.services.catalog_service import TileCatalog
from app.services.query_service import QueryService


def load_point_query_scene():
    scene_path = PROJECT_ROOT.parent / "local-embed-engine" / "scripts" / "point_query_scene.py"
    spec = importlib.util.spec_from_file_location("point_query_scene", scene_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_point_query_scene


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


def create_cluster_test_geotiff(path: Path, with_zero_pixel: bool = False) -> None:
    create_embedding_test_geotiff(path, dim=64, with_zero_pixel=with_zero_pixel)


def create_embedding_test_geotiff(path: Path, dim: int, with_zero_pixel: bool = False) -> None:
    _write_embedding_test_geotiff(path, dim=dim, with_zero_pixel=with_zero_pixel, positive_y_scale=False)


def create_positive_y_embedding_test_geotiff(path: Path, dim: int) -> None:
    _write_embedding_test_geotiff(path, dim=dim, with_zero_pixel=False, positive_y_scale=True)


def _write_embedding_test_geotiff(path: Path, dim: int, with_zero_pixel: bool, positive_y_scale: bool) -> None:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x0, y0 = transformer.transform(121.5400, 29.8720)
    x1, y1 = transformer.transform(121.5480, 29.8640)
    width = 8
    height = 8
    pixel_size_x = (x1 - x0) / width
    pixel_size_y = (y0 - y1) / height
    transform_value = from_origin(x0, y0, pixel_size_x, pixel_size_y)
    if positive_y_scale:
        transform_value = Affine(pixel_size_x, 0.0, x0, 0.0, pixel_size_y, y1)

    data = np.zeros((dim, height, width), dtype=np.int16)
    prototypes = []
    chunk = max(1, dim // 4)
    for prototype_index in range(4):
        base = np.full(dim, 10, dtype=np.int16)
        start = min(prototype_index * chunk, max(dim - 1, 0))
        end = min(dim, start + chunk)
        base[start:end] = 90
        prototypes.append(base)

    for row in range(height):
        for col in range(width):
            if row < 4 and col < 4:
                base = prototypes[0]
            elif row < 4 and col >= 4:
                base = prototypes[1]
            elif row >= 4 and col < 4:
                base = prototypes[2]
            else:
                base = prototypes[3]
            noise = ((row + col) % 3) - 1
            vector = np.clip(base + noise, -120, 120)
            data[:, row, col] = vector.astype(np.int16)

    data[:, 0, 0] = -9999
    if with_zero_pixel:
        data[:, 3, 3] = 0

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=dim,
        dtype="int16",
        crs="EPSG:32651",
        transform=transform_value,
        nodata=-9999,
    ) as ds:
        ds.write(data)


def pixel_center_lonlat(path: Path, row: int, col: int) -> tuple[float, float]:
    with rasterio.open(path) as ds:
        x, y = ds.xy(row, col, offset="center")
        lon, lat = transform(ds.crs, "EPSG:4326", [x], [y])
        return float(lon[0]), float(lat[0])


class TifSearchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "tiles"
        self.data_dir.mkdir()
        self.boundary_kml = self.root / "ningbo.kml"
        create_test_kml(self.boundary_kml)
        create_cluster_test_geotiff(self.data_dir / "cluster.tiff")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _settings(self) -> Settings:
        return Settings(
            service_name="Test TIF Search",
            host="127.0.0.1",
            port=8011,
            year=2024,
            data_dir=self.data_dir,
            boundary_kml_path=self.boundary_kml,
            search_block_size=4,
            max_bbox_area_km2=100.0,
            embedding_dim=64,
        )

    def _service(self) -> QueryService:
        settings = self._settings()
        boundary = BoundaryService.from_kml(settings.boundary_kml_path)
        catalog = TileCatalog.scan(settings.data_dir, expected_band_count=settings.embedding_dim)
        return QueryService(
            year=settings.year,
            boundary=boundary,
            catalog=catalog,
            search_block_size=settings.search_block_size,
            max_bbox_area_km2=settings.max_bbox_area_km2,
            embedding_dim=settings.embedding_dim,
        )

    def test_boundary_and_catalog_scan(self) -> None:
        boundary = BoundaryService.from_kml(self.boundary_kml)
        self.assertTrue(boundary.contains_point(121.544, 29.868))
        self.assertFalse(boundary.contains_point(121.60, 30.10))
        clipped = boundary.clip_bbox([121.539, 29.863, 121.545, 29.870])
        self.assertIsNotNone(clipped)
        catalog = TileCatalog.scan(self.data_dir, expected_band_count=64)
        hits = list(catalog.iter_intersecting_bbox([121.5400, 29.8640, 121.5480, 29.8720]))
        self.assertEqual(len(hits), 1)
        self.assertEqual(catalog.embedding_dim, 64)

    def test_point_query_service_validates_boundary_and_pixels(self) -> None:
        service = self._service()
        payload = service.get_embedding_by_point(121.546, 29.868)
        self.assertEqual(len(payload["embedding"]), 64)
        vector = np.asarray(payload["embedding"], dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

        with self.assertRaisesRegex(ValueError, "outside Ningbo boundary"):
            service.get_embedding_by_point(121.60, 30.10)

        zero_dir = self.root / "zero_tiles"
        zero_dir.mkdir()
        create_cluster_test_geotiff(zero_dir / "zero.tiff", with_zero_pixel=True)
        settings = self._settings()
        settings = Settings(
            service_name=settings.service_name,
            host=settings.host,
            port=settings.port,
            year=settings.year,
            data_dir=zero_dir,
            boundary_kml_path=settings.boundary_kml_path,
            search_block_size=settings.search_block_size,
            max_bbox_area_km2=settings.max_bbox_area_km2,
            embedding_dim=settings.embedding_dim,
        )
        boundary = BoundaryService.from_kml(settings.boundary_kml_path)
        catalog = TileCatalog.scan(settings.data_dir, expected_band_count=settings.embedding_dim)
        zero_service = QueryService(
            year=settings.year,
            boundary=boundary,
            catalog=catalog,
            search_block_size=settings.search_block_size,
            max_bbox_area_km2=settings.max_bbox_area_km2,
            embedding_dim=settings.embedding_dim,
        )
        zero_lon, zero_lat = pixel_center_lonlat(zero_dir / "zero.tiff", 3, 3)
        with self.assertRaisesRegex(ValueError, "all-zero embedding pixel"):
            zero_service.get_embedding_by_point(zero_lon, zero_lat)

    def test_api_endpoints_and_validation(self) -> None:
        app = create_app(self._settings())
        with TestClient(app) as client:
            point_response = client.post("/embedding/by-point", json={"lon": 121.546, "lat": 29.868})
            self.assertEqual(point_response.status_code, 200)
            point_payload = point_response.json()
            self.assertEqual(len(point_payload["embedding"]), 64)

            search_response = client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"],
                    "top_k": 3,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                },
            )
            self.assertEqual(search_response.status_code, 200)
            search_payload = search_response.json()
            self.assertGreaterEqual(search_payload["result_count"], 1)
            self.assertLessEqual(search_payload["result_count"], 3)
            self.assertEqual(len(search_payload["results"][0]["embedding"]), 64)
            self.assertGreaterEqual(search_payload["results"][0]["score"], search_payload["results"][-1]["score"])

            filtered_response = client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"],
                    "top_k": 10,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                    "min_distance_m": 300,
                    "min_score": 0.0,
                },
            )
            self.assertEqual(filtered_response.status_code, 200)
            filtered_payload = filtered_response.json()
            self.assertGreaterEqual(filtered_payload["result_count"], 1)
            self.assertLess(filtered_payload["result_count"], 10)

            missing_bbox = client.post(
                "/search/by-embedding",
                json={"embedding": point_payload["embedding"], "top_k": 3},
            )
            self.assertEqual(missing_bbox.status_code, 422)

            invalid_embedding = client.post(
                "/search/by-embedding",
                json={"embedding": [1, 2, 3], "top_k": 3, "bbox": [121.5400, 29.8640, 121.5480, 29.8720]},
            )
            self.assertEqual(invalid_embedding.status_code, 422)

            too_large_bbox = client.post(
                "/embedding/by-point",
                json={"lon": 121.546, "lat": 29.868},
            )
            self.assertEqual(too_large_bbox.status_code, 200)

        tight_settings = Settings(
            service_name="Tight Limit",
            host="127.0.0.1",
            port=8011,
            year=2024,
            data_dir=self.data_dir,
            boundary_kml_path=self.boundary_kml,
            search_block_size=4,
            max_bbox_area_km2=0.2,
            embedding_dim=64,
        )
        tight_app = create_app(tight_settings)
        with TestClient(tight_app) as tight_client:
            point_payload = tight_client.post("/embedding/by-point", json={"lon": 121.546, "lat": 29.868}).json()
            too_large_bbox = tight_client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"],
                    "top_k": 3,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                },
            )
            self.assertEqual(too_large_bbox.status_code, 422)

    def test_auto_detects_non_64_embedding_dim(self) -> None:
        data_dir = self.root / "tiles_768"
        data_dir.mkdir()
        create_embedding_test_geotiff(data_dir / "cluster_768.tiff", dim=768)

        settings = Settings(
            service_name="Auto Detect 768",
            host="127.0.0.1",
            port=8011,
            year=2024,
            data_dir=data_dir,
            boundary_kml_path=self.boundary_kml,
            search_block_size=4,
            max_bbox_area_km2=100.0,
            embedding_dim=None,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            point_response = client.post("/embedding/by-point", json={"lon": 121.546, "lat": 29.868})
            self.assertEqual(point_response.status_code, 200)
            point_payload = point_response.json()
            self.assertEqual(len(point_payload["embedding"]), 768)

            search_response = client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"],
                    "top_k": 3,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                },
            )
            self.assertEqual(search_response.status_code, 200)
            self.assertEqual(len(search_response.json()["results"][0]["embedding"]), 768)

            wrong_dim = client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"][:-1],
                    "top_k": 3,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                },
            )
            self.assertEqual(wrong_dim.status_code, 422)

    def test_search_handles_positive_y_scale_transform(self) -> None:
        data_dir = self.root / "tiles_positive_y"
        data_dir.mkdir()
        create_positive_y_embedding_test_geotiff(data_dir / "cluster_positive_y.tiff", dim=64)

        settings = Settings(
            service_name="Positive Y",
            host="127.0.0.1",
            port=8011,
            year=2024,
            data_dir=data_dir,
            boundary_kml_path=self.boundary_kml,
            search_block_size=4,
            max_bbox_area_km2=100.0,
            embedding_dim=None,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            point_response = client.post("/embedding/by-point", json={"lon": 121.546, "lat": 29.868})
            self.assertEqual(point_response.status_code, 200)
            point_payload = point_response.json()

            search_response = client.post(
                "/search/by-embedding",
                json={
                    "embedding": point_payload["embedding"],
                    "top_k": 5,
                    "bbox": [121.5400, 29.8640, 121.5480, 29.8720],
                    "min_distance_m": 50,
                    "min_score": 0.0,
                },
            )
            self.assertEqual(search_response.status_code, 200)
            payload = search_response.json()
            self.assertGreaterEqual(payload["result_count"], 1)

    def test_point_query_scene_compatibility(self) -> None:
        run_point_query_scene = load_point_query_scene()
        service = self._service()
        result = run_point_query_scene(
            [
                {"lon": 121.5425, "lat": 29.8695},
                {"lon": 121.5435, "lat": 29.8685},
            ],
            fetch_embedding_by_point=service.get_embedding_by_point,
            search_by_embedding=service.search_by_embedding,
            top_k=5,
            min_distance_m=0,
            min_score=0,
            search_radius_km=2,
            year=service.year,
        )
        self.assertEqual(result["scene"], "point-query")
        self.assertEqual(result["point_count"], 2)
        self.assertGreater(result["result_count"], 0)
        self.assertEqual(len(result["results"][0]["embedding"]), 64)

    def test_compare_with_local_embed_engine_on_small_bbox(self) -> None:
        service = self._service()
        point = service.get_embedding_by_point(121.546, 29.868)
        new_result = service.search_by_embedding(
            embedding=point["embedding"],
            top_k=5,
            bbox=[121.5400, 29.8640, 121.5480, 29.8720],
            min_distance_m=0,
            min_score=0.0,
        )

        derived_dir = self.root / "derived_old"
        code = textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            import sys

            root = Path({str((PROJECT_ROOT.parent / "local-embed-engine").resolve())!r})
            sys.path.insert(0, str(root))

            from app.services.boundary_service import BoundaryService
            from app.services.build_service import build_index
            from app.services.catalog_service import TileCatalog
            from app.services.index_service import IndexBundle
            from app.services.query_service import QueryService

            data_dir = Path({str(self.data_dir)!r})
            boundary_kml = Path({str(self.boundary_kml)!r})
            derived_dir = Path({str(derived_dir)!r})

            build_index(data_dir, boundary_kml, derived_dir, block_size=4)
            boundary = BoundaryService.from_kml(boundary_kml, cache_path=derived_dir / "ningbo_boundary.geojson")
            catalog = TileCatalog.scan(data_dir)
            bundle = IndexBundle.load(
                metadata_path=derived_dir / "metadata.parquet",
                embeddings_path=derived_dir / "embeddings.npy",
                build_info_path=derived_dir / "build_info.json",
                index_path=derived_dir / "faiss.index",
            )
            service = QueryService(year=2024, boundary=boundary, catalog=catalog, index_bundle=bundle)
            result = service.search_by_embedding(
                {json.dumps(point["embedding"])},
                5,
                [121.5400, 29.8640, 121.5480, 29.8720],
                0,
                0.0,
            )
            print(json.dumps(result))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        old_result = json.loads(completed.stdout.strip().splitlines()[-1])

        self.assertEqual(new_result["result_count"], old_result["result_count"])
        self.assertEqual(len(new_result["results"]), len(old_result["results"]))
        for new_item, old_item in zip(new_result["results"], old_result["results"]):
            self.assertAlmostEqual(new_item["lon"], old_item["lon"], places=6)
            self.assertAlmostEqual(new_item["lat"], old_item["lat"], places=6)
            self.assertAlmostEqual(new_item["score"], old_item["score"], places=5)


if __name__ == "__main__":
    unittest.main()
