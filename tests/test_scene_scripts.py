from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from app.config import Settings
from app.services.boundary_service import BoundaryService
from app.services.build_service import build_index
from app.services.catalog_service import TileCatalog
from app.services.index_service import IndexBundle
from app.services.query_service import QueryService
from scripts.point_query_scene import run_point_query_scene
from scripts.region_cluster_scene import run_region_cluster_scene


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


def create_cluster_test_geotiff(path: Path) -> None:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x0, y0 = transformer.transform(121.5400, 29.8720)
    x1, y1 = transformer.transform(121.5480, 29.8640)
    width = 8
    height = 8
    pixel_size_x = (x1 - x0) / width
    pixel_size_y = (y0 - y1) / height
    transform = from_origin(x0, y0, pixel_size_x, pixel_size_y)

    data = np.zeros((64, height, width), dtype=np.int8)
    prototypes = [
        np.concatenate([np.full(16, 90), np.full(48, 10)]),
        np.concatenate([np.full(16, 10), np.full(16, 90), np.full(32, 10)]),
        np.concatenate([np.full(32, 10), np.full(16, 90), np.full(16, 10)]),
        np.concatenate([np.full(48, 10), np.full(16, 90)]),
    ]

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
            data[:, row, col] = vector.astype(np.int8)

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


class SceneScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "tiles"
        self.data_dir.mkdir()
        self.boundary_kml = self.root / "宁波市.kml"
        self.derived_dir = self.root / "derived"
        create_test_kml(self.boundary_kml)
        create_cluster_test_geotiff(self.data_dir / "cluster.tiff")
        build_index(self.data_dir, self.boundary_kml, self.derived_dir, block_size=4)
        self.service = self._create_service()

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

    def _create_service(self) -> QueryService:
        settings = self._settings()
        boundary = BoundaryService.from_kml(settings.boundary_kml_path, cache_path=settings.boundary_cache_path)
        catalog = TileCatalog.scan(settings.data_dir)
        index_bundle = IndexBundle.load(
            metadata_path=settings.metadata_path,
            embeddings_path=settings.embeddings_path,
            build_info_path=settings.build_info_path,
            index_path=settings.index_path,
        )
        return QueryService(
            year=settings.year,
            boundary=boundary,
            catalog=catalog,
            index_bundle=index_bundle,
        )

    def test_point_query_scene_runs_end_to_end(self) -> None:
        result = run_point_query_scene(
            [
                {"lon": 121.5425, "lat": 29.8695},
                {"lon": 121.5435, "lat": 29.8685},
            ],
            fetch_embedding_by_point=self.service.get_embedding_by_point,
            search_by_embedding=self.service.search_by_embedding,
            top_k=5,
            min_distance_m=0,
            min_score=0,
            search_radius_km=2,
            year=self.service.year,
        )

        self.assertEqual(result["scene"], "point-query")
        self.assertEqual(result["point_count"], 2)
        self.assertEqual(len(result["avg_embedding"]), 64)
        self.assertEqual(result["top_k"], 5)
        self.assertGreater(result["result_count"], 0)
        self.assertEqual(len(result["query_points"]), 2)
        self.assertEqual(len(result["results"][0]["embedding"]), 64)

    def test_region_cluster_scene_selects_adaptive_k(self) -> None:
        result = run_region_cluster_scene(
            [121.5400, 29.8640, 121.5480, 29.8720],
            get_embeddings_by_bbox=self.service.get_embeddings_by_bbox,
            total_samples=64,
            cluster_budget=64,
            year=self.service.year,
        )

        self.assertEqual(result["scene"], "region-cluster")
        self.assertGreater(result["count"], 0)
        self.assertIn(result["selected_k"], [3, 4, 5, 6])
        self.assertEqual(len(result["labels"]), result["count"])
        self.assertEqual(len(result["centroids"]), result["selected_k"])
        self.assertEqual(result["cluster_sample_size"], result["count"])
        self.assertIn(str(result["selected_k"]), result["k_scores"])

    def test_region_cluster_scene_falls_back_to_single_cluster_for_tiny_sample(self) -> None:
        result = run_region_cluster_scene(
            [121.5400, 29.8640, 121.5410, 29.8650],
            get_embeddings_by_bbox=self.service.get_embeddings_by_bbox,
            total_samples=1,
            cluster_budget=10,
            year=self.service.year,
        )

        self.assertGreaterEqual(result["count"], 0)
        if result["count"] <= 1:
            self.assertEqual(result["selected_k"], 1 if result["count"] == 1 else 0)


if __name__ == "__main__":
    unittest.main()
