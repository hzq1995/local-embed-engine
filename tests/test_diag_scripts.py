from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def write_metadata(path: Path, records: list[dict]) -> None:
    table = pa.table(
        {
            "id": [int(item["id"]) for item in records],
            "lon": [float(item["lon"]) for item in records],
            "lat": [float(item["lat"]) for item in records],
            "tile_path": [str(item["tile_path"]) for item in records],
            "row": [int(item["row"]) for item in records],
            "col": [int(item["col"]) for item in records],
        }
    )
    pq.write_table(table, path)


def write_embeddings(path: Path, embeddings: np.ndarray) -> None:
    np.save(path, embeddings.astype(np.float32))


def write_build_info(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_script(script_name: str, derived_dir: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), str(derived_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report_name = {
        "diag_derived_overview.py": "derived_overview.json",
        "diag_point_query_sampling.py": "point_query_sampling.json",
        "diag_bbox_sampling_scan.py": "bbox_sampling_scan.json",
        "plot_metadata_distribution.py": "metadata_distribution.json",
    }[script_name]
    return json.loads((derived_dir / "diagnostics" / report_name).read_text(encoding="utf-8"))


class DiagnosticScriptsTests(unittest.TestCase):
    def test_diag_derived_overview_reports_mismatch_and_build_info(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            derived_dir = Path(temp_dir)
            records = [
                {"id": 0, "lon": 0.0, "lat": 0.0, "tile_path": "tile_a.tif", "row": 0, "col": 0},
                {"id": 1, "lon": 0.005, "lat": 0.005, "tile_path": "tile_a.tif", "row": 0, "col": 1},
                {"id": 2, "lon": 5.0, "lat": 5.0, "tile_path": "tile_b.tif", "row": 1, "col": 0},
                {"id": 3, "lon": 5.005, "lat": 5.005, "tile_path": "tile_b.tif", "row": 1, "col": 1},
                {"id": 4, "lon": 10.0, "lat": 10.0, "tile_path": "tile_c.tif", "row": 2, "col": 0},
            ]
            embeddings = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            write_metadata(derived_dir / "metadata.parquet", records)
            write_embeddings(derived_dir / "embeddings.npy", embeddings)
            write_build_info(
                derived_dir / "build_info.json",
                {
                    "vector_count": 5,
                    "tile_count": 3,
                    "index_type": "numpy_exact",
                    "data_dir": "/tmp/data",
                    "build_time": "2026-01-01T00:00:00Z",
                },
            )

            report = run_script("diag_derived_overview.py", derived_dir)

            self.assertTrue(report["build_info_present"])
            self.assertEqual(report["metadata_row_count"], 5)
            self.assertEqual(report["embedding_count"], 4)
            self.assertFalse(report["row_count_match"])
            self.assertEqual(report["invalid_id_count"], 1)
            self.assertAlmostEqual(report["embedding_sample"]["sample_zero_ratio"], 0.5)
            self.assertTrue(report["build_info_comparison"]["vector_count_matches_metadata"])
            self.assertFalse(report["build_info_comparison"]["vector_count_matches_embeddings"])

    def test_diag_point_query_sampling_runs_without_build_info(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            derived_dir = Path(temp_dir)
            records = [
                {"id": 0, "lon": 0.0, "lat": 0.0, "tile_path": "tile_a.tif", "row": 0, "col": 0},
                {"id": 1, "lon": 0.005, "lat": 0.005, "tile_path": "tile_a.tif", "row": 0, "col": 1},
                {"id": 2, "lon": 10.0, "lat": 10.0, "tile_path": "tile_b.tif", "row": 1, "col": 0},
                {"id": 3, "lon": 10.005, "lat": 10.005, "tile_path": "tile_b.tif", "row": 1, "col": 1},
                {"id": 4, "lon": 20.0, "lat": 20.0, "tile_path": "tile_c.tif", "row": 2, "col": 0},
                {"id": 5, "lon": 20.005, "lat": 20.005, "tile_path": "tile_c.tif", "row": 2, "col": 1},
            ]
            embeddings = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            write_metadata(derived_dir / "metadata.parquet", records)
            write_embeddings(derived_dir / "embeddings.npy", embeddings)

            report = run_script("diag_point_query_sampling.py", derived_dir)

            self.assertFalse(report["build_info_present"])
            self.assertEqual(report["metadata_sample_point_count"], 6)
            self.assertEqual(report["grid_point_count"], 25)
            self.assertGreaterEqual(report["local_hit_count"], 6)
            self.assertGreater(report["global_fallback_count"], 0)
            self.assertGreaterEqual(report["zero_return_count"], 2)
            self.assertEqual(report["metadata_self_hit_ratio"], 1.0)
            self.assertTrue(any(item["used_global_fallback"] for item in report["query_results"]))

    def test_diag_bbox_sampling_scan_reports_zero_and_mixed_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            derived_dir = Path(temp_dir)
            records = [
                {"id": 0, "lon": 0.10, "lat": 0.10, "tile_path": "tile_a.tif", "row": 0, "col": 0},
                {"id": 1, "lon": 0.35, "lat": 0.35, "tile_path": "tile_a.tif", "row": 0, "col": 1},
                {"id": 2, "lon": 0.60, "lat": 0.60, "tile_path": "tile_a.tif", "row": 0, "col": 2},
                {"id": 3, "lon": 5.00, "lat": 5.00, "tile_path": "tile_b.tif", "row": 1, "col": 0},
                {"id": 4, "lon": 5.30, "lat": 5.30, "tile_path": "tile_b.tif", "row": 1, "col": 1},
                {"id": 5, "lon": 5.60, "lat": 5.60, "tile_path": "tile_b.tif", "row": 1, "col": 2},
                {"id": 6, "lon": 10.00, "lat": 10.00, "tile_path": "tile_c.tif", "row": 2, "col": 0},
                {"id": 7, "lon": 10.01, "lat": 10.01, "tile_path": "tile_c.tif", "row": 2, "col": 1},
                {"id": 8, "lon": 10.02, "lat": 10.02, "tile_path": "tile_c.tif", "row": 2, "col": 2},
            ]
            embeddings = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            write_metadata(derived_dir / "metadata.parquet", records)
            write_embeddings(derived_dir / "embeddings.npy", embeddings)
            write_build_info(
                derived_dir / "build_info.json",
                {
                    "vector_count": 9,
                    "tile_count": 3,
                    "index_type": "numpy_exact",
                    "data_dir": "/tmp/data",
                    "build_time": "2026-01-01T00:00:00Z",
                },
            )

            report = run_script("diag_bbox_sampling_scan.py", derived_dir)

            self.assertTrue(report["build_info_present"])
            self.assertEqual(report["bbox_count"], 36)
            classifications = {
                item["classification"]
                for item in report["bbox_reports"]
                if item["candidate_count"] > 0
            }
            self.assertIn("all_zero", classifications)
            self.assertIn("mixed", classifications)
            self.assertTrue(
                any(
                    item["classification"] == "all_zero" and item["selected_zero_ratio"] == 1.0
                    for item in report["bbox_reports"]
                )
            )

    def test_plot_metadata_distribution_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            derived_dir = Path(temp_dir)
            records = [
                {"id": 0, "lon": 121.0, "lat": 29.0, "tile_path": "tile_a.tif", "row": 0, "col": 0},
                {"id": 1, "lon": 121.2, "lat": 29.1, "tile_path": "tile_a.tif", "row": 0, "col": 1},
                {"id": 2, "lon": 121.4, "lat": 29.2, "tile_path": "tile_b.tif", "row": 1, "col": 0},
                {"id": 3, "lon": 121.6, "lat": 29.4, "tile_path": "tile_b.tif", "row": 1, "col": 1},
            ]
            embeddings = np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            )
            write_metadata(derived_dir / "metadata.parquet", records)
            write_embeddings(derived_dir / "embeddings.npy", embeddings)

            report = run_script("plot_metadata_distribution.py", derived_dir)

            image_path = derived_dir / "diagnostics" / "metadata_distribution.png"
            self.assertEqual(report["metadata_row_count"], 4)
            self.assertTrue(image_path.exists())
            self.assertGreater(image_path.stat().st_size, 0)
            self.assertEqual(report["output_image_path"], str(image_path))


if __name__ == "__main__":
    unittest.main()
