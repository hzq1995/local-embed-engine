from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def create_test_geotiff(path: Path, *, band_count: int, zero_block: bool = False) -> None:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x0, y0 = transformer.transform(121.5400, 29.8720)
    x1, y1 = transformer.transform(121.5480, 29.8640)
    width = 4
    height = 4
    pixel_size_x = (x1 - x0) / width
    pixel_size_y = (y0 - y1) / height
    transform = from_origin(x0, y0, pixel_size_x, pixel_size_y)

    data = np.zeros((band_count, height, width), dtype=np.int16)
    for band in range(band_count):
        data[band, :, :] = band + 1
    data[:, 0, 0] = -9999
    if zero_block:
        data[:, 1, 1] = 0
        data[:, 1, 2] = 0

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=band_count,
        dtype="int16",
        crs="EPSG:32651",
        transform=transform,
        nodata=-9999,
    ) as ds:
        ds.write(data)


def run_script(script_name: str, tif_dir: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), str(tif_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report_name = {
        "diag_tif_overview.py": "tif_overview.json",
        "diag_tif_pixel_sampling.py": "tif_pixel_sampling.json",
        "diag_tif_pixel_sampling_parallel.py": "tif_pixel_sampling_parallel.json",
    }[script_name]
    payload = json.loads((tif_dir / "diagnostics" / report_name).read_text(encoding="utf-8"))
    payload["_stdout"] = completed.stdout
    return payload


class TifDiagnosticScriptsTests(unittest.TestCase):
    def test_diag_tif_overview_reports_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tif_dir = Path(temp_dir)
            create_test_geotiff(tif_dir / "a.tiff", band_count=64, zero_block=True)
            create_test_geotiff(tif_dir / "b.tiff", band_count=768, zero_block=False)

            report = run_script("diag_tif_overview.py", tif_dir)

            self.assertEqual(report["tile_count"], 2)
            self.assertEqual(report["band_count_distribution"], {"64": 1, "768": 1})
            self.assertEqual(len(report["tiles"]), 2)
            self.assertIsNotNone(report["overall_bbox_wgs84"])

    def test_diag_tif_pixel_sampling_reports_zero_and_nodata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tif_dir = Path(temp_dir)
            create_test_geotiff(tif_dir / "a.tiff", band_count=64, zero_block=True)
            create_test_geotiff(tif_dir / "b.tiff", band_count=64, zero_block=False)

            report = run_script("diag_tif_pixel_sampling.py", tif_dir)

            self.assertEqual(report["tile_count"], 2)
            self.assertGreater(report["sampled_pixel_total"], 0)
            self.assertGreater(report["nodata_pixel_total"], 0)
            self.assertGreater(report["all_zero_pixel_total"], 0)
            self.assertGreater(report["nonzero_valid_pixel_total"], 0)
            self.assertEqual(len(report["tiles"]), 2)

    def test_diag_tif_pixel_sampling_parallel_reports_each_tif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tif_dir = Path(temp_dir)
            create_test_geotiff(tif_dir / "a.tiff", band_count=64, zero_block=True)
            create_test_geotiff(tif_dir / "b.tiff", band_count=64, zero_block=False)

            report = run_script("diag_tif_pixel_sampling_parallel.py", tif_dir)

            self.assertEqual(report["tile_count"], 2)
            self.assertIn("parallel_workers", report)
            self.assertGreaterEqual(report["parallel_workers"], 1)
            self.assertIn("a.tiff", report["_stdout"])
            self.assertIn("b.tiff", report["_stdout"])
            self.assertGreater(report["all_zero_pixel_total"], 0)
            self.assertGreater(report["nodata_pixel_total"], 0)


if __name__ == "__main__":
    unittest.main()
