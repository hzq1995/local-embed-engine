from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


GRID_SIDE = 16
MAX_WORKERS = 16
OUTPUT_NAME = "tif_pixel_sampling_parallel.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample raw GeoTIFF pixels in parallel for zero/nodata diagnostics.")
    parser.add_argument("tif_dir", type=Path)
    return parser.parse_args()


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {path}")
    return path


def sample_positions(length: int, side: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    count = min(length, side)
    if count == length:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=count, dtype=np.int64))


def inspect_tif(tif_path_str: str) -> dict:
    tif_path = Path(tif_path_str)
    with rasterio.open(tif_path) as ds:
        row_positions = sample_positions(int(ds.height), GRID_SIDE)
        col_positions = sample_positions(int(ds.width), GRID_SIDE)
        nodata = ds.nodata

        sampled_count = 0
        nodata_pixel_count = 0
        all_zero_pixel_count = 0
        nonzero_valid_pixel_count = 0
        examples: list[dict] = []

        for row in row_positions.tolist():
            for col in col_positions.tolist():
                values = ds.read(window=Window(col, row, 1, 1))[:, 0, 0].astype(np.float32, copy=False)
                sampled_count += 1
                is_nodata = bool(nodata is not None and np.any(values == nodata))
                is_all_zero = bool(np.all(values == 0))
                if is_nodata:
                    nodata_pixel_count += 1
                elif is_all_zero:
                    all_zero_pixel_count += 1
                else:
                    nonzero_valid_pixel_count += 1

                if len(examples) < 8:
                    examples.append(
                        {
                            "row": int(row),
                            "col": int(col),
                            "is_nodata": is_nodata,
                            "is_all_zero": is_all_zero,
                            "value_min": float(values.min()) if values.size else None,
                            "value_max": float(values.max()) if values.size else None,
                        }
                    )

        return {
            "path": str(tif_path),
            "width": int(ds.width),
            "height": int(ds.height),
            "band_count": int(ds.count),
            "dtype": str(ds.dtypes[0]) if ds.dtypes else "unknown",
            "nodata": None if nodata is None else float(nodata),
            "sampled_pixel_count": int(sampled_count),
            "nodata_pixel_count": int(nodata_pixel_count),
            "all_zero_pixel_count": int(all_zero_pixel_count),
            "nonzero_valid_pixel_count": int(nonzero_valid_pixel_count),
            "nodata_ratio": float(nodata_pixel_count / sampled_count) if sampled_count else 0.0,
            "all_zero_ratio": float(all_zero_pixel_count / sampled_count) if sampled_count else 0.0,
            "nonzero_valid_ratio": float(nonzero_valid_pixel_count / sampled_count) if sampled_count else 0.0,
            "examples": examples,
        }


def print_tile_summary(tile_report: dict, completed: int, total: int) -> None:
    print(
        f"[{completed}/{total}] {tile_report['path']} "
        f"sampled={tile_report['sampled_pixel_count']} "
        f"nodata_ratio={tile_report['nodata_ratio']:.6f} "
        f"all_zero_ratio={tile_report['all_zero_ratio']:.6f} "
        f"nonzero_valid_ratio={tile_report['nonzero_valid_ratio']:.6f}"
    )


def scan_tifs_parallel(tif_dir: Path) -> dict:
    tif_paths = sorted(tif_dir.glob("*.tif*"))
    tile_reports: list[dict] = []
    sampled_pixel_total = 0
    nodata_total = 0
    all_zero_total = 0
    nonzero_valid_total = 0

    worker_count = min(MAX_WORKERS, max(1, len(tif_paths)))
    completed = 0

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(inspect_tif, str(tif_path)): str(tif_path)
            for tif_path in tif_paths
        }
        for future in as_completed(futures):
            tile_report = future.result()
            completed += 1
            tile_reports.append(tile_report)
            sampled_pixel_total += int(tile_report["sampled_pixel_count"])
            nodata_total += int(tile_report["nodata_pixel_count"])
            all_zero_total += int(tile_report["all_zero_pixel_count"])
            nonzero_valid_total += int(tile_report["nonzero_valid_pixel_count"])
            print_tile_summary(tile_report, completed, len(tif_paths))

    tile_reports.sort(key=lambda item: item["path"])
    return {
        "tif_dir": str(tif_dir),
        "parallel_workers": int(worker_count),
        "tile_count": len(tile_reports),
        "grid_side": GRID_SIDE,
        "sampled_pixel_total": int(sampled_pixel_total),
        "nodata_pixel_total": int(nodata_total),
        "all_zero_pixel_total": int(all_zero_total),
        "nonzero_valid_pixel_total": int(nonzero_valid_total),
        "nodata_ratio": float(nodata_total / sampled_pixel_total) if sampled_pixel_total else 0.0,
        "all_zero_ratio": float(all_zero_total / sampled_pixel_total) if sampled_pixel_total else 0.0,
        "nonzero_valid_ratio": float(nonzero_valid_total / sampled_pixel_total) if sampled_pixel_total else 0.0,
        "tiles": tile_reports,
    }


def write_report(tif_dir: Path, report: dict) -> Path:
    diagnostics_dir = tif_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def print_summary(report: dict, output_path: Path) -> None:
    print(f"Report written to {output_path}")
    print(
        f"tile_count={report['tile_count']} workers={report['parallel_workers']} "
        f"sampled_pixels={report['sampled_pixel_total']} "
        f"nodata_ratio={report['nodata_ratio']:.6f} all_zero_ratio={report['all_zero_ratio']:.6f}"
    )
    print(f"nonzero_valid_ratio={report['nonzero_valid_ratio']:.6f}")


def main() -> None:
    args = parse_args()
    tif_dir = require_dir(args.tif_dir.resolve())
    report = scan_tifs_parallel(tif_dir)
    output_path = write_report(tif_dir, report)
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
