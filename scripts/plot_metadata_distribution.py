from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


BIN_SIZE = 768
OUTPUT_IMAGE_NAME = "metadata_distribution.png"
OUTPUT_JSON_NAME = "metadata_distribution.json"
BATCH_SIZE = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a coarse metadata lon/lat distribution map.")
    parser.add_argument("derived_dir", type=Path)
    return parser.parse_args()


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def load_optional_build_info(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        "vector_count": payload.get("vector_count"),
        "tile_count": payload.get("tile_count"),
        "index_type": payload.get("index_type"),
        "data_dir": payload.get("data_dir"),
        "build_time": payload.get("build_time"),
    }


def progress(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="row")


def scan_metadata_bounds(metadata_path: Path) -> tuple[int, list[float] | None]:
    parquet_file = pq.ParquetFile(metadata_path)
    row_count = int(parquet_file.metadata.num_rows)
    lon_min = None
    lon_max = None
    lat_min = None
    lat_max = None

    progress_bar = progress(row_count, "Scan plot bbox")
    try:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["lon", "lat"]):
            payload = batch.to_pydict()
            lons = np.asarray(payload["lon"], dtype=np.float64)
            lats = np.asarray(payload["lat"], dtype=np.float64)
            if len(lons) == 0:
                continue
            batch_lon_min = float(lons.min())
            batch_lon_max = float(lons.max())
            batch_lat_min = float(lats.min())
            batch_lat_max = float(lats.max())
            lon_min = batch_lon_min if lon_min is None else min(lon_min, batch_lon_min)
            lon_max = batch_lon_max if lon_max is None else max(lon_max, batch_lon_max)
            lat_min = batch_lat_min if lat_min is None else min(lat_min, batch_lat_min)
            lat_max = batch_lat_max if lat_max is None else max(lat_max, batch_lat_max)
            if progress_bar is not None:
                progress_bar.update(int(len(lons)))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    bbox = None if lon_min is None else [lon_min, lat_min, lon_max, lat_max]
    return row_count, bbox


def accumulate_histogram(metadata_path: Path, bbox: list[float], bins: int) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)
    histogram = np.zeros((bins, bins), dtype=np.uint64)
    parquet_file = pq.ParquetFile(metadata_path)
    progress_bar = progress(int(parquet_file.metadata.num_rows), "Build histogram")
    try:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["lon", "lat"]):
            payload = batch.to_pydict()
            lons = np.asarray(payload["lon"], dtype=np.float64)
            lats = np.asarray(payload["lat"], dtype=np.float64)
            if len(lons) == 0:
                continue

            lon_indices = np.clip(np.floor((lons - min_lon) / lon_span * bins).astype(np.int64), 0, bins - 1)
            lat_indices = np.clip(np.floor((lats - min_lat) / lat_span * bins).astype(np.int64), 0, bins - 1)
            flat_indices = lat_indices * bins + lon_indices
            counts = np.bincount(flat_indices, minlength=bins * bins)
            histogram += counts.reshape(bins, bins).astype(np.uint64)
            if progress_bar is not None:
                progress_bar.update(int(len(lons)))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return histogram


def render_png(output_path: Path, histogram: np.ndarray, bbox: list[float]) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    display = np.log1p(histogram.astype(np.float64))
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    image = ax.imshow(
        display,
        origin="lower",
        extent=[min_lon, max_lon, min_lat, max_lat],
        cmap="inferno",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title("Metadata Distribution (log1p density)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log1p(count)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_report(
    derived_dir: Path,
    row_count: int,
    bbox: list[float] | None,
    histogram: np.ndarray | None,
    build_info: dict[str, Any] | None,
) -> dict[str, Any]:
    nonzero_bins = int(np.count_nonzero(histogram)) if histogram is not None else 0
    max_bin_count = int(histogram.max()) if histogram is not None and histogram.size > 0 else 0
    return {
        "derived_dir": str(derived_dir),
        "metadata_path": str(derived_dir / "metadata.parquet"),
        "build_info_path": str(derived_dir / "build_info.json"),
        "build_info_present": build_info is not None,
        "build_info": build_info,
        "metadata_row_count": int(row_count),
        "metadata_bbox": bbox,
        "histogram_bins": int(BIN_SIZE),
        "nonzero_histogram_cells": nonzero_bins,
        "max_histogram_cell_count": max_bin_count,
        "output_image_path": str(derived_dir / "diagnostics" / OUTPUT_IMAGE_NAME),
    }


def write_outputs(derived_dir: Path, report: dict[str, Any], histogram: np.ndarray | None, bbox: list[float] | None) -> tuple[Path, Path | None]:
    diagnostics_dir = derived_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    report_path = diagnostics_dir / OUTPUT_JSON_NAME
    image_path = diagnostics_dir / OUTPUT_IMAGE_NAME
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if histogram is not None and bbox is not None:
        render_png(image_path, histogram, bbox)
        return report_path, image_path
    return report_path, None


def print_summary(report: dict[str, Any], report_path: Path, image_path: Path | None) -> None:
    print(f"Report written to {report_path}")
    if image_path is not None:
        print(f"PNG written to {image_path}")
    print(
        "metadata_rows="
        f"{report['metadata_row_count']} bins={report['histogram_bins']} "
        f"nonzero_cells={report['nonzero_histogram_cells']} max_cell_count={report['max_histogram_cell_count']}"
    )


def main() -> None:
    args = parse_args()
    derived_dir = args.derived_dir.resolve()
    metadata_path = require_path(derived_dir / "metadata.parquet")
    build_info = load_optional_build_info(derived_dir / "build_info.json")

    row_count, bbox = scan_metadata_bounds(metadata_path)
    histogram = accumulate_histogram(metadata_path, bbox, BIN_SIZE) if bbox is not None else None
    report = build_report(derived_dir, row_count, bbox, histogram, build_info)
    report_path, image_path = write_outputs(derived_dir, report, histogram, bbox)
    print_summary(report, report_path, image_path)


if __name__ == "__main__":
    main()
