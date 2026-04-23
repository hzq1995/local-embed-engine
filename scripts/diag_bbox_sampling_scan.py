from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DIAG_GRID_ROWS = 6
DIAG_GRID_COLS = 6
TOTAL_SAMPLES = 400
OUTPUT_NAME = "bbox_sampling_scan.json"
BATCH_SIZE = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay bbox sampling offline.")
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


def metadata_bbox(metadata_path: Path) -> tuple[int, list[float] | None]:
    parquet_file = pq.ParquetFile(metadata_path)
    row_count = int(parquet_file.metadata.num_rows)
    lon_min = None
    lon_max = None
    lat_min = None
    lat_max = None

    progress_bar = progress(row_count, "Scan bbox")
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


def build_diag_boxes(bbox: list[float] | None) -> list[dict[str, Any]]:
    if bbox is None:
        return []
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)
    lon_step = lon_span / DIAG_GRID_COLS
    lat_step = lat_span / DIAG_GRID_ROWS
    boxes: list[dict[str, Any]] = []
    for row_index in range(DIAG_GRID_ROWS):
        for col_index in range(DIAG_GRID_COLS):
            box_min_lon = min_lon + col_index * lon_step
            box_max_lon = min_lon + (col_index + 1) * lon_step
            box_min_lat = min_lat + row_index * lat_step
            box_max_lat = min_lat + (row_index + 1) * lat_step
            local_lon_span = max(box_max_lon - box_min_lon, 1e-9)
            local_lat_span = max(box_max_lat - box_min_lat, 1e-9)
            aspect = local_lon_span / local_lat_span
            grid_rows = max(2, int(math.sqrt(TOTAL_SAMPLES / aspect)))
            grid_cols = max(2, int(math.ceil(TOTAL_SAMPLES / grid_rows)))
            boxes.append(
                {
                    "bbox_id": int(row_index * DIAG_GRID_COLS + col_index),
                    "diag_row": int(row_index),
                    "diag_col": int(col_index),
                    "bbox": [float(box_min_lon), float(box_min_lat), float(box_max_lon), float(box_max_lat)],
                    "grid_rows": int(grid_rows),
                    "grid_cols": int(grid_cols),
                    "lon_step": float(local_lon_span / grid_cols),
                    "lat_step": float(local_lat_span / grid_rows),
                    "candidate_count": 0,
                    "selected_cells": {},
                }
            )
    return boxes


def populate_boxes(metadata_path: Path, boxes: list[dict[str, Any]], bbox: list[float] | None) -> None:
    if not boxes or bbox is None:
        return
    min_lon, min_lat, max_lon, max_lat = bbox
    outer_lon_step = max((max_lon - min_lon) / DIAG_GRID_COLS, 1e-9)
    outer_lat_step = max((max_lat - min_lat) / DIAG_GRID_ROWS, 1e-9)
    parquet_file = pq.ParquetFile(metadata_path)
    progress_bar = progress(int(parquet_file.metadata.num_rows), "Replay bbox sampling")
    try:
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["id", "lon", "lat", "tile_path", "row", "col"],
        ):
            payload = batch.to_pydict()
            ids = np.asarray(payload["id"], dtype=np.int64)
            lons = np.asarray(payload["lon"], dtype=np.float64)
            lats = np.asarray(payload["lat"], dtype=np.float64)
            rows = np.asarray(payload["row"], dtype=np.int64)
            cols = np.asarray(payload["col"], dtype=np.int64)
            tile_paths = payload["tile_path"]
            if len(ids) == 0:
                continue

            outer_cols = np.clip(np.floor((lons - min_lon) / outer_lon_step).astype(np.int64), 0, DIAG_GRID_COLS - 1)
            outer_rows = np.clip(np.floor((lats - min_lat) / outer_lat_step).astype(np.int64), 0, DIAG_GRID_ROWS - 1)
            box_ids = outer_rows * DIAG_GRID_COLS + outer_cols

            for unique_box_id in np.unique(box_ids):
                box = boxes[int(unique_box_id)]
                mask = box_ids == unique_box_id
                local_ids = ids[mask]
                local_lons = lons[mask]
                local_lats = lats[mask]
                local_rows = rows[mask]
                local_cols = cols[mask]
                local_tile_paths = np.asarray(tile_paths, dtype=object)[mask]
                box["candidate_count"] += int(len(local_ids))
                if len(local_ids) == 0:
                    continue

                box_min_lon, box_min_lat, box_max_lon, box_max_lat = box["bbox"]
                fine_cols = np.clip(
                    np.floor((local_lons - box_min_lon) / box["lon_step"]).astype(np.int64),
                    0,
                    box["grid_cols"] - 1,
                )
                fine_rows = np.clip(
                    np.floor((box_max_lat - local_lats) / box["lat_step"]).astype(np.int64),
                    0,
                    box["grid_rows"] - 1,
                )
                cell_keys = fine_rows * box["grid_cols"] + fine_cols
                selected_cells: dict[int, dict[str, Any]] = box["selected_cells"]

                for idx, cell_key in enumerate(cell_keys.tolist()):
                    if cell_key in selected_cells:
                        continue
                    selected_cells[cell_key] = {
                        "id": int(local_ids[idx]),
                        "lon": float(local_lons[idx]),
                        "lat": float(local_lats[idx]),
                        "tile_path": str(local_tile_paths[idx]),
                        "row": int(local_rows[idx]),
                        "col": int(local_cols[idx]),
                        "grid_row_index": int(fine_rows[idx]),
                        "grid_col_index": int(fine_cols[idx]),
                    }
            if progress_bar is not None:
                progress_bar.update(int(len(ids)))
    finally:
        if progress_bar is not None:
            progress_bar.close()


def classify_box(selected_zero_ratio: float | None, selected_count: int, valid_selected_count: int) -> str:
    if selected_count == 0:
        return "empty"
    if valid_selected_count == 0:
        return "invalid"
    if selected_zero_ratio is None:
        return "invalid"
    if selected_zero_ratio >= 0.95:
        return "all_zero"
    if selected_zero_ratio <= 0.05:
        return "nonzero"
    return "mixed"


def build_report(
    derived_dir: Path,
    embedding_count: int,
    metadata_row_count: int,
    overall_bbox: list[float] | None,
    boxes: list[dict[str, Any]],
    embeddings_path: Path,
    build_info: dict[str, Any] | None,
) -> dict[str, Any]:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    bbox_reports: list[dict[str, Any]] = []

    for box in boxes:
        selected_records = list(box["selected_cells"].values())
        selected_count = len(selected_records)
        selected_ids = [int(item["id"]) for item in selected_records]
        valid_records = [item for item in selected_records if 0 <= int(item["id"]) < embedding_count]
        invalid_records = [item for item in selected_records if not (0 <= int(item["id"]) < embedding_count)]

        zero_examples = []
        nonzero_examples = []
        zero_count = 0
        for record in valid_records:
            vector = np.asarray(embeddings[int(record["id"])], dtype=np.float32)
            is_zero = bool(np.all(vector == 0))
            if is_zero:
                zero_count += 1
                if len(zero_examples) < 3:
                    zero_examples.append(record)
            elif len(nonzero_examples) < 3:
                nonzero_examples.append(record)

        valid_selected_count = len(valid_records)
        zero_ratio = None if valid_selected_count == 0 else float(zero_count / valid_selected_count)
        total_cells = int(box["grid_rows"] * box["grid_cols"])
        empty_cell_ratio = float(max(total_cells - selected_count, 0) / total_cells) if total_cells > 0 else 0.0
        classification = classify_box(zero_ratio, selected_count, valid_selected_count)
        severity_score = float((zero_ratio or 0.0) * 0.7 + empty_cell_ratio * 0.3)

        bbox_reports.append(
            {
                "bbox_id": int(box["bbox_id"]),
                "diag_row": int(box["diag_row"]),
                "diag_col": int(box["diag_col"]),
                "bbox": box["bbox"],
                "grid_rows": int(box["grid_rows"]),
                "grid_cols": int(box["grid_cols"]),
                "candidate_count": int(box["candidate_count"]),
                "selected_count": int(selected_count),
                "valid_selected_count": int(valid_selected_count),
                "invalid_selected_count": int(len(invalid_records)),
                "selected_zero_ratio": zero_ratio,
                "empty_cell_ratio": float(empty_cell_ratio),
                "classification": classification,
                "severity_score": severity_score,
                "zero_examples": zero_examples,
                "nonzero_examples": nonzero_examples,
                "selected_id_examples": selected_ids[:10],
            }
        )

    bbox_reports.sort(
        key=lambda item: (
            item["severity_score"],
            item["candidate_count"],
            item["selected_count"],
        ),
        reverse=True,
    )
    return {
        "derived_dir": str(derived_dir),
        "metadata_path": str(derived_dir / "metadata.parquet"),
        "embeddings_path": str(derived_dir / "embeddings.npy"),
        "build_info_path": str(derived_dir / "build_info.json"),
        "build_info_present": build_info is not None,
        "build_info": build_info,
        "metadata_row_count": int(metadata_row_count),
        "embedding_count": int(embedding_count),
        "metadata_bbox": overall_bbox,
        "diag_grid_rows": DIAG_GRID_ROWS,
        "diag_grid_cols": DIAG_GRID_COLS,
        "bbox_count": int(len(bbox_reports)),
        "bbox_reports": bbox_reports,
    }


def write_report(derived_dir: Path, report: dict[str, Any]) -> Path:
    diagnostics_dir = derived_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    top = report["bbox_reports"][0] if report["bbox_reports"] else None
    print(f"Report written to {output_path}")
    print(
        "bbox_count="
        f"{report['bbox_count']} metadata_rows={report['metadata_row_count']} embeddings={report['embedding_count']}"
    )
    if top is not None:
        print(
            "top_bbox="
            f"{top['bbox_id']} classification={top['classification']} "
            f"zero_ratio={top['selected_zero_ratio']} empty_ratio={top['empty_cell_ratio']:.6f}"
        )


def main() -> None:
    args = parse_args()
    derived_dir = args.derived_dir.resolve()
    metadata_path = require_path(derived_dir / "metadata.parquet")
    embeddings_path = require_path(derived_dir / "embeddings.npy")
    build_info = load_optional_build_info(derived_dir / "build_info.json")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    embedding_count = int(embeddings.shape[0]) if embeddings.ndim >= 1 else 0
    metadata_row_count, overall_bbox = metadata_bbox(metadata_path)
    boxes = build_diag_boxes(overall_bbox)
    populate_boxes(metadata_path, boxes, overall_bbox)
    report = build_report(
        derived_dir,
        embedding_count,
        metadata_row_count,
        overall_bbox,
        boxes,
        embeddings_path,
        build_info,
    )
    output_path = write_report(derived_dir, report)
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
