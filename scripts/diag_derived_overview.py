from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SAMPLE_SIZE = 4096
TOP_TILE_LIMIT = 20
OUTPUT_NAME = "derived_overview.json"
BATCH_SIZE = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect derived embeddings and metadata.")
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


def sample_indices(length: int, limit: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=limit, dtype=np.int64))


def inspect_embeddings(embeddings_path: Path) -> tuple[np.memmap, dict[str, Any]]:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    embedding_count = int(embeddings.shape[0]) if embeddings.ndim >= 1 else 0
    dim = int(embeddings.shape[1]) if embeddings.ndim >= 2 else 0
    indices = sample_indices(embedding_count, SAMPLE_SIZE)

    if len(indices) == 0:
        return embeddings, {
            "embedding_count": embedding_count,
            "embedding_dim": dim,
            "embedding_dtype": str(embeddings.dtype),
            "embedding_nbytes": int(getattr(embeddings, "nbytes", 0)),
            "sample_size": 0,
            "sample_zero_count": 0,
            "sample_zero_ratio": 0.0,
            "sample_norm_min": None,
            "sample_norm_max": None,
            "sample_norm_mean": None,
            "sample_value_min": None,
            "sample_value_max": None,
            "sample_examples": [],
        }

    sampled = np.asarray(embeddings[indices], dtype=np.float32)
    if sampled.ndim == 1:
        sampled = sampled.reshape(1, -1)
    norms = np.linalg.norm(sampled, axis=1)
    zero_mask = np.all(sampled == 0, axis=1)
    sample_examples = []
    for index, norm_value, is_zero in zip(indices[:5], norms[:5], zero_mask[:5]):
        sample_examples.append(
            {
                "id": int(index),
                "norm": float(norm_value),
                "is_all_zero": bool(is_zero),
            }
        )

    return embeddings, {
        "embedding_count": embedding_count,
        "embedding_dim": dim,
        "embedding_dtype": str(embeddings.dtype),
        "embedding_nbytes": int(getattr(embeddings, "nbytes", 0)),
        "sample_size": int(sampled.shape[0]),
        "sample_zero_count": int(zero_mask.sum()),
        "sample_zero_ratio": float(zero_mask.mean()),
        "sample_norm_min": float(norms.min()),
        "sample_norm_max": float(norms.max()),
        "sample_norm_mean": float(norms.mean()),
        "sample_value_min": float(sampled.min()),
        "sample_value_max": float(sampled.max()),
        "sample_examples": sample_examples,
    }


def inspect_metadata(metadata_path: Path, embedding_count: int) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(metadata_path)
    total_rows = int(parquet_file.metadata.num_rows)
    row_count = 0
    lon_min = None
    lon_max = None
    lat_min = None
    lat_max = None
    tile_counter: Counter[str] = Counter()
    invalid_id_examples: list[dict[str, Any]] = []
    invalid_id_count = 0

    progress_bar = progress(total_rows, "Scan metadata")
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

            row_count += int(len(ids))
            batch_lon_min = float(lons.min())
            batch_lon_max = float(lons.max())
            batch_lat_min = float(lats.min())
            batch_lat_max = float(lats.max())
            lon_min = batch_lon_min if lon_min is None else min(lon_min, batch_lon_min)
            lon_max = batch_lon_max if lon_max is None else max(lon_max, batch_lon_max)
            lat_min = batch_lat_min if lat_min is None else min(lat_min, batch_lat_min)
            lat_max = batch_lat_max if lat_max is None else max(lat_max, batch_lat_max)
            tile_counter.update(tile_paths)

            invalid_mask = (ids < 0) | (ids >= embedding_count)
            invalid_positions = np.flatnonzero(invalid_mask)
            invalid_id_count += int(invalid_mask.sum())
            for position in invalid_positions[: max(0, 5 - len(invalid_id_examples))]:
                invalid_id_examples.append(
                    {
                        "id": int(ids[position]),
                        "lon": float(lons[position]),
                        "lat": float(lats[position]),
                        "tile_path": str(tile_paths[position]),
                        "row": int(rows[position]),
                        "col": int(cols[position]),
                    }
                )
            if progress_bar is not None:
                progress_bar.update(int(len(ids)))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    tile_counts = [
        {"tile_path": tile_path, "count": int(count)}
        for tile_path, count in tile_counter.most_common(TOP_TILE_LIMIT)
    ]
    return {
        "metadata_row_count": int(row_count),
        "metadata_bbox": None if lon_min is None else [lon_min, lat_min, lon_max, lat_max],
        "tile_path_counts": tile_counts,
        "tile_path_unique_count": int(len(tile_counter)),
        "invalid_id_count": int(invalid_id_count),
        "invalid_id_examples": invalid_id_examples,
    }


def build_report(
    derived_dir: Path,
    embedding_stats: dict[str, Any],
    metadata_stats: dict[str, Any],
    build_info: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata_row_count = int(metadata_stats["metadata_row_count"])
    embedding_count = int(embedding_stats["embedding_count"])
    report = {
        "derived_dir": str(derived_dir),
        "metadata_path": str(derived_dir / "metadata.parquet"),
        "embeddings_path": str(derived_dir / "embeddings.npy"),
        "build_info_path": str(derived_dir / "build_info.json"),
        "build_info_present": build_info is not None,
        "metadata_row_count": metadata_row_count,
        "embedding_count": embedding_count,
        "embedding_dim": int(embedding_stats["embedding_dim"]),
        "embedding_dtype": embedding_stats["embedding_dtype"],
        "embedding_nbytes": int(embedding_stats["embedding_nbytes"]),
        "row_count_match": metadata_row_count == embedding_count,
        "metadata_bbox": metadata_stats["metadata_bbox"],
        "tile_path_unique_count": int(metadata_stats["tile_path_unique_count"]),
        "tile_path_counts": metadata_stats["tile_path_counts"],
        "invalid_id_count": int(metadata_stats["invalid_id_count"]),
        "invalid_id_examples": metadata_stats["invalid_id_examples"],
        "embedding_sample": {
            "sample_size": int(embedding_stats["sample_size"]),
            "sample_zero_count": int(embedding_stats["sample_zero_count"]),
            "sample_zero_ratio": float(embedding_stats["sample_zero_ratio"]),
            "sample_norm_min": embedding_stats["sample_norm_min"],
            "sample_norm_max": embedding_stats["sample_norm_max"],
            "sample_norm_mean": embedding_stats["sample_norm_mean"],
            "sample_value_min": embedding_stats["sample_value_min"],
            "sample_value_max": embedding_stats["sample_value_max"],
            "sample_examples": embedding_stats["sample_examples"],
        },
        "build_info": build_info,
        "build_info_comparison": None,
    }
    if build_info is not None:
        build_vector_count = build_info.get("vector_count")
        report["build_info_comparison"] = {
            "vector_count_matches_metadata": build_vector_count == metadata_row_count,
            "vector_count_matches_embeddings": build_vector_count == embedding_count,
            "tile_count_matches_observed_tiles": (
                build_info.get("tile_count") == metadata_stats["tile_path_unique_count"]
            ),
        }
    return report


def write_report(derived_dir: Path, report: dict[str, Any]) -> Path:
    diagnostics_dir = derived_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    bbox = report["metadata_bbox"]
    bbox_text = "empty" if bbox is None else ", ".join(f"{value:.6f}" for value in bbox)
    print(f"Report written to {output_path}")
    print(
        "metadata_rows="
        f"{report['metadata_row_count']} embeddings={report['embedding_count']} "
        f"dim={report['embedding_dim']} row_count_match={report['row_count_match']}"
    )
    print(
        "sample_zero_ratio="
        f"{report['embedding_sample']['sample_zero_ratio']:.6f} invalid_ids={report['invalid_id_count']}"
    )
    print(f"metadata_bbox={bbox_text}")


def main() -> None:
    args = parse_args()
    derived_dir = args.derived_dir.resolve()
    metadata_path = require_path(derived_dir / "metadata.parquet")
    embeddings_path = require_path(derived_dir / "embeddings.npy")
    build_info = load_optional_build_info(derived_dir / "build_info.json")

    _embeddings, embedding_stats = inspect_embeddings(embeddings_path)
    metadata_stats = inspect_metadata(metadata_path, int(embedding_stats["embedding_count"]))
    report = build_report(derived_dir, embedding_stats, metadata_stats, build_info)
    output_path = write_report(derived_dir, report)
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
