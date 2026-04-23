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


LOCAL_WINDOW_DEG = 0.01
METADATA_SAMPLE_POINTS = 12
GRID_SIDE = 5
OUTPUT_NAME = "point_query_sampling.json"
BATCH_SIZE = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay point-query sampling offline.")
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


def metadata_sample_positions(row_count: int) -> np.ndarray:
    if row_count <= 0:
        return np.asarray([], dtype=np.int64)
    count = min(row_count, METADATA_SAMPLE_POINTS)
    if count == row_count:
        return np.arange(row_count, dtype=np.int64)
    return np.unique(np.linspace(0, row_count - 1, num=count, dtype=np.int64))


def summarize_metadata(metadata_path: Path) -> tuple[int, list[float] | None, list[dict[str, Any]]]:
    parquet_file = pq.ParquetFile(metadata_path)
    row_count = int(parquet_file.metadata.num_rows)
    targets = metadata_sample_positions(row_count)
    collected: list[dict[str, Any]] = []
    lon_min = None
    lon_max = None
    lat_min = None
    lat_max = None
    global_offset = 0
    target_index = 0

    progress_bar = progress(row_count, "Sample metadata")
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

            batch_lon_min = float(lons.min())
            batch_lon_max = float(lons.max())
            batch_lat_min = float(lats.min())
            batch_lat_max = float(lats.max())
            lon_min = batch_lon_min if lon_min is None else min(lon_min, batch_lon_min)
            lon_max = batch_lon_max if lon_max is None else max(lon_max, batch_lon_max)
            lat_min = batch_lat_min if lat_min is None else min(lat_min, batch_lat_min)
            lat_max = batch_lat_max if lat_max is None else max(lat_max, batch_lat_max)

            batch_end = global_offset + len(ids)
            while target_index < len(targets) and int(targets[target_index]) < batch_end:
                local_index = int(targets[target_index] - global_offset)
                collected.append(
                    {
                        "source": "metadata",
                        "query_lon": float(lons[local_index]),
                        "query_lat": float(lats[local_index]),
                        "expected_id": int(ids[local_index]),
                        "expected_tile_path": str(tile_paths[local_index]),
                        "expected_row": int(rows[local_index]),
                        "expected_col": int(cols[local_index]),
                        "metadata_ordinal": int(targets[target_index]),
                    }
                )
                target_index += 1
            global_offset = batch_end
            if progress_bar is not None:
                progress_bar.update(int(len(ids)))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    bbox = None if lon_min is None else [lon_min, lat_min, lon_max, lat_max]
    return row_count, bbox, collected


def build_grid_points(bbox: list[float] | None) -> list[dict[str, Any]]:
    if bbox is None:
        return []
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)
    points: list[dict[str, Any]] = []
    for row_index in range(GRID_SIDE):
        for col_index in range(GRID_SIDE):
            lon = min_lon + lon_span * (col_index + 0.5) / GRID_SIDE
            lat = min_lat + lat_span * (row_index + 0.5) / GRID_SIDE
            points.append(
                {
                    "source": "grid",
                    "query_lon": float(lon),
                    "query_lat": float(lat),
                    "grid_row": int(row_index),
                    "grid_col": int(col_index),
                }
            )
    return points


def run_point_matching(
    metadata_path: Path,
    query_points: list[dict[str, Any]],
    embeddings: np.ndarray,
    embedding_count: int,
) -> list[dict[str, Any]]:
    if not query_points:
        return []

    parquet_file = pq.ParquetFile(metadata_path)
    query_lons = np.asarray([item["query_lon"] for item in query_points], dtype=np.float64)
    query_lats = np.asarray([item["query_lat"] for item in query_points], dtype=np.float64)
    query_count = len(query_points)

    local_best_dist2 = np.full(query_count, np.inf, dtype=np.float64)
    global_best_dist2 = np.full(query_count, np.inf, dtype=np.float64)
    local_best: list[dict[str, Any] | None] = [None] * query_count
    global_best: list[dict[str, Any] | None] = [None] * query_count

    progress_bar = progress(int(parquet_file.metadata.num_rows), "Replay point query")
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

            diff_lon = lons[None, :] - query_lons[:, None]
            diff_lat = lats[None, :] - query_lats[:, None]
            dist2 = diff_lon * diff_lon + diff_lat * diff_lat

            global_indices = np.argmin(dist2, axis=1)
            global_values = dist2[np.arange(query_count), global_indices]
            global_updates = np.flatnonzero(global_values < global_best_dist2)
            for query_index in global_updates.tolist():
                candidate_index = int(global_indices[query_index])
                global_best_dist2[query_index] = float(global_values[query_index])
                global_best[query_index] = {
                    "id": int(ids[candidate_index]),
                    "lon": float(lons[candidate_index]),
                    "lat": float(lats[candidate_index]),
                    "tile_path": str(tile_paths[candidate_index]),
                    "row": int(rows[candidate_index]),
                    "col": int(cols[candidate_index]),
                }

            local_mask = (np.abs(diff_lon) <= LOCAL_WINDOW_DEG) & (np.abs(diff_lat) <= LOCAL_WINDOW_DEG)
            if not np.any(local_mask):
                if progress_bar is not None:
                    progress_bar.update(int(len(ids)))
                continue
            masked_dist2 = np.where(local_mask, dist2, np.inf)
            local_indices = np.argmin(masked_dist2, axis=1)
            local_values = masked_dist2[np.arange(query_count), local_indices]
            local_updates = np.flatnonzero(local_values < local_best_dist2)
            for query_index in local_updates.tolist():
                if not np.isfinite(local_values[query_index]):
                    continue
                candidate_index = int(local_indices[query_index])
                local_best_dist2[query_index] = float(local_values[query_index])
                local_best[query_index] = {
                    "id": int(ids[candidate_index]),
                    "lon": float(lons[candidate_index]),
                    "lat": float(lats[candidate_index]),
                    "tile_path": str(tile_paths[candidate_index]),
                    "row": int(rows[candidate_index]),
                    "col": int(cols[candidate_index]),
                }
            if progress_bar is not None:
                progress_bar.update(int(len(ids)))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    results: list[dict[str, Any]] = []
    for query_index, query_point in enumerate(query_points):
        chosen = local_best[query_index] if local_best[query_index] is not None else global_best[query_index]
        used_global_fallback = local_best[query_index] is None
        if chosen is None:
            result = dict(query_point)
            result.update(
                {
                    "used_global_fallback": used_global_fallback,
                    "matched_id": None,
                    "matched_lon": None,
                    "matched_lat": None,
                    "matched_tile_path": None,
                    "matched_row": None,
                    "matched_col": None,
                    "distance_to_matched_point_m": None,
                    "match_is_valid": False,
                    "embedding_is_all_zero": None,
                    "matched_embedding_norm": None,
                    "self_hit": False,
                }
            )
            results.append(result)
            continue

        matched_id = int(chosen["id"])
        valid = 0 <= matched_id < embedding_count
        if valid:
            embedding = np.asarray(embeddings[matched_id], dtype=np.float32)
            is_zero = bool(np.all(embedding == 0))
            embedding_norm = float(np.linalg.norm(embedding))
        else:
            is_zero = None
            embedding_norm = None

        result = dict(query_point)
        result.update(
            {
                "used_global_fallback": used_global_fallback,
                "matched_id": matched_id,
                "matched_lon": float(chosen["lon"]),
                "matched_lat": float(chosen["lat"]),
                "matched_tile_path": str(chosen["tile_path"]),
                "matched_row": int(chosen["row"]),
                "matched_col": int(chosen["col"]),
                "distance_to_matched_point_m": float(
                    haversine_distance_m(
                        float(query_point["query_lon"]),
                        float(query_point["query_lat"]),
                        float(chosen["lon"]),
                        float(chosen["lat"]),
                    )
                ),
                "match_is_valid": bool(valid),
                "embedding_is_all_zero": is_zero,
                "matched_embedding_norm": embedding_norm,
                "self_hit": bool(query_point.get("expected_id") == matched_id),
            }
        )
        results.append(result)
    return results


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return 6_371_000.0 * c


def summarize_results(
    derived_dir: Path,
    bbox: list[float] | None,
    metadata_row_count: int,
    embedding_count: int,
    query_points: list[dict[str, Any]],
    results: list[dict[str, Any]],
    build_info: dict[str, Any] | None,
) -> dict[str, Any]:
    local_hit_count = sum(1 for item in results if not item["used_global_fallback"])
    fallback_results = [item for item in results if item["used_global_fallback"]]
    valid_results = [item for item in results if item["match_is_valid"]]
    zero_results = [item for item in valid_results if item["embedding_is_all_zero"]]
    metadata_results = [item for item in results if item["source"] == "metadata" and item["match_is_valid"]]
    self_hit_ratio = (
        float(sum(1 for item in metadata_results if item["self_hit"]) / len(metadata_results))
        if metadata_results
        else None
    )
    fallback_distances = [
        float(item["distance_to_matched_point_m"])
        for item in fallback_results
        if item["distance_to_matched_point_m"] is not None
    ]
    anomalies = [
        item
        for item in results
        if item["used_global_fallback"] or item["embedding_is_all_zero"] or not item["match_is_valid"]
    ][:15]
    return {
        "derived_dir": str(derived_dir),
        "metadata_path": str(derived_dir / "metadata.parquet"),
        "embeddings_path": str(derived_dir / "embeddings.npy"),
        "build_info_path": str(derived_dir / "build_info.json"),
        "build_info_present": build_info is not None,
        "build_info": build_info,
        "metadata_row_count": int(metadata_row_count),
        "embedding_count": int(embedding_count),
        "metadata_bbox": bbox,
        "query_point_count": int(len(query_points)),
        "metadata_sample_point_count": int(sum(1 for item in query_points if item["source"] == "metadata")),
        "grid_point_count": int(sum(1 for item in query_points if item["source"] == "grid")),
        "local_hit_count": int(local_hit_count),
        "global_fallback_count": int(len(fallback_results)),
        "zero_return_count": int(len(zero_results)),
        "invalid_match_count": int(sum(1 for item in results if not item["match_is_valid"])),
        "metadata_self_hit_ratio": self_hit_ratio,
        "fallback_distance_stats_m": {
            "count": int(len(fallback_distances)),
            "min": None if not fallback_distances else float(min(fallback_distances)),
            "max": None if not fallback_distances else float(max(fallback_distances)),
            "mean": None if not fallback_distances else float(sum(fallback_distances) / len(fallback_distances)),
        },
        "query_results": results,
        "anomaly_samples": anomalies,
    }


def write_report(derived_dir: Path, report: dict[str, Any]) -> Path:
    diagnostics_dir = derived_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    print(f"Report written to {output_path}")
    print(
        "query_points="
        f"{report['query_point_count']} local_hits={report['local_hit_count']} "
        f"global_fallbacks={report['global_fallback_count']}"
    )
    print(
        "zero_returns="
        f"{report['zero_return_count']} invalid_matches={report['invalid_match_count']} "
        f"metadata_self_hit_ratio={report['metadata_self_hit_ratio']}"
    )


def main() -> None:
    args = parse_args()
    derived_dir = args.derived_dir.resolve()
    metadata_path = require_path(derived_dir / "metadata.parquet")
    embeddings_path = require_path(derived_dir / "embeddings.npy")
    build_info = load_optional_build_info(derived_dir / "build_info.json")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    embedding_count = int(embeddings.shape[0]) if embeddings.ndim >= 1 else 0

    metadata_row_count, bbox, sampled_metadata_points = summarize_metadata(metadata_path)
    query_points = sampled_metadata_points + build_grid_points(bbox)
    results = run_point_matching(metadata_path, query_points, embeddings, embedding_count)
    report = summarize_results(
        derived_dir,
        bbox,
        metadata_row_count,
        embedding_count,
        query_points,
        results,
        build_info,
    )
    output_path = write_report(derived_dir, report)
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
