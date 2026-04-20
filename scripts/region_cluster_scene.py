from __future__ import annotations

from typing import Any, Callable

import numpy as np


GetEmbeddingsByBbox = Callable[[list[float], int], dict[str, Any]]


def run_region_cluster_scene(
    bbox: list[float],
    *,
    get_embeddings_by_bbox: GetEmbeddingsByBbox,
    total_samples: int = 5000,
    cluster_budget: int = 1000,
    k_min: int = 3,
    k_max: int = 6,
    random_seed: int = 42,
    max_iter: int = 100,
    year: int | None = None,
) -> dict[str, Any]:
    """对 bbox 范围内采样到的 embedding 做聚类，并自动选择最合适的 K。"""
    if len(bbox) != 4:
        raise ValueError("bbox must be [minLon, minLat, maxLon, maxLat].")
    min_lon, min_lat, max_lon, max_lat = [float(item) for item in bbox]
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError("bbox min coordinates must be <= max coordinates.")
    if total_samples < 1:
        raise ValueError("total_samples must be >= 1.")
    if cluster_budget < 1:
        raise ValueError("cluster_budget must be >= 1.")
    if k_min < 2 or k_max < k_min:
        raise ValueError("k_min/k_max must satisfy 2 <= k_min <= k_max.")

    bbox_data = get_embeddings_by_bbox([min_lon, min_lat, max_lon, max_lat], total_samples)
    if bbox_data["count"] == 0:
        result = {
            "scene": "region-cluster",
            "count": 0,
            "selected_k": 0,
            "k_candidates": [],
            "k_scores": {},
            "cluster_sample_size": 0,
            "labels": [],
            "centroids": [],
            "lons": [],
            "lats": [],
            "grid_rows": 0,
            "grid_cols": 0,
            "grid_row_indices": [],
            "grid_col_indices": [],
            "effective_bbox": bbox_data["effective_bbox"],
        }
        if year is not None:
            result["year"] = year
        return result

    all_embeddings = np.asarray(bbox_data["embeddings"], dtype=np.float32)
    training_embeddings = _evenly_sample_embeddings(all_embeddings, cluster_budget)
    sample_size = int(training_embeddings.shape[0])
    k_candidates = _select_candidate_ks(sample_size, k_min, k_max)

    if not k_candidates:
        centroids = np.mean(training_embeddings, axis=0, keepdims=True, dtype=np.float32)
        labels = np.zeros(int(all_embeddings.shape[0]), dtype=np.int32)
        selected_k = 1
        k_scores: dict[int, float] = {}
    else:
        best_score = None
        best_centroids = None
        selected_k = k_candidates[0]
        k_scores = {}
        for k in k_candidates:
            centroids = _fit_kmeans(training_embeddings, k, random_seed=random_seed, max_iter=max_iter)
            labels = _assign_to_centroids(training_embeddings, centroids)
            score = _silhouette_score(training_embeddings, labels)
            k_scores[k] = float(score)
            if best_score is None or score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and k < selected_k):
                best_score = score
                best_centroids = centroids
                selected_k = k
        assert best_centroids is not None
        centroids = best_centroids
        labels = _assign_to_centroids(all_embeddings, centroids)

    result = {
        "scene": "region-cluster",
        "count": int(bbox_data["count"]),
        "selected_k": int(selected_k),
        "k_candidates": k_candidates,
        "k_scores": {str(k): float(score) for k, score in k_scores.items()},
        "cluster_sample_size": sample_size,
        "labels": labels.astype(int).tolist(),
        "centroids": np.asarray(centroids, dtype=np.float32).tolist(),
        "lons": bbox_data["lons"],
        "lats": bbox_data["lats"],
        "grid_rows": int(bbox_data["grid_rows"]),
        "grid_cols": int(bbox_data["grid_cols"]),
        "grid_row_indices": bbox_data["grid_row_indices"],
        "grid_col_indices": bbox_data["grid_col_indices"],
        "effective_bbox": bbox_data["effective_bbox"],
    }
    if year is not None:
        result["year"] = year
    return result


def _evenly_sample_embeddings(embeddings: np.ndarray, budget: int) -> np.ndarray:
    if embeddings.shape[0] <= budget:
        return embeddings
    indices = np.linspace(0, embeddings.shape[0] - 1, num=budget, dtype=np.int64)
    return embeddings[indices]


def _select_candidate_ks(sample_size: int, k_min: int, k_max: int) -> list[int]:
    upper = min(k_max, sample_size - 1)
    if upper < 2:
        return []
    lower = min(max(k_min, 2), upper)
    return list(range(lower, upper + 1))


def _fit_kmeans(vectors: np.ndarray, k: int, *, random_seed: int, max_iter: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed + k)
    centroids = _init_kmeans_plus_plus(vectors, k, rng)
    labels = np.zeros(vectors.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        new_labels = _assign_to_centroids(vectors, centroids)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if not np.any(mask):
                centroids[cluster_id] = vectors[rng.integers(0, vectors.shape[0])]
                continue
            centroids[cluster_id] = vectors[mask].mean(axis=0)

    return centroids.astype(np.float32, copy=False)


def _init_kmeans_plus_plus(vectors: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = vectors.shape[0]
    centroids = np.empty((k, vectors.shape[1]), dtype=np.float32)
    first_index = int(rng.integers(0, n_samples))
    centroids[0] = vectors[first_index]
    closest_dist_sq = np.sum((vectors - centroids[0]) ** 2, axis=1)

    for centroid_index in range(1, k):
        total = float(np.sum(closest_dist_sq))
        if total <= 0:
            centroids[centroid_index] = vectors[int(rng.integers(0, n_samples))]
            continue
        threshold = float(rng.random()) * total
        cumulative = np.cumsum(closest_dist_sq)
        chosen_index = int(np.searchsorted(cumulative, threshold, side="left"))
        chosen_index = min(chosen_index, n_samples - 1)
        centroids[centroid_index] = vectors[chosen_index]
        dist_sq = np.sum((vectors - centroids[centroid_index]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)

    return centroids


def _assign_to_centroids(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def _silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
    if vectors.shape[0] < 2:
        return -1.0
    unique_labels = np.unique(labels)
    if unique_labels.shape[0] < 2:
        return -1.0

    distances = np.sqrt(np.sum((vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=2)).astype(np.float32)
    silhouettes = np.zeros(vectors.shape[0], dtype=np.float32)

    for index in range(vectors.shape[0]):
        label = labels[index]
        same_mask = labels == label
        same_mask[index] = False
        if np.any(same_mask):
            a = float(np.mean(distances[index, same_mask]))
        else:
            silhouettes[index] = 0.0
            continue

        b = np.inf
        for other_label in unique_labels:
            if int(other_label) == int(label):
                continue
            other_mask = labels == other_label
            if not np.any(other_mask):
                continue
            b = min(b, float(np.mean(distances[index, other_mask])))

        if not np.isfinite(b):
            silhouettes[index] = 0.0
            continue
        denom = max(a, b)
        silhouettes[index] = 0.0 if denom <= 0 else (b - a) / denom

    return float(np.mean(silhouettes))
