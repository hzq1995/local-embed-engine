from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


FetchEmbeddingByPoint = Callable[[float, float], dict[str, Any]]
SearchByEmbedding = Callable[[list[float], int, list[float] | None, float, float], dict[str, Any]]


def run_point_query_scene(
    points: list[dict[str, float]],
    *,
    fetch_embedding_by_point: FetchEmbeddingByPoint,
    search_by_embedding: SearchByEmbedding,
    top_k: int = 20,
    min_distance_m: float = 50.0,
    min_score: float = 0.9,
    search_radius_km: float = 5.0,
    year: int | None = None,
) -> dict[str, Any]:
    """将多个点选得到的 embedding 求均值，再用均值向量做检索。

    参数：
        points: 外部传入的点位输入，每个元素都必须包含 ``lon`` 和 ``lat``。
        fetch_embedding_by_point: 后端提供的单点 embedding 查询函数。
        search_by_embedding: 后端提供的 embedding 检索函数。
        top_k: 最多返回多少条检索结果。
        min_distance_m: 返回结果之间的最小间隔距离（米）。
        min_score: 相似度下限。
        search_radius_km: 以所有点的中心为中心点时的搜索半径（公里）。
        year: 可选，用于透传年份等业务元数据。
    """
    if not points:
        raise ValueError("points must contain at least one {lon, lat} item.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if min_distance_m < 0:
        raise ValueError("min_distance_m must be >= 0.")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0, 1].")
    if search_radius_km <= 0:
        raise ValueError("search_radius_km must be > 0.")

    point_records: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    for idx, point in enumerate(points, start=1):
        if "lon" not in point or "lat" not in point:
            raise ValueError(f"point #{idx} must contain lon and lat.")
        lon = float(point["lon"])
        lat = float(point["lat"])
        record = fetch_embedding_by_point(lon, lat)
        point_records.append(record)
        embeddings.append(np.asarray(record["embedding"], dtype=np.float32))

    avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0, dtype=np.float32)
    center_lat = sum(float(item["lat"]) for item in point_records) / len(point_records)
    center_lon = sum(float(item["lon"]) for item in point_records) / len(point_records)
    search_bbox = _build_square_bbox(center_lon=center_lon, center_lat=center_lat, radius_km=search_radius_km)

    search_result = search_by_embedding(
        avg_embedding.tolist(),
        top_k,
        search_bbox,
        min_distance_m,
        min_score,
    )

    result = {
        "scene": "point-query",
        "point_count": len(point_records),
        "query_points": [
            {
                "lon": float(item["lon"]),
                "lat": float(item["lat"]),
                "tile_path": item.get("tile_path"),
                "row": int(item["row"]) if "row" in item else None,
                "col": int(item["col"]) if "col" in item else None,
            }
            for item in point_records
        ],
        "avg_embedding": avg_embedding.astype(np.float32).tolist(),
        "search_center": {"lon": center_lon, "lat": center_lat},
        "search_bbox": search_bbox,
        "top_k": int(search_result["top_k"]),
        "result_count": int(search_result["result_count"]),
        "results": search_result["results"],
    }
    if year is not None:
        result["year"] = year
    return result


def _build_square_bbox(*, center_lon: float, center_lat: float, radius_km: float) -> list[float]:
    half_m = radius_km * 1000.0
    d_lat = half_m / 111_320.0
    cos_lat = math.cos(math.radians(center_lat))
    if abs(cos_lat) < 1e-9:
        cos_lat = 1e-9
    d_lon = half_m / (111_320.0 * cos_lat)
    return [
        float(round(center_lon - d_lon, 6)),
        float(round(center_lat - d_lat, 6)),
        float(round(center_lon + d_lon, 6)),
        float(round(center_lat + d_lat, 6)),
    ]
