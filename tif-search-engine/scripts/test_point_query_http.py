from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


@dataclass(frozen=True, slots=True)
class QueryPoint:
    name: str
    lat: float
    lon: float


DEFAULT_POINTS = [
    QueryPoint(name="Q1", lat=29.87747, lon=122.02161),
    QueryPoint(name="Q2", lat=29.87734, lon=122.02450),
    QueryPoint(name="Q3", lat=29.87356, lon=122.02259),
]


def _post_json(base_url: str, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{path} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {base_url}: {exc}") from exc


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


def run_flow(
    *,
    base_url: str,
    top_k: int,
    min_distance_m: float,
    min_score: float,
    search_radius_km: float,
) -> dict:
    point_records: list[dict] = []
    embeddings: list[np.ndarray] = []

    for point in DEFAULT_POINTS:
        response = _post_json(
            base_url,
            "/embedding/by-point",
            {"lon": point.lon, "lat": point.lat},
        )
        point_records.append({"name": point.name, **response})
        embeddings.append(np.asarray(response["embedding"], dtype=np.float32))

    avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0, dtype=np.float32)
    center_lon = sum(point.lon for point in DEFAULT_POINTS) / len(DEFAULT_POINTS)
    center_lat = sum(point.lat for point in DEFAULT_POINTS) / len(DEFAULT_POINTS)
    search_bbox = _build_square_bbox(
        center_lon=center_lon,
        center_lat=center_lat,
        radius_km=search_radius_km,
    )

    search_response = _post_json(
        base_url,
        "/search/by-embedding",
        {
            "embedding": avg_embedding.astype(np.float32).tolist(),
            "top_k": top_k,
            "bbox": search_bbox,
            "min_distance_m": min_distance_m,
            "min_score": min_score,
        },
    )

    return {
        "points": point_records,
        "avg_embedding_dim": int(avg_embedding.shape[0]),
        "search_center": {"lon": center_lon, "lat": center_lat},
        "search_bbox": search_bbox,
        "search_response": search_response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test point-query flow over tif-search-engine HTTP APIs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-distance-m", type=float, default=50.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--search-radius-km", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args()

    try:
        result = run_flow(
            base_url=args.base_url,
            top_k=args.top_k,
            min_distance_m=args.min_distance_m,
            min_score=args.min_score,
            search_radius_km=args.search_radius_km,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print("Point query flow completed.")
    print(f"Base URL: {args.base_url}")
    print(f"Embedding dim: {result['avg_embedding_dim']}")
    print(f"Search center: {result['search_center']}")
    print(f"Search bbox: {result['search_bbox']}")
    print(f"Result count: {result['search_response']['result_count']}")
    print("Query points:")
    for point in result["points"]:
        print(
            f"  {point['name']}: lon={point['lon']:.5f}, lat={point['lat']:.5f}, "
            f"tile={point['tile_path']}, row={point['row']}, col={point['col']}"
        )
    print("Top results:")
    for item in result["search_response"]["results"][:5]:
        print(
            f"  rank={item['rank']} score={item['score']:.6f} "
            f"lon={item['lon']:.6f} lat={item['lat']:.6f} "
            f"row={item['row']} col={item['col']}"
        )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
