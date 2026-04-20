from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclass(slots=True)
class IndexBundle:
    metadata: pd.DataFrame
    embeddings: np.ndarray
    build_info: dict
    faiss_index: object | None = None

    @property
    def vector_count(self) -> int:
        return int(len(self.metadata))

    @property
    def index_loaded(self) -> bool:
        return self.vector_count > 0

    @property
    def index_type(self) -> str:
        if self.faiss_index is not None:
            return str(self.build_info.get("index_type", "faiss"))
        return str(self.build_info.get("index_type", "numpy_exact"))

    @classmethod
    def load(
        cls,
        metadata_path: Path,
        embeddings_path: Path,
        build_info_path: Path,
        index_path: Path,
    ) -> "IndexBundle":
        if not metadata_path.exists() or not embeddings_path.exists() or not build_info_path.exists():
            return cls(
                metadata=pd.DataFrame(columns=["id", "lon", "lat", "tile_path", "row", "col"]),
                embeddings=np.zeros((0, 64), dtype=np.float32),
                build_info={},
                faiss_index=None,
            )

        metadata = pd.read_parquet(metadata_path)
        embeddings = np.load(embeddings_path, mmap_mode="r")
        build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
        faiss_index = None
        if faiss is not None and index_path.exists():
            faiss_index = faiss.read_index(str(index_path))
        return cls(metadata=metadata, embeddings=embeddings, build_info=build_info, faiss_index=faiss_index)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        bbox: Optional[list[float]] = None,
        min_distance_m: float = 0,
        min_score: float = 0.0,
    ) -> list[dict]:
        if self.vector_count == 0:
            return []

        query = normalize_embeddings(query.reshape(1, -1))[0]
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            mask = (
                (self.metadata["lon"] >= min_lon)
                & (self.metadata["lon"] <= max_lon)
                & (self.metadata["lat"] >= min_lat)
                & (self.metadata["lat"] <= max_lat)
            )
            candidate_ids = self.metadata.loc[mask, "id"].to_numpy(dtype=np.int64)
            return self._search_exact(query, top_k, candidate_ids, min_distance_m=min_distance_m, min_score=min_score)

        if self.faiss_index is not None and min_distance_m <= 0 and min_score <= 0:
            scores, indices = self.faiss_index.search(query.reshape(1, -1), top_k)
            return self._serialize_matches(indices[0], scores[0])

        return self._search_exact(query, top_k, None, min_distance_m=min_distance_m, min_score=min_score)

    def _search_exact(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_ids: Optional[np.ndarray],
        min_distance_m: float = 0,
        min_score: float = 0.0,
    ) -> list[dict]:
        if candidate_ids is None:
            candidate_ids = self.metadata["id"].to_numpy(dtype=np.int64)
        if len(candidate_ids) == 0:
            return []
        candidate_vectors = self.embeddings[candidate_ids]
        scores = candidate_vectors @ query
        order = np.argsort(scores)[::-1]
        ordered_ids = candidate_ids[order]
        ordered_scores = scores[order]
        if min_score > 0:
            keep_mask = ordered_scores >= min_score
            ordered_ids = ordered_ids[keep_mask]
            ordered_scores = ordered_scores[keep_mask]
        if min_distance_m > 0:
            ordered_ids, ordered_scores = self._apply_min_distance_filter(
                ordered_ids,
                ordered_scores,
                min_distance_m,
                top_k,
            )
        else:
            ordered_ids = ordered_ids[:top_k]
            ordered_scores = ordered_scores[:top_k]
        return self._serialize_matches(ordered_ids, ordered_scores)

    def _apply_min_distance_filter(
        self,
        ordered_ids: np.ndarray,
        ordered_scores: np.ndarray,
        min_distance_m: float,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        kept_ids: list[int] = []
        kept_scores: list[float] = []
        kept_points: list[tuple[float, float]] = []

        for index, score in zip(ordered_ids, ordered_scores):
            row = self.metadata.iloc[int(index)]
            lon = float(row["lon"])
            lat = float(row["lat"])
            if any(_haversine_distance_m(lon, lat, kept_lon, kept_lat) < min_distance_m for kept_lon, kept_lat in kept_points):
                continue
            kept_ids.append(int(index))
            kept_scores.append(float(score))
            kept_points.append((lon, lat))
            if len(kept_ids) >= top_k:
                break

        return np.asarray(kept_ids, dtype=np.int64), np.asarray(kept_scores, dtype=np.float32)

    def _serialize_matches(self, indices, scores) -> list[dict]:
        results: list[dict] = []
        for rank, (index, score) in enumerate(zip(indices, scores), start=1):
            if index is None or int(index) < 0:
                continue
            row = self.metadata.iloc[int(index)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "lon": float(row["lon"]),
                    "lat": float(row["lat"]),
                    "embedding": self.embeddings[int(index)].astype(np.float32).tolist(),
                    "tile_path": str(row["tile_path"]),
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                }
            )
        return results


def _haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1 - a, 0)))
    return 6_371_000 * c
