from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


NPY_MEMORY_LOAD_THRESHOLD_BYTES = 20 * 1024**3


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclass(slots=True)
class CoarseIndex:
    embeddings: np.ndarray
    ids: np.ndarray
    projection: np.ndarray
    info: dict
    block_rows: int = 250_000

    @property
    def loaded(self) -> bool:
        return self.embeddings.shape[0] > 0

    @property
    def vector_count(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    @property
    def stride(self) -> int | None:
        value = self.info.get("stride")
        return int(value) if value is not None else None


@dataclass(slots=True)
class IndexBundle:
    metadata: pd.DataFrame
    embeddings: np.ndarray
    build_info: dict
    faiss_index: object | None = None
    coarse_index: CoarseIndex | None = None

    @property
    def vector_count(self) -> int:
        return int(len(self.metadata))

    @property
    def index_loaded(self) -> bool:
        return self.vector_count > 0

    @property
    def index_type(self) -> str:
        if self.coarse_index is not None and self.coarse_index.loaded:
            return str(self.build_info.get("index_type", "numpy_exact")) + "+coarse"
        if self.faiss_index is not None:
            return str(self.build_info.get("index_type", "faiss"))
        return str(self.build_info.get("index_type", "numpy_exact"))

    @property
    def coarse_index_loaded(self) -> bool:
        return self.coarse_index is not None and self.coarse_index.loaded

    @classmethod
    def load(
        cls,
        metadata_path: Path,
        embeddings_path: Path,
        build_info_path: Path,
        index_path: Path,
        coarse_embeddings_path: Path | None = None,
        coarse_ids_path: Path | None = None,
        coarse_projection_path: Path | None = None,
        coarse_info_path: Path | None = None,
        coarse_block_rows: int = 250_000,
    ) -> "IndexBundle":
        if not metadata_path.exists() or not embeddings_path.exists() or not build_info_path.exists():
            return cls(
                metadata=pd.DataFrame(columns=["id", "lon", "lat", "tile_path", "row", "col"]),
                embeddings=np.zeros((0, 64), dtype=np.float32),
                build_info={},
                faiss_index=None,
            )

        metadata = pd.read_parquet(metadata_path)
        embeddings = _load_npy_array(embeddings_path)
        build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
        faiss_index = None
        if faiss is not None and index_path.exists():
            faiss_index = faiss.read_index(str(index_path))
        coarse_index = _load_coarse_index(
            coarse_embeddings_path=coarse_embeddings_path,
            coarse_ids_path=coarse_ids_path,
            coarse_projection_path=coarse_projection_path,
            coarse_info_path=coarse_info_path,
            coarse_block_rows=coarse_block_rows,
        )
        return cls(
            metadata=metadata,
            embeddings=embeddings,
            build_info=build_info,
            faiss_index=faiss_index,
            coarse_index=coarse_index,
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        bbox: Optional[list[float]] = None,
        min_distance_m: float = 0,
        min_score: float = 0.0,
        search_mode: str = "fine",
    ) -> list[dict]:
        if self.vector_count == 0:
            return []

        query = normalize_embeddings(query.reshape(1, -1))[0]
        if search_mode == "coarse":
            return self._search_coarse(query, top_k, bbox, min_distance_m=min_distance_m, min_score=min_score)

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

    def _search_coarse(
        self,
        query: np.ndarray,
        top_k: int,
        bbox: Optional[list[float]],
        min_distance_m: float = 0,
        min_score: float = 0.0,
    ) -> list[dict]:
        coarse = self.coarse_index
        if coarse is None or not coarse.loaded:
            raise ValueError("Coarse index is not available; build coarse index first.")
        if query.shape[0] != coarse.projection.shape[0]:
            raise ValueError(
                f"query embedding dim {query.shape[0]} does not match coarse projection source dim "
                f"{coarse.projection.shape[0]}."
            )

        projected = query.astype(np.float32, copy=False) @ coarse.projection.astype(np.float32, copy=False)
        projected = normalize_embeddings(projected.reshape(1, -1))[0]

        candidate_mask = None
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            coarse_ids = coarse.ids.astype(np.int64, copy=False)
            coarse_md = self.metadata.iloc[coarse_ids]
            candidate_mask = (
                (coarse_md["lon"].to_numpy() >= min_lon)
                & (coarse_md["lon"].to_numpy() <= max_lon)
                & (coarse_md["lat"].to_numpy() >= min_lat)
                & (coarse_md["lat"].to_numpy() <= max_lat)
            )
            if not np.any(candidate_mask):
                return []

        keep_count = max(top_k, min(top_k * 20, max(top_k, 500)))
        best_scores = np.empty((0,), dtype=np.float32)
        best_positions = np.empty((0,), dtype=np.int64)
        block_rows = max(int(coarse.block_rows), 1)
        total = coarse.vector_count

        for start in range(0, total, block_rows):
            end = min(start + block_rows, total)
            if candidate_mask is None:
                local_positions = np.arange(start, end, dtype=np.int64)
                block = coarse.embeddings[start:end]
            else:
                local_mask = candidate_mask[start:end]
                if not np.any(local_mask):
                    continue
                local_offsets = np.flatnonzero(local_mask).astype(np.int64)
                local_positions = start + local_offsets
                block = coarse.embeddings[start:end][local_offsets]
            scores = (np.asarray(block, dtype=np.float32) @ projected) / 127.0
            if min_score > 0:
                score_mask = scores >= min_score
                if not np.any(score_mask):
                    continue
                scores = scores[score_mask]
                local_positions = local_positions[score_mask]
            if scores.size == 0:
                continue
            best_scores = np.concatenate([best_scores, scores.astype(np.float32, copy=False)])
            best_positions = np.concatenate([best_positions, local_positions])
            if best_scores.size > keep_count:
                keep = np.argpartition(best_scores, -keep_count)[-keep_count:]
                best_scores = best_scores[keep]
                best_positions = best_positions[keep]

        if best_scores.size == 0:
            return []
        order = np.argsort(best_scores)[::-1]
        ordered_positions = best_positions[order]
        ordered_scores = best_scores[order]
        ordered_ids = coarse.ids[ordered_positions].astype(np.int64, copy=False)
        if min_distance_m > 0:
            ordered_positions, ordered_ids, ordered_scores = self._apply_min_distance_filter_with_positions(
                ordered_positions,
                ordered_ids,
                ordered_scores,
                min_distance_m,
                top_k,
            )
        else:
            ordered_ids = ordered_ids[:top_k]
            ordered_scores = ordered_scores[:top_k]
        return self._serialize_coarse_matches(ordered_positions[: len(ordered_ids)], ordered_ids, ordered_scores)

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

    def _apply_min_distance_filter_with_positions(
        self,
        ordered_positions: np.ndarray,
        ordered_ids: np.ndarray,
        ordered_scores: np.ndarray,
        min_distance_m: float,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kept_positions: list[int] = []
        kept_ids: list[int] = []
        kept_scores: list[float] = []
        kept_points: list[tuple[float, float]] = []

        for coarse_position, index, score in zip(ordered_positions, ordered_ids, ordered_scores):
            row = self.metadata.iloc[int(index)]
            lon = float(row["lon"])
            lat = float(row["lat"])
            if any(_haversine_distance_m(lon, lat, kept_lon, kept_lat) < min_distance_m for kept_lon, kept_lat in kept_points):
                continue
            kept_positions.append(int(coarse_position))
            kept_ids.append(int(index))
            kept_scores.append(float(score))
            kept_points.append((lon, lat))
            if len(kept_ids) >= top_k:
                break

        return (
            np.asarray(kept_positions, dtype=np.int64),
            np.asarray(kept_ids, dtype=np.int64),
            np.asarray(kept_scores, dtype=np.float32),
        )

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

    def _serialize_coarse_matches(self, coarse_positions, indices, scores) -> list[dict]:
        if self.coarse_index is None:
            return []
        results: list[dict] = []
        for rank, (coarse_position, index, score) in enumerate(zip(coarse_positions, indices, scores), start=1):
            if index is None or int(index) < 0:
                continue
            row = self.metadata.iloc[int(index)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "lon": float(row["lon"]),
                    "lat": float(row["lat"]),
                    "embedding": self.coarse_index.embeddings[int(coarse_position)].astype(np.float32).tolist(),
                    "tile_path": str(row["tile_path"]),
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                }
            )
        return results


def _load_coarse_index(
    *,
    coarse_embeddings_path: Path | None,
    coarse_ids_path: Path | None,
    coarse_projection_path: Path | None,
    coarse_info_path: Path | None,
    coarse_block_rows: int,
) -> CoarseIndex | None:
    paths = [coarse_embeddings_path, coarse_ids_path, coarse_projection_path, coarse_info_path]
    if any(path is None for path in paths):
        return None
    assert coarse_embeddings_path is not None
    assert coarse_ids_path is not None
    assert coarse_projection_path is not None
    assert coarse_info_path is not None
    if not (
        coarse_embeddings_path.exists()
        and coarse_ids_path.exists()
        and coarse_projection_path.exists()
        and coarse_info_path.exists()
    ):
        return None
    embeddings = _load_npy_array(coarse_embeddings_path)
    ids = _load_npy_array(coarse_ids_path)
    projection = _load_npy_array(coarse_projection_path)
    info = json.loads(coarse_info_path.read_text(encoding="utf-8"))
    expected_projection_hash = info.get("projection_sha256")
    if expected_projection_hash and expected_projection_hash != _sha256(coarse_projection_path):
        raise ValueError("coarse_projection.npy sha256 does not match coarse_info.json.")
    if embeddings.ndim != 2:
        raise ValueError("coarse_embeddings_i8.npy must be a 2D array.")
    if ids.ndim != 1 or ids.shape[0] != embeddings.shape[0]:
        raise ValueError("coarse_ids.npy length must match coarse_embeddings_i8.npy rows.")
    if projection.ndim != 2 or projection.shape[1] != embeddings.shape[1]:
        raise ValueError("coarse_projection.npy shape must match coarse embedding dimension.")
    return CoarseIndex(
        embeddings=embeddings,
        ids=ids,
        projection=projection,
        info=info,
        block_rows=coarse_block_rows,
    )


def _load_npy_array(path: Path) -> np.ndarray:
    if path.stat().st_size < NPY_MEMORY_LOAD_THRESHOLD_BYTES:
        return np.load(path)
    return np.load(path, mmap_mode="r")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
