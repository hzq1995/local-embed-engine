from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.index_service import normalize_embeddings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_projection(source_dim: int, reduced_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0, size=(source_dim, reduced_dim)).astype(np.float32)
    norms = np.linalg.norm(projection, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return projection / norms


def _select_coarse_ids(metadata: pd.DataFrame, stride: int) -> np.ndarray:
    if stride <= 1:
        return metadata["id"].to_numpy(dtype=np.int64)
    mask = ((metadata["row"].to_numpy(dtype=np.int64) % stride) == 0) & (
        (metadata["col"].to_numpy(dtype=np.int64) % stride) == 0
    )
    ids = metadata.loc[mask, "id"].to_numpy(dtype=np.int64)
    if ids.size == 0:
        ids = metadata["id"].to_numpy(dtype=np.int64)[::stride]
    return ids


def build_coarse_index(
    derived_dir: Path,
    stride: int = 4,
    reduced_dim: int = 128,
    block_rows: int = 250_000,
    projection_seed: int = 20240424,
) -> dict:
    metadata_path = derived_dir / "metadata.parquet"
    embeddings_path = derived_dir / "embeddings.npy"
    build_info_path = derived_dir / "build_info.json"
    coarse_embeddings_path = derived_dir / "coarse_embeddings_i8.npy"
    coarse_ids_path = derived_dir / "coarse_ids.npy"
    coarse_projection_path = derived_dir / "coarse_projection.npy"
    coarse_info_path = derived_dir / "coarse_info.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {embeddings_path}")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if reduced_dim < 1:
        raise ValueError("reduced_dim must be >= 1")

    metadata = pd.read_parquet(metadata_path)
    source_embeddings = np.load(embeddings_path, mmap_mode="r")
    if source_embeddings.ndim != 2:
        raise ValueError("embeddings.npy must be a 2D array.")
    source_count = int(source_embeddings.shape[0])
    source_dim = int(source_embeddings.shape[1])
    if reduced_dim > source_dim:
        raise ValueError("reduced_dim must be <= source embedding dimension.")

    coarse_ids = _select_coarse_ids(metadata, stride)
    if np.any(coarse_ids < 0) or np.any(coarse_ids >= source_count):
        raise ValueError("metadata contains ids outside embeddings.npy row range.")

    use_random_projection = reduced_dim != source_dim
    if use_random_projection:
        projection = _build_projection(source_dim, reduced_dim, projection_seed)
    else:
        projection = np.eye(source_dim, dtype=np.float32)
    np.save(coarse_projection_path, projection.astype(np.float32, copy=False))
    np.save(coarse_ids_path, coarse_ids.astype(np.int64, copy=False))

    output = np.lib.format.open_memmap(
        coarse_embeddings_path,
        mode="w+",
        dtype=np.int8,
        shape=(int(coarse_ids.shape[0]), reduced_dim),
    )
    for start in range(0, int(coarse_ids.shape[0]), block_rows):
        end = min(start + block_rows, int(coarse_ids.shape[0]))
        ids = coarse_ids[start:end]
        vectors = np.asarray(source_embeddings[ids], dtype=np.float32)
        if use_random_projection:
            projected = normalize_embeddings(vectors @ projection)
        else:
            projected = normalize_embeddings(vectors)
        quantized = np.clip(np.rint(projected * 127.0), -127, 127).astype(np.int8)
        output[start:end] = quantized
        print(f"[build_coarse_index] rows {start}:{end} / {coarse_ids.shape[0]}", flush=True)
    output.flush()
    del output

    source_build_info = {}
    if build_info_path.exists():
        source_build_info = json.loads(build_info_path.read_text(encoding="utf-8"))

    info = {
        "build_time": datetime.now(timezone.utc).isoformat(),
        "source_metadata_path": str(metadata_path),
        "source_embeddings_path": str(embeddings_path),
        "source_build_info_path": str(build_info_path),
        "source_vector_count": source_count,
        "source_dim": source_dim,
        "coarse_vector_count": int(coarse_ids.shape[0]),
        "reduced_dim": reduced_dim,
        "stride": stride,
        "projection_seed": projection_seed,
        "quantization": "symmetric_int8_scale_127",
        "coarse_embeddings_path": str(coarse_embeddings_path),
        "coarse_ids_path": str(coarse_ids_path),
        "coarse_projection_path": str(coarse_projection_path),
        "projection_sha256": _sha256(coarse_projection_path),
        "source_index_type": source_build_info.get("index_type"),
    }
    coarse_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sparse low-dim int8 coarse index from derived embeddings.")
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--reduced-dim", type=int, default=128)
    parser.add_argument("--block-rows", type=int, default=250_000)
    parser.add_argument("--projection-seed", type=int, default=20240424)
    args = parser.parse_args()

    info = build_coarse_index(
        derived_dir=args.derived_dir,
        stride=args.stride,
        reduced_dim=args.reduced_dim,
        block_rows=args.block_rows,
        projection_seed=args.projection_seed,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
