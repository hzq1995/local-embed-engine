from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.build_service import build_index


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build Ningbo local AEF index.")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--boundary-kml", type=Path, default=settings.boundary_kml_path)
    parser.add_argument("--output-dir", type=Path, default=settings.derived_dir)
    parser.add_argument("--year", type=int, default=settings.year)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--flush-rows", type=int, default=250000)
    parser.add_argument("--with-faiss", action="store_true", help="Build FAISS index (disabled by default, only embeddings.npy)")
    args = parser.parse_args()

    result = build_index(
        data_dir=args.data_dir,
        boundary_kml_path=args.boundary_kml,
        output_dir=args.output_dir,
        year=args.year,
        block_size=args.block_size,
        flush_rows=args.flush_rows,
        skip_faiss=not args.with_faiss,
    )
    print(
        f"Built index in {result.output_dir} with {result.vector_count} vectors from "
        f"{result.tile_count} tiles using {result.index_type}."
    )


if __name__ == "__main__":
    main()
