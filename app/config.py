from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    service_name: str
    host: str
    port: int
    year: int
    data_dir: Path
    boundary_kml_path: Path
    derived_dir: Path
    boundary_cache_path: Path
    metadata_path: Path
    embeddings_path: Path
    index_path: Path
    build_info_path: Path


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[1]
    derived_dir = Path(os.getenv("LOCAL_AEF_DERIVED_DIR", root_dir / "data" / "derived"))
    return Settings(
        service_name=os.getenv("LOCAL_AEF_SERVICE_NAME", "Ningbo Local Embed Engine"),
        host=os.getenv("LOCAL_AEF_HOST", "0.0.0.0"),
        port=int(os.getenv("LOCAL_AEF_PORT", "8010")),
        year=int(os.getenv("LOCAL_AEF_YEAR", "2024")),
        data_dir=Path(os.getenv("LOCAL_AEF_DATA_DIR", "/mnt_llm_A100_V1/aef-zhejiang/2024/51N")),
        boundary_kml_path=Path(os.getenv("LOCAL_AEF_BOUNDARY_KML", root_dir / "宁波市.kml")),
        derived_dir=derived_dir,
        boundary_cache_path=Path(os.getenv("LOCAL_AEF_BOUNDARY_CACHE", derived_dir / "ningbo_boundary.geojson")),
        metadata_path=Path(os.getenv("LOCAL_AEF_METADATA_PATH", derived_dir / "metadata.parquet")),
        embeddings_path=Path(os.getenv("LOCAL_AEF_EMBEDDINGS_PATH", derived_dir / "embeddings.npy")),
        index_path=Path(os.getenv("LOCAL_AEF_INDEX_PATH", derived_dir / "faiss.index")),
        build_info_path=Path(os.getenv("LOCAL_AEF_BUILD_INFO_PATH", derived_dir / "build_info.json")),
    )
