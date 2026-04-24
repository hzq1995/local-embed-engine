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
    search_block_size: int
    max_bbox_area_km2: float
    embedding_dim: int | None = None


def _parse_optional_int(raw_value: str | None) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    return int(raw_value)


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[1]
    return Settings(
        service_name=os.getenv("TIF_SEARCH_SERVICE_NAME", "Ningbo TIF Search Engine"),
        host=os.getenv("TIF_SEARCH_HOST", "0.0.0.0"),
        port=int(os.getenv("TIF_SEARCH_PORT", "8010")),
        year=int(os.getenv("TIF_SEARCH_YEAR", "2024")),
        data_dir=Path(os.getenv("TIF_SEARCH_DATA_DIR", "/mnt_llm_A100_V1/aef-zhejiang/2024/51N")),
        boundary_kml_path=Path(os.getenv("TIF_SEARCH_BOUNDARY_KML", root_dir / "data" / "ningbo.kml")),
        search_block_size=int(os.getenv("TIF_SEARCH_BLOCK_SIZE", "512")),
        max_bbox_area_km2=float(os.getenv("TIF_SEARCH_MAX_BBOX_AREA_KM2", "500")),
        embedding_dim=_parse_optional_int(os.getenv("TIF_SEARCH_EMBEDDING_DIM")),
    )
