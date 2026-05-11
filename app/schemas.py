from __future__ import annotations

from typing import Literal, List, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    year: int
    index_loaded: bool
    vector_count: int
    build_time: Optional[str] = None
    index_type: Optional[str] = None
    coarse_index_loaded: bool = False
    coarse_vector_count: int = 0
    coarse_embedding_dim: Optional[int] = None
    coarse_stride: Optional[int] = None


class IndexInfoResponse(BaseModel):
    year: int
    data_dir: str
    boundary_kml_path: str
    boundary_cache_path: str
    metadata_path: str
    embeddings_path: str
    index_path: str
    tile_count: int
    vector_count: int
    build_time: Optional[str] = None
    index_type: Optional[str] = None
    coarse_index_loaded: bool = False
    coarse_vector_count: int = 0
    coarse_embedding_dim: Optional[int] = None
    coarse_stride: Optional[int] = None


class PointRequest(BaseModel):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)


class PointEmbeddingResponse(BaseModel):
    year: int
    lon: float
    lat: float
    embedding: List[float]
    tile_path: str
    row: int
    col: int


class EmbeddingSearchRequest(BaseModel):
    embedding: List[float]
    search_mode: Literal["fine", "coarse"] = "coarse"
    top_k: int = Field(default=10, ge=1, le=1000)
    min_distance_m: float = Field(default=0, ge=0)
    min_score: float = Field(default=0.0, ge=0, le=1)
    bbox: Optional[List[float]] = None

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: List[float]) -> List[float]:
        if len(value) == 0:
            raise ValueError("embedding must not be empty.")
        return [float(item) for item in value]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("bbox must be [minLon, minLat, maxLon, maxLat].")
        min_lon, min_lat, max_lon, max_lat = value
        if min_lon > max_lon or min_lat > max_lat:
            raise ValueError("bbox min coordinates must be <= max coordinates.")
        return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


class SearchResult(BaseModel):
    rank: int
    score: float
    lon: float
    lat: float
    embedding: List[float]
    tile_path: str
    row: int
    col: int


class EmbeddingSearchResponse(BaseModel):
    top_k: int
    result_count: int
    results: List[SearchResult]


class BboxEmbeddingRequest(BaseModel):
    bbox: List[float]
    total_samples: int = Field(default=5000, ge=100, le=50000)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: List[float]) -> List[float]:
        if len(value) != 4:
            raise ValueError("bbox must be [minLon, minLat, maxLon, maxLat].")
        min_lon, min_lat, max_lon, max_lat = value
        if min_lon > max_lon or min_lat > max_lat:
            raise ValueError("bbox min coordinates must be <= max coordinates.")
        return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


class BboxEmbeddingResponse(BaseModel):
    count: int
    lons: List[float]
    lats: List[float]
    embeddings: List[List[float]]
    grid_rows: int
    grid_cols: int
    grid_row_indices: List[int]
    grid_col_indices: List[int]
    effective_bbox: List[float]
