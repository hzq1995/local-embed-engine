from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


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
    top_k: int = Field(default=10, ge=1, le=1000)
    min_distance_m: float = Field(default=0, ge=0)
    min_score: float = Field(default=0.0, ge=0, le=1)
    bbox: List[float]

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: List[float]) -> List[float]:
        if len(value) == 0:
            raise ValueError("embedding must not be empty.")
        return [float(item) for item in value]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: List[float]) -> List[float]:
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
