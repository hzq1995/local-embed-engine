from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.schemas import (
    EmbeddingSearchRequest,
    EmbeddingSearchResponse,
    PointEmbeddingResponse,
    PointRequest,
)
from app.services.boundary_service import BoundaryService
from app.services.catalog_service import TileCatalog
from app.services.query_service import QueryService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        boundary = BoundaryService.from_kml(settings.boundary_kml_path)
        catalog = TileCatalog.scan(settings.data_dir, expected_band_count=settings.embedding_dim)
        app.state.query_service = QueryService(
            year=settings.year,
            boundary=boundary,
            catalog=catalog,
            search_block_size=settings.search_block_size,
            max_bbox_area_km2=settings.max_bbox_area_km2,
            embedding_dim=catalog.embedding_dim,
        )
        yield

    app = FastAPI(title=settings.service_name, version="0.1.0", lifespan=lifespan)

    @app.post("/embedding/by-point", response_model=PointEmbeddingResponse)
    def embedding_by_point(request: PointRequest) -> PointEmbeddingResponse:
        service = app.state.query_service
        try:
            return PointEmbeddingResponse.model_validate(service.get_embedding_by_point(request.lon, request.lat))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/search/by-embedding", response_model=EmbeddingSearchResponse)
    def search_by_embedding(request: EmbeddingSearchRequest) -> EmbeddingSearchResponse:
        service = app.state.query_service
        try:
            payload = service.search_by_embedding(
                embedding=request.embedding,
                top_k=request.top_k,
                bbox=request.bbox,
                min_distance_m=request.min_distance_m,
                min_score=request.min_score,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EmbeddingSearchResponse.model_validate(payload)

    return app


app = create_app()
