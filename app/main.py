from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.schemas import (
    BboxEmbeddingRequest,
    BboxEmbeddingResponse,
    EmbeddingSearchRequest,
    EmbeddingSearchResponse,
    HealthResponse,
    IndexInfoResponse,
    PointEmbeddingResponse,
    PointRequest,
)
from app.services.boundary_service import BoundaryService
from app.services.catalog_service import TileCatalog
from app.services.index_service import IndexBundle
from app.services.query_service import QueryService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.derived_dir.mkdir(parents=True, exist_ok=True)
        boundary = BoundaryService.from_kml(settings.boundary_kml_path, cache_path=settings.boundary_cache_path)
        catalog = TileCatalog.scan(settings.data_dir) if settings.data_dir.exists() else TileCatalog([])
        index_bundle = IndexBundle.load(
            metadata_path=settings.metadata_path,
            embeddings_path=settings.embeddings_path,
            build_info_path=settings.build_info_path,
            index_path=settings.index_path,
            coarse_embeddings_path=settings.coarse_embeddings_path,
            coarse_ids_path=settings.coarse_ids_path,
            coarse_projection_path=settings.coarse_projection_path,
            coarse_info_path=settings.coarse_info_path,
            coarse_block_rows=settings.coarse_block_rows,
        )
        app.state.settings = settings
        app.state.query_service = QueryService(
            year=settings.year,
            boundary=boundary,
            catalog=catalog,
            index_bundle=index_bundle,
        )
        yield

    app = FastAPI(title=settings.service_name, version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        service = app.state.query_service
        build_info = service.index_bundle.build_info
        return HealthResponse(
            status="ok",
            service=settings.service_name,
            year=settings.year,
            index_loaded=service.index_bundle.index_loaded,
            vector_count=service.index_bundle.vector_count,
            build_time=build_info.get("build_time"),
            index_type=service.index_bundle.index_type if service.index_bundle.vector_count else build_info.get("index_type"),
            coarse_index_loaded=service.index_bundle.coarse_index_loaded,
            coarse_vector_count=service.index_bundle.coarse_index.vector_count if service.index_bundle.coarse_index else 0,
            coarse_embedding_dim=service.index_bundle.coarse_index.embedding_dim if service.index_bundle.coarse_index else None,
            coarse_stride=service.index_bundle.coarse_index.stride if service.index_bundle.coarse_index else None,
        )

    @app.get("/index/info", response_model=IndexInfoResponse)
    def index_info() -> IndexInfoResponse:
        service = app.state.query_service
        build_info = service.index_bundle.build_info
        return IndexInfoResponse(
            year=settings.year,
            data_dir=str(settings.data_dir),
            boundary_kml_path=str(settings.boundary_kml_path),
            boundary_cache_path=str(settings.boundary_cache_path),
            metadata_path=str(settings.metadata_path),
            embeddings_path=str(settings.embeddings_path),
            index_path=str(settings.index_path),
            tile_count=int(build_info.get("tile_count", 0)),
            vector_count=service.index_bundle.vector_count,
            build_time=build_info.get("build_time"),
            index_type=service.index_bundle.index_type if service.index_bundle.vector_count else build_info.get("index_type"),
            coarse_index_loaded=service.index_bundle.coarse_index_loaded,
            coarse_vector_count=service.index_bundle.coarse_index.vector_count if service.index_bundle.coarse_index else 0,
            coarse_embedding_dim=service.index_bundle.coarse_index.embedding_dim if service.index_bundle.coarse_index else None,
            coarse_stride=service.index_bundle.coarse_index.stride if service.index_bundle.coarse_index else None,
        )

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
            return EmbeddingSearchResponse.model_validate(
                service.search_by_embedding(
                    request.embedding,
                    request.top_k,
                    request.bbox,
                    request.min_distance_m,
                    request.min_score,
                    request.search_mode,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/embedding/by-bbox", response_model=BboxEmbeddingResponse)
    def embedding_by_bbox(request: BboxEmbeddingRequest) -> BboxEmbeddingResponse:
        service = app.state.query_service
        return BboxEmbeddingResponse.model_validate(
            service.get_embeddings_by_bbox(request.bbox, request.total_samples)
        )

    return app


app = create_app()
