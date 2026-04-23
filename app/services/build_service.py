from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from app.services.boundary_service import BoundaryService
from app.services.catalog_service import TileCatalog
from app.services.index_service import normalize_embeddings

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(slots=True)
class BuildResult:
    tile_count: int
    vector_count: int
    output_dir: Path
    index_type: str


def _iter_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row_off in range(0, height, block_size):
        for col_off in range(0, width, block_size):
            yield Window(col_off=col_off, row_off=row_off, width=min(block_size, width - col_off), height=min(block_size, height - row_off))


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _format_window(window: Window) -> str:
    return (
        f"row_off={int(window.row_off)} col_off={int(window.col_off)} "
        f"width={int(window.width)} height={int(window.height)}"
    )


def _log_build(message: str) -> None:
    print(f"[build_index] {message}", flush=True)


class IncrementalBuildWriter:
    def __init__(self, output_dir: Path, flush_rows: int = 250_000):
        self.output_dir = output_dir
        self.flush_rows = flush_rows
        self.temp_dir = Path(tempfile.mkdtemp(prefix="build_tmp_", dir=output_dir))
        self.metadata_tmp_path = self.temp_dir / "metadata.parquet"
        self.parquet_writer: pq.ParquetWriter | None = None
        self.embedding_chunk_paths: list[Path] = []
        self.metadata_buffer: list[dict] = []
        self.embedding_buffer: list[np.ndarray] = []
        self.total_rows = 0
        self.chunk_index = 0
        self.dim: int | None = None

    def append(self, metadata_rows: list[dict], embeddings: np.ndarray) -> None:
        if len(metadata_rows) == 0:
            return
        if self.dim is None:
            self.dim = embeddings.shape[1]
        self.metadata_buffer.extend(metadata_rows)
        self.embedding_buffer.append(embeddings.astype(np.float32, copy=False))
        buffered_rows = sum(chunk.shape[0] for chunk in self.embedding_buffer)
        if buffered_rows >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.metadata_buffer:
            return

        metadata_df = pd.DataFrame(
            self.metadata_buffer,
            columns=["id", "lon", "lat", "tile_path", "row", "col"],
        )
        table = pa.Table.from_pandas(metadata_df, preserve_index=False)
        if self.parquet_writer is None:
            self.parquet_writer = pq.ParquetWriter(self.metadata_tmp_path, table.schema)
        self.parquet_writer.write_table(table)

        embeddings = np.vstack(self.embedding_buffer).astype(np.float32, copy=False)
        chunk_path = self.temp_dir / f"embeddings_{self.chunk_index:05d}.npy"
        np.save(chunk_path, embeddings)
        self.embedding_chunk_paths.append(chunk_path)

        self.total_rows += len(metadata_df)
        self.chunk_index += 1
        self.metadata_buffer.clear()
        self.embedding_buffer.clear()

    def finalize(self, metadata_path: Path, embeddings_path: Path) -> int:
        self.flush()
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            shutil.move(self.metadata_tmp_path, metadata_path)
        else:
            empty_df = pd.DataFrame(columns=["id", "lon", "lat", "tile_path", "row", "col"])
            empty_df.to_parquet(metadata_path, index=False)

        dim = self.dim if self.dim is not None else 64
        output = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.total_rows, dim),
        )
        cursor = 0
        for chunk_path in self.embedding_chunk_paths:
            chunk = np.load(chunk_path, mmap_mode="r")
            next_cursor = cursor + chunk.shape[0]
            output[cursor:next_cursor] = chunk
            cursor = next_cursor
        output.flush()
        del output
        return self.total_rows

    def cleanup(self) -> None:
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def build_index(
    data_dir: Path,
    boundary_kml_path: Path,
    output_dir: Path,
    year: int = 2024,
    block_size: int = 512,
    flush_rows: int = 250_000,
    skip_faiss: bool = True,
) -> BuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_cache_path = output_dir / "ningbo_boundary.geojson"
    boundary = BoundaryService.from_kml(boundary_kml_path, cache_path=boundary_cache_path)
    catalog = TileCatalog.scan(data_dir)
    candidate_tiles = list(catalog.iter_intersecting_bbox(boundary.bbox))

    next_id = 0
    selected_tiles = []
    writer = IncrementalBuildWriter(output_dir=output_dir, flush_rows=flush_rows)
    metadata_path = output_dir / "metadata.parquet"
    embeddings_path = output_dir / "embeddings.npy"
    index_path = output_dir / "faiss.index"
    build_info_path = output_dir / "build_info.json"

    try:
        tile_iterator = _progress(candidate_tiles, desc="Tiles", unit="tile")
        total_candidate_tiles = len(candidate_tiles)
        for tile_number, tile in enumerate(tile_iterator, start=1):
            with rasterio.open(tile.path) as ds:
                tile_bounds = rasterio.warp.transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
                if not boundary.geometry.intersects(box(*tile_bounds)):
                    continue
                selected_tiles.append(str(tile.path))

                windows = list(_iter_windows(ds.width, ds.height, block_size))
                tile_start_rows = writer.total_rows + len(writer.metadata_buffer)
                _log_build(
                    "tile_start "
                    f"tile_index={tile_number}/{total_candidate_tiles} "
                    f"path={tile.path} width={ds.width} height={ds.height} "
                    f"bands={ds.count} nodata={ds.nodata} blocks={len(windows)}"
                )
                window_iterator = _progress(
                    windows,
                    desc=f"Blocks {tile.path.name}",
                    unit="block",
                    leave=False,
                )
                for block_number, window in enumerate(window_iterator, start=1):
                    block = ds.read(window=window)
                    nodata = ds.nodata
                    valid_mask = np.ones((int(window.height), int(window.width)), dtype=bool)
                    if nodata is not None:
                        valid_mask &= np.all(block != nodata, axis=0)
                    valid_pixel_count = int(valid_mask.sum())
                    _log_build(
                        "block_scan "
                        f"tile={tile.path.name} block={block_number}/{len(windows)} "
                        f"{_format_window(window)} valid_pixels={valid_pixel_count} "
                        f"buffered_vectors={writer.total_rows + len(writer.metadata_buffer)}"
                    )
                    if not np.any(valid_mask):
                        continue

                    rows, cols = np.where(valid_mask)
                    rows = rows + int(window.row_off)
                    cols = cols + int(window.col_off)
                    lons, lats = catalog.transform_pixel_centers_to_wgs84(ds, rows, cols)
                    inside_mask = np.asarray(boundary.contains_xy(lons, lats), dtype=bool)
                    inside_pixel_count = int(inside_mask.sum())
                    _log_build(
                        "block_filter "
                        f"tile={tile.path.name} block={block_number}/{len(windows)} "
                        f"{_format_window(window)} inside_pixels={inside_pixel_count} "
                        f"valid_pixels={valid_pixel_count}"
                    )
                    if not np.any(inside_mask):
                        continue

                    rows = rows[inside_mask]
                    cols = cols[inside_mask]
                    lons = lons[inside_mask]
                    lats = lats[inside_mask]
                    local_rows = rows - int(window.row_off)
                    local_cols = cols - int(window.col_off)
                    raw_vectors = block[:, local_rows, local_cols].T.astype(np.float32)
                    zero_mask = np.all(raw_vectors == 0, axis=1)
                    zero_count = int(zero_mask.sum())
                    if zero_count > 0:
                        first_zero_index = int(np.flatnonzero(zero_mask)[0])
                        error_message = (
                            "Encountered all-zero vector while building index: "
                            f"tile_path={tile.path}, {_format_window(window)}, "
                            f"valid_pixels={valid_pixel_count}, inside_pixels={inside_pixel_count}, "
                            f"zero_vector_count={zero_count}, "
                            f"row={int(rows[first_zero_index])}, col={int(cols[first_zero_index])}, "
                            f"lon={float(lons[first_zero_index])}, lat={float(lats[first_zero_index])}"
                        )
                        _log_build(f"zero_vector_error {error_message}")
                        raise ValueError(error_message)

                    vectors = normalize_embeddings(raw_vectors)
                    _log_build(
                        "block_ready "
                        f"tile={tile.path.name} block={block_number}/{len(windows)} "
                        f"{_format_window(window)} vectors={len(rows)} "
                        f"total_after_append={writer.total_rows + len(writer.metadata_buffer) + len(rows)}"
                    )

                    metadata_rows = []
                    for row, col, lon, lat in zip(rows, cols, lons, lats):
                        metadata_rows.append(
                            {
                                "id": next_id,
                                "lon": float(lon),
                                "lat": float(lat),
                                "tile_path": str(tile.path),
                                "row": int(row),
                                "col": int(col),
                            }
                        )
                        next_id += 1

                    writer.append(metadata_rows, vectors)

                if tqdm is not None:
                    tile_iterator.set_postfix(vectors=writer.total_rows + len(writer.metadata_buffer))
                tile_written_rows = writer.total_rows + len(writer.metadata_buffer) - tile_start_rows
                _log_build(
                    f"tile_done tile_index={tile_number}/{total_candidate_tiles} "
                    f"path={tile.path} blocks={len(windows)} "
                    f"tile_vectors={tile_written_rows} total_vectors={writer.total_rows + len(writer.metadata_buffer)}"
                )

        total_rows = writer.finalize(metadata_path, embeddings_path)
    finally:
        writer.cleanup()

    index_type = "numpy_exact"
    if not skip_faiss and faiss is not None and total_rows > 0:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        for start in _progress(range(0, total_rows, flush_rows), desc="FAISS", unit="chunk"):
            index.add(np.asarray(embeddings[start:start + flush_rows], dtype=np.float32))
        faiss.write_index(index, str(index_path))
        index_type = "faiss_hnsw_ip"

    build_info = {
        "year": year,
        "data_dir": str(data_dir),
        "boundary_kml_path": str(boundary_kml_path),
        "boundary_cache_path": str(boundary_cache_path),
        "metadata_path": str(metadata_path),
        "embeddings_path": str(embeddings_path),
        "index_path": str(index_path),
        "build_time": datetime.now(timezone.utc).isoformat(),
        "tile_count": len(selected_tiles),
        "vector_count": int(total_rows),
        "selected_tiles": selected_tiles,
        "index_type": index_type,
        "block_size": block_size,
        "flush_rows": flush_rows,
    }
    build_info_path.write_text(json.dumps(build_info, indent=2), encoding="utf-8")

    return BuildResult(
        tile_count=len(selected_tiles),
        vector_count=int(total_rows),
        output_dir=output_dir,
        index_type=index_type,
    )
