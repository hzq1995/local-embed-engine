from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.services.boundary_service import BoundaryService
from app.services.catalog_service import TileCatalog
from app.services.index_service import IndexBundle


@dataclass(slots=True)
class QueryService:
    year: int
    boundary: BoundaryService
    catalog: TileCatalog
    index_bundle: IndexBundle

    def get_embedding_by_point(self, lon: float, lat: float) -> dict:
        if not self.boundary.contains_point(lon, lat):
            raise ValueError("Point is outside Ningbo boundary.")
        result = self._fetch_embedding_from_index(lon, lat)
        return {"year": self.year, "lon": lon, "lat": lat, **result}

    def _fetch_embedding_from_index(self, lon: float, lat: float) -> dict:
        """Fetch the nearest embedding from the pre-built index (embeddings.npy + metadata).

        Uses a small bbox window, falling back to a full scan if nothing is found nearby.
        """
        md = self.index_bundle.metadata
        if len(md) == 0:
            raise ValueError("Index is empty; run build first.")

        # Search within a ~1 km window (~0.01 deg) first for speed
        window = 0.01
        mask = (
            (md["lon"] >= lon - window)
            & (md["lon"] <= lon + window)
            & (md["lat"] >= lat - window)
            & (md["lat"] <= lat + window)
        )
        subset = md[mask]
        if len(subset) == 0:
            subset = md  # fall back to full scan

        lons = subset["lon"].to_numpy()
        lats = subset["lat"].to_numpy()
        # Approximate squared distance in degrees (sufficient for nearest-pixel selection)
        dist2 = (lons - lon) ** 2 + (lats - lat) ** 2
        local_idx = int(np.argmin(dist2))
        row = subset.iloc[local_idx]
        pixel_id = int(row["id"])
        embedding = self.index_bundle.embeddings[pixel_id].astype(np.float32).tolist()
        return {
            "tile_path": str(row.get("tile_path", "")),
            "row": int(row.get("row", -1)),
            "col": int(row.get("col", -1)),
            "embedding": embedding,
        }

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int,
        bbox: list[float] | None,
        min_distance_m: float,
        min_score: float = 0.0,
    ) -> dict:
        effective_bbox = self.boundary.bbox
        if bbox is not None:
            effective_bbox = self.boundary.clip_bbox(bbox)
            if effective_bbox is None:
                return {"top_k": top_k, "result_count": 0, "results": []}
        query = np.asarray(embedding, dtype=np.float32)
        results = self.index_bundle.search(
            query=query,
            top_k=top_k,
            bbox=effective_bbox,
            min_distance_m=min_distance_m,
            min_score=min_score,
        )
        return {"top_k": top_k, "result_count": len(results), "results": results}

    def get_embeddings_by_bbox(
        self,
        bbox: list[float],
        total_samples: int = 5000,
    ) -> dict:
        """Return embeddings sampled on a regular lon/lat grid over the bbox.

        The bbox is clipped to the boundary, then a grid of approximately
        *total_samples* cells is constructed.  For each cell the first
        candidate point (by metadata order) is selected as the representative.
        Returns grid layout metadata so the frontend can render contiguous cells.
        """
        _empty = {
            "count": 0, "lons": [], "lats": [], "embeddings": [],
            "grid_rows": 0, "grid_cols": 0,
            "grid_row_indices": [], "grid_col_indices": [],
            "effective_bbox": [],
        }

        effective_bbox = self.boundary.clip_bbox(bbox)
        if effective_bbox is None:
            return _empty

        min_lon, min_lat, max_lon, max_lat = effective_bbox
        md = self.index_bundle.metadata
        mask = (
            (md["lon"] >= min_lon)
            & (md["lon"] <= max_lon)
            & (md["lat"] >= min_lat)
            & (md["lat"] <= max_lat)
        )
        candidates = md[mask]

        if len(candidates) == 0:
            return {**_empty, "effective_bbox": list(effective_bbox)}

        # Build a regular grid sized to ~total_samples cells, preserving aspect ratio
        lon_span = max(max_lon - min_lon, 1e-9)
        lat_span = max(max_lat - min_lat, 1e-9)
        aspect = lon_span / lat_span
        n_rows = max(2, int(math.sqrt(total_samples / aspect)))
        n_cols = max(2, int(math.ceil(total_samples / n_rows)))
        lon_step = lon_span / n_cols
        lat_step = lat_span / n_rows

        # Assign each candidate to its grid cell (vectorized)
        lons_arr = candidates["lon"].to_numpy()
        lats_arr = candidates["lat"].to_numpy()
        ids_arr = candidates["id"].to_numpy(dtype=np.int64)

        col_idx = np.clip(np.floor((lons_arr - min_lon) / lon_step).astype(int), 0, n_cols - 1)
        # row 0 = northernmost row (max_lat side), so invert lat axis
        row_idx = np.clip(np.floor((max_lat - lats_arr) / lat_step).astype(int), 0, n_rows - 1)
        cell_keys = row_idx * n_cols + col_idx

        # np.unique returns sorted unique keys and the index of their first occurrence
        _, first_occ = np.unique(cell_keys, return_index=True)

        sel_ids = ids_arr[first_occ]
        sel_lons = lons_arr[first_occ].tolist()
        sel_lats = lats_arr[first_occ].tolist()
        sel_row_idx = row_idx[first_occ].tolist()
        sel_col_idx = col_idx[first_occ].tolist()
        embeddings = self.index_bundle.embeddings[sel_ids].astype(np.float32).tolist()

        return {
            "count": len(sel_ids),
            "lons": sel_lons,
            "lats": sel_lats,
            "embeddings": embeddings,
            "grid_rows": n_rows,
            "grid_cols": n_cols,
            "grid_row_indices": sel_row_idx,
            "grid_col_indices": sel_col_idx,
            "effective_bbox": list(effective_bbox),
        }
