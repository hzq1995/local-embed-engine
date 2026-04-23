from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


OUTPUT_NAME = "tif_overview.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw GeoTIFF inventory and coverage.")
    parser.add_argument("tif_dir", type=Path)
    return parser.parse_args()


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {path}")
    return path


def progress(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="file")


def scan_tifs(tif_dir: Path) -> dict:
    tif_paths = sorted(tif_dir.glob("*.tif*"))
    tile_reports: list[dict] = []
    band_counter: Counter[int] = Counter()
    dtype_counter: Counter[str] = Counter()
    crs_counter: Counter[str] = Counter()
    nodata_counter: Counter[str] = Counter()
    total_pixels = 0
    overall_bbox = None

    progress_bar = progress(len(tif_paths), "Scan tif inventory")
    try:
        for tif_path in tif_paths:
            with rasterio.open(tif_path) as ds:
                bounds_wgs84 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
                count = int(ds.count)
                dtype = str(ds.dtypes[0]) if ds.dtypes else "unknown"
                crs = str(ds.crs)
                nodata = None if ds.nodata is None else float(ds.nodata)
                width = int(ds.width)
                height = int(ds.height)
                pixel_count = width * height

                band_counter[count] += 1
                dtype_counter[dtype] += 1
                crs_counter[crs] += 1
                nodata_counter["none" if nodata is None else str(nodata)] += 1
                total_pixels += pixel_count

                min_lon, min_lat, max_lon, max_lat = [float(v) for v in bounds_wgs84]
                if overall_bbox is None:
                    overall_bbox = [min_lon, min_lat, max_lon, max_lat]
                else:
                    overall_bbox[0] = min(overall_bbox[0], min_lon)
                    overall_bbox[1] = min(overall_bbox[1], min_lat)
                    overall_bbox[2] = max(overall_bbox[2], max_lon)
                    overall_bbox[3] = max(overall_bbox[3], max_lat)

                tile_reports.append(
                    {
                        "path": str(tif_path),
                        "width": width,
                        "height": height,
                        "pixel_count": pixel_count,
                        "band_count": count,
                        "dtype": dtype,
                        "crs": crs,
                        "nodata": nodata,
                        "bounds_wgs84": [min_lon, min_lat, max_lon, max_lat],
                    }
                )
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return {
        "tif_dir": str(tif_dir),
        "tile_count": len(tif_paths),
        "total_pixel_count": int(total_pixels),
        "overall_bbox_wgs84": overall_bbox,
        "band_count_distribution": {str(key): int(value) for key, value in sorted(band_counter.items())},
        "dtype_distribution": dict(dtype_counter),
        "crs_distribution": dict(crs_counter),
        "nodata_distribution": dict(nodata_counter),
        "tiles": tile_reports,
    }


def write_report(tif_dir: Path, report: dict) -> Path:
    diagnostics_dir = tif_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def print_summary(report: dict, output_path: Path) -> None:
    print(f"Report written to {output_path}")
    print(
        f"tile_count={report['tile_count']} total_pixels={report['total_pixel_count']} "
        f"band_count_distribution={report['band_count_distribution']}"
    )
    print(f"overall_bbox_wgs84={report['overall_bbox_wgs84']}")


def main() -> None:
    args = parse_args()
    tif_dir = require_dir(args.tif_dir.resolve())
    report = scan_tifs(tif_dir)
    output_path = write_report(tif_dir, report)
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
