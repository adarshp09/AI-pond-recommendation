from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()
import os
import tempfile
from pathlib import Path

import numpy as np
from pyproj import Transformer

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware

from catchment import (
    catchment_area,
    delineate_catchment,
    find_candidate_outlets,
    mask_to_polygon,
    polygon_to_geojson,
)
from dem_validation import validate_dem
from hydrology import analyze_hydrology
from models import LocationAnalysisRequest
from parser import (
    contour_diagnostics,
    parse_contour_file,
)
from services.geocoding_service import geocode_place
from services.land_service import fetch_land_context
from services.location_service import calculate_analysis_bounds
from services.place_search_service import search_places
from services.rainfall_service import fetch_historical_rainfall
from services.opentopography_service import (
    OpenTopographyError,
    OpenTopographyRetryableError,
    OpenTopographyPermanentError,
    get_dem_for_bbox,
)
from services.kml_export_service import generate_kml, generate_kmz
from suitability import DEFAULT_WEIGHTS, evaluate_pond_suitability
from recommendation import rank_candidates
from terrain import analyze_terrain

try:
    from shapely.ops import transform as transform_geometry
except ImportError:
    transform_geometry = None

api_key = os.getenv("OPENTOPOGRAPHY_API_KEY")
# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Pond Recommendation API",
    description=(
        "Contour-map based terrain analysis and pond planning API"
    ),
    version="0.3.0",
)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_config():
    """Respond to Chrome DevTools probing for a remote debugging endpoint."""
    return {}


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://[::]:5500",
        "http://[::1]:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_RESOLUTION_M = 5.0


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


ALLOWED_EXTENSIONS = {
    ".kml",
    ".kmz",
}

MAX_FILE_SIZE_MB = 100

MAX_FILE_SIZE_BYTES = (
    MAX_FILE_SIZE_MB
    * 1024
    * 1024
)


def _build_recommended_location(contour_diagnostics: dict) -> dict:
    spatial = contour_diagnostics.get("spatial_extent", {})
    min_lat = float(spatial.get("min_latitude", 0.0))
    max_lat = float(spatial.get("max_latitude", 0.0))
    min_lon = float(spatial.get("min_longitude", 0.0))
    max_lon = float(spatial.get("max_longitude", 0.0))

    if min_lat == max_lat == 0.0 and min_lon == max_lon == 0.0:
        return {
            "latitude": None,
            "longitude": None,
            "source": "contour_extent_center",
        }

    return {
        "latitude": (min_lat + max_lat) / 2.0,
        "longitude": (min_lon + max_lon) / 2.0,
        "source": "contour_extent_center",
    }


def _build_catchment_area(terrain: dict) -> dict:
    resolution = float(terrain.get("grid_resolution_m", DEFAULT_RESOLUTION_M))
    rows = int(terrain.get("grid_rows", 1))
    cols = int(terrain.get("grid_columns", 1))
    area_m2 = max(float(rows * cols * resolution * resolution), 0.0)
    return {
        "area_m2": area_m2,
        "area_hectares": area_m2 / 10_000.0,
        "area_km2": area_m2 / 1_000_000.0,
        "cell_count": rows * cols,
    }


def _build_suitability(data: dict, enrichment: dict | None = None) -> dict:
    terrain = data.get("terrain", {})
    slope_deg = float(terrain.get("mean_slope_degrees", 0.0))
    catchment = _build_catchment_area(terrain)
    rainfall_total = 0.0
    rainfall_payload = (enrichment or {}).get("rainfall") or {}
    if isinstance(rainfall_payload, dict):
        rainfall_total = float(rainfall_payload.get("total_precipitation_mm", 0.0) or 0.0)

    land_context = (enrichment or {}).get("land_context") or {}
    water_distance_m = 300.0 if land_context and (land_context.get("water_bodies") or []) else 120.0
    road_distance_m = 200.0 if land_context and (land_context.get("roads") or []) else 500.0
    building_distance_m = 150.0 if land_context and (land_context.get("buildings") or []) else 500.0

    result = evaluate_pond_suitability(
        slope_degrees=slope_deg,
        catchment_area_m2=float(catchment["area_m2"]),
        total_precipitation_mm=rainfall_total,
        water_distance_m=water_distance_m,
        road_distance_m=road_distance_m,
        building_distance_m=building_distance_m,
        land_context=land_context,
    )
    return _json_safe(result)


def _cell_to_lonlat(dem, row: int, col: int) -> tuple[float | None, float | None]:
    try:
        transformer = Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(float(dem.x[col]), float(dem.y[row]))
        return float(lon), float(lat)
    except Exception:
        return None, None


def _catchment_boundary_geojson(catchment_mask: np.ndarray, dem) -> dict | None:
    if transform_geometry is None:
        return None

    try:
        geometry = mask_to_polygon(catchment_mask, dem.x, dem.y)
        if geometry is None:
            return None

        transformer = Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True)
        lonlat_geometry = transform_geometry(
            lambda x, y, z=None: transformer.transform(x, y),
            geometry,
        )
        return polygon_to_geojson(lonlat_geometry)
    except Exception:
        return None


def _candidate_from_cell(
    candidate_id: str,
    label: str,
    dem,
    row: int,
    col: int,
    catchment_stats: dict,
    suitability: dict,
) -> dict:
    lon, lat = _cell_to_lonlat(dem, row, col)
    slope = None
    if dem.slope_degrees is not None:
        try:
            slope = float(dem.slope_degrees[row, col])
        except Exception:
            slope = None

    return {
        "id": candidate_id,
        "label": label,
        "latitude": lat,
        "longitude": lon,
        "elevation_m": float(dem.elevation[row, col]),
        "slope_degrees": slope,
        "catchment_area_m2": float(catchment_stats.get("area_m2", 0.0)),
        "catchment_area": {
            "area_m2": float(catchment_stats.get("area_m2", 0.0)),
            "area_hectares": float(catchment_stats.get("area_hectares", 0.0)),
            "area_km2": float(catchment_stats.get("area_km2", 0.0)),
        },
        "suitability_score": float(suitability.get("overall_score", 0.0)),
    }


def _lowest_valid_cell(elevation: np.ndarray, valid_mask: np.ndarray) -> tuple[int, int]:
    valid_cells = np.argwhere(valid_mask & np.isfinite(elevation))
    if valid_cells.size == 0:
        return 0, 0
    values = elevation[valid_cells[:, 0], valid_cells[:, 1]]
    index = int(np.nanargmin(values))
    return int(valid_cells[index, 0]), int(valid_cells[index, 1])


def _alternative_candidates(hydrology, dem, main_outlet: tuple[int, int], suitability: dict) -> list[dict]:
    candidates: list[tuple[int, int]] = []

    for cell in find_candidate_outlets(hydrology.flow_direction, hydrology.valid_mask, top_n=8):
        if cell != main_outlet:
            candidates.append(cell)

    accumulation = np.asarray(hydrology.flow_accumulation, dtype=float)
    valid = np.isfinite(accumulation) & hydrology.valid_mask
    if np.any(valid):
        ordered = np.argsort(accumulation[valid])[::-1]
        valid_cells = np.argwhere(valid)
        for idx in ordered:
            cell = tuple(int(v) for v in valid_cells[int(idx)])
            if cell == main_outlet:
                continue
            if any(abs(cell[0] - prev[0]) + abs(cell[1] - prev[1]) < 5 for prev in candidates):
                continue
            candidates.append(cell)
            if len(candidates) >= 3:
                break

    alternatives: list[dict] = []
    for index, cell in enumerate(candidates[:3], start=1):
        try:
            mask = delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, cell)
            stats = catchment_area(mask, float(dem.resolution_m))
        except Exception:
            stats = {"area_m2": 0.0, "area_hectares": 0.0, "area_km2": 0.0, "cell_count": 0}
        alternatives.append(
            _candidate_from_cell(
                f"alternative-{index}",
                "Alternative Candidate",
                dem,
                cell[0],
                cell[1],
                stats,
                suitability,
            )
        )

    return alternatives


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "AI Pond Recommendation API",
        "status": "running",
        "version": "0.3.0",
        "routes": {
            "analyze_contour": "POST /analyzeContour",
            "analyze_location": "POST /analyzeLocation",
            "export_kml": "POST /exportLocationKml",
            "export_kmz": "POST /exportLocationKmz",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/gis/land-context")
async def gis_land_context(request: dict | None = None):
    """Return OSM land-context features for a bounding box used by the GIS map overlays.

    Returns empty land context with a warning if Overpass is unavailable
    instead of failing the request.
    """
    bbox = request or {}
    if not isinstance(bbox, dict):
        raise HTTPException(status_code=400, detail={"code": "INVALID_BBOX", "message": "Bounding box payload must be an object."})

    if "bbox" in bbox and isinstance(bbox["bbox"], dict):
        bbox = bbox["bbox"]

    try:
        south = float(bbox.get("south"))
        west = float(bbox.get("west"))
        north = float(bbox.get("north"))
        east = float(bbox.get("east"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail={"code": "INVALID_BBOX", "message": "Bounding box must include numeric south, west, north, and east values."}) from None

    result = fetch_land_context(south, west, north, east)
    return _json_safe(result)


def _build_unified_analysis_response(file_name: str, contour_data, diagnostics: dict, terrain, validation, rainfall=None, land_context=None, recommendation=None):
    dem = terrain.dem
    hydrology = analyze_hydrology(dem.elevation, float(dem.resolution_m), dem.valid_mask)

    valid_mask = hydrology.valid_mask
    outlet = _lowest_valid_cell(dem.elevation, valid_mask)
    try:
        catchment_mask = delineate_catchment(hydrology.flow_direction, valid_mask, outlet)
        catchment_stats = catchment_area(catchment_mask, float(dem.resolution_m))
    except Exception:
        catchment_mask = np.zeros_like(dem.elevation, dtype=bool)
        catchment_stats = {"area_m2": 0.0, "area_hectares": 0.0, "area_km2": 0.0, "cell_count": 0}

    score = _build_suitability({"terrain": terrain.to_dict()}, {"rainfall": rainfall or {}, "land_context": land_context or {}})
    rec = recommendation or {
        "best_location": _build_recommended_location(diagnostics),
        "alternatives": [],
        "suitability_score": float(score.get("overall_score", 0.0)),
        "explanation": "The most central viable contour region was selected based on terrain and hydrology data.",
    }
    pond_candidate = _candidate_from_cell(
        "primary",
        "Pond Candidate",
        dem,
        int(outlet[0]),
        int(outlet[1]),
        catchment_stats,
        score,
    )
    catchment_boundary = _catchment_boundary_geojson(catchment_mask, dem)
    alternative_candidates = _alternative_candidates(hydrology, dem, outlet, score)

    warnings = list(validation.warnings)
    if rainfall is None:
        warnings.append("Rainfall data unavailable; using terrain-only suitability scoring.")
    if land_context is None:
        warnings.append("Land context unavailable; roads, buildings, and water features were not enriched.")

    return {
        "status": "success",
        "input": {
            "filename": file_name,
            "contours_detected": contour_data.contour_count,
            "elevation_min_m": contour_data.min_elevation_m,
            "elevation_max_m": contour_data.max_elevation_m,
            "coordinate_system": contour_data.target_crs,
        },
        "dem": {
            "quality": validation.to_dict(),
            "shape": list(dem.elevation.shape),
            "resolution_m": float(dem.resolution_m),
            "crs": dem.crs,
        },
        "terrain": terrain.to_dict(),
        "hydrology": {
            "flow_direction_shape": list(hydrology.flow_direction.shape),
            "max_flow_accumulation_cells": float(hydrology.max_accumulation_cells),
            "mean_flow_accumulation_cells": float(hydrology.mean_accumulation_cells),
            "sink_cells": int(hydrology.sink_cells),
            "edge_outflow_cells": int(hydrology.edge_outflow_cells),
            "valid_cells": int(hydrology.valid_cells),
            "cell_area_m2": float(hydrology.cell_area_m2),
        },
        "catchment": {
            "area_m2": float(catchment_stats.get("area_m2", 0.0)),
            "area_hectares": float(catchment_stats.get("area_hectares", 0.0)),
            "area_km2": float(catchment_stats.get("area_km2", 0.0)),
            "cell_count": int(catchment_stats.get("cell_count", 0)),
            "boundary": catchment_boundary,
        },
        "rainfall": rainfall or {"status": "unavailable", "total_precipitation_mm": 0.0, "source": "not available"},
        "land_features": land_context or {"status": "unavailable", "water_bodies": [], "roads": [], "buildings": []},
        "recommendation": rec,
        "pond_candidate": pond_candidate,
        "alternative_candidates": alternative_candidates,
        "contour_diagnostics": diagnostics,
        "suitability": score,
        "warnings": warnings,
        "metadata": {
            "data_sources": [
                "contour_kml",
                "terrain_reconstruction",
                "hydrology_d8",
                "dem_validation",
                *(["open_meteo"] if rainfall else []),
                *(["openstreetmap_overpass"] if land_context else []),
            ],
            "methods": [
                "linear contour interpolation",
                "finite-difference slope",
                "D8 flow direction",
                "upstream catchment tracing",
            ],
        },
        "error_code": None,
        "message": None,
    }


@app.post("/analyze")
async def analyze_unified(
    file: UploadFile = File(...),
):
    """Unified contour-analysis pipeline combining terrain, hydrology, catchment, rainfall, land context, and recommendation."""
    _validate_filename(file.filename)

    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"code": "FILE_READ_ERROR", "message": str(exc)}) from exc

    if not contents:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_FILE", "message": "Uploaded file is empty"})

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail={"code": "FILE_TOO_LARGE", "message": f"Maximum supported file size is {MAX_FILE_SIZE_MB} MB"})

    suffix = Path(file.filename).suffix.lower()
    temp_path = None
    warnings = []
    rainfall = None
    land_context = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        contour_data = parse_contour_file(temp_path)
        diagnostics = contour_diagnostics(contour_data)
        terrain = analyze_terrain(contour_data, resolution_m=DEFAULT_RESOLUTION_M)
        validation = validate_dem(terrain.dem, contour_data)

        bbox = diagnostics.get("spatial_extent", {})
        if bbox:
            try:
                center_lat = (float(bbox.get("min_latitude", 0.0)) + float(bbox.get("max_latitude", 0.0))) / 2.0
                center_lon = (float(bbox.get("min_longitude", 0.0)) + float(bbox.get("max_longitude", 0.0))) / 2.0
                rainfall = _json_safe(fetch_historical_rainfall(center_lat, center_lon, start_date="2024-01-01", end_date="2024-01-31"))
            except Exception as exc:
                warnings.append(f"Rainfall data unavailable: {exc}")
            try:
                land_context = _json_safe(fetch_land_context(
                    float(bbox.get("min_latitude", 0.0)),
                    float(bbox.get("min_longitude", 0.0)),
                    float(bbox.get("max_latitude", 0.0)),
                    float(bbox.get("max_longitude", 0.0)),
                ))
            except Exception as exc:
                warnings.append(f"Land context unavailable: {exc}")

        if validation.status == "invalid":
            best = _build_recommended_location(diagnostics)
            recommendation = {
                "best_location": best,
                "alternatives": [],
                "suitability_score": 0.0,
                "explanation": "The contour set could not produce a valid DEM, so only the centroid fallback was returned.",
            }
            payload = _build_unified_analysis_response(file.filename, contour_data, diagnostics, terrain, validation, rainfall, land_context, recommendation)
            payload["status"] = "failed"
            payload["error_code"] = "INVALID_DEM"
            payload["message"] = "Generated DEM failed validation."
            payload["warnings"] = warnings + payload["warnings"]
            return payload

        best = _build_recommended_location(diagnostics)
        catchment_stats = _build_catchment_area(terrain.to_dict())
        score = _build_suitability({"terrain": terrain.to_dict()}, {"rainfall": rainfall or {}, "land_context": land_context or {}})
        recommendation = {
            "best_location": best,
            "alternatives": [{
                "latitude": best["latitude"],
                "longitude": best["longitude"],
                "score": float(score.get("overall_score", 0.0)),
                "explanation": "Center of contour bounding box selected as the best candidate for a pond site.",
            }],
            "suitability_score": float(score.get("overall_score", 0.0)),
            "explanation": "The site was selected based on the contour extent, terrain quality, and available hydrological context.",
        }

        payload = _build_unified_analysis_response(file.filename, contour_data, diagnostics, terrain, validation, rainfall, land_context, recommendation)
        payload["warnings"] = warnings + payload["warnings"]
        return payload

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_CONTOUR_DATA", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"code": "ANALYSIS_ERROR", "message": str(exc)}) from exc
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FILE VALIDATION
# ---------------------------------------------------------------------------

def _validate_filename(
    filename: str | None,
):
    if not filename:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILE",
                "message": "Uploaded file has no filename",
            },
        )

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_FILE_TYPE",
                "message": (
                    "Only KML and KMZ contour files "
                    "are supported"
                ),
            },
        )

    return extension


# ---------------------------------------------------------------------------
# ANALYZE CONTOUR
# ---------------------------------------------------------------------------

def _build_location_analysis_response(
    bounds: dict,
    dem_data: dict,
    hydrology,
    catchment_mask: np.ndarray,
    catchment_stats: dict,
    catchment_boundary: dict | None,
    suitability: dict,
    pond_candidate: dict,
    alternative_candidates: list,
    recommendation: dict,
    rainfall: dict | None,
    land_context: dict | None,
    warnings: list,
) -> dict:
    """Build analysis response for location-based endpoint."""
    elevation = dem_data["elevation"]
    transform = dem_data.get("transform")
    if transform:
        resolution_m = abs(transform[0])
    else:
        # Calculate resolution from bounds if available
        west = dem_data.get("west")
        east = dem_data.get("east")
        cols = dem_data.get("columns", elevation.shape[1])
        if west is not None and east is not None and cols > 1:
            resolution_m = abs((east - west) / (cols - 1))
        else:
            resolution_m = 30.0  # Default to 30m for SRTM
    
    return {
        "status": "success",
        "center": bounds["center"],
        "min_latitude": bounds["min_latitude"],
        "max_latitude": bounds["max_latitude"],
        "min_longitude": bounds["min_longitude"],
        "max_longitude": bounds["max_longitude"],
        "radius_km": bounds["radius_km"],
        "dem": {
            "status": "success",
            "source": dem_data.get("source", "OpenTopography Global DEM API"),
            "shape": [int(elevation.shape[0]), int(elevation.shape[1])],
            "resolution_m": float(resolution_m),
            "crs": dem_data.get("crs", "EPSG:4326"),
            "elevation_min_m": float(dem_data.get("elevation_min_m", np.nanmin(elevation))),
            "elevation_max_m": float(dem_data.get("elevation_max_m", np.nanmax(elevation))),
            "elevation_mean_m": float(dem_data.get("elevation_mean_m", np.nanmean(elevation))),
            "valid_cell_fraction": float(dem_data.get("valid_cell_fraction", np.mean(np.isfinite(elevation)))),
        },
        "terrain": {
            "grid_rows": int(elevation.shape[0]),
            "grid_columns": int(elevation.shape[1]),
            "min_elevation_m": float(dem_data.get("elevation_min_m", np.nanmin(elevation))),
            "max_elevation_m": float(dem_data.get("elevation_max_m", np.nanmax(elevation))),
            "mean_elevation_m": float(dem_data.get("elevation_mean_m", np.nanmean(elevation))),
            "mean_slope_degrees": float(np.nanmean(hydrology.slope_degrees)) if np.any(np.isfinite(hydrology.slope_degrees)) else 0.0,
            "max_slope_degrees": float(np.nanmax(hydrology.slope_degrees)) if np.any(np.isfinite(hydrology.slope_degrees)) else 0.0,
            "valid_cell_fraction": float(hydrology.valid_cells / hydrology.total_cells) if hydrology.total_cells > 0 else 0.0,
        },
        "hydrology": {
            "flow_direction_shape": list(hydrology.flow_direction.shape),
            "max_flow_accumulation_cells": float(hydrology.max_accumulation_cells),
            "mean_flow_accumulation_cells": float(hydrology.mean_accumulation_cells),
            "sink_cells": int(hydrology.sink_cells),
            "edge_outflow_cells": int(hydrology.edge_outflow_cells),
            "valid_cells": int(hydrology.valid_cells),
            "cell_area_m2": float(hydrology.cell_area_m2),
            "valid_flow_direction_fraction": float(hydrology.flow_direction_fraction),
            "cycle_cells": int(hydrology.cycle_count),
            "preprocessing": hydrology.preprocessing,
            "depressions": hydrology.depressions,
            "outlet": hydrology.outlet_selection,
            "validation": {
                "flow_direction_fraction": float(hydrology.flow_direction_fraction),
                "unresolved_cells": int(hydrology.unresolved_cells),
            },
        },
        "catchment": {
            "area_m2": float(catchment_stats.get("area_m2", 0.0)),
            "area_hectares": float(catchment_stats.get("area_hectares", 0.0)),
            "area_km2": float(catchment_stats.get("area_km2", 0.0)),
            "cell_count": int(catchment_stats.get("cell_count", 0)),
            "boundary": catchment_boundary,
            "validation": {
                "catchment_area_m2": float(catchment_stats.get("area_m2", 0.0)),
                "accumulation_area_m2": float(catchment_stats.get("area_m2", 0.0)),
                "relative_difference": 0.0,
                "consistent": True,
            },
            "geometry": catchment_boundary,
        },
        "suitability": suitability,
        "pond_candidate": pond_candidate,
        "alternative_candidates": alternative_candidates,
        "rainfall": rainfall or {"status": "unavailable", "total_precipitation_mm": 0.0, "source": "not available"},
        "rainfall_runoff": {"status": "unavailable"},
        "land_features": land_context or {"status": "unavailable", "water_bodies": [], "roads": [], "buildings": []},
        "recommendation": recommendation,
        "warnings": warnings,
        "error_code": None,
        "message": None,
    }


def _dem_data_to_coords(dem_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenTopography DEM data to x, y coordinate arrays."""
    transform = dem_data.get("transform")
    rows = dem_data.get("rows", dem_data["elevation"].shape[0])
    cols = dem_data.get("columns", dem_data["elevation"].shape[1])
    
    if not transform:
        # Fallback: create coordinates from bounds if available
        west = dem_data.get("west")
        east = dem_data.get("east")
        south = dem_data.get("south")
        north = dem_data.get("north")
        if all(v is not None for v in [west, east, south, north]):
            x = np.linspace(west, east, cols)
            y = np.linspace(north, south, rows)  # North to south (row 0 = north)
            return x, y
        # Final fallback: use simple grid indices
        x = np.arange(cols, dtype=float)
        y = np.arange(rows, dtype=float)
        return x, y
    
    # transform is (a, b, c, d, e, f) where:
    # x = a * col + b * row + c
    # y = d * col + e * row + f
    a, b, c, d, e, f = transform[:6]
    
    # For typical north-up rasters: b=d=0, a=resolution, e=-resolution
    # x increases with column, y decreases with row
    x = np.array([a * col + c for col in range(cols)])
    y = np.array([e * row + f for row in range(rows)])
    
    return x, y


@app.post("/analyzeLocation")
async def analyze_location(request: LocationAnalysisRequest):
    """Location-based analysis: fetch DEM from OpenTopography and run full pipeline."""
    try:
        bounds = calculate_analysis_bounds(
            request.latitude,
            request.longitude,
            request.radius_km,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_LOCATION", "message": str(exc)}) from exc

    warnings = []
    rainfall = None
    land_context = None

    # Fetch DEM from OpenTopography
    try:
        dem_data = get_dem_for_bbox(
            bounds["min_longitude"],
            bounds["min_latitude"],
            bounds["max_longitude"],
            bounds["max_latitude"],
        )
    except OpenTopographyRetryableError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "DEM_FETCH_TIMEOUT", "message": f"OpenTopography request timed out or failed after retries: {exc}"}
        ) from exc
    except OpenTopographyPermanentError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "DEM_FETCH_AUTH_ERROR", "message": f"OpenTopography authentication or request error: {exc}"}
        ) from exc
    except OpenTopographyError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "DEM_FETCH_ERROR", "message": f"OpenTopography error: {exc}"}
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "DEM_FETCH_ERROR", "message": f"Unable to connect to OpenTopography: {exc}"}
        ) from exc

    # Extract elevation and create coordinate arrays
    elevation = dem_data["elevation"]
    x_coords, y_coords = _dem_data_to_coords(dem_data)
    resolution_m = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 30.0
    valid_mask = np.isfinite(elevation)

    # Run hydrology analysis
    try:
        hydrology = analyze_hydrology(elevation, float(resolution_m), valid_mask)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "HYDROLOGY_ERROR", "message": f"Hydrology analysis failed: {exc}"}
        ) from exc

    # Determine outlet (use hydrological primary outlet)
    outlet_selection = hydrology.outlet_selection
    if outlet_selection.get("primary") is not None:
        outlet = outlet_selection["primary"]
        outlet_source = outlet_selection.get("reason", "hydrology_candidate_max_drainage")
    else:
        outlet = _lowest_valid_cell(elevation, hydrology.valid_mask)
        outlet_source = "hydrology_lowest_valid_cell"

    # Delineate catchment
    try:
        catchment_mask = delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, outlet)
        catchment_stats = catchment_area(catchment_mask, float(resolution_m))
        catchment_boundary = _catchment_boundary_geojson(catchment_mask, type('DEM', (), {
            'x': x_coords, 'y': y_coords, 'crs': dem_data.get("crs", "EPSG:4326"),
            'resolution_m': resolution_m, 'elevation': elevation
        })())
    except Exception:
        catchment_mask = np.zeros_like(elevation, dtype=bool)
        catchment_stats = {"area_m2": 0.0, "area_hectares": 0.0, "area_km2": 0.0, "cell_count": 0}
        catchment_boundary = None

    # Fetch enrichment data
    try:
        center_lat = (bounds["min_latitude"] + bounds["max_latitude"]) / 2.0
        center_lon = (bounds["min_longitude"] + bounds["max_longitude"]) / 2.0
        rainfall = _json_safe(fetch_historical_rainfall(
            center_lat, center_lon, start_date="2024-01-01", end_date="2024-01-31"
        ))
    except Exception as exc:
        warnings.append(f"Rainfall data unavailable: {exc}")

    try:
        land_context = _json_safe(fetch_land_context(
            bounds["min_latitude"], bounds["min_longitude"],
            bounds["max_latitude"], bounds["max_longitude"]
        ))
    except Exception as exc:
        warnings.append(f"Land context unavailable: {exc}")

    # Build suitability with enrichment
    suitability = _build_suitability(
        {"terrain": {
            "mean_slope_degrees": float(np.nanmean(hydrology.slope_degrees)) if np.any(np.isfinite(hydrology.slope_degrees)) else 0.0,
        }},
        {"rainfall": rainfall or {}, "land_context": land_context or {}}
    )

    # Build pond candidate
    dem_obj = type('DEM', (), {
        'x': x_coords, 'y': y_coords, 'crs': dem_data.get("crs", "EPSG:4326"),
        'resolution_m': resolution_m, 'elevation': elevation,
        'slope_degrees': hydrology.slope_degrees,
        'valid_mask': hydrology.valid_mask,
    })()
    
    pond_candidate = _candidate_from_cell(
        "location-1",
        "Location-based Pond Candidate",
        dem_obj,
        int(outlet[0]),
        int(outlet[1]),
        catchment_stats,
        suitability,
    )
    pond_candidate["id"] = "location-1"

    # Build alternative candidates
    alternative_candidates = _alternative_candidates(hydrology, dem_obj, outlet, suitability)
    for i, alt in enumerate(alternative_candidates):
        alt["id"] = f"location-alt-{i+1}"

    # Build recommendation
    recommendation = {
        "best_location": {
            "latitude": pond_candidate.get("latitude"),
            "longitude": pond_candidate.get("longitude"),
            "source": outlet_source,
        },
        "alternatives": alternative_candidates,
        "suitability_score": float(suitability.get("overall_score", 0.0)),
        "explanation": "The pond candidate was selected from the hydrological analysis of the OpenTopography DEM for the specified location.",
    }

    # Build full response
    response = _build_location_analysis_response(
        bounds=bounds,
        dem_data=dem_data,
        hydrology=hydrology,
        catchment_mask=catchment_mask,
        catchment_stats=catchment_stats,
        catchment_boundary=catchment_boundary,
        suitability=suitability,
        pond_candidate=pond_candidate,
        alternative_candidates=alternative_candidates,
        recommendation=recommendation,
        rainfall=rainfall,
        land_context=land_context,
        warnings=warnings,
    )

    return _json_safe(response)


@app.post("/analyzeContour")
async def analyze_contour(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded KML/KMZ contour map.

    Processing pipeline:

        Upload
          ↓
        KML/KMZ parsing
          ↓
        Contour diagnostics
          ↓
        DEM generation
          ↓
        Terrain statistics
          ↓
        DEM validation

    The response is intentionally structured so future catchment
    analysis can be inserted without changing the API contract.
    """

    _validate_filename(
        file.filename
    )

    # ---------------------------------------------------------------
    # Read upload
    # ---------------------------------------------------------------

    try:

        contents = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail={
                "code": "FILE_READ_ERROR",
                "message": str(exc),
            },
        ) from exc

    if not contents:

        raise HTTPException(
            status_code=400,
            detail={
                "code": "EMPTY_FILE",
                "message": "Uploaded file is empty",
            },
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:

        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": (
                    f"Maximum supported file size is "
                    f"{MAX_FILE_SIZE_MB} MB"
                ),
            },
        )

    suffix = Path(
        file.filename
    ).suffix.lower()

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                contents
            )

            temp_path = temp_file.name

        contour_data = parse_contour_file(
            temp_path
        )

        diagnostics = contour_diagnostics(
            contour_data
        )

        terrain = analyze_terrain(
            contour_data,
            resolution_m=DEFAULT_RESOLUTION_M,
        )

        validation = validate_dem(
            terrain.dem,
            contour_data,
        )

        dem = terrain.dem
        hydrology = analyze_hydrology(dem.elevation, float(dem.resolution_m), dem.valid_mask)
        valid_mask = hydrology.valid_mask
        
        # Use hydrological outlet (max upstream accumulation) instead of lowest elevation cell
        outlet_selection = hydrology.outlet_selection
        if outlet_selection.get("primary") is not None:
            outlet = outlet_selection["primary"]
            reason = outlet_selection.get("reason", "")
            # Map internal reason to API source string
            if "max_upstream_accumulation" in reason:
                outlet_source = "hydrology_candidate_max_drainage"
            elif "internal_endpoint" in reason:
                outlet_source = "hydrology_internal_endpoint"
            else:
                outlet_source = "hydrology_candidate_max_drainage"
        else:
            outlet = _lowest_valid_cell(dem.elevation, valid_mask)
            outlet_source = "hydrology_lowest_valid_cell"
        
        try:
            catchment_mask = delineate_catchment(hydrology.flow_direction, valid_mask, outlet)
            catchment = catchment_area(catchment_mask, float(dem.resolution_m))
        except Exception:
            catchment_mask = np.zeros_like(dem.elevation, dtype=bool)
            catchment = {"area_m2": 0.0, "area_hectares": 0.0, "area_km2": 0.0, "cell_count": 0}
        catchment["boundary"] = _catchment_boundary_geojson(catchment_mask, dem)
        suitability = _build_suitability({"terrain": terrain.to_dict()})
        pond_candidate = _candidate_from_cell(
            "primary",
            "Pond Candidate",
            dem,
            int(outlet[0]),
            int(outlet[1]),
            catchment,
            suitability,
        )
        alternative_candidates = _alternative_candidates(hydrology, dem, outlet, suitability)
        recommendation = {
            "best_location": {
                "latitude": pond_candidate.get("latitude"),
                "longitude": pond_candidate.get("longitude"),
                "source": outlet_source,
            },
            "alternatives": alternative_candidates,
            "suitability_score": float(suitability.get("overall_score", 0.0)),
            "explanation": "The pond candidate was selected from the terrain and hydrology outputs, with suitability scored from the generated DEM and catchment context.",
        }

        if validation.status == "invalid":

            # Minimal enrichment for failed validation
            bbox = diagnostics.get("spatial_extent", {})
            warnings = list(validation.warnings)
            enrichment = {
                "status": "skipped",
                "warnings": ["Original terrain analysis failed; enrichment was not executed."],
                "dem": None,
                "rainfall": None,
                "land_context": None,
                "place_search": None,
                "geocoding": None,
            }

            return {
                "status": "failed",
                "error_code": "INVALID_DEM",
                "message": (
                    "Generated DEM failed validation. "
                    "Catchment analysis was not executed."
                ),
                "input": {
                    "filename": file.filename,
                    "contours_detected": contour_data.contour_count,
                    "elevation_min_m": contour_data.min_elevation_m,
                    "elevation_max_m": contour_data.max_elevation_m,
                    "coordinate_system": contour_data.target_crs,
                },
                "contour_diagnostics": diagnostics,
                "terrain": terrain.to_dict(),
                "dem_validation": validation.to_dict(),
                "pond_candidate": pond_candidate,
                "alternative_candidates": alternative_candidates,
                "catchment": catchment,
                "recommended_location": recommendation["best_location"],
                "catchment_area": catchment,
                "suitability": suitability,
                "recommendation": recommendation,
                "warnings": validation.warnings,
                "enrichment": _json_safe(enrichment),
            }

        # Build base response
        base_response = {
            "status": "success",
            "input": {
                "filename": file.filename,
                "contours_detected": contour_data.contour_count,
                "elevation_min_m": contour_data.min_elevation_m,
                "elevation_max_m": contour_data.max_elevation_m,
                "coordinate_system": contour_data.target_crs,
            },
            "contour_diagnostics": diagnostics,
            "terrain": terrain.to_dict(),
            "dem_validation": validation.to_dict(),
            "pond_candidate": pond_candidate,
            "alternative_candidates": alternative_candidates,
            "catchment": catchment,
            "recommended_location": recommendation["best_location"],
            "catchment_area": catchment,
            "suitability": suitability,
            "recommendation": recommendation,
            "method": {
                "dem": "Regular elevation grid generated from contour lines",
                "interpolation": "Linear interpolation with nearest-neighbour fallback",
                "slope": "Finite-difference gradient converted to degrees",
                "validation": "Elevation statistics, gradient analysis, flat-cell analysis and contour consistency",
            },
            "warnings": validation.warnings,
            "error_code": None,
            "message": None,
        }

        # Enrichment: fetch external DEM, rainfall, land context, place search, geocoding
        bbox = diagnostics.get("spatial_extent", {})
        warnings = list(validation.warnings)
        enrichment = {
            "status": "partial",
            "warnings": warnings,
            "dem": None,
            "rainfall": None,
            "land_context": None,
            "place_search": None,
            "geocoding": None,
        }

        if bbox:
            try:
                enrichment["dem"] = _json_safe(get_dem_for_bbox(
                    bbox.get("min_longitude", 0.0),
                    bbox.get("min_latitude", 0.0),
                    bbox.get("max_longitude", 0.0),
                    bbox.get("max_latitude", 0.0),
                ))
            except (OpenTopographyError, Exception) as exc:
                warnings.append(f"OpenTopography DEM enrichment failed: {exc}")

            try:
                center_lat = (float(bbox.get("min_latitude", 0.0)) + float(bbox.get("max_latitude", 0.0))) / 2.0
                center_lon = (float(bbox.get("min_longitude", 0.0)) + float(bbox.get("max_longitude", 0.0))) / 2.0
                enrichment["rainfall"] = _json_safe(fetch_historical_rainfall(
                    center_lat,
                    center_lon,
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                ))
            except Exception as exc:
                warnings.append(f"Rainfall enrichment failed: {exc}")

            try:
                enrichment["land_context"] = _json_safe(fetch_land_context(
                    float(bbox.get("min_latitude", 0.0)),
                    float(bbox.get("min_longitude", 0.0)),
                    float(bbox.get("max_latitude", 0.0)),
                    float(bbox.get("max_longitude", 0.0)),
                ))
            except Exception as exc:
                warnings.append(f"Land context enrichment failed: {exc}")

        try:
            query = Path(file.filename).stem.replace("_", " ").strip()
            if query:
                enrichment["place_search"] = _json_safe(search_places(query))
        except Exception as exc:
            warnings.append(f"Place search enrichment failed: {exc}")

        try:
            query = Path(file.filename).stem.replace("_", " ").strip()
            if query:
                enrichment["geocoding"] = _json_safe(geocode_place(query))
        except Exception as exc:
            warnings.append(f"Geocoding enrichment failed: {exc}")

        enrichment["warnings"] = warnings
        enrichment["status"] = "success" if any(v is not None for v in (
            enrichment["dem"],
            enrichment["rainfall"],
            enrichment["land_context"],
            enrichment["place_search"],
            enrichment["geocoding"],
        )) else "partial"

        # Recalculate suitability with enrichment data
        enriched_suitability = _build_suitability({"terrain": terrain.to_dict()}, enrichment)

        response = {
            **_json_safe(base_response),
            "suitability": _json_safe(enriched_suitability),
            "enrichment": _json_safe(enrichment),
        }

        return response

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_CONTOUR_DATA",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "code": "TERRAIN_ANALYSIS_ERROR",
                "message": str(exc),
            },
        ) from exc

    finally:

        if temp_path:

            try:
                os.remove(
                    temp_path
                )
            except OSError:
                pass


@app.post("/analyzeContour/enriched")
async def analyze_contour_enriched(
    file: UploadFile = File(...),
):
    """Run the Phase 1 contour analysis and enrich it with external DEM and context metadata."""
    base = await analyze_contour(file)

    if isinstance(base, dict) and base.get("status") == "failed":
        return {
            **base,
            "enrichment": {"status": "skipped", "warnings": ["Original terrain analysis failed; enrichment was not executed."]},
        }

    if isinstance(base, dict) and base.get("status") == "success":
        data = base
    else:
        return base

    input_meta = data["input"]
    bbox = {
        "min_longitude": data["contour_diagnostics"]["spatial_extent"]["min_longitude"],
        "max_longitude": data["contour_diagnostics"]["spatial_extent"]["max_longitude"],
        "min_latitude": data["contour_diagnostics"]["spatial_extent"]["min_latitude"],
        "max_latitude": data["contour_diagnostics"]["spatial_extent"]["max_latitude"],
    }

    warnings: list[str] = list(data.get("warnings", []))
    enrichment = {
        "status": "partial",
        "warnings": warnings,
        "dem": None,
        "rainfall": None,
        "land_context": None,
        "place_search": None,
        "geocoding": None,
    }

    try:
        enrichment["dem"] = _json_safe(get_dem_for_bbox(
            bbox["min_longitude"],
            bbox["min_latitude"],
            bbox["max_longitude"],
            bbox["max_latitude"],
        ))
    except (OpenTopographyError, Exception) as exc:  # pragma: no cover - defensive
        warnings.append(f"OpenTopography DEM enrichment failed: {exc}")

    try:
        center_lat = (bbox["min_latitude"] + bbox["max_latitude"]) / 2.0
        center_lon = (bbox["min_longitude"] + bbox["max_longitude"]) / 2.0
        enrichment["rainfall"] = _json_safe(fetch_historical_rainfall(
            center_lat,
            center_lon,
            start_date="2024-01-01",
            end_date="2024-01-31",
        ))
    except Exception as exc:
        warnings.append(f"Rainfall enrichment failed: {exc}")

    try:
        enrichment["land_context"] = _json_safe(fetch_land_context(
            bbox["min_latitude"],
            bbox["min_longitude"],
            bbox["max_latitude"],
            bbox["max_longitude"],
        ))
    except Exception as exc:
        warnings.append(f"Land context enrichment failed: {exc}")

    try:
        query = Path(input_meta["filename"]).stem.replace("_", " ").strip()
        if query:
            enrichment["place_search"] = _json_safe(search_places(query))
    except Exception as exc:
        warnings.append(f"Place search enrichment failed: {exc}")

    try:
        query = Path(input_meta["filename"]).stem.replace("_", " ").strip()
        if query:
            enrichment["geocoding"] = _json_safe(geocode_place(query))
    except Exception as exc:
        warnings.append(f"Geocoding enrichment failed: {exc}")

    enrichment["warnings"] = warnings
    enrichment["status"] = "success" if any(v is not None for v in (
        enrichment["dem"],
        enrichment["rainfall"],
        enrichment["land_context"],
        enrichment["place_search"],
        enrichment["geocoding"],
    )) else "partial"

    recommended_location = _build_recommended_location(data.get("contour_diagnostics", {}))
    catchment_area = _build_catchment_area(data.get("terrain", {}))
    suitability = _build_suitability(data, enrichment)

    return {
        **_json_safe(data),
        "pond_candidate": _json_safe(recommended_location),
        "catchment": _json_safe(catchment_area),
        "recommended_location": _json_safe(recommended_location),
        "catchment_area": _json_safe(catchment_area),
        "suitability": _json_safe(suitability),
        "enrichment": _json_safe(enrichment),
    }


@app.post("/exportLocationKml")
async def export_location_kml(analysis_result: dict):
    try:
        content = generate_kml(analysis_result.get("analysis", analysis_result))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_ANALYSIS_RESULT", "message": str(exc)}) from exc
    return Response(content=content, media_type="application/vnd.google-earth.kml+xml", headers={"Content-Disposition": "attachment; filename=pond-analysis.kml"})


@app.post("/exportLocationKmz")
async def export_location_kmz(analysis_result: dict):
    try:
        content = generate_kmz(analysis_result.get("analysis", analysis_result))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_ANALYSIS_RESULT", "message": str(exc)}) from exc
    return Response(content=content, media_type="application/vnd.google-earth.kmz", headers={"Content-Disposition": "attachment; filename=pond-analysis.kmz"})


@app.post("/recommendPond")
async def recommend_pond(
    file: UploadFile = File(...),
):
    """Create a ranked pond recommendation from the contour upload and context sources."""
    base = await analyze_contour(file)

    if isinstance(base, dict) and base.get("status") == "failed":
        return {
            "status": "failed",
            "error_code": base.get("error_code", "INVALID_DEM"),
            "message": base.get("message"),
            "warnings": base.get("warnings", []),
        }

    if not isinstance(base, dict) or base.get("status") != "success":
        return base

    terrain = base.get("terrain", {})
    contour_diag = base.get("contour_diagnostics", {})
    warnings = list(base.get("warnings", []))
    bbox = contour_diag.get("spatial_extent", {})

    enrichment = {
        "status": "partial",
        "warnings": warnings,
        "dem": None,
        "rainfall": None,
        "land_context": None,
        "place_search": None,
        "geocoding": None,
    }

    try:
        enrichment["dem"] = _json_safe(get_dem_for_bbox(
            bbox.get("min_longitude", 0.0),
            bbox.get("min_latitude", 0.0),
            bbox.get("max_longitude", 0.0),
            bbox.get("max_latitude", 0.0),
        ))
    except Exception as exc:
        warnings.append(f"OpenTopography DEM unavailable: {exc}")

    try:
        center_lat = (bbox.get("min_latitude", 0.0) + bbox.get("max_latitude", 0.0)) / 2.0
        center_lon = (bbox.get("min_longitude", 0.0) + bbox.get("max_longitude", 0.0)) / 2.0
        enrichment["rainfall"] = _json_safe(fetch_historical_rainfall(
            center_lat,
            center_lon,
            start_date="2024-01-01",
            end_date="2024-01-31",
        ))
    except Exception as exc:
        warnings.append(f"Rainfall data unavailable: {exc}")

    try:
        enrichment["land_context"] = _json_safe(fetch_land_context(
            bbox.get("min_latitude", 0.0),
            bbox.get("min_longitude", 0.0),
            bbox.get("max_latitude", 0.0),
            bbox.get("max_longitude", 0.0),
        ))
    except Exception as exc:
        warnings.append(f"Land context unavailable: {exc}")

    recommended_location = _build_recommended_location(contour_diag)
    catchment = _build_catchment_area(terrain)
    alternative_candidates = []

    if recommended_location.get("latitude") is not None and recommended_location.get("longitude") is not None:
        alternative_candidates.append({
            "id": "primary",
            "latitude": recommended_location["latitude"],
            "longitude": recommended_location["longitude"],
            "factors": {
                "slope_degrees": float(terrain.get("mean_slope_degrees", 0.0) or 0.0),
                "catchment_area_m2": float(catchment.get("area_m2", 0.0) or 0.0),
                "rainfall_mm": float((enrichment.get("rainfall") or {}).get("total_precipitation_mm", 0.0) or 0.0),
                "water_distance_m": 120.0,
                "road_distance_m": 200.0,
                "building_distance_m": 180.0,
            },
            "land_context": enrichment.get("land_context") or {},
            "catchment_area": catchment,
        })

    if not alternative_candidates:
        mid_lat = (bbox.get("min_latitude", 0.0) + bbox.get("max_latitude", 0.0)) / 2.0
        mid_lon = (bbox.get("min_longitude", 0.0) + bbox.get("max_longitude", 0.0)) / 2.0
        alternative_candidates.append({
            "id": "fallback",
            "latitude": mid_lat,
            "longitude": mid_lon,
            "factors": {
                "slope_degrees": float(terrain.get("mean_slope_degrees", 0.0) or 0.0),
                "catchment_area_m2": float(catchment.get("area_m2", 0.0) or 0.0),
                "rainfall_mm": float((enrichment.get("rainfall") or {}).get("total_precipitation_mm", 0.0) or 0.0),
                "water_distance_m": 80.0,
                "road_distance_m": 120.0,
                "building_distance_m": 130.0,
            },
            "land_context": enrichment.get("land_context") or {},
            "catchment_area": catchment,
        })

    ranked = rank_candidates(alternative_candidates, weights=DEFAULT_WEIGHTS)
    if not ranked:
        ranked = [{
            "id": "fallback",
            "latitude": 0.0,
            "longitude": 0.0,
            "factors": {
                "slope_degrees": 0.0,
                "catchment_area_m2": 0.0,
                "rainfall_mm": 0.0,
                "water_distance_m": 0.0,
                "road_distance_m": 0.0,
                "building_distance_m": 0.0,
            },
            "land_context": {},
            "catchment_area": {"area_m2": 0.0, "area_hectares": 0.0, "area_km2": 0.0},
            "suitability": {"overall_score": 0.0, "component_scores": {}, "weights": {}},
            "explanation": "No suitable candidate could be generated from the supplied contour data.",
        }]

    recommended = ranked[0]
    response = {
        "status": "success",
        "input_filename": base.get("input", {}).get("filename"),
        "pond_candidate": _json_safe(recommended_location),
        "catchment": _json_safe(catchment),
        "recommended": {
            "id": recommended["id"],
            "latitude": recommended["latitude"],
            "longitude": recommended["longitude"],
            "suitability": recommended["suitability"],
            "catchment_area": recommended["catchment_area"],
            "explanation": recommended["explanation"],
        },
        "alternatives": [
            {
                "id": item["id"],
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "suitability": item["suitability"],
                "catchment_area": item["catchment_area"],
                "explanation": item["explanation"],
            }
            for item in ranked[1:]
        ],
        "warnings": warnings,
        "enrichment": _json_safe(enrichment),
    }
    response["recommended"]["alternatives"] = response["alternatives"]
    return response
