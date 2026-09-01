from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class Contour:
    elevation_m: float
    coordinates: List[Tuple[float, float]]
    name: Optional[str] = None


@dataclass
class ContourData:
    contours: List[Contour]
    source_crs: str
    target_crs: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_elevation_m: float
    max_elevation_m: float
    longitude_min: float
    longitude_max: float
    latitude_min: float
    latitude_max: float

    @property
    def contour_count(self) -> int:
        return len(self.contours)

    @property
    def elevations(self) -> np.ndarray:
        return np.asarray(
            [float(contour.elevation_m) for contour in self.contours],
            dtype=float,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contour_count": self.contour_count,
            "source_crs": self.source_crs,
            "target_crs": self.target_crs,
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_elevation_m": self.min_elevation_m,
            "max_elevation_m": self.max_elevation_m,
        }


@dataclass
class DEM:
    elevation: np.ndarray
    x: np.ndarray
    y: np.ndarray
    resolution_m: float
    crs: str
    slope_degrees: Optional[np.ndarray] = None
    gradient_m_per_m: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.elevation.shape),
            "resolution_m": float(self.resolution_m),
            "crs": self.crs,
            "min_elevation_m": float(np.nanmin(self.elevation)),
            "max_elevation_m": float(np.nanmax(self.elevation)),
            "mean_elevation_m": float(np.nanmean(self.elevation)),
        }


@dataclass
class DEMValidationResult:
    status: str
    score: float
    shape: Dict[str, int]
    resolution_m: float
    valid_cell_fraction: float
    nan_fraction: float
    elevation: Dict[str, float]
    local_elevation_change: Dict[str, float]
    slope: Dict[str, float]
    gradient: Dict[str, float]
    flat_fraction: float
    contour_consistency: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": float(self.score),
            "shape": self.shape,
            "resolution_m": float(self.resolution_m),
            "valid_cell_fraction": float(self.valid_cell_fraction),
            "nan_fraction": float(self.nan_fraction),
            "elevation": self.elevation,
            "local_elevation_change": self.local_elevation_change,
            "slope": self.slope,
            "gradient": self.gradient,
            "flat_fraction": float(self.flat_fraction),
            "contour_consistency": self.contour_consistency,
            "warnings": list(self.warnings),
        }


@dataclass
class TerrainResult:
    dem: DEM
    mean_slope_degrees: float
    max_slope_degrees: float
    valid_cell_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_resolution_m": float(self.dem.resolution_m),
            "grid_rows": int(self.dem.elevation.shape[0]),
            "grid_columns": int(self.dem.elevation.shape[1]),
            "min_elevation_m": float(np.nanmin(self.dem.elevation)),
            "max_elevation_m": float(np.nanmax(self.dem.elevation)),
            "mean_elevation_m": float(np.nanmean(self.dem.elevation)),
            "mean_slope_degrees": float(self.mean_slope_degrees),
            "max_slope_degrees": float(self.max_slope_degrees),
            "valid_cell_fraction": float(self.valid_cell_fraction),
        }


class SuitabilityFactors(BaseModel):
    slope_degrees: float = Field(..., ge=0.0, le=90.0)
    catchment_area_m2: float = Field(..., ge=0.0)
    rainfall_mm: float = Field(..., ge=0.0)
    water_distance_m: float = Field(..., ge=0.0)
    road_distance_m: float = Field(..., ge=0.0)
    building_distance_m: float = Field(..., ge=0.0)


class CandidateScore(BaseModel):
    id: str
    latitude: float
    longitude: float
    factors: SuitabilityFactors
    suitability: Dict[str, Any]
    explanation: str
    catchment_area: Dict[str, float]


class PondRecommendation(BaseModel):
    id: str
    latitude: float
    longitude: float
    suitability: Dict[str, Any]
    catchment_area: Dict[str, float]
    explanation: str
    alternatives: List[CandidateScore] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    status: str = "success"
    input_filename: Optional[str] = None
    recommended: PondRecommendation
    alternatives: List[CandidateScore] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    enrichment: Dict[str, Any] = Field(default_factory=dict)


class LocationAnalysisRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    radius_km: float = Field(..., gt=0.0, le=100.0)
