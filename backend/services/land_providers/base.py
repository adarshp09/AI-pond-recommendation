"""Land-context provider contract.

A *land provider* answers one question: for a WGS84 bounding box, what water, road
and building features exist? Providers are selected and sequenced by
``services.land_service``; the geographic-context engine stays provider-agnostic and
only ever sees the normalised payload produced here.

The payload contract is frozen and backward compatible with the original Overpass
implementation::

    {"bbox": {south,west,north,east},
     "water_bodies": [...], "roads": [...], "buildings": [...],
     "source": "<human readable>"}

Providers may add status metadata (``provider``, ``status``, ``coverage``,
``feature_counts``, ``warning``) which consumers may inspect but must not require.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Provider status vocabulary (never conflate these — missing vs empty vs failed).
STATUS_AVAILABLE = "available"          # dataset used, query executed
STATUS_EMPTY = "empty"                  # dataset used, query validly returned no features
STATUS_OUT_OF_COVERAGE = "out_of_coverage"
STATUS_STALE = "stale"
STATUS_MISSING = "missing"              # no dataset configured / not present
STATUS_UNREADABLE = "unreadable"        # dataset present but corrupt/unreadable
STATUS_UNAVAILABLE = "unavailable"      # provider could not be used at all
STATUS_ERROR = "error"                  # unexpected failure while querying

# Coverage vocabulary.
COVERAGE_FULL = "full"
COVERAGE_PARTIAL = "partial"
COVERAGE_NONE = "none"


def bbox_dict(south: float, west: float, north: float, east: float) -> Dict[str, float]:
    return {"south": float(south), "west": float(west), "north": float(north), "east": float(east)}


def bbox_relation(
    query: tuple[float, float, float, float],
    coverage: tuple[float, float, float, float],
) -> str:
    """Classify ``query`` (s,w,n,e) against a ``coverage`` box."""

    qs, qw, qn, qe = query
    cs, cw, cn, ce = coverage
    if qs >= cs and qw >= cw and qn <= cn and qe <= ce:
        return COVERAGE_FULL
    if qn < cs or qs > cn or qe < cw or qw > ce:
        return COVERAGE_NONE
    return COVERAGE_PARTIAL


@dataclass
class LandContextResult:
    """Normalised provider result (never fabricates features)."""

    bbox: Dict[str, float]
    water_bodies: List[Dict[str, Any]] = field(default_factory=list)
    roads: List[Dict[str, Any]] = field(default_factory=list)
    buildings: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""
    provider: str = ""
    status: str = STATUS_AVAILABLE
    coverage: Optional[str] = None
    warning: Optional[str] = None
    feature_counts: Optional[Dict[str, int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "bbox": dict(self.bbox),
            "water_bodies": list(self.water_bodies),
            "roads": list(self.roads),
            "buildings": list(self.buildings),
            "source": self.source,
            "provider": self.provider,
            "status": self.status,
        }
        if self.coverage is not None:
            payload["coverage"] = self.coverage
        if self.warning:
            payload["warning"] = self.warning
        payload["feature_counts"] = self.feature_counts or {
            "water_bodies": len(self.water_bodies),
            "roads": len(self.roads),
            "buildings": len(self.buildings),
        }
        payload.update(self.extra)
        return payload


class ProviderUnavailable(Exception):
    """The provider cannot serve this query (missing/stale/out-of-coverage/error)."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class LandProvider:
    """Minimal provider protocol. Concrete providers implement ``fetch``."""

    name: str = "provider"

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def covers(self, south: float, west: float, north: float, east: float) -> bool:  # pragma: no cover
        return False

    def fetch(self, south: float, west: float, north: float, east: float) -> LandContextResult:  # pragma: no cover
        raise NotImplementedError
