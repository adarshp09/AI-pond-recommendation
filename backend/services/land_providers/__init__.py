"""Land-context providers (Phase 02).

* :class:`LocalOSMProvider` — local GeoPackage, offline (this package).
* Overpass remains implemented in ``services.land_service`` and is used as the
  optional fallback.
"""

from .base import (
    COVERAGE_FULL,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    STATUS_AVAILABLE,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_MISSING,
    STATUS_OUT_OF_COVERAGE,
    STATUS_STALE,
    STATUS_UNAVAILABLE,
    STATUS_UNREADABLE,
    LandContextResult,
    LandProvider,
    ProviderUnavailable,
    bbox_dict,
    bbox_relation,
)
from .local_osm import LocalOSMProvider

__all__ = [
    "COVERAGE_FULL",
    "COVERAGE_NONE",
    "COVERAGE_PARTIAL",
    "STATUS_AVAILABLE",
    "STATUS_EMPTY",
    "STATUS_ERROR",
    "STATUS_MISSING",
    "STATUS_OUT_OF_COVERAGE",
    "STATUS_STALE",
    "STATUS_UNAVAILABLE",
    "STATUS_UNREADABLE",
    "LandContextResult",
    "LandProvider",
    "LocalOSMProvider",
    "ProviderUnavailable",
    "bbox_dict",
    "bbox_relation",
]
