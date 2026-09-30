"""Lightweight, non-invasive instrumentation for the pond recommendation pipeline.

This module provides:

* request/job/dataset correlation identifiers,
* per-stage timing with success/failure and input/output sizes,
* structured JSON logging (never logs secrets or file contents),
* a bounded in-memory registry of recent traces for tests and benchmarks.

The instrumentation is deliberately additive: nothing here mutates pipeline
results or API response schemas unless a caller explicitly asks for the
instrumentation payload (``attach_payload``).

Stage names (canonical order)::

    upload_validation -> parsing -> diagnostics -> DEM -> DEM_validation
    -> hydrology -> catchment -> suitability -> recommendation
    -> enrichment -> export
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

LOGGER_NAME = "pond_recommendation"

_REQUEST_ID_HEADER = "x-request-id"
_JOB_ID_HEADER = "x-job-id"
_DATASET_ID_HEADER = "x-dataset-id"
_INSTRUMENTATION_HEADER = "x-pond-instrumentation"
_INSTRUMENTATION_ENV = "POND_INSTRUMENTATION"

MAX_RECENT_TRACES = 256

STAGE_NAMES = (
    "upload_validation",
    "parsing",
    "diagnostics",
    "DEM",
    "DEM_validation",
    "hydrology",
    "catchment",
    "suitability",
    "recommendation",
    "enrichment",
    "export",
)

_LOGGER = logging.getLogger(LOGGER_NAME)


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _now() -> float:
    return time.perf_counter()


def new_request_id() -> str:
    return uuid.uuid4().hex


def new_job_id() -> str:
    return uuid.uuid4().hex


def dataset_id_from_bytes(contents: bytes, filename: str = "") -> str:
    """Return a stable, non-reversible dataset identifier.

    Only a truncated SHA-256 digest is kept so raw upload contents are never
    exposed through logs or responses.
    """

    hasher = hashlib.sha256()
    hasher.update(contents or b"")
    hasher.update((filename or "").encode("utf-8", "ignore"))
    return "sha256:" + hasher.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stage + trace records
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    stage: str
    duration_ms: float
    success: bool
    input_size: Any = None
    output_size: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    external_api_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_ms": round(float(self.duration_ms), 6),
            "success": bool(self.success),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
            "error": self.error,
            "external_api_ms": (
                None if self.external_api_ms is None else round(float(self.external_api_ms), 6)
            ),
        }


@dataclass
class PipelineTrace:
    request_id: str
    job_id: str
    dataset_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    emit: bool = False
    stages: List[StageRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    _start_perf: float = field(default_factory=_now, repr=False)

    def add(self, record: StageRecord) -> None:
        self.stages.append(record)
        if record.warnings:
            self.warnings.extend(record.warnings)

    def stage(self, name: str) -> Optional[StageRecord]:
        for record in reversed(self.stages):
            if record.stage == name:
                return record
        return None

    def total_duration_ms(self) -> float:
        return round((_now() - self._start_perf) * 1000.0, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "emit": self.emit,
            "total_duration_ms": self.total_duration_ms(),
            "stages": [record.to_dict() for record in self.stages],
            "warnings": list(self.warnings),
            "metrics": self.metrics,
        }

    def stage_summary(self) -> Dict[str, float]:
        return {record.stage: round(float(record.duration_ms), 6) for record in self.stages}


# ---------------------------------------------------------------------------
# Mutable stage handle handed to callers inside a ``with stage(...)`` block
# ---------------------------------------------------------------------------

class Stage:
    def __init__(self, name: str, input_size: Any = None) -> None:
        self.name = name
        self.input_size = input_size
        self.output_size: Any = None
        self.metrics: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.external_api_ms: Optional[float] = None

    def set_output(self, output_size: Any = None, **metrics: Any) -> "Stage":
        self.output_size = output_size
        for key, value in metrics.items():
            self.metrics[key] = value
        return self

    def add_metric(self, key: str, value: Any) -> "Stage":
        self.metrics[key] = value
        return self

    def warn(self, message: str) -> "Stage":
        self.warnings.append(str(message))
        return self

    def add_external_api_ms(self, duration_ms: float) -> "Stage":
        if duration_ms is None:
            return self
        self.external_api_ms = float(duration_ms) + float(self.external_api_ms or 0.0)
        return self

    def time_external(self, callable_obj, *args: Any, **kwargs: Any) -> Any:
        """Call ``callable_obj`` and accumulate its wall-clock latency."""
        start = _now()
        try:
            result = callable_obj(*args, **kwargs)
        finally:
            self.add_external_api_ms((_now() - start) * 1000.0)
        return result


# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

_CURRENT_TRACE: contextvars.ContextVar[Optional[PipelineTrace]] = contextvars.ContextVar(
    "pond_trace", default=None
)

_RECENT: "OrderedDict[str, PipelineTrace]" = OrderedDict()
_RECENT_LOCK = threading.Lock()


def current_trace() -> Optional[PipelineTrace]:
    return _CURRENT_TRACE.get()


def set_trace(trace: PipelineTrace):
    return _CURRENT_TRACE.set(trace)


def reset_trace(token) -> None:
    _CURRENT_TRACE.reset(token)


def register_trace(trace: PipelineTrace) -> PipelineTrace:
    with _RECENT_LOCK:
        _RECENT[trace.request_id] = trace
        _RECENT.move_to_end(trace.request_id)
        while len(_RECENT) > MAX_RECENT_TRACES:
            _RECENT.popitem(last=False)
    return trace


def get_trace(request_id: str) -> Optional[PipelineTrace]:
    with _RECENT_LOCK:
        return _RECENT.get(request_id)


def recent_traces(limit: Optional[int] = None) -> List[PipelineTrace]:
    with _RECENT_LOCK:
        traces = list(_RECENT.values())
    if limit is not None:
        return traces[-limit:]
    return traces


def clear_traces() -> None:
    with _RECENT_LOCK:
        _RECENT.clear()


def begin_request(
    request_id: Optional[str] = None,
    job_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    emit: bool = False,
) -> PipelineTrace:
    trace = PipelineTrace(
        request_id=request_id or new_request_id(),
        job_id=job_id or new_job_id(),
        dataset_id=dataset_id,
        emit=bool(emit),
    )
    register_trace(trace)
    return trace


def instrumentation_requested(
    headers: Optional[Any] = None,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    if headers is not None:
        try:
            if _truthy(headers.get(_INSTRUMENTATION_HEADER)):
                return True
        except Exception:
            pass
    environ = env if env is not None else os.environ
    return _truthy(environ.get(_INSTRUMENTATION_ENV))


def header_value(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    return value or None


def extract_ids_from_headers(headers: Any) -> Dict[str, Optional[str]]:
    return {
        "request_id": header_value(headers, _REQUEST_ID_HEADER),
        "job_id": header_value(headers, _JOB_ID_HEADER),
        "dataset_id": header_value(headers, _DATASET_ID_HEADER),
    }


def set_dataset_id(trace: Optional[PipelineTrace], dataset_id: Optional[str]) -> None:
    if trace is not None and dataset_id:
        trace.dataset_id = dataset_id


# ---------------------------------------------------------------------------
# Structured logging (never emits secrets or file contents)
# ---------------------------------------------------------------------------

def _log(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event}
    payload.update({key: value for key, value in fields.items() if value is not None})
    if _LOGGER.isEnabledFor(level):
        _LOGGER.log(level, json.dumps(payload, default=str, sort_keys=True))


def _record_to_log(record: StageRecord, trace: Optional[PipelineTrace]) -> None:
    fields: Dict[str, Any] = {
        "request_id": trace.request_id if trace else None,
        "job_id": trace.job_id if trace else None,
        "dataset_id": trace.dataset_id if trace else None,
        "stage": record.stage,
        "duration_ms": round(float(record.duration_ms), 6),
        "success": bool(record.success),
        "input_size": record.input_size,
        "output_size": record.output_size,
        "external_api_ms": (
            None if record.external_api_ms is None else round(float(record.external_api_ms), 6)
        ),
    }
    if record.metrics:
        fields["metrics"] = record.metrics
    if record.error:
        fields["error"] = record.error
    level = logging.INFO if record.success else logging.WARNING
    _log(level, "pipeline_stage", **fields)


# ---------------------------------------------------------------------------
# Stage instrumentation
# ---------------------------------------------------------------------------

@contextmanager
def stage(name: str, *, input_size: Any = None) -> Iterator[Stage]:
    """Time a pipeline stage, record success/failure, and log it.

    Never swallows exceptions: any error is re-raised after being recorded.
    """

    handle = Stage(name, input_size=input_size)
    start = _now()
    success = True
    error: Optional[str] = None
    try:
        yield handle
    except Exception as exc:  # noqa: BLE001 - recorded then re-raised
        success = False
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = (_now() - start) * 1000.0
        record = StageRecord(
            stage=name,
            duration_ms=duration_ms,
            success=success,
            input_size=handle.input_size,
            output_size=handle.output_size,
            metrics=dict(handle.metrics),
            warnings=list(handle.warnings),
            error=error,
            external_api_ms=handle.external_api_ms,
        )
        trace = _CURRENT_TRACE.get()
        if trace is not None:
            trace.add(record)
        _record_to_log(record, trace)


def record_stage(
    name: str,
    duration_ms: float,
    *,
    success: bool = True,
    input_size: Any = None,
    output_size: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[str] = None,
    external_api_ms: Optional[float] = None,
) -> StageRecord:
    """Manually record a stage that was timed elsewhere."""

    record = StageRecord(
        stage=name,
        duration_ms=float(duration_ms),
        success=bool(success),
        input_size=input_size,
        output_size=output_size,
        metrics=dict(metrics or {}),
        warnings=list(warnings or []),
        error=error,
        external_api_ms=external_api_ms,
    )
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.add(record)
    _record_to_log(record, trace)
    return record


# ---------------------------------------------------------------------------
# Response payload attachment (opt-in, additive, non-breaking)
# ---------------------------------------------------------------------------

def should_emit(trace: Optional[PipelineTrace]) -> bool:
    return bool(trace is not None and trace.emit)


def attach_payload(
    payload: Dict[str, Any],
    *,
    trace: Optional[PipelineTrace] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach correlation IDs and (optionally) the trace to a response payload.

    When instrumentation was not requested the payload is returned unchanged so
    existing API schemas and frontend behaviour are preserved byte-for-byte.
    """

    active = trace if trace is not None else current_trace()
    if not should_emit(active):
        return payload

    payload["request_id"] = active.request_id
    payload["job_id"] = active.job_id
    payload["dataset_id"] = active.dataset_id
    payload["instrumentation"] = {
        "request_id": active.request_id,
        "job_id": active.job_id,
        "dataset_id": active.dataset_id,
        "total_duration_ms": active.total_duration_ms(),
        "stages": [record.to_dict() for record in active.stages],
        "stage_summary_ms": active.stage_summary(),
        "warnings": list(active.warnings),
        "data_quality": metrics or {},
    }
    return payload


def reset_for_testing() -> None:
    clear_traces()
    _CURRENT_TRACE.set(None)


__all__ = [
    "STAGE_NAMES",
    "LOGGER_NAME",
    "Stage",
    "StageRecord",
    "PipelineTrace",
    "attach_payload",
    "begin_request",
    "clear_traces",
    "current_trace",
    "dataset_id_from_bytes",
    "extract_ids_from_headers",
    "get_trace",
    "instrumentation_requested",
    "new_job_id",
    "new_request_id",
    "recent_traces",
    "record_stage",
    "register_trace",
    "reset_for_testing",
    "reset_trace",
    "set_dataset_id",
    "set_trace",
    "should_emit",
    "stage",
]
