"""Phase 05 — stdlib-first observability: JSON logging + stage timing.

No third-party dependencies: only :mod:`logging`, :mod:`json`, :mod:`time`.

* :class:`JSONLogFormatter` renders every record as a single JSON line with a
  fixed core schema (``timestamp`` ISO-8601, ``level``, ``logger_name``,
  ``message``) plus any structured ``extra`` fields (``stage``,
  ``duration_ms``, ...).
* :func:`configure_json_logging` installs the formatter on a logger (the root
  logger by default).
* :class:`track_time` measures a block/function and emits a structured
  ``"Stage completed"`` record carrying ``stage`` and ``duration_ms``. It works
  as both a context manager and a decorator.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

# Phase 08 — log correlation. Guarded so logging works without OpenTelemetry.
try:  # pragma: no cover - trivial import branch
    from opentelemetry import trace as _otel_trace
except Exception:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]


def _current_span_ids() -> Optional[tuple[str, str]]:
    """Return ``(trace_id, span_id)`` (32/16-hex) for the active span, else ``None``."""

    if _otel_trace is None:
        return None
    try:
        span_context = _otel_trace.get_current_span().get_span_context()
        if span_context is not None and span_context.is_valid:
            return format(span_context.trace_id, "032x"), format(span_context.span_id, "016x")
    except Exception:  # pragma: no cover - defensive
        pass
    return None

# Attributes present on every LogRecord (excluded from the structured extras).
_RESERVED_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)

DEFAULT_LOGGER_NAME = "pond_recommendation"


class JSONLogFormatter(logging.Formatter):
    """Render a :class:`logging.LogRecord` as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self._iso_timestamp(record.created),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
        }

        # Structured fields passed via ``extra=``.
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = value

        span_ids = _current_span_ids()
        if span_ids is not None:
            payload["trace_id"], payload["span_id"] = span_ids

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, sort_keys=True)

    @staticmethod
    def _iso_timestamp(created: float) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()


def configure_json_logging(
    logger: Optional[logging.Logger] = None,
    level: str | int = logging.INFO,
    stream: Any = None,
) -> logging.Logger:
    """Install :class:`JSONLogFormatter` on ``logger`` (root by default)."""

    import sys

    target = logger if logger is not None else logging.getLogger()
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JSONLogFormatter())

    target.handlers[:] = [handler]
    target.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    return target


class track_time:
    """Measure a block or function and log the elapsed time as JSON.

    Usage::

        with track_time("candidate_generation"):
            ...

        @track_time("candidate_generation")
        def generate(...): ...

    Emits ``{"message": "Stage completed", "stage": <name>,
    "duration_ms": <float>, ...}`` through ``logger_name``.
    """

    def __init__(self, stage: str, logger_name: str = DEFAULT_LOGGER_NAME) -> None:
        self.stage = stage
        self.logger_name = logger_name
        self._start: Optional[float] = None

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "track_time":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = (time.perf_counter() - self._start) * 1000.0 if self._start is not None else 0.0
        logging.getLogger(self.logger_name).info(
            "Stage completed",
            extra={"stage": self.stage, "duration_ms": round(duration_ms, 3), "success": exc_type is None},
        )
        return False  # never swallow exceptions

    # -- decorator ---------------------------------------------------------
    def __call__(self, func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with track_time(self.stage, self.logger_name):
                return func(*args, **kwargs)

        return wrapper


__all__ = [
    "DEFAULT_LOGGER_NAME",
    "JSONLogFormatter",
    "configure_json_logging",
    "track_time",
]
