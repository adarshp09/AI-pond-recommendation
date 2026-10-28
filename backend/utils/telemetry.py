"""Phase 08 — OpenTelemetry tracing + Prometheus metrics (stdlib-first, guarded).

Everything here degrades gracefully if the optional observability libraries are not
installed:

* Prometheus metrics become no-op stubs (the app keeps running; ``/metrics`` reports
  the library is unavailable).
* ``trace_stage`` still times the block and records the Prometheus histogram, but
  creates no spans.
* ``setup_telemetry`` logs and returns without registering exporters.

When the libraries ARE installed (see ``requirements.txt``) and ``OTEL_ENABLED=true``,
a ``TracerProvider`` with a ``BatchSpanProcessor(OTLPSpanExporter(...))`` is installed
and FastAPI is instrumented.

Verified conventions (Phase 08 research):
* ``OTLPSpanExporter`` takes the full signal URL — ``<base>/v1/traces``.
* ``FastAPIInstrumentor.instrument_app(app)`` for manual FastAPI instrumentation.
* ``trace.get_current_span().get_span_context()`` → ``trace_id`` (32-hex) /
  ``span_id`` (16-hex); guard with ``span_context.is_valid``.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("pond_recommendation")

# ---------------------------------------------------------------------------
# Prometheus metrics (guarded)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import branch
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROM_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None  # type: ignore[assignment]


class _NoopMetric:
    """Minimal stand-in so callers never need to check availability."""

    def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
        return self

    def inc(self, *args: Any, **kwargs: Any) -> None:
        return None

    def observe(self, *args: Any, **kwargs: Any) -> None:
        return None


if _PROM_AVAILABLE:
    pond_requests_total = Counter(
        "pond_requests_total",
        "Total HTTP requests handled by the AI Pond Recommendation API",
        ["method", "path", "status"],
    )
    pond_stage_duration_seconds = Histogram(
        "pond_stage_duration_seconds",
        "Duration of pond-recommendation pipeline stages, in seconds",
        ["stage"],
    )
else:  # pragma: no cover
    pond_requests_total = _NoopMetric()
    pond_stage_duration_seconds = _NoopMetric()


def render_metrics() -> tuple[Optional[bytes], Optional[str]]:
    """Return ``(payload, content_type)`` for ``GET /metrics`` or ``(None, None)``."""

    if not _PROM_AVAILABLE or generate_latest is None:
        return None, None
    try:
        return generate_latest(), CONTENT_TYPE_LATEST
    except Exception:  # pragma: no cover - defensive
        return None, None


# ---------------------------------------------------------------------------
# OpenTelemetry tracing (guarded)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import branch
    from opentelemetry import trace as _otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

    class _PlaceholderStatus:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _PlaceholderStatusCode:  # noqa: N801 - mirrors OTel naming
        OK = "OK"
        ERROR = "ERROR"

    Status = _PlaceholderStatus  # type: ignore[assignment,misc]
    StatusCode = _PlaceholderStatusCode  # type: ignore[assignment,misc]


def otel_available() -> bool:
    return _OTEL_AVAILABLE


def _get_tracer():
    """Return a tracer, or ``None`` when the SDK is unavailable.

    Kept as a module function so tests can monkeypatch it with a tracer bound to an
    ``InMemorySpanExporter`` (avoids mutating the process-global tracer provider).
    """

    if not _OTEL_AVAILABLE:
        return None
    try:
        from config import settings

        name = getattr(settings, "OTEL_SERVICE_NAME", "ai-pond-recommendation")
    except Exception:  # pragma: no cover - defensive
        name = "ai-pond-recommendation"
    try:
        return _otel_trace.get_tracer(name)
    except Exception:  # pragma: no cover - defensive
        return None


def setup_telemetry(app: Any) -> None:
    """Configure tracing + FastAPI instrumentation per ``settings``.

    No-op (with an info log) when ``OTEL_ENABLED`` is false.
    """

    try:
        from config import settings
    except Exception:  # pragma: no cover - defensive
        logger.info("otel_disabled")
        return

    if not getattr(settings, "OTEL_ENABLED", False):
        logger.info("otel_disabled", extra={"otel_enabled": False})
        return

    if not _OTEL_AVAILABLE:
        logger.warning(
            "otel_enabled_but_sdk_unavailable",
            extra={"hint": "pip install -r requirements.txt"},
        )
        return

    endpoint = f"{str(settings.OTEL_EXPORTER_OTLP_ENDPOINT).rstrip('/')}/v1/traces"
    try:
        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: settings.OTEL_SERVICE_NAME})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _otel_trace.set_tracer_provider(provider)
    except Exception as exc:
        logger.warning("otel_setup_failed", extra={"error": str(exc)})
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:  # pragma: no cover - optional instrumentation
        logger.warning("otel_fastapi_instrumentation_failed", extra={"error": str(exc)})

    logger.info(
        "otel_enabled",
        extra={"endpoint": endpoint, "service": settings.OTEL_SERVICE_NAME},
    )


@contextmanager
def trace_stage(stage_name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Trace a pipeline stage + record its duration in the Prometheus histogram.

    * Wraps execution in ``tracer.start_as_current_span(stage_name)`` when a tracer
      is available.
    * Sets ``attributes`` on the span.
    * Records ``pond_stage_duration_seconds.labels(stage=stage_name)``.
    * On success sets ``StatusCode.OK``; on exception sets ``StatusCode.ERROR``,
      records the exception, and **re-raises**.
    """

    start = time.perf_counter()
    tracer = _get_tracer()
    span_cm = tracer.start_as_current_span(stage_name) if tracer is not None else nullcontext()

    try:
        with span_cm as span:
            if span is not None and attributes:
                try:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                except Exception:  # pragma: no cover - defensive
                    pass
            try:
                yield span
            except Exception as exc:
                if span is not None:
                    try:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                    except Exception:  # pragma: no cover - defensive
                        pass
                raise
            else:
                if span is not None:
                    try:
                        span.set_status(Status(StatusCode.OK))
                    except Exception:  # pragma: no cover - defensive
                        pass
    finally:
        duration = time.perf_counter() - start
        try:
            pond_stage_duration_seconds.labels(stage=stage_name).observe(duration)
        except Exception:  # pragma: no cover - defensive
            pass


__all__ = [
    "otel_available",
    "pond_requests_total",
    "pond_stage_duration_seconds",
    "render_metrics",
    "setup_telemetry",
    "trace_stage",
]
