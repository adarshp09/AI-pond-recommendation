"""Phase 08 — telemetry tests: Prometheus metrics, OTel tracing, log correlation.

Tracing tests require the OpenTelemetry SDK and are skipped when it is absent
(this environment does not install it). Metrics tests use the installed
``prometheus_client``.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from config import settings
from backend.utils import telemetry
from backend.utils.telemetry import trace_stage


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint_enabled(client):
    pytest.importorskip("prometheus_client")

    client.get("/health")  # generate at least one counted request
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "pond_requests_total" in body
    assert "pond_stage_duration_seconds" in body


def test_metrics_endpoint_disabled_returns_404(client, monkeypatch):
    monkeypatch.setitem(settings.__dict__, "PROMETHEUS_METRICS_ENABLED", False)
    response = client.get("/metrics")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "METRICS_DISABLED"


def test_requests_total_counts_handled_requests(client):
    pytest.importorskip("prometheus_client")
    from prometheus_client import REGISTRY

    before = REGISTRY.get_sample_value(
        "pond_requests_total", {"method": "GET", "path": "/health", "status": "200"}
    ) or 0.0
    client.get("/health")
    after = REGISTRY.get_sample_value(
        "pond_requests_total", {"method": "GET", "path": "/health", "status": "200"}
    )
    assert after is not None and after > before


def test_setup_telemetry_disabled_is_noop(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "OTEL_ENABLED", False)
    telemetry.setup_telemetry(None)  # must not raise or register exporters


def test_trace_stage_without_tracer_still_times(monkeypatch):
    monkeypatch.setattr(telemetry, "_get_tracer", lambda: None)
    with trace_stage("trace_test_no_tracer"):
        pass  # timing-only; no span, no exception


# ---------------------------------------------------------------------------
# OpenTelemetry tracing (skipped when the SDK is unavailable)
# ---------------------------------------------------------------------------

@pytest.fixture
def span_exporter(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("phase08-tests")
    monkeypatch.setattr(telemetry, "_get_tracer", lambda: tracer)
    yield exporter, tracer
    exporter.clear()


def test_trace_stage_creates_span_with_attributes(span_exporter):
    from opentelemetry.trace import StatusCode

    exporter, _ = span_exporter
    with trace_stage("trace_test_ok", {"candidate_count": 3}):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "trace_test_ok"
    assert spans[0].attributes.get("candidate_count") == 3
    assert spans[0].status.status_code == StatusCode.OK


def test_trace_stage_error_sets_status_and_reraises(span_exporter):
    from opentelemetry.trace import StatusCode

    exporter, _ = span_exporter
    with pytest.raises(ValueError, match="boom"):
        with trace_stage("trace_test_err"):
            raise ValueError("boom")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


def test_trace_stage_records_prometheus_histogram(span_exporter):
    pytest.importorskip("prometheus_client")
    from prometheus_client import REGISTRY

    stage = "trace_test_histogram_unique"
    with trace_stage(stage):
        pass

    count = REGISTRY.get_sample_value("pond_stage_duration_seconds_count", {"stage": stage})
    assert count == 1.0


def test_json_log_carries_trace_and_span_ids(span_exporter):
    from backend.utils.observability import JSONLogFormatter

    _, tracer = span_exporter
    stream = io.StringIO()
    logger = logging.getLogger("otel.correlation.test")
    logger.handlers[:] = []
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    with tracer.start_as_current_span("correlated"):
        logger.info("inside span")

    logger.handlers[:] = []
    record = json.loads(stream.getvalue().strip())
    assert len(record["trace_id"]) == 32
    assert len(record["span_id"]) == 16
    int(record["trace_id"], 16)  # valid hex
    int(record["span_id"], 16)


def test_json_log_omits_trace_ids_without_span():
    from backend.utils.observability import JSONLogFormatter

    stream = io.StringIO()
    logger = logging.getLogger("otel.correlation.nospan")
    logger.handlers[:] = []
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("no span here")
    logger.handlers[:] = []

    record = json.loads(stream.getvalue().strip())
    assert "trace_id" not in record
    assert "span_id" not in record
