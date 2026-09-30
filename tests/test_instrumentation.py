"""Unit and integration tests for the pipeline instrumentation layer."""

from __future__ import annotations

import pytest

import backend.instrumentation as instrumentation
from backend.instrumentation import (
    STAGE_NAMES,
    attach_payload,
    begin_request,
    current_trace,
    dataset_id_from_bytes,
    instrumentation_requested,
    recent_traces,
    record_stage,
    reset_for_testing,
    reset_trace,
    set_trace,
    stage,
)


@pytest.fixture(autouse=True)
def clean_instrumentation():
    reset_for_testing()
    yield
    reset_for_testing()


def test_stage_records_success_duration_and_sizes():
    trace = begin_request(request_id="req-1", job_id="job-1")
    token = set_trace(trace)
    try:
        with stage("parsing", input_size=10) as handle:
            handle.set_output(output_size=3, contour_count=3)
    finally:
        reset_trace(token)

    record = trace.stage("parsing")
    assert record is not None
    assert record.success is True
    assert record.duration_ms >= 0.0
    assert record.input_size == 10
    assert record.output_size == 3
    assert record.metrics["contour_count"] == 3


def test_stage_records_failure_and_reraises():
    trace = begin_request(request_id="req-2")
    token = set_trace(trace)
    try:
        with pytest.raises(ValueError):
            with stage("DEM"):
                raise ValueError("boom")
    finally:
        reset_trace(token)

    record = trace.stage("DEM")
    assert record is not None
    assert record.success is False
    assert "boom" in (record.error or "")


def test_record_stage_manual_entry():
    trace = begin_request(request_id="req-3")
    token = set_trace(trace)
    try:
        record_stage("export", 1.5, output_size=42, metrics={"format": "kml"})
    finally:
        reset_trace(token)

    record = trace.stage("export")
    assert record is not None
    assert record.duration_ms == 1.5
    assert record.metrics["format"] == "kml"


def test_external_call_latency_is_accumulated():
    trace = begin_request(request_id="req-4")
    token = set_trace(trace)
    try:
        with stage("enrichment") as handle:
            handle.time_external(lambda: "ok")
            handle.time_external(lambda: "ok")
    finally:
        reset_trace(token)

    record = trace.stage("enrichment")
    assert record.external_api_ms is not None
    assert record.external_api_ms >= 0.0


def test_dataset_id_is_deterministic_and_opaque():
    first = dataset_id_from_bytes(b"contour-bytes", "pond.kml")
    second = dataset_id_from_bytes(b"contour-bytes", "pond.kml")
    different = dataset_id_from_bytes(b"other-bytes", "pond.kml")

    assert first == second
    assert first != different
    assert first.startswith("sha256:")
    assert "contour-bytes" not in first


def test_attach_payload_is_noop_without_emit():
    trace = begin_request(request_id="req-5", emit=False)
    payload = {"status": "success"}
    result = attach_payload(payload, trace=trace)
    assert result is payload
    assert "instrumentation" not in result


def test_attach_payload_includes_ids_and_stages_when_emit():
    trace = begin_request(request_id="req-6", job_id="job-6", dataset_id="ds-6", emit=True)
    token = set_trace(trace)
    try:
        with stage("hydrology"):
            pass
    finally:
        reset_trace(token)

    payload = attach_payload({"status": "success"}, trace=trace)
    assert payload["request_id"] == "req-6"
    assert payload["job_id"] == "job-6"
    assert payload["dataset_id"] == "ds-6"
    assert "hydrology" in payload["instrumentation"]["stage_summary_ms"]


def test_registry_is_bounded():
    for index in range(instrumentation.MAX_RECENT_TRACES + 50):
        begin_request(request_id=f"req-bound-{index}")
    assert len(recent_traces()) <= instrumentation.MAX_RECENT_TRACES


def test_instrumentation_requested_parses_header_and_env():
    assert instrumentation_requested({"x-pond-instrumentation": "true"}, env={}) is True
    assert instrumentation_requested({"x-pond-instrumentation": "0"}, env={}) is False
    assert instrumentation_requested(None, env={"POND_INSTRUMENTATION": "yes"}) is True
    assert instrumentation_requested(None, env={}) is False


def test_stage_names_are_canonical():
    assert "upload_validation" in STAGE_NAMES
    assert STAGE_NAMES[0] == "upload_validation"
    assert STAGE_NAMES[-1] == "export"


def test_middleware_adds_and_echoes_request_id(client):
    response = client.get("/health")
    request_id = response.headers.get("X-Request-ID")
    assert request_id

    echoed = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})
    assert echoed.headers["X-Request-ID"] == "caller-supplied-id"


def test_opt_in_response_contains_end_to_end_trace(client, sample_kml_bytes):
    response = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
        headers={"X-Pond-Instrumentation": "true", "X-Job-ID": "job-abc"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-abc"
    assert payload["request_id"]
    summary = payload["instrumentation"]["stage_summary_ms"]
    for expected in ["upload_validation", "parsing", "diagnostics", "DEM", "DEM_validation", "hydrology"]:
        assert expected in summary
    # Verify data quality structure exists
    dq = payload["instrumentation"]["data_quality"]
    assert "external_data" in dq
    assert "geographic_context" in dq
    assert dq["geographic_context"]["candidates"] is not None
    assert len(dq["geographic_context"]["candidates"]) > 0


def test_default_response_schema_is_unchanged(client, sample_kml_bytes):
    response = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    )
    payload = response.json()
    assert "instrumentation" not in payload
    assert "request_id" not in payload
    assert "job_id" not in payload
    assert "dataset_id" not in payload


def test_no_secrets_or_file_contents_are_logged(client, sample_kml_bytes, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger=instrumentation.LOGGER_NAME):
        client.post(
            "/analyzeContour",
            files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
            headers={"X-Pond-Instrumentation": "true"},
        )

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "pipeline_stage" in joined
    # The raw KML body must never appear in logs.
    assert "opengis.net/kml" not in joined
