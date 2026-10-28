"""Phase 05 — observability tests: JSON logging, stage timing, cache metrics.

Stdlib + repo modules only; no external services.
"""

from __future__ import annotations

import io
import json
import logging
import time

import pytest

import main as main_module
from config import settings
from backend.utils.observability import (
    JSONLogFormatter,
    configure_json_logging,
    track_time,
)


def _capture(logger_name, emit):
    """Attach a JSON handler to ``logger_name``, run ``emit``, return parsed lines."""

    stream = io.StringIO()
    logger = logging.getLogger(logger_name)
    logger.handlers[:] = []
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        emit(logger)
    finally:
        logger.handlers[:] = []
    return [json.loads(line) for line in stream.getvalue().strip().splitlines() if line]


# ---------------------------------------------------------------------------
# JSONLogFormatter
# ---------------------------------------------------------------------------

def test_json_formatter_core_fields_and_extras():
    def emit(logger):
        logger.info("hello", extra={"stage": "candidate_generation", "duration_ms": 12.5})

    records = _capture("obs.test.core", emit)
    assert len(records) == 1
    record = records[0]

    assert record["level"] == "INFO"
    assert record["logger_name"] == "obs.test.core"
    assert record["message"] == "hello"
    assert record["stage"] == "candidate_generation"
    assert record["duration_ms"] == 12.5

    # ISO-8601 UTC timestamp.
    assert "timestamp" in record
    assert "T" in record["timestamp"]
    assert record["timestamp"].endswith("+00:00")


def test_json_formatter_omits_reserved_and_handles_exceptions():
    def emit(logger):
        try:
            raise ValueError("kaboom")
        except ValueError:
            logger.exception("failed")

    records = _capture("obs.test.exc", emit)
    record = records[0]
    assert record["message"] == "failed"
    assert "kaboom" in record.get("exception", "")
    # Reserved LogRecord internals must not leak into the payload.
    assert "exc_info" not in record
    assert "lineno" not in record
    assert "pathname" not in record


# ---------------------------------------------------------------------------
# track_time
# ---------------------------------------------------------------------------

def test_track_time_context_manager_logs_stage_and_duration():
    def emit(_logger):
        with track_time("candidate_generation", logger_name="obs.test.ctx"):
            time.sleep(0.01)

    records = _capture("obs.test.ctx", emit)
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "Stage completed"
    assert record["stage"] == "candidate_generation"
    assert record["duration_ms"] >= 0.0
    assert record["success"] is True


def test_track_time_marks_failure_without_swallowing():
    def emit(_logger):
        with pytest.raises(RuntimeError):
            with track_time("boom_stage", logger_name="obs.test.fail"):
                raise RuntimeError("x")

    records = _capture("obs.test.fail", emit)
    assert records[0]["stage"] == "boom_stage"
    assert records[0]["success"] is False


def test_track_time_decorator():
    @track_time("decorated_stage", logger_name="obs.test.dec")
    def double(value):
        return value * 2

    def emit(_logger):
        assert double(3) == 6

    records = _capture("obs.test.dec", emit)
    assert records[0]["message"] == "Stage completed"
    assert records[0]["stage"] == "decorated_stage"


# ---------------------------------------------------------------------------
# configure_json_logging + main wiring
# ---------------------------------------------------------------------------

def test_configure_json_logging_installs_formatter():
    stream = io.StringIO()
    logger = logging.getLogger("obs.test.configure")
    logger.handlers[:] = []
    configure_json_logging(logger, level="INFO", stream=stream)
    logger.info("x", extra={"k": 1})
    record = json.loads(stream.getvalue().strip())
    assert record["message"] == "x"
    assert record["k"] == 1
    logger.handlers[:] = []


def test_main_configures_json_logging_when_enabled(monkeypatch):
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        monkeypatch.setitem(settings.__dict__, "LOG_FORMAT", "json")
        main_module._configure_logging()
        assert any(isinstance(h.formatter, JSONLogFormatter) for h in root.handlers)
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Cache metrics
# ---------------------------------------------------------------------------

def test_cache_metrics_increment(tmp_path, monkeypatch):
    monkeypatch.setitem(settings.__dict__, "CACHE_ENABLED", True)
    monkeypatch.setitem(settings.__dict__, "CACHE_DIR", str(tmp_path))

    from services import cache as cache_mod

    cache_mod.reset_cache_metrics()
    cache = cache_mod.FileCache(cache_dir=str(tmp_path), enabled=True)

    assert cache.get("p", {"a": 1}) is None      # miss
    cache.set("p", {"a": 1}, "v")
    assert cache.get("p", {"a": 1}) == "v"       # hit

    metrics = cache_mod.cache_metrics()
    assert metrics["misses"] == 1
    assert metrics["hits"] == 1
    assert metrics["hit_rate"] == pytest.approx(0.5)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(settings.__dict__, "CACHE_ENABLED", True)
    monkeypatch.setitem(settings.__dict__, "CACHE_DIR", str(tmp_path))
    from services import cache as cache_mod

    fresh = cache_mod.FileCache(cache_dir=str(tmp_path), enabled=True)
    monkeypatch.setattr(cache_mod, "_default_cache", fresh)
    cache_mod.reset_cache_metrics()
    return fresh


def test_cache_stats_endpoint_includes_metrics(client, tmp_cache):
    from services.cache import get_cache

    get_cache().get("missing", {"k": 1})          # a miss
    response = client.get("/cache/stats")

    assert response.status_code == 200
    body = response.json()
    assert "metrics" in body
    assert {"hits", "misses", "hit_rate"}.issubset(body["metrics"].keys())
    assert body["metrics"]["misses"] >= 1
