from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("pond_recommendation")


@contextmanager
def timed_stage(stage_name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("pipeline_stage_completed", extra={"stage": stage_name, "duration_seconds": round(elapsed, 6)})


def log_stage_duration(stage_name: str, duration_seconds: float) -> None:
    logger.info("pipeline_stage_summary", extra={"stage": stage_name, "duration_seconds": round(duration_seconds, 6)})


def measure_stage(stage_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            log_stage_duration(stage_name, time.perf_counter() - start)
            return result

        return wrapper

    return decorator
