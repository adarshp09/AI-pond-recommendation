from __future__ import annotations

import json
import logging
import sys
from typing import Any

from config import redact_secret, settings


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(level: str | None = None) -> logging.Logger:
    level_name = (level or settings.LOG_LEVEL or "INFO").upper()
    logger = logging.getLogger("pond_recommendation")
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    return logger


def log_event(level: int, message: str, **context: Any) -> None:
    logger = logging.getLogger("pond_recommendation")
    safe_context = {}
    for key, value in context.items():
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            safe_context[key] = redact_secret(str(value))
        else:
            safe_context[key] = value
    logger.log(level, message, extra={"context": safe_context})


configure_logging()
