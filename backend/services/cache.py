from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings


class CacheError(Exception):
    pass


class FileCache:
    def __init__(self, cache_dir: str | None = None, ttl_seconds: int | None = None, enabled: bool | None = None):
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.ttl_seconds = int(ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS)
        self.enabled = bool(enabled if enabled is not None else settings.CACHE_ENABLED)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, prefix: str, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{prefix}_{digest}.json"

    def get(self, prefix: str, payload: Any) -> Any:
        if not self.enabled:
            return None
        cache_path = self.cache_dir / self._make_key(prefix, payload)
        if not cache_path.exists():
            return None
        try:
            entry = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        timestamp = float(entry.get("timestamp", 0.0))
        if time.time() - timestamp > self.ttl_seconds:
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return entry.get("value")

    def set(self, prefix: str, payload: Any, value: Any) -> None:
        if not self.enabled:
            return
        cache_path = self.cache_dir / self._make_key(prefix, payload)
        try:
            cache_path.write_text(
                json.dumps({"timestamp": time.time(), "value": value}, default=str, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            raise CacheError("Unable to write cache entry")

    def clear(self) -> None:
        if self.cache_dir.exists():
            for child in self.cache_dir.iterdir():
                try:
                    child.unlink()
                except OSError:
                    pass

    def bypass(self) -> "FileCache":
        return FileCache(cache_dir=str(self.cache_dir), ttl_seconds=self.ttl_seconds, enabled=False)


_default_cache = FileCache()


def get_cache(cache_dir: str | None = None, ttl_seconds: int | None = None, enabled: bool | None = None) -> FileCache:
    if cache_dir is None and ttl_seconds is None and enabled is None:
        return _default_cache
    return FileCache(cache_dir=cache_dir, ttl_seconds=ttl_seconds, enabled=enabled)


def cached_response(prefix: str, payload: Any, value_factory, *, cache_dir: str | None = None, ttl_seconds: int | None = None, enabled: bool | None = None):
    cache = get_cache(cache_dir=cache_dir, ttl_seconds=ttl_seconds, enabled=enabled)
    cached_value = cache.get(prefix, payload)
    if cached_value is not None:
        return cached_value
    value = value_factory()
    cache.set(prefix, payload, value)
    return value


def cache_bypass() -> bool:
    return os.getenv("CACHE_BYPASS", "0").lower() in {"1", "true", "yes", "on"}
