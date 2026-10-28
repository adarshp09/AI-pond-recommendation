from __future__ import annotations

import hashlib
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings


class CacheError(Exception):
    pass


class FileCache:
    """Thread-safe file-based cache with LRU eviction and bounded storage."""

    def __init__(
        self,
        cache_dir: str | None = None,
        ttl_seconds: int | None = None,
        enabled: bool | None = None,
        max_entries: int | None = None,
        max_size_bytes: int | None = None,
    ):
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.ttl_seconds = int(ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS)
        self.enabled = bool(enabled if enabled is not None else settings.CACHE_ENABLED)
        self.max_entries = max_entries or int(os.getenv("CACHE_MAX_ENTRIES", "10000"))
        self.max_size_bytes = max_size_bytes or int(os.getenv("CACHE_MAX_SIZE_MB", "100")) * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # LRU tracking: key -> (last_access_time, file_size)
        self._lru: Dict[str, tuple[float, int]] = {}
        self._lru_lock = threading.RLock()
        self._current_size = 0
        self._load_lru_index()

    def _load_lru_index(self) -> None:
        """Build LRU index from existing cache files on startup."""
        if not self.cache_dir.exists():
            return
        with self._lru_lock:
            for child in self.cache_dir.iterdir():
                if child.suffix == ".json":
                    try:
                        stat = child.stat()
                        # Extract prefix from filename for LRU tracking
                        self._lru[child.name] = (stat.st_atime, stat.st_size)
                        self._current_size += stat.st_size
                    except OSError:
                        pass
            # Enforce limits on startup
            self._enforce_limits()

    def _make_key(self, prefix: str, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{prefix}_{digest}.json"

    def _update_lru(self, filename: str, file_size: int) -> None:
        """Update LRU tracking for a cache entry."""
        with self._lru_lock:
            self._lru[filename] = (time.time(), file_size)

    def _enforce_limits(self) -> None:
        """Enforce max_entries and max_size_bytes by evicting LRU entries."""
        with self._lru_lock:
            # Evict by count
            while len(self._lru) > self.max_entries:
                self._evict_lru()

            # Evict by size
            while self._current_size > self.max_size_bytes and self._lru:
                self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._lru:
            return
        lru_file = min(self._lru.items(), key=lambda kv: kv[1][0])[0]
        file_size = self._lru[lru_file][1]
        try:
            (self.cache_dir / lru_file).unlink(missing_ok=True)
        except OSError:
            pass
        self._current_size -= file_size
        del self._lru[lru_file]

    def get(self, prefix: str, payload: Any, ttl_seconds: int | None = None) -> Any:
        if not self.enabled:
            return None
        cache_path = self.cache_dir / self._make_key(prefix, payload)
        if not cache_path.exists():
            record_cache_miss()
            return None
        try:
            entry = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            record_cache_miss()
            return None
        timestamp = float(entry.get("timestamp", 0.0))
        effective_ttl = self.ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        if time.time() - timestamp > effective_ttl:
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass
            with self._lru_lock:
                self._lru.pop(cache_path.name, None)
            record_cache_miss()
            return None
        # Update LRU on successful access
        try:
            file_size = cache_path.stat().st_size
            self._update_lru(cache_path.name, file_size)
        except OSError:
            pass
        record_cache_hit()
        return entry.get("value")

    def set(self, prefix: str, payload: Any, value: Any) -> None:
        if not self.enabled:
            return
        cache_path = self.cache_dir / self._make_key(prefix, payload)
        try:
            payload_json = json.dumps({"timestamp": time.time(), "value": value}, default=str, sort_keys=True)
            cache_path.write_text(payload_json, encoding="utf-8")
        except OSError:
            raise CacheError("Unable to write cache entry")
        # Update LRU and enforce limits
        try:
            file_size = cache_path.stat().st_size
            self._update_lru(cache_path.name, file_size)
            with self._lru_lock:
                self._current_size += file_size
        except OSError:
            pass
        self._enforce_limits()

    def clear(self) -> None:
        with self._lru_lock:
            if self.cache_dir.exists():
                for child in self.cache_dir.iterdir():
                    try:
                        child.unlink()
                    except OSError:
                        pass
            self._lru.clear()
            self._current_size = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lru_lock:
            return {
                "entries": len(self._lru),
                "size_bytes": self._current_size,
                "max_entries": self.max_entries,
                "max_size_bytes": self.max_size_bytes,
                "enabled": self.enabled,
                "ttl_seconds": self.ttl_seconds,
            }

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        if not self.cache_dir.exists():
            return 0
        removed = 0
        now = time.time()
        with self._lru_lock:
            for filename, (_, size) in list(self._lru.items()):
                cache_path = self.cache_dir / filename
                if not cache_path.exists():
                    self._lru.pop(filename, None)
                    continue
                try:
                    entry = json.loads(cache_path.read_text(encoding="utf-8"))
                    timestamp = float(entry.get("timestamp", 0.0))
                    if now - timestamp > self.ttl_seconds:
                        cache_path.unlink(missing_ok=True)
                        self._current_size -= size
                        self._lru.pop(filename, None)
                        removed += 1
                except (json.JSONDecodeError, OSError):
                    # Corrupt or unreadable, remove it
                    cache_path.unlink(missing_ok=True)
                    self._current_size -= size
                    self._lru.pop(filename, None)
                    removed += 1
        return removed

    def bypass(self) -> "FileCache":
        return FileCache(
            cache_dir=str(self.cache_dir),
            ttl_seconds=self.ttl_seconds,
            enabled=False,
            max_entries=self.max_entries,
            max_size_bytes=self.max_size_bytes,
        )


_default_cache = FileCache()


def get_cache(
    cache_dir: str | None = None,
    ttl_seconds: int | None = None,
    enabled: bool | None = None,
    max_entries: int | None = None,
    max_size_bytes: int | None = None,
) -> FileCache:
    if cache_dir is None and ttl_seconds is None and enabled is None and max_entries is None and max_size_bytes is None:
        return _default_cache
    return FileCache(
        cache_dir=cache_dir,
        ttl_seconds=ttl_seconds,
        enabled=enabled,
        max_entries=max_entries,
        max_size_bytes=max_size_bytes,
    )


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


# In-memory hit/miss counters (Phase 05). Process-local, monotonic until reset;
# exposed via GET /cache/stats. Never persisted.
_CACHE_METRICS = {"hits": 0, "misses": 0}
_CACHE_METRICS_LOCK = threading.Lock()


def record_cache_hit() -> None:
    with _CACHE_METRICS_LOCK:
        _CACHE_METRICS["hits"] += 1


def record_cache_miss() -> None:
    with _CACHE_METRICS_LOCK:
        _CACHE_METRICS["misses"] += 1


def cache_metrics() -> dict:
    with _CACHE_METRICS_LOCK:
        hits = _CACHE_METRICS["hits"]
        misses = _CACHE_METRICS["misses"]
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": (float(hits) / total) if total else None,
    }


def reset_cache_metrics() -> None:
    with _CACHE_METRICS_LOCK:
        _CACHE_METRICS["hits"] = 0
        _CACHE_METRICS["misses"] = 0


def cache_is_enabled() -> bool:
    """Honour ``config.settings.CACHE_ENABLED`` (resolved at call time) + bypass env.

    Resolving at call time means tests/operators can disable caching without
    re-importing the module, and providers gate on this instead of the cached
    singleton's construction-time ``enabled`` flag.
    """

    try:
        from config import settings

        enabled = bool(getattr(settings, "CACHE_ENABLED", True))
    except Exception:  # pragma: no cover - defensive
        enabled = True
    return enabled and not cache_bypass()


# Per-key in-process single-flight locks: concurrent callers computing the same
# cache key share one computation instead of stampeding the provider.
_SINGLEFLIGHT_LOCKS: Dict[str, threading.Lock] = {}
_SINGLEFLIGHT_GUARD = threading.Lock()


def _singleflight_lock(filename: str) -> threading.Lock:
    with _SINGLEFLIGHT_GUARD:
        lock = _SINGLEFLIGHT_LOCKS.get(filename)
        if lock is None:
            lock = threading.Lock()
            _SINGLEFLIGHT_LOCKS[filename] = lock
        return lock


def get_or_set(
    prefix: str,
    payload: Any,
    value_factory,
    *,
    ttl_seconds: int | None = None,
    cache: "FileCache | None" = None,
):
    """Return the cached value for ``(prefix, payload)`` or compute and store it.

    * No caching (``CACHE_ENABLED`` false / bypass) => the factory is called directly.
    * A cache miss computes under a per-key lock so N concurrent callers perform
      one provider call (single-flight), then store the value.
    * Only successful values are stored; if ``value_factory`` raises, nothing is
      cached and the exception propagates (failures are never cached as success).
    """

    if not cache_is_enabled():
        return value_factory()

    active_cache = cache if cache is not None else get_cache()
    filename = active_cache._make_key(prefix, payload)
    with _singleflight_lock(filename):
        cached = active_cache.get(prefix, payload, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached
        value = value_factory()
        active_cache.set(prefix, payload, value)
        return value
