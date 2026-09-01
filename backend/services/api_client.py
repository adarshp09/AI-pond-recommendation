from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

from config import redact_url, redact_secret, settings
from services.cache import cache_bypass, cached_response

logger = logging.getLogger("pond_recommendation")


class APIClientError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, timeout: float | None = None, retries: int | None = None, cache_enabled: bool | None = None):
        self.timeout = float(timeout if timeout is not None else settings.HTTP_TIMEOUT_SECONDS)
        self.retries = int(retries if retries is not None else settings.HTTP_RETRY_COUNT)
        self.cache_enabled = bool(cache_enabled if cache_enabled is not None else settings.CACHE_ENABLED)

    def request(self, method: str, url: str, *, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, use_cache: bool = True) -> httpx.Response:
        safe_url = redact_url(url)
        logger.debug("http_request_started", extra={"method": method.upper(), "url": safe_url, "cache": use_cache})

        cache_payload = {"method": method.upper(), "url": safe_url, "params": params or {}, "data": data or {}, "headers": headers or {}}
        if use_cache and not cache_bypass():
            cached = cached_response("http_request", cache_payload, lambda: None, enabled=self.cache_enabled)
            if cached is not None:
                if isinstance(cached, dict) and cached.get("status_code") is not None:
                    return httpx.Response(cached["status_code"], content=cached.get("content", b""), headers=cached.get("headers", {}), request=httpx.Request(method, url))

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = httpx.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                if response.is_error:
                    if attempt < self.retries:
                        time.sleep(2**attempt)
                        continue
                    raise APIClientError(f"HTTP {response.status_code}: {response.text[:200]}")
                if use_cache and not cache_bypass():
                    cached_response(
                        "http_request",
                        cache_payload,
                        lambda: {
                            "status_code": response.status_code,
                            "content": response.content,
                            "headers": dict(response.headers),
                        },
                        enabled=self.cache_enabled,
                    )
                logger.debug("http_request_completed", extra={"method": method.upper(), "url": safe_url, "status": response.status_code})
                return response
            except (httpx.TimeoutException, httpx.HTTPError, APIClientError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                logger.warning("http_request_failed", extra={"method": method.upper(), "url": safe_url, "error": str(exc)})
                raise APIClientError(str(exc)) from exc

        if last_error is not None:
            logger.warning("http_request_failed_final", extra={"method": method.upper(), "url": safe_url, "error": str(last_error)})
            raise APIClientError(str(last_error))
        raise APIClientError(f"Request failed for {safe_url}")

    def get(self, url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, use_cache: bool = True) -> httpx.Response:
        return self.request("GET", url, params=params, headers=headers, use_cache=use_cache)

    def post(self, url: str, *, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, use_cache: bool = False) -> httpx.Response:
        return self.request("POST", url, params=params, data=data, headers=headers, use_cache=use_cache)


def get_api_client(timeout: float | None = None, retries: int | None = None, cache_enabled: bool | None = None) -> ApiClient:
    return ApiClient(timeout=timeout, retries=retries, cache_enabled=cache_enabled)
