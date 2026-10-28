#!/usr/bin/env python3
"""Static UI server that also reverse-proxies the API to the backend.

Why: when the UI (:6000) and the API (:5000) are separate ports, the *browser*
needs to reach both. Behind a NAT/gateway that only forwards one port, the page
loads but every API call fails. This server keeps the two processes on their own
ports (as configured) while presenting them to the browser as a single origin:

    browser ──► :6000 ──┬─► static files from frontend/
                        └─► /analyze*, /gis/*, /docs, ... ──► http://127.0.0.1:5000

Standard library only — no extra dependencies.

    python scripts/frontend_server.py --port 6000 --api http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths handled by the API rather than by the static file server.
API_PREFIXES = (
    "/analyze",
    "/export",
    "/recommendPond",
    "/gis",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/.well-known",
)

# Headers that must not be forwarded verbatim.
_HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding", "keep-alive"}
_TIMEOUT_S = 600.0


def is_api_path(path: str) -> bool:
    """True when a request path belongs to the API (not a static asset).

    Prefix match, so ``/analyze`` covers ``/analyzeContour``, ``/analyzeLocation``
    and ``/analyzeContour/enriched``. No file under ``frontend/`` shares these
    prefixes, so static assets are never swallowed.
    """

    base = path.split("?", 1)[0].split("#", 1)[0]
    return any(base.startswith(prefix) for prefix in API_PREFIXES)


class _Handler(http.server.SimpleHTTPRequestHandler):
    api_target = "http://127.0.0.1:5000"
    protocol_version = "HTTP/1.1"

    # -- proxy -------------------------------------------------------------
    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        request = urllib.request.Request(self.api_target + self.path, data=body, method=method)
        for name, value in self.headers.items():
            if name.lower() not in _HOP_BY_HOP:
                request.add_header(name, value)

        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                payload = response.read()
                self._relay(response.status, response.headers, payload)
        except urllib.error.HTTPError as exc:
            self._relay(exc.code, exc.headers, exc.read())
        except urllib.error.URLError as exc:
            self._relay(
                502,
                {"Content-Type": "application/json"},
                b'{"detail":{"code":"UPSTREAM_UNAVAILABLE","message":"API unreachable at '
                + self.api_target.encode()
                + b'"}}',
            )
            sys.stderr.write(f"[proxy] upstream error: {exc}\n")

    def _relay(self, status: int, headers, payload: bytes) -> None:
        self.send_response(status)
        for name, value in headers.items():
            if name.lower() not in _HOP_BY_HOP:
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # -- routing -----------------------------------------------------------
    def _dispatch(self, method: str) -> None:
        if is_api_path(self.path):
            self._proxy(method)
        elif method == "GET":
            requested = self.translate_path(self.path)
            if self.path.endswith("/"):
                self.path += "index.html"
            elif not Path(requested).is_file() and Path(requested + "/index.html").is_file():
                self.path += "/index.html"
            super().do_GET()
        else:
            self.send_error(405, "Method not allowed for static assets")

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def do_OPTIONS(self):  # noqa: N802
        self._dispatch("OPTIONS")

    def log_message(self, fmt, *args):
        # keep the log, but mark proxied calls so the origin is obvious
        prefix = "proxy" if is_api_path(self.path) else "static"
        sys.stderr.write(f"[{prefix}] {self.address_string()} {fmt % args}\n")


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(port: int, api_target: str, directory: Path) -> None:
    # `directory` must be passed to the constructor (SimpleHTTPRequestHandler
    # defaults it to os.getcwd(), overriding any class attribute).
    handler_cls = type("BoundHandler", (_Handler,), {"api_target": api_target})
    handler = functools.partial(handler_cls, directory=str(directory))
    with _Server(("0.0.0.0", port), handler) as httpd:
        print(f"frontend  : http://0.0.0.0:{port}/  (static: {directory})", flush=True)
        print(f"api proxy : /analyze*, /gis/*, /docs … -> {api_target}", flush=True)
        httpd.serve_forever()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Static UI server + API reverse proxy.")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--api", default="http://127.0.0.1:5000", help="backend base URL")
    parser.add_argument("--directory", type=Path, default=PROJECT_ROOT / "frontend")
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        print(f"error: no such directory: {args.directory}", file=sys.stderr)
        return 1
    try:
        serve(args.port, args.api.rstrip("/"), args.directory)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
