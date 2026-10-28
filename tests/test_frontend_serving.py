"""Serving the web UI from the API (single-origin deployment).

The UI is mounted at ``/ui/`` so one port serves both the page and the API —
which is what makes the buttons work behind a NAT that only forwards one port.

Offline and deterministic: no network, no external services.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app

PROJECT_ROOT_UI_TITLE = "<title>AI Pond Recommendation</title>"


@pytest.fixture
def raw_client():
    """A client that does NOT follow redirects (so we can see the 307)."""
    return TestClient(app, follow_redirects=False)


def test_browser_root_redirects_to_ui(raw_client):
    response = raw_client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_root_still_returns_json_for_api_clients(client):
    body = client.get("/").json()
    assert body["service"] == "AI Pond Recommendation API"
    assert "analyze_contour" in body["routes"]


def test_ui_index_is_served(raw_client):
    response = raw_client.get("/ui/")
    assert response.status_code == 200
    assert PROJECT_ROOT_UI_TITLE in response.text
    assert "app.js" in response.text


def test_ui_assets_are_served(raw_client):
    app_js = raw_client.get("/ui/app.js")
    assert app_js.status_code == 200
    assert "API_BASE_URL" in app_js.text


def test_api_routes_are_not_shadowed_by_the_mount(client):
    # /ui must not capture the API surface.
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/analyzeContour", files={"file": ("x.txt", b"nope", "text/plain")}).status_code in (400, 413, 422)


def test_ui_defaults_to_same_origin_and_keeps_fallbacks():
    """app.js targets the page origin; ?api= / localhost remain as overrides."""
    app_js = TestClient(app).get("/ui/app.js").text
    assert "return pageLocation.origin;" in app_js
    assert "DEFAULT_API_PORT = 5000" in app_js
    assert 'get("api")' in app_js


def _load_frontend_server():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "frontend_server.py"
    spec = importlib.util.spec_from_file_location("frontend_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_route_classification():
    """Only API paths are proxied; UI assets are served statically."""
    module = _load_frontend_server()
    for path in ("/analyzeContour", "/analyzeLocation", "/analyze", "/analyzeContour/enriched",
                 "/gis/land-context", "/health", "/docs", "/redoc", "/openapi.json",
                 "/exportLocationKml", "/exportLocationKmz", "/recommendPond"):
        assert module.is_api_path(path), path
    for path in ("/", "/index.html", "/app.js", "/style.css", "/favicon.ico"):
        assert not module.is_api_path(path), path


def test_proxy_forwards_requests_and_serves_assets(tmp_path):
    """End-to-end: the frontend server proxies API calls and serves the UI."""
    import functools
    import http.server
    import threading
    import urllib.request

    # upstream "API"
    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"status":"ok","via":"upstream"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            body = b'{"status":"success","upstream_body_seen":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    upstream = http.server.HTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    (tmp_path / "index.html").write_text("<title>UI</title>", encoding="utf-8")

    module = _load_frontend_server()
    handler_cls = type("H", (module._Handler,), {
        "api_target": f"http://127.0.0.1:{upstream.server_address[1]}",
    })
    proxy = module._Server(("127.0.0.1", 0), functools.partial(handler_cls, directory=str(tmp_path)))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    try:
        base = f"http://127.0.0.1:{proxy.server_address[1]}"
        with urllib.request.urlopen(base + "/", timeout=10) as response:
            assert b"<title>UI</title>" in response.read()
        with urllib.request.urlopen(base + "/health", timeout=10) as response:
            assert b"upstream" in response.read()
        post = urllib.request.Request(base + "/analyzeContour", data=b"x", method="POST")
        with urllib.request.urlopen(post, timeout=10) as response:
            assert b"success" in response.read()
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_cors_allows_the_separate_ui_port(client):
    """Frontend :6000 -> backend :5000 is cross-origin and must be allowed."""
    preflight = client.options(
        "/analyzeContour",
        headers={
            "Origin": "http://10.1.75.79:6000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://10.1.75.79:6000"


def test_cors_still_rejects_public_origins(client):
    preflight = client.options(
        "/analyzeContour",
        headers={
            "Origin": "http://evil.example.com:6000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in preflight.headers
