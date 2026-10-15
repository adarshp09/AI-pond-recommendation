#!/usr/bin/env python3
"""Phase 01 — download a regional OpenStreetMap extract (Geofabrik).

Downloads an ``.osm.pbf`` extract to ``data/osm/raw`` and records provenance
(source URL, retrieval timestamp, byte size, SHA-256, optional MD5). Validates the
HTTP response, the expected size, and the PBF magic header so a partial or broken
download is never accepted.

This is an operator/preprocessing tool. It is NEVER invoked during API requests.

Examples
--------
    # Default: Geofabrik India "central-zone" (covers the sample study area).
    python scripts/download_osm_data.py

    # Explicit source.
    python scripts/download_osm_data.py \
        --url https://download.geofabrik.de/asia/india/central-zone-latest.osm.pbf

    # Just verify an existing file against its .md5 sidecar.
    python scripts/download_osm_data.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = os.getenv(
    "OSM_EXTRACT_URL",
    "https://download.geofabrik.de/asia/india/central-zone-latest.osm.pbf",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "osm" / "raw"
USER_AGENT = os.getenv("OSM_DOWNLOAD_USER_AGENT", "AI-Pond-App/1.0 (osm dataset builder)")
CHUNK = 1024 * 256
MIN_BYTES = 1024  # a real extract is far larger; guards against HTML error pages


class DownloadError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pbf(path: Path) -> None:
    """A valid .osm.pbf starts with a 4-byte length then the 'OSMHeader' blob."""

    with path.open("rb") as handle:
        header = handle.read(16)
    if len(header) < 16 or b"OSMHeader" not in header[4:16]:
        raise DownloadError(f"{path.name} is not a valid OSM PBF (missing OSMHeader magic)")


def _fetch_text(url: str, timeout: float = 20.0) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace").strip()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _expected_md5(url: str) -> str | None:
    """Geofabrik publishes ``<url>.md5`` as ``<hash>  <filename>``."""

    text = _fetch_text(url + ".md5")
    if not text:
        return None
    token = text.split()[0].strip().lower()
    return token if len(token) == 32 else None


def download(url: str, destination: Path, *, force: bool = False, timeout: float = 60.0) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[download] {destination} exists — verifying instead of re-downloading", file=sys.stderr)
        record = verify(destination)
        record["retrieved_at"] = _now_iso()
        record["url"] = url
        record["cached"] = True
        sidecar = destination.with_suffix(destination.suffix + ".download.json")
        sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record

    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise DownloadError(f"unexpected HTTP status {status} for {url}")
            expected = response.headers.get("Content-Length")
            total = int(expected) if expected and expected.isdigit() else None
            received = 0
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"download failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    if total is not None and received != total:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"incomplete download: got {received} of {total} bytes")
    if received < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"download too small ({received} bytes) — likely an error page")

    _validate_pbf(tmp)
    tmp.replace(destination)

    record = {
        "url": url,
        "file": str(destination),
        "filename": destination.name,
        "retrieved_at": _now_iso(),
        "bytes": received,
        "sha256": _sha256(destination),
        "elapsed_s": round(elapsed, 2),
    }
    expected_md5 = _expected_md5(url)
    if expected_md5:
        actual = _md5(destination)
        record["md5_expected"] = expected_md5
        record["md5_actual"] = actual
        record["md5_verified"] = actual == expected_md5
        if actual != expected_md5:
            raise DownloadError(
                f"MD5 mismatch for {destination.name}: {actual} != {expected_md5}"
            )
    sidecar = destination.with_suffix(destination.suffix + ".download.json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def verify(destination: Path) -> dict:
    if not destination.exists():
        raise DownloadError(f"{destination} does not exist")
    _validate_pbf(destination)
    return {
        "file": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "md5": _md5(destination),
        "verified_at": _now_iso(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a Geofabrik OSM extract.")
    parser.add_argument("--url", default=DEFAULT_URL, help="source .osm.pbf URL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filename", default=None, help="override output filename")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    parser.add_argument("--verify-only", action="store_true", help="validate without downloading")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    filename = args.filename or os.path.basename(args.url.split("?")[0])
    destination = args.output_dir / filename

    try:
        if args.verify_only:
            record = verify(destination)
        else:
            record = download(args.url, destination, force=args.force, timeout=args.timeout)
    except DownloadError as exc:
        print(f"[download] FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
