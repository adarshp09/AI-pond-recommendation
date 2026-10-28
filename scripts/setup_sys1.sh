#!/usr/bin/env bash
# AI Pond Recommendation — one-shot setup / run helper for a Linux host (SYS1).
#
# Handles the quirks of this box: python3 -m venv has no ensurepip/pip, so
# dependencies are installed with the *system* pip into the venv's site-packages
# (user-level, no sudo, no system-level install).
#
# Usage:
#   scripts/setup_sys1.sh system-deps      # install OS packages (apt; needs root/sudo)
#   scripts/setup_sys1.sh deps             # create/refresh .venv + dependencies
#   scripts/setup_sys1.sh env              # create .env (prompts for API key)
#   scripts/setup_sys1.sh env --api-key X  # create .env with a given key
#   scripts/setup_sys1.sh dataset          # download + prepare the local OSM GeoPackage
#   scripts/setup_sys1.sh start            # start backend (:6000) + frontend (:5000)
#   scripts/setup_sys1.sh stop             # stop both
#   scripts/setup_sys1.sh status           # show what is running / reachable
#   scripts/setup_sys1.sh test             # run the test suite (per-file: 512MB cgroup)
#   scripts/setup_sys1.sh smoke            # API smoke test (local_first/local_only/overpass_only)
#   scripts/setup_sys1.sh all              # deps + env + start
#
# A Makefile at the repo root wraps these targets: `make install`, `make run`, ...
# Docker is NOT required by this script.
#
# Env overrides: API_PORT (6000), WEB_PORT (5000), VENV (.venv), LOG_DIR (/tmp)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-$REPO_ROOT/.venv}"
API_PORT="${API_PORT:-6000}"
WEB_PORT="${WEB_PORT:-5000}"
LOG_DIR="${LOG_DIR:-/tmp}"
API_LOG="$LOG_DIR/pond_server.log"
WEB_LOG="$LOG_DIR/pond_frontend.log"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  # Prefer the distro interpreter: login shells here put linuxbrew/conda Python
  # (3.14) first on PATH, which must not be used to build this venv.
  if [ -x /usr/bin/python3 ]; then PY=/usr/bin/python3; else PY=python3; fi
fi

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

venv_python() { echo "$VENV/bin/python"; }

py_version() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }

# The venv is only reusable if it was built by the interpreter we intend to use.
# A mismatched pyvenv.cfg (e.g. someone ran `python3 -m venv` from a login shell
# where linuxbrew/conda Python comes first) also invalidates it.
venv_is_ok() {
  [ -x "$(venv_python)" ] || return 1
  local want have cfg
  want="$(py_version "$PY")"
  have="$(py_version "$(venv_python)")"
  [ -n "$want" ] && [ "$want" = "$have" ] || return 1
  cfg="$(sed -n 's/^version *= *//p' "$VENV/pyvenv.cfg" 2>/dev/null)"
  case "$cfg" in ""|"$want".*) return 0 ;; *) return 1 ;; esac
}

site_packages() {
  local py; py="$(venv_python)"
  [ -x "$py" ] || return 1
  "$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}

# --------------------------------------------------------------------------
# Container -> native fallback. If the shell was set up for docker compose
# (service hostnames, /app paths) the values are rewritten for this host.
# config.native_service_url() does the same for service URLs in the app; this
# covers the env vars the launcher itself passes to uvicorn.
sanitize_container_env() {
  case "${CACHE_DIR:-}" in
    /app/*)
      warn "remapping CACHE_DIR ${CACHE_DIR} -> $REPO_ROOT/.cache"
      export CACHE_DIR="$REPO_ROOT/.cache" ;;
  esac
  case "${LOCAL_OSM_GPKG_PATH:-}" in
    /app/*)
      export LOCAL_OSM_GPKG_PATH="$REPO_ROOT/data/osm/prepared/land_context.gpkg" ;;
  esac
  case "${LOCAL_OSM_MANIFEST_PATH:-}" in
    /app/*)
      export LOCAL_OSM_MANIFEST_PATH="$REPO_ROOT/data/osm/manifest.json" ;;
  esac
  case "${OTEL_EXPORTER_OTLP_ENDPOINT:-}" in
    http://jaeger:*|https://jaeger:*)
      export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318" ;;
  esac
}

# --------------------------------------------------------------------------
APT_PACKAGES="python3 python3-venv python3-dev build-essential curl libgdal-dev libgeos-dev libproj-dev"

cmd_system_deps() {
  say "==> system dependencies"
  echo "  python : $(command -v "$PY" >/dev/null 2>&1 && "$PY" --version 2>&1 || echo 'not found')"
  echo "  apt    : $APT_PACKAGES"

  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found — install the equivalents for your distro:"
    warn "  GDAL/GEOS/PROJ development headers, python3-venv, build-essential, curl"
    return 0
  fi

  if [ "$(id -u)" = "0" ]; then
    apt-get update && apt-get install -y $APT_PACKAGES || die "apt-get failed"
  elif command -v sudo >/dev/null 2>&1; then
    warn "using sudo (you may be prompted for a password)"
    sudo apt-get update && sudo apt-get install -y $APT_PACKAGES || die "apt-get failed"
  else
    warn "no root/sudo — run manually: sudo apt-get install -y $APT_PACKAGES"
  fi

  echo "  note: prebuilt manylinux wheels for pyogrio/rasterio bundle GDAL, so the"
  echo "        -dev packages are only needed when pip must build from source."
}

# --------------------------------------------------------------------------
cmd_deps() {
  say "==> dependencies"
  say "using interpreter: $PY ($("$PY" --version 2>&1))"

  if ! venv_is_ok; then
    if [ -d "$VENV" ]; then
      say "existing $VENV was built by a different interpreter — recreating"
      rm -rf "$VENV"
    else
      say "creating venv at $VENV"
    fi
    "$PY" -m venv --without-pip "$VENV" || die "venv creation failed"
  fi

  local sp; sp="$(site_packages)" || die "cannot locate venv site-packages"
  mkdir -p "$sp"

  # Use the chosen interpreter's own pip (the login shell's `pip` may belong to a
  # different Python entirely). --target keeps it user-level; no sudo, PEP 668 safe.
  say "installing into $sp"
  "$PY" -m pip install --quiet --no-input --retries 10 --timeout 60 \
    --target "$sp" -r requirements.txt pyogrio osmium rasterio \
    || die "pip install failed (network?)"

  say "verifying imports"
  "$(venv_python)" - <<'PY' || die "dependency verification failed"
import importlib
import importlib.util
import sys
required = ["fastapi", "uvicorn", "numpy", "scipy", "shapely", "pyproj",
            "geopandas", "pyogrio", "rasterio", "lxml", "httpx", "requests",
            "dotenv", "pytest", "osmium"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
for m in required:
    if m not in missing:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', 'ok')}")
if missing:
    print("MISSING:", missing, file=sys.stderr)
    sys.exit(1)
PY
}

# --------------------------------------------------------------------------
cmd_env() {
  local api_key=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --api-key) api_key="${2:-}"; shift 2 ;;
      *) die "unknown option for env: $1" ;;
    esac
  done

  if [ -f .env ] && [ -z "$api_key" ]; then
    say "==> .env already exists (leaving it alone): $REPO_ROOT/.env"
    return 0
  fi

  say "==> creating .env"
  if [ -n "$api_key" ]; then
    {
      echo "# Generated by scripts/setup_sys1.sh"
      echo "OPENTOPOGRAPHY_API_KEY=$api_key"
    } > .env
  elif [ -f .env.example ]; then
    cp .env.example .env
  else
    printf 'OPENTOPOGRAPHY_API_KEY=\n' > .env
  fi

  if [ -n "$api_key" ]; then
    chmod 600 .env
  fi

  if ! grep -qE '^OPENTOPOGRAPHY_API_KEY=.+' .env; then
    warn "OPENTOPOGRAPHY_API_KEY is empty in .env"
    warn "/analyzeContour works without it; /analyzeLocation requires a valid key."
    warn "Add it:  scripts/setup_sys1.sh env --api-key <KEY>"
  fi
}

# --------------------------------------------------------------------------
cmd_dataset() {
  say "==> local OSM dataset"
  local gpkg="data/osm/prepared/land_context.gpkg"
  if [ -f "$gpkg" ]; then
    say "dataset already present: $gpkg ($(du -h "$gpkg" | cut -f1))"
    return 0
  fi
  say "downloading Geofabrik extract (this takes a while)…"
  "$(venv_python)" scripts/download_osm_data.py || die "download failed"
  say "preparing GeoPackage (relation-aware clip; several minutes)…"
  "$(venv_python)" scripts/prepare_osm_data.py || die "prepare failed"
  say "dataset ready: $gpkg"
}

# --------------------------------------------------------------------------
port_busy() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

cmd_start() {
  say "==> starting services"
  cmd_stop >/dev/null 2>&1
  sanitize_container_env

  [ -x "$(venv_python)" ] || die "no venv — run: scripts/setup_sys1.sh deps"
  [ -f .env ] || warn ".env missing — /analyzeLocation will fail (see: setup_sys1.sh env)"

  if [ "${SINGLE_PORT:-0}" = "1" ]; then
    # One port serves the UI (/ui/) AND the API -> same origin, no CORS, and only
    # one port has to be reachable/forwarded. Best default behind NAT.
    port_busy "$WEB_PORT" && die "port $WEB_PORT already in use"
    setsid --fork "$(venv_python)" -m uvicorn backend.main:app \
        --host 0.0.0.0 --port "$WEB_PORT" > "$API_LOG" 2>&1 < /dev/null
    local i
    for i in $(seq 1 20); do
      curl -sf -m 2 "http://127.0.0.1:$WEB_PORT/health" >/dev/null 2>&1 && break
      sleep 0.5
    done
    echo
    echo "  app (UI + API): port $WEB_PORT   (log $API_LOG)"
    echo "  open          : http://<host>:$WEB_PORT/   (→ /ui/)"
    echo
    cmd_status
    return
  fi

  port_busy "$API_PORT" && die "port $API_PORT already in use"
  port_busy "$WEB_PORT" && die "port $WEB_PORT already in use"

  # setsid --fork detaches fully: the parent returns immediately, so this works
  # interactively *and* over a non-interactive ssh command.
  setsid --fork "$(venv_python)" -m uvicorn backend.main:app \
      --host 0.0.0.0 --port "$API_PORT" > "$API_LOG" 2>&1 < /dev/null
  # Static UI + API reverse proxy on one origin: the browser only needs this port.
  setsid --fork "$(venv_python)" scripts/frontend_server.py \
      --port "$WEB_PORT" --api "http://127.0.0.1:$API_PORT" \
      --directory frontend > "$WEB_LOG" 2>&1 < /dev/null

  local i
  for i in $(seq 1 20); do
    curl -sf -m 2 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
  for i in $(seq 1 20); do
    curl -sf -m 2 "http://127.0.0.1:$WEB_PORT/openapi.json" >/dev/null 2>&1 && break
    sleep 0.5
  done

  echo
  echo "  api     : http://0.0.0.0:$API_PORT   (log $API_LOG)"
  echo "  ui      : http://0.0.0.0:$WEB_PORT   (log $WEB_LOG) — proxies the API"
  echo "  open    : http://<host-ip>:$WEB_PORT    (browser needs ONLY this port)"
  echo
  cmd_status
}

# --------------------------------------------------------------------------
cmd_stop() {
  pkill -f '[u]vicorn backend.main' 2>/dev/null
  # any UI server this script may have started (proxy or plain static)
  pkill -f '[f]rontend_server.py' 2>/dev/null
  pkill -f '[h]ttp.server' 2>/dev/null
  sleep 1
  if pgrep -f '[u]vicorn backend.main' >/dev/null || pgrep -f '[f]rontend_server.py' >/dev/null || pgrep -f '[h]ttp.server' >/dev/null; then
    warn "some processes still running"
  else
    say "stopped"
  fi
}

# --------------------------------------------------------------------------
host_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}

port_listening() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"; }

cmd_status() {
  say "==> status"
  local ip; ip="$(host_ip)"
  local api_pids web_pids
  api_pids="$(pgrep -f '[u]vicorn backend.main' | tr '\n' ' ')"
  web_pids="$(pgrep -f '[f]rontend_server.py|[h]ttp.server' | tr '\n' ' ')"

  # Does the UI port also answer API calls? (single-port, or the proxy server)
  local api_on_web=0
  if port_listening "$WEB_PORT" && curl -sf -m 3 "http://127.0.0.1:$WEB_PORT/openapi.json" >/dev/null 2>&1; then
    api_on_web=1
  fi

  local mode_short="stopped" mode="stopped"
  if [ -n "$web_pids" ] && [ "$api_on_web" = "1" ]; then
    mode_short="proxy"; mode="ui :$WEB_PORT proxies api :$API_PORT (browser needs only :$WEB_PORT)"
  elif [ -n "$web_pids" ]; then
    mode_short="two-port"; mode="two-port (api :$API_PORT, ui :$WEB_PORT) — browser needs BOTH"
  elif [ "$api_on_web" = "1" ]; then
    mode_short="single-port"; mode="single-port (UI + API on :$WEB_PORT)"
  elif port_listening "$API_PORT"; then
    mode_short="api-only"; mode="api-only (:$API_PORT)"
  fi

  echo "  mode     : $mode"

  if [ -n "$api_pids" ]; then
    echo "  api      : running (pid ${api_pids% })"
  else
    echo "  api      : not running"
  fi

  # The UI is either its own process or served by the API — never report it as
  # "not running" just because there is no separate static server.
  if [ -n "$web_pids" ] && [ "$api_on_web" = "1" ]; then
    echo "  ui       : running — frontend server on :$WEB_PORT (proxies the API, pid ${web_pids% })"
  elif [ -n "$web_pids" ]; then
    echo "  ui       : running — static server on :$WEB_PORT (pid ${web_pids% })"
  elif [ "$api_on_web" = "1" ]; then
    echo "  ui       : running — served by the API at :$WEB_PORT/ui/  (no separate process)"
  else
    echo "  ui       : not reachable"
  fi

  check() { printf '  http://%-30s -> %s\n' "$1" "$(curl -s -m 4 -o /dev/null -w '%{http_code}' "http://$1" 2>/dev/null || echo '---')"; }

  if port_listening "$API_PORT"; then
    check "127.0.0.1:$API_PORT/health"
    check "$ip:$API_PORT/health"
  else
    echo "  api port :$API_PORT not listening"
  fi

  if [ -n "$web_pids" ]; then
    check "127.0.0.1:$WEB_PORT/"
    check "$ip:$WEB_PORT/"
    check "127.0.0.1:$WEB_PORT/health"          # reaches the API through the proxy
  elif [ "$api_on_web" = "1" ]; then
    check "127.0.0.1:$WEB_PORT/ui/"
    check "$ip:$WEB_PORT/ui/"
  elif ! port_listening "$WEB_PORT"; then
    echo "  ui port :$WEB_PORT not listening ($mode_short)"
  fi

  command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -E ":($API_PORT|$WEB_PORT) " | sed 's/^/  /'
}

# --------------------------------------------------------------------------
cmd_test() {
  [ -x "$(venv_python)" ] || die "no venv — run: scripts/setup_sys1.sh deps"
  say "==> tests (per-file: this host's cgroup caps memory at 512 MiB)"
  local fail=0 failed="" line total=0
  total="$("$(venv_python)" -m pytest -q -p no:cacheprovider tests/ --co 2>/dev/null | tail -1)"
  for f in tests/test_*.py; do
    line="$(OPENTOPOGRAPHY_API_KEY="${OPENTOPOGRAPHY_API_KEY:-smoke-test}" \
      "$(venv_python)" -m pytest -q -p no:cacheprovider "$f" 2>&1 | tail -1)"
    case "$line" in *failed*) fail=$((fail+1)); failed="$failed $f"; printf '  FAIL %s :: %s\n' "$f" "$line";; esac
  done
  echo "  collected: $total"
  if [ "$fail" -eq 0 ]; then
    say "all test files passed"
  else
    warn "$fail file(s) failed:$failed"
    return 1
  fi
}

cmd_smoke() {
  for mode in local_first local_only overpass_only; do
    echo "--- $mode ---"
    "$(venv_python)" scripts/api_smoke_test.py --mode "$mode" || true
  done
}

# --------------------------------------------------------------------------
case "${1:-}" in
  system-deps) shift; cmd_system_deps "$@" ;;
  deps)    shift; cmd_deps "$@" ;;
  env)     shift; cmd_env "$@" ;;
  dataset) shift; cmd_dataset "$@" ;;
  start)   shift; cmd_start "$@" ;;
  stop)    shift; cmd_stop "$@" ;;
  status)  shift; cmd_status "$@" ;;
  test)    shift; cmd_test "$@" ;;
  smoke)   shift; cmd_smoke "$@" ;;
  all)     shift; cmd_deps && cmd_env "$@" && cmd_start ;;
  ""|-h|--help)
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *) die "unknown command: $1 (try --help)" ;;
esac
