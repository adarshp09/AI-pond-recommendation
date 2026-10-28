# AI Pond Recommendation — native (no Docker) task runner.
#
# Everything delegates to scripts/setup_sys1.sh, which knows this host's quirks
# (512 MiB cgroup, venv built without pip). Docker is NOT required.
#
#   make install     create .venv and install dependencies
#   make env         create .env (set OPENTOPOGRAPHY_API_KEY for /analyzeLocation)
#   make dataset     download + prepare data/osm/prepared/land_context.gpkg
#   make run         start backend (:6000) + UI (:5000)
#   make run-single  start UI+API on one port (:5000) — best behind NAT
#   make status      show what is running / reachable
#   make test        run the test suite (per-file; host caps memory at 512 MiB)
#   make stop        stop the server(s)
#   make system-deps install OS packages via apt (needs root/sudo)
#   make clean       remove .venv and .cache (keeps data/)
#   make docker-up   docker compose up --build (when Docker is available)

# $(CURDIR) is used instead of $(dir ...) because this repo's path contains
# spaces, which make's dir/strip helpers would split into separate words.
# Run make from the repo root (or use `make -C <repo-root>`).
REPO_ROOT := $(CURDIR)
SETUP     := $(REPO_ROOT)/scripts/setup_sys1.sh

.DEFAULT_GOAL := help
.PHONY: help native system-deps install env dataset run run-single stop status test smoke clean docker-up

help:
	@sed -n '2,16p' $(firstword $(MAKEFILE_LIST)) | sed 's/^# \{0,1\}//'

system-deps:
	@bash "$(SETUP)" system-deps

install:
	@bash "$(SETUP)" deps

env:
	@bash "$(SETUP)" env

dataset:
	@bash "$(SETUP)" dataset

run:
	@bash "$(SETUP)" start

run-single:
	@SINGLE_PORT=1 bash "$(SETUP)" start

stop:
	@bash "$(SETUP)" stop

status:
	@bash "$(SETUP)" status

test:
	@bash "$(SETUP)" test

smoke:
	@bash "$(SETUP)" smoke

native: install env run

clean:
	@rm -rf "$(REPO_ROOT)/.venv" "$(REPO_ROOT)/.cache"
	@echo "removed .venv and .cache (data/ untouched)"

docker-up:
	@cd "$(REPO_ROOT)" && docker compose up --build
