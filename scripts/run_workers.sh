#!/usr/bin/env sh
# Runs what is queued, then exits. Fetch first (it feeds the parse queue), then parse.
# Nothing polls the database while this is not running, so Neon's compute can sleep.
set -e
cd "$(dirname "$0")/.."
PYTHON=python
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python

for queue in fetch ingest; do
    "$PYTHON" -m src.jobs.workers "$queue" --drain
done
