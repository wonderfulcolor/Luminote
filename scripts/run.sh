#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
port="${LUMINOTE_PORT:-8765}"
PYTHONPATH=. .venv/bin/python -m uvicorn luminote.api:app --host 127.0.0.1 --port "$port"
