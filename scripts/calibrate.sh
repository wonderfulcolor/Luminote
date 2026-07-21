#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHONPATH=. .venv/bin/python scripts/calibrate.py
