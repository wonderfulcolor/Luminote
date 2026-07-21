#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/generate_public_color_map.py
echo "Setup complete. Run ./scripts/calibrate.sh once, then ./scripts/run.sh"
