#!/usr/bin/env bash
# Clean the DEGIRO exports in raw_data/ and upload them to S3.
# Run inside your Distrobox container. Pass --dry-run to preview the upload.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

.venv/bin/python -m pytest -q tests          # refuse to run on a broken cleaner
.venv/bin/python src/cleaner.py
.venv/bin/python src/uploader.py "$@"