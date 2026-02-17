#!/usr/bin/env bash
set -euo pipefail

ruff check .
gdlint $(git ls-files '*.gd')
pytest
