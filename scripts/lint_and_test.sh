#!/usr/bin/env bash
set -euo pipefail

ruff check .

if command -v gdlint >/dev/null 2>&1; then
  gdlint $(git ls-files '*.gd')
elif [[ "${CI:-}" == "true" ]]; then
  echo "gdlint is required in CI but was not found on PATH" >&2
  exit 1
else
  echo "⚠️  gdlint not found; skipping GDScript lint locally"
fi

pytest
