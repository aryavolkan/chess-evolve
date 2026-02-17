#!/usr/bin/env bash
set -euo pipefail

if ! command -v gdlint >/dev/null 2>&1; then
  if [[ "${CI:-}" == "true" ]]; then
    echo "gdlint is required in CI but was not found on PATH" >&2
    exit 1
  fi

  echo "⚠️  gdlint not found; skipping GDScript lint locally"
  exit 0
fi

# shellcheck disable=SC2046
gdlint $(git ls-files '*.gd')
