#!/usr/bin/env bash
# Run Ruff lint + format in one go.
# Usage:
#   bash tools/run_lint.sh
set -euo pipefail

ensure_tool () {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "⚠️  '$1' not found in PATH. Installing into current venv..."
    pip install "$1"
  fi
}

ensure_tool ruff

if [[ "${1:-}" == "--check" ]]; then
  echo "🔎 Running: ruff check ."
  ruff check .
  echo "✅ Lint check passed."
  exit 0
fi

echo "🧹 Running: ruff check . --fix"
ruff check . --fix

echo "🖊️  Running: ruff format ."
ruff format .

echo "✅ Codebase is clean and formatted."
