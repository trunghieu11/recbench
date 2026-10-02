#!/usr/bin/env bash
# Smoke-tier time-to-endpoint timer. Uses a local raw fixture unless DATA_DIR already has a split.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
python -m recbench.ttm --data-dir "${DATA_DIR:-$ROOT/runs/ttm}" --steps "${STEPS:-20}"
