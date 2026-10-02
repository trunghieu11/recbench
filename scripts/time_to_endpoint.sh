#!/usr/bin/env bash
# Measure raw data -> working API on a synthetic log (no downloads). Prints JSON timings.
#   ./scripts/time_to_endpoint.sh            # EASE
#   METHOD=sasrec ./scripts/time_to_endpoint.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -d "$ROOT/.venv" ]] && source "$ROOT/.venv/bin/activate"
PYTHONWARNINGS=ignore python -m recbench.ttm --data-dir "$ROOT/runs/ttm" --method "${METHOD:-ease}"
