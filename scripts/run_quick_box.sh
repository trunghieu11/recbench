#!/usr/bin/env bash
# Run the quick-tier bake-off on a rented box, after ./scripts/setup_box.sh and with the splits in data/splits.
#
#   ./scripts/run_quick_box.sh                          # every dataset in order, until all are done
#   ./scripts/run_quick_box.sh --stop-after-dataset     # finish the current dataset (and its confirmations), then stop
#   ./scripts/run_quick_box.sh --deadline-hours 4       # start no new job after 4 hours
#   ./scripts/run_quick_box.sh --datasets movielens-25m --methods ease,itemknn
#   ./scripts/run_quick_box.sh --price-per-hour 0.55   # show the session's cost on the status page
#
# Run it inside tmux, so it survives a dropped SSH connection (vast.ai sessions already start in tmux; elsewhere run
# `tmux new -s quick` first). Watch it from another tmux window (Ctrl-b c):
#   source .venv/bin/activate && python -m recbench.queue status --config configs/benchmarks/quick.yaml
# or through the status page: ./scripts/monitor_box.sh here, ./scripts/open_tunnel.sh <host> on the laptop.
# Running it again resumes: finished jobs are kept, interrupted ones continue from their finished trials.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"
[[ -d .venv ]] || { echo "No .venv: run ./scripts/setup_box.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export RECBENCH_ROOT="$ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$ROOT/runs/mlflow}"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cap_threads

CONFIG=configs/benchmarks/quick.yaml
check_disk 20   # GB free for runs, reports and bundles
missing=()
for dataset in $(quick_datasets "$CONFIG"); do
  for tier in quick quick-val full full-val; do
    [[ -f "$DATA_DIR/splits/$dataset/$tier/meta.json" ]] || missing+=("$dataset/$tier")
  done
done
if (( ${#missing[@]} )); then
  echo "warning: splits not prepared yet: ${missing[*]}" >&2
  echo "         their jobs are recorded as missing_split and run again the next time you start this script." >&2
fi

mkdir -p runs/logs
log="runs/logs/quick-queue-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Logging to $log. Status page: reports/queue/quick.html (./scripts/monitor_box.sh serves it)."
python -m recbench.queue run --config "$CONFIG" "$@" 2>&1 | tee -a "$log"
