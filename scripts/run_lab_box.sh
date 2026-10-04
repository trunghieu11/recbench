#!/usr/bin/env bash
# Run the improvement lab on a rented box: its baseline (every lab method on every dataset), then your experiments.
# The lab trains on CPUs only, with the same machine profile as on the laptop (configs/hardware/lab-cpu.yaml), so its
# results are comparable wherever they ran.
#
#   ./scripts/run_lab_box.sh                                   # the baseline: 12 methods x 5 datasets
#   ./scripts/run_lab_box.sh itemknn:no-time-knobs ease:edlae  # the baseline (skipped if done), then these experiments
#   LAB_WORKERS=4 ./scripts/run_lab_box.sh                     # jobs at a time (default: a quarter of the cores, 1 to 8)
#
# Needs ./scripts/setup_box.sh and the quick splits (./scripts/upload_splits.sh on the laptop). Experiments that use
# your own code need your branch on the box first: git fetch && git switch lab/03-ease. Run this inside tmux; running
# it again resumes. On the laptop, ./scripts/fetch_lab_results.sh <host> brings the results home.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"
[[ -d .venv ]] || { echo "No .venv: run ./scripts/setup_box.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export RECBENCH_ROOT="$ROOT"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cap_threads

CONFIG=configs/benchmarks/lab.yaml
check_disk 10   # GB free for the lab's runs and reports
missing=()
for dataset in $(quick_datasets "$CONFIG"); do
  for tier in quick quick-val; do
    [[ -f "$DATA_DIR/splits/$dataset/$tier/meta.json" ]] || missing+=("$dataset/$tier")
  done
done
if (( ${#missing[@]} )); then
  echo "Splits missing: ${missing[*]}. Upload them from the laptop: ./scripts/upload_splits.sh <host>" >&2
  exit 1
fi

cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
workers="${LAB_WORKERS:-$(( cores / 4 < 1 ? 1 : (cores / 4 > 8 ? 8 : cores / 4) ))}"
mkdir -p runs/lab/logs
log="runs/lab/logs/lab-box-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Logging to $log; $workers jobs at a time. Watch it from another tmux window: python -m recbench.lab status"
python -m recbench.lab baseline --cpu-workers "$workers" 2>&1 | tee -a "$log"
for spec in "$@"; do
  method="${spec%%:*}"
  experiment="${spec#*:}"
  [[ "$method" != "$spec" && -n "$experiment" ]] || { echo "Write experiments as method:experiment, got $spec" >&2; exit 1; }
  python -m recbench.lab run --method "$method" --experiment "$experiment" --workers 5 2>&1 | tee -a "$log"
done
python -m recbench.lab scoreboard 2>&1 | tail -n 3
echo "Done. On the laptop: ./scripts/fetch_lab_results.sh <host>"
