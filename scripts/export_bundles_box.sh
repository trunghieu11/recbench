#!/usr/bin/env bash
# On the rented box: write the serving bundle of every method confirmed on full data that does not have one yet
# (runs/tuning/full/<dataset>/<method>.confirm.json -> data/bundles/<dataset>/full/<method>/), with the settings its
# confirmation chose. Confirmations write their bundle themselves; this fills the gaps, for example the first
# bake-off (2026-10-04), which ran before they did.
#
#   ./scripts/export_bundles_box.sh                     # every confirmed method without a bundle
#   ./scripts/export_bundles_box.sh hm lastfm           # only these datasets
#   FORCE=1 ./scripts/export_bundles_box.sh hm          # write them again, even where a bundle exists
#
# Needs ./scripts/setup_box.sh, the full splits (./scripts/upload_splits.sh) and the job summaries
# (./scripts/upload_results.sh). Run it inside tmux; running it again skips the bundles already written. Then fetch
# them on the laptop with ./scripts/fetch_results.sh <host>.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"
[[ -d .venv ]] || { echo "No .venv: run ./scripts/setup_box.sh first." >&2; exit 1; }
[[ -d runs/tuning/full ]] || { echo "No runs/tuning/full: upload the job summaries first (./scripts/upload_results.sh on the laptop)." >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export RECBENCH_ROOT="$ROOT"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cap_threads

datasets=("$@")
if (( ! $# )); then
  for dir in runs/tuning/full/*/; do datasets+=("$(basename "$dir")"); done
fi
mkdir -p runs/logs
log="runs/logs/export-bundles-$(date -u +%Y%m%dT%H%M%SZ).log"
failed=()
for dataset in "${datasets[@]}"; do
  for summary in runs/tuning/full/"$dataset"/*.confirm.json; do
    [[ -e "$summary" ]] || continue
    method="$(basename "$summary" .confirm.json)"
    if ! python -c "import json, sys; sys.exit(json.load(open(sys.argv[1])).get('status') != 'finished')" "$summary"; then
      echo "[export] $dataset/$method: its confirmation did not finish; skipped" | tee -a "$log"
      continue
    fi
    if [[ -z "${FORCE:-}" && -f "$DATA_DIR/bundles/$dataset/full/$method/manifest.json" ]]; then
      echo "[export] $dataset/$method: bundle exists; skipped" | tee -a "$log"
      continue
    fi
    echo "[export] $(date +%H:%M:%S) $dataset/$method ..." | tee -a "$log"
    if python -m recbench.export --dataset "$dataset" --method "$method" --tier full --from-confirm 2>&1 | tee -a "$log"; then
      echo "[export] $(date +%H:%M:%S) $dataset/$method done" | tee -a "$log"
    else
      failed+=("$dataset/$method")
      echo "[export] $dataset/$method FAILED; the others go on" | tee -a "$log"
    fi
  done
done
if (( ${#failed[@]} )); then
  echo "Failed: ${failed[*]} (details in $log)." >&2
  exit 1
fi
echo "Bundles are in $DATA_DIR/bundles/<dataset>/full/. Fetch them on the laptop: ./scripts/fetch_results.sh <host>"
