#!/usr/bin/env bash
# Bring the lab's results from a rented box to the laptop: its runs are imported into the lab's MLflow store
# (runs/lab/mlflow), its job summaries copied (runs/lab/tuning), and the scoreboard and its docs pages rebuilt.
# Safe to run while the box is still working (runs in progress are imported by a later fetch) and to run again.
#
#   ./scripts/fetch_lab_results.sh vast-gpu
#   ./scripts/fetch_lab_results.sh vast-gpu /workspace/recbench
set -euo pipefail
HOST="${1:?usage: fetch_lab_results.sh <ssh-host> [remote repository path]}"
REMOTE="${2:-recbench}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"
mkdir -p runs/lab-box runs/lab/tuning reports/lab-box

store="$(fetch_mlflow_store "$HOST" "$REMOTE/runs/lab/mlflow" runs/lab-box/mlflow)"  # one folder per box; imported below
rsync -a --exclude journal.log "$HOST:$REMOTE/runs/lab/tuning/" runs/lab/tuning/ # job summaries replace older ones
rsync -a "$HOST:$REMOTE/runs/lab/logs/" runs/lab-box/logs/ 2>/dev/null || true
rsync -a "$HOST:$REMOTE/reports/lab/" reports/lab-box/ 2>/dev/null || true

# shellcheck disable=SC1091
source .venv/bin/activate
if [[ -n "$store" ]]; then python -m recbench.import_runs "$store" --workspace lab; fi
python -m recbench.lab scoreboard --docs | tail -n 3
echo "Lab results imported: python -m recbench.lab status, and docs/generated/lab/ for the scoreboard pages."
