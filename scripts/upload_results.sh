#!/usr/bin/env bash
# Copy the bake-off's job summaries (runs/tuning/quick and runs/tuning/full) from the laptop to a new rented box.
# A box that did not run the bake-off itself needs them before any follow-up work: the queue reads them to know
# which jobs are done (without them it would tune every method again), the re-rankers read their generators'
# settings from them, and scripts/export_bundles_box.sh reads the confirmed settings.
#
#   ./scripts/upload_results.sh vast-gpu                       # the repository is at ~/recbench on the box
#   ./scripts/upload_results.sh vast-gpu /workspace/recbench
#
# Re-running only sends what changed. Archived summaries (runs/tuning/*/*/archive/) stay on the laptop.
set -euo pipefail
HOST="${1:?usage: upload_results.sh <ssh-host> [remote repository path]}"
REMOTE="${2:-recbench}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -d runs/tuning/quick ]] || { echo "runs/tuning/quick is missing: fetch the bake-off's results first (./scripts/fetch_results.sh)." >&2; exit 1; }

ssh "$HOST" "mkdir -p '$REMOTE/runs/tuning'"
for tier in quick full; do
  if [[ -d "runs/tuning/$tier" ]]; then
    rsync -a --exclude archive "runs/tuning/$tier" "$HOST:$REMOTE/runs/tuning/"
    echo "uploaded runs/tuning/$tier ($(find "runs/tuning/$tier" -name '*.json' -not -path '*/archive/*' | wc -l | tr -d ' ') job summaries)"
  fi
done
echo "The job summaries are on $HOST:$REMOTE/runs/tuning."
