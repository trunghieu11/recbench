#!/usr/bin/env bash
# Copy the splits prepared on the laptop to the rented box, in the order the queue needs them: every dataset's
# small quick tiers first (about 0.5 GB, so the queue can start at once), then the full tiers dataset by dataset
# (about 8 GB, needed only when a dataset's top methods are confirmed).
#
#   ./scripts/upload_splits.sh vast-gpu                       # the repository is at ~/recbench on the box
#   ./scripts/upload_splits.sh vast-gpu /workspace/recbench
#   QUICK_ONLY=1 ./scripts/upload_splits.sh vast-gpu          # only the quick tiers: enough for the improvement lab
#
# "vast-gpu" is an SSH host name from ~/.ssh/config (or user@address). Re-running only sends what changed.
set -euo pipefail
HOST="${1:?usage: upload_splits.sh <ssh-host> [remote repository path]}"
REMOTE="${2:-recbench}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"
read -r -a DATASETS <<< "$(quick_datasets)"

TIERS="quick quick-val full full-val"
PHASES=("quick quick-val" "full full-val")
if [[ -n "${QUICK_ONLY:-}" ]]; then TIERS="quick quick-val"; PHASES=("quick quick-val"); fi
for d in "${DATASETS[@]}"; do
  for t in $TIERS; do
    [[ -f "data/splits/$d/$t/meta.json" ]] || { echo "data/splits/$d/$t is missing: prepare it first (docs/start/quick-tier-box.md)." >&2; exit 1; }
  done
done
ssh "$HOST" "mkdir -p $(printf "'$REMOTE/data/splits/%s' " "${DATASETS[@]}")"
for phase in "${PHASES[@]}"; do
  for d in "${DATASETS[@]}"; do
    for t in $phase; do
      rsync -a --exclude cache "data/splits/$d/$t" "$HOST:$REMOTE/data/splits/$d/"
      echo "$(date +%H:%M:%S) uploaded $d/$t ($(du -sh "data/splits/$d/$t" | cut -f1))"
    done
  done
done
echo "All splits are on $HOST:$REMOTE/data/splits."
