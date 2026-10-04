#!/usr/bin/env bash
# Bring the quick-tier results from the rented box to the laptop, import them, and rebuild the reports and docs.
# Safe to run while the box is still working (runs in progress are skipped until they end) and to run again.
# It also copies the serving bundles of the confirmed methods (data/bundles/<dataset>/full/), used to deploy the API.
#
#   ./scripts/fetch_results.sh vast-gpu
#   ./scripts/fetch_results.sh vast-gpu /workspace/recbench
set -euo pipefail
HOST="${1:?usage: fetch_results.sh <ssh-host> [remote repository path]}"
REMOTE="${2:-recbench}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs/mlflow-box runs/tuning runs/queue-box runs/logs-box reports/box data/bundles

rsync -a "$HOST:$REMOTE/runs/mlflow/" runs/mlflow-box/            # never copied over runs/mlflow: imported instead
for tier in quick full; do                                         # job summaries (trials, best settings, timings)
  if ssh "$HOST" "test -d $REMOTE/runs/tuning/$tier"; then rsync -a "$HOST:$REMOTE/runs/tuning/$tier" runs/tuning/; fi
done
rsync -a "$HOST:$REMOTE/runs/tuning/journal.log" runs/tuning/journal-box.log 2>/dev/null || true
rsync -a "$HOST:$REMOTE/runs/queue/" runs/queue-box/
rsync -a "$HOST:$REMOTE/runs/logs/" runs/logs-box/
rsync -a "$HOST:$REMOTE/reports/" reports/box/ 2>/dev/null || true
for dataset in $(ssh "$HOST" "ls $REMOTE/data/bundles 2>/dev/null" || true); do   # bundles of confirmed methods
  if ssh "$HOST" "test -d $REMOTE/data/bundles/$dataset/full"; then
    mkdir -p "data/bundles/$dataset"
    rsync -a "$HOST:$REMOTE/data/bundles/$dataset/full" "data/bundles/$dataset/"
  fi
done

# shellcheck disable=SC1091
source .venv/bin/activate
python -m recbench.import_runs runs/mlflow-box
python -m recbench.report.build --tier quick --tuning tuned --out reports/quick-tuned --docs
python -m recbench.report.build --tier full --tuning tuned --out reports/full-tuned --docs
python -m recbench.dictionary.build
python -m recbench.queue status --config configs/benchmarks/quick.yaml --state runs/queue-box/quick.json --html reports/queue/quick.html
echo "Reports: reports/quick-tuned/report.html and reports/full-tuned/report.html; docs fragments in docs/generated/."
