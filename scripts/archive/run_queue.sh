#!/usr/bin/env bash
# Run individual method:dataset jobs in a fixed order: cheap and informative first, long jobs last.
# Finished jobs are skipped by the runner (resume). Each job is capped by timeout_minutes in the config.
# No new job starts after the deadline in runs/BREADTH_DEADLINE (epoch seconds).
#
#   ./scripts/run_queue.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export RECBENCH_ROOT="$ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$ROOT/runs/mlflow}"
export PYTHONUNBUFFERED=1
# This host has 192 cores. OpenBLAS is built for 128 threads and aborts iALS (exit -11) above that.
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

CONFIG="${CONFIG:-$ROOT/configs/benchmarks/archive/gpu-12h.yaml}"
DEADLINE_FILE="$ROOT/runs/BREADTH_DEADLINE"
LOG="$ROOT/runs/logs/breadth-12h.log"
mkdir -p "$ROOT/runs/logs" "$ROOT/reports"
DEADLINE="$(cat "$DEADLINE_FILE")"

VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "$VRAM_MB" -ge 46000 ]]; then PRESET=48gb; elif [[ "$VRAM_MB" -ge 23000 ]]; then PRESET=24gb; else
  echo "GPU has ${VRAM_MB} MB; need at least 24 GB." >&2
  exit 1
fi

# DIN on movielens-25m is left out: it runs out of GPU memory during scoring at the 48gb preset.
# HSTU on movielens-25m and hm is left out: it needs far more than the 2-hour cap.
# BERT4Rec and S3-Rec (RecBole) score the whole catalog in the loss, so their cost grows with the
# item count: steam 15k, movielens 59k, hm 105k, lastfm 177k, retailrocket 235k.
JOBS=(
  sasrec:lastfm sasrec:steam sasrec:retailrocket sasrec:hm
  bert4rec:steam s3rec:steam
  din:lastfm din:steam din:retailrocket din:hm
  lightgcn:lastfm lightgcn:steam
  xsimgcl:lastfm xsimgcl:steam xsimgcl:retailrocket
  bert4rec:movielens-25m s3rec:movielens-25m
  hstu:lastfm hstu:steam hstu:retailrocket
  bert4rec:hm s3rec:hm
  s3rec:lastfm bert4rec:retailrocket s3rec:retailrocket
  lightgcn:hm
  xsimgcl:movielens-25m xsimgcl:hm
)

echo "queue start preset=$PRESET deadline=$(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ) $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
for job in "${JOBS[@]}"; do
  method="${job%%:*}"
  dataset="${job#*:}"
  if (( $(date -u +%s) >= DEADLINE )); then
    echo "deadline reached before $job" | tee -a "$LOG"
    break
  fi
  echo "START $method $dataset $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  python -m recbench.runner --config "$CONFIG" --preset "$PRESET" --methods "$method" --datasets "$dataset" | tee -a "$LOG"
  echo "END $method $dataset $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
done
python -m recbench.report.build --tier full --out "$ROOT/reports/breadth-12h" --docs || true
echo "queue finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
touch "$ROOT/runs/BREADTH_DONE"
