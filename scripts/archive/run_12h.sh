#!/usr/bin/env bash
# Cover as many dataset/method pairs as possible, then stop.
# Fast methods run across all datasets before slower ones start.
# HSTU is omitted. Each pair is capped by timeout_minutes in the config.
# A wall-clock deadline stops the queue between methods (default 12 hours).
#
#   ./scripts/run_12h.sh
#   HOURS=12 ./scripts/run_12h.sh
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
HOURS="${HOURS:-12}"
DEADLINE_FILE="$ROOT/runs/BREADTH_DEADLINE"
mkdir -p "$ROOT/runs/logs" "$ROOT/reports"
if [[ ! -f "$DEADLINE_FILE" ]]; then
  date -u -d "+${HOURS} hours" +%s > "$DEADLINE_FILE"
fi
DEADLINE="$(cat "$DEADLINE_FILE")"
LOG="$ROOT/runs/logs/breadth-12h.log"

VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "$VRAM_MB" -ge 46000 ]]; then PRESET=48gb; elif [[ "$VRAM_MB" -ge 23000 ]]; then PRESET=24gb; else
  echo "GPU has ${VRAM_MB} MB; need at least 24 GB." >&2
  exit 1
fi

# Fast and already-proven methods first, so every dataset gets them before the slow models.
METHODS=(
  random most_popular itemknn ease ials bpr_mf
  text_hash_tower multimodal_tower tiger_lite dcnv2
  lightgcn xsimgcl sasrec din bert4rec s3rec
)

echo "breadth start preset=$PRESET deadline=$(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ) $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
for method in "${METHODS[@]}"; do
  now="$(date -u +%s)"
  if (( now >= DEADLINE )); then
    echo "deadline reached before $method" | tee -a "$LOG"
    break
  fi
  echo "START $method $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  python -m recbench.runner --config "$CONFIG" --preset "$PRESET" --methods "$method" | tee -a "$LOG"
  echo "END $method $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
done
python -m recbench.report.build --tier full --out "$ROOT/reports/breadth-12h" --docs || true
echo "breadth finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
touch "$ROOT/runs/BREADTH_DONE"
