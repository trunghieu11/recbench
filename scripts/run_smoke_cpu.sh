#!/usr/bin/env bash
# Smoke tier on your laptop (CPU only): download -> clean -> split -> train + evaluate every method
# -> report -> regenerate the docs pages. Small user samples (~50K events per dataset).
#
#   ./scripts/run_smoke_cpu.sh
#   ./scripts/run_smoke_cpu.sh --datasets movielens-25m --methods most_popular,ease,sasrec
#   ./scripts/run_smoke_cpu.sh --serve          # then open http://127.0.0.1:8080/dashboard
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"

CONFIG="${CONFIG:-$ROOT/configs/benchmarks/smoke-cpu.yaml}"
DATASETS="" METHODS="" SERVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --serve) SERVE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ROOT/runs/logs"
exec > >(tee -a "$ROOT/runs/logs/smoke-$STAMP.log") 2>&1

setup_env "cpu,docs"
check_disk 30
check_kaggle "${DATASETS:-hm,retailrocket}"
export CUDA_VISIBLE_DEVICES=""

python -m recbench.pipeline.prepare --config "$CONFIG" ${DATASETS:+--datasets "$DATASETS"}
python -m recbench.runner --config "$CONFIG" ${DATASETS:+--datasets "$DATASETS"} ${METHODS:+--methods "$METHODS"}
python -m recbench.report.build --tier smoke --out "$ROOT/reports/smoke-$STAMP" --docs
python -m recbench.dictionary.build

echo "Report:  $ROOT/reports/smoke-$STAMP/report.md"
echo "MLflow:  mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI --port 5001"
if [[ "$SERVE" -eq 1 ]]; then
  RECBENCH_TIER=smoke exec python -m uvicorn recbench.serving.app:app --host 127.0.0.1 --port 8080
fi
