#!/usr/bin/env bash
# Smoke-tier benchmark on the laptop. CPU only.
# Usage:
#   ./scripts/run_smoke_cpu.sh
#   ./scripts/run_smoke_cpu.sh --datasets movielens-25m,hm
#   ./scripts/run_smoke_cpu.sh --serve
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-$ROOT/data}"
MLFLOW_DIR="${MLFLOW_DIR:-$ROOT/runs/mlflow}"
LOG_DIR="$ROOT/runs/smoke-cpu"
CONFIG="${CONFIG:-$ROOT/configs/benchmarks/smoke-cpu.yaml}"
DATASETS="${DATASETS:-movielens-25m,retailrocket,hm,lastfm,steam}"
METHODS="${METHODS:-xsimgcl,bert4rec,s3rec,dcnv2,din,multimodal_tower,tiger,hstu,generative_llm}"
SERVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve) SERVE=1; shift ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$DATA_DIR" "$MLFLOW_DIR" "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

FREE_GB="$(df -Pk "$DATA_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [[ "$FREE_GB" -lt 40 ]]; then
  echo "Need at least 40 GB free under $DATA_DIR (have ${FREE_GB} GB)." >&2
  exit 1
fi

if [[ "$DATASETS" == *hm* || "$DATASETS" == *retailrocket* ]]; then
  if [[ ! -f "${HOME}/.kaggle/kaggle.json" && -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
    echo "Kaggle credentials missing. Put kaggle.json in ~/.kaggle/, or export KAGGLE_API_TOKEN, or export KAGGLE_USERNAME and KAGGLE_KEY." >&2
    exit 1
  fi
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip
python -m pip install -e ".[cpu]"

export DATA_DIR MLFLOW_TRACKING_URI="file:${MLFLOW_DIR}"
export CUDA_VISIBLE_DEVICES=""
bash "$ROOT/scripts/fetch_third_party.sh"

IFS=',' read -ra DS <<< "$DATASETS"
READY=()
for d in "${DS[@]}"; do
  if python -m recbench.pipeline.prepare --config "$CONFIG" --datasets "$d"; then
    READY+=("$d")
  else
    echo "Prepare failed for $d; continuing with the other datasets." >&2
  fi
done
if [[ ${#READY[@]} -eq 0 ]]; then
  echo "No dataset prepared." >&2
  exit 1
fi
READY_CSV="$(IFS=','; echo "${READY[*]}")"

python -m recbench.runner --config "$CONFIG" --datasets "$READY_CSV" --methods "$METHODS"
python -m recbench.dictionary.build --tracking-uri "$MLFLOW_TRACKING_URI"
REPORT_DIR="$ROOT/reports/smoke-$STAMP"
python -m recbench.report.build \
  --wiring \
  --tracking-uri "$MLFLOW_TRACKING_URI" \
  --datasets "$DATASETS" \
  --methods "$METHODS" \
  --out "$REPORT_DIR"

echo "Smoke wiring run finished. Log: $LOG"
echo "Report: $REPORT_DIR/report.md"
echo "Report: $REPORT_DIR/report.html"

if [[ "$SERVE" -eq 1 ]]; then
  exec python -m uvicorn recbench.serving.app:app --host 127.0.0.1 --port 8000
fi
