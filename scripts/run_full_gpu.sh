#!/usr/bin/env bash
# Full-tier benchmark on a GPU machine: download, train, evaluate, rebuild the dictionary.
# Usage:
#   ./scripts/run_full_gpu.sh
#   ./scripts/run_full_gpu.sh --datasets hm,steam --methods hstu,tiger
#   ./scripts/run_full_gpu.sh --serve
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-$ROOT/data}"
MLFLOW_DIR="${MLFLOW_DIR:-$ROOT/runs/mlflow}"
LOG_DIR="$ROOT/runs/gpu-full"
CONFIG="${CONFIG:-$ROOT/configs/benchmarks/gpu-full.yaml}"
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

echo "Log: $LOG"
echo "Data: $DATA_DIR"
echo "MLflow: $MLFLOW_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "run_full_gpu.sh is for the Ubuntu GPU machine. On this laptop use scripts/run_smoke_cpu.sh." >&2
  exit 1
fi
if [[ -f /etc/os-release ]] && ! grep -q '^ID=ubuntu$' /etc/os-release; then
  echo "This machine is Linux but not Ubuntu. 22.04 or 24.04 is the tested OS. Continuing anyway." >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No GPU visible. This script is for the stronger machine." >&2
  exit 1
fi
nvidia-smi -L

VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "$VRAM_MB" -lt 23000 ]]; then
  echo "GPU has ${VRAM_MB} MB. Full tier needs at least 24 GB." >&2
  exit 1
fi
if [[ "$VRAM_MB" -ge 46000 ]]; then
  PRESET=48gb
else
  PRESET=24gb
fi
echo "Using gpu_full preset: $PRESET"

FREE_GB="$(df -Pk "$DATA_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [[ "$FREE_GB" -lt 200 ]]; then
  echo "Need at least 200 GB free under $DATA_DIR (have ${FREE_GB} GB)." >&2
  exit 1
fi

if [[ "$DATASETS" == *hm* || "$DATASETS" == *retailrocket* ]]; then
  if [[ ! -f "${HOME}/.kaggle/kaggle.json" && -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
    echo "Kaggle credentials missing. H&M and Retailrocket will not download." >&2
    echo "Put kaggle.json in ~/.kaggle/, or export KAGGLE_API_TOKEN, or export KAGGLE_USERNAME and KAGGLE_KEY." >&2
    exit 1
  fi
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip
python -m pip install -e ".[gpu]"

export DATA_DIR MLFLOW_TRACKING_URI="file:${MLFLOW_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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

python -m recbench.runner --config "$CONFIG" --datasets "$READY_CSV" --methods "$METHODS" --preset "$PRESET"
python -m recbench.dictionary.build --tracking-uri "$MLFLOW_TRACKING_URI"
REPORT_DIR="$ROOT/reports/full-$STAMP"
python -m recbench.report.build \
  --tracking-uri "$MLFLOW_TRACKING_URI" \
  --tier full \
  --preset "$PRESET" \
  --out "$REPORT_DIR"

echo "Finished. Full-tier ranks are under $ROOT/site. Log: $LOG"
echo "Report: $REPORT_DIR/report.md"
echo "Report: $REPORT_DIR/report.html"
echo "MLflow UI: mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI --port 5000"

if [[ "$SERVE" -eq 1 ]]; then
  exec python -m uvicorn recbench.serving.app:app --host 0.0.0.0 --port 8000
fi
