#!/usr/bin/env bash
# Full tier on your Ubuntu NVIDIA machine: complete datasets, at most 10K warm eval users each.
#
#   ./scripts/run_full_gpu.sh
#   ./scripts/run_full_gpu.sh --datasets hm --methods sasrec,hstu
#
# The model-size preset is picked from the GPU memory: 24gb (>= 23 GB) or 48gb (>= 46 GB).
# Copy results back to your laptop with: rsync -a gpu-box:recommendation_benchmark/runs/mlflow/ runs/mlflow/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"

CONFIG="${CONFIG:-$ROOT/configs/benchmarks/gpu-full.yaml}"
DATASETS="" METHODS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || { echo "run_full_gpu.sh is for the Linux GPU machine; on the laptop use run_smoke_cpu.sh." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || { echo "No NVIDIA GPU visible (nvidia-smi failed)." >&2; exit 1; }
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "$VRAM_MB" -ge 46000 ]]; then PRESET=48gb; elif [[ "$VRAM_MB" -ge 23000 ]]; then PRESET=24gb; else
  echo "GPU has ${VRAM_MB} MB; the full tier needs at least 24 GB." >&2; exit 1; fi
echo "GPU: $(nvidia-smi -L | head -n 1)  ->  preset $PRESET"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ROOT/runs/logs"
exec > >(tee -a "$ROOT/runs/logs/full-$STAMP.log") 2>&1

setup_env "gpu,docs"
check_disk 80
check_kaggle "${DATASETS:-hm,retailrocket}"
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU (install a CUDA build of torch)'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

python -m recbench.pipeline.prepare --config "$CONFIG" ${DATASETS:+--datasets "$DATASETS"}
python -m recbench.runner --config "$CONFIG" --preset "$PRESET" ${DATASETS:+--datasets "$DATASETS"} ${METHODS:+--methods "$METHODS"}
python -m recbench.report.build --tier full --out "$ROOT/reports/full-$STAMP" --docs
python -m recbench.dictionary.build
echo "Report: $ROOT/reports/full-$STAMP/report.md"
