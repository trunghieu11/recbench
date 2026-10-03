#!/usr/bin/env bash
# Prepare a freshly rented GPU box (Ubuntu + NVIDIA driver, e.g. vast.ai) for quick-tier or full-tier runs.
#
#   git clone https://github.com/trunghieu11/recbench.git && cd recbench && ./scripts/setup_box.sh
#
# Then upload the splits prepared on your laptop (faster than downloading and re-splitting the raw data):
#   rsync -a --info=progress2 data/splits/ box:recbench/data/splits/       # from the laptop
# and start the queue inside tmux, so it survives a dropped SSH connection:
#   tmux new -s quick 'source .venv/bin/activate && python -m recbench.queue run --config configs/benchmarks/quick.yaml'
#   python -m recbench.queue status --config configs/benchmarks/quick.yaml          # any time, from another shell
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"

[[ "$(uname -s)" == "Linux" ]] || { echo "setup_box.sh is for the rented Linux box; on the laptop use install.md." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "warning: no NVIDIA GPU visible; GPU jobs will run on the CPU" >&2

# SuiteSparse for SANSA (scalable EASE); the box usually runs as root, otherwise sudo is used.
if ! ldconfig -p 2>/dev/null | grep -q libcholmod; then
  SUDO=""; [[ "$(id -u)" -eq 0 ]] || SUDO="sudo"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq libsuitesparse-dev tmux rsync >/dev/null
fi

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export SUITESPARSE_INCLUDE_DIR="${SUITESPARSE_INCLUDE_DIR:-/usr/include/suitesparse}"
setup_env "gpu,sansa"   # creates .venv, installs recbench with CUDA PyTorch, caps BLAS threads, fetches third-party code

python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF
mkdir -p runs/logs
pip_log="runs/logs/pip-freeze-$(date -u +%Y%m%dT%H%M%SZ).txt"
uv pip freeze > "$pip_log"
echo "Environment recorded in $pip_log"
echo "Threads per process: OPENBLAS=$OPENBLAS_NUM_THREADS OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS (cores: $(nproc))"
echo "Next: upload data/splits from the laptop, then start the queue in tmux (see the top of this script)."
