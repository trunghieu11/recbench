#!/usr/bin/env bash
# Prepare a freshly rented GPU box (Ubuntu + NVIDIA driver, e.g. vast.ai) for quick-tier or full-tier runs.
#
#   git clone https://github.com/trunghieu11/recbench.git && cd recbench && ./scripts/setup_box.sh
#
# Then copy the splits prepared on the laptop (or prepare them here), and start the queue inside tmux.
# The full walkthrough: docs/start/quick-tier-box.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_common.sh"

[[ "$(uname -s)" == "Linux" ]] || { echo "setup_box.sh is for the rented Linux box; on the laptop use install.md." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "warning: no NVIDIA GPU visible; GPU jobs will run on the CPU" >&2

# SuiteSparse and a C compiler for SANSA (scikit-sparse builds from source), plus tmux and rsync.
# The box usually runs as root, otherwise sudo is used.
if ! ldconfig -p 2>/dev/null | grep -q libcholmod || ! command -v gcc >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
  SUDO=""; [[ "$(id -u)" -eq 0 ]] || SUDO="sudo"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq libsuitesparse-dev build-essential tmux rsync >/dev/null
fi

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export SUITESPARSE_INCLUDE_DIR="${SUITESPARSE_INCLUDE_DIR:-/usr/include/suitesparse}"
# PyTorch's default Linux build needs a recent NVIDIA driver (CUDA 13 in 2.14). uv picks the PyTorch build that
# matches this machine's driver instead.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
setup_env "gpu"   # creates .venv, installs recbench with CUDA PyTorch, caps BLAS threads, fetches third-party code
# SANSA is optional: if its build fails, the run goes on and SANSA is recorded as unsupported.
uv pip install -q -e ".[gpu,sansa]" || echo "warning: SANSA could not be installed; its jobs will be skipped" >&2

python - <<'EOF'
import shutil, subprocess, sys
import torch
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda, "| CUDA available:", torch.cuda.is_available(),
      "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if shutil.which("nvidia-smi") and not torch.cuda.is_available():
    driver = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    line = next((l for l in driver.splitlines() if "CUDA Version" in l), "").strip()
    sys.exit("PyTorch cannot use the GPU. The driver reports: " + line + "\nRent a box whose driver supports a newer CUDA "
             "(vast.ai search filter 'CUDA' >= 12.8), or reinstall PyTorch for this driver, e.g.\n"
             "  uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu126")
EOF
# The text-embedding model (about 90 MB), downloaded once here instead of by several jobs at the same time.
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
  && echo "Sentence-transformer model cached." || echo "warning: could not download the text model; text_knn will fail" >&2
python -c "import lightgbm, optuna; print('lightgbm', lightgbm.__version__, '| optuna', optuna.__version__)"
python -c "import sansa, sksparse; print('sansa ready')" 2>/dev/null || echo "SANSA not available." >&2
mkdir -p runs/logs
pip_log="runs/logs/pip-freeze-$(date -u +%Y%m%dT%H%M%SZ).txt"
uv pip freeze > "$pip_log"
echo "Environment recorded in $pip_log"
echo "Threads per process: OPENBLAS=$OPENBLAS_NUM_THREADS OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS (cores: $(nproc))"
echo "Next: put the splits in data/splits, then: tmux new -s quick ./scripts/run_quick_box.sh (docs/start/quick-tier-box.md)."
