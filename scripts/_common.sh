# Shared setup for the run scripts (sourced, not executed).
setup_env() {
  local extras="$1"
  if [[ ! -d "$ROOT/.venv" ]]; then
    if command -v uv >/dev/null 2>&1; then uv venv "$ROOT/.venv" --python 3.12; else python3 -m venv "$ROOT/.venv"; fi
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  if command -v uv >/dev/null 2>&1; then uv pip install -q -e ".[${extras}]"; else python -m pip install -q -e ".[${extras}]"; fi
  export DATA_DIR="${DATA_DIR:-$ROOT/data}"
  export RECBENCH_ROOT="$ROOT"
  export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$ROOT/runs/mlflow}"
  export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
  bash "$ROOT/scripts/fetch_third_party.sh"
}

check_kaggle() {
  if [[ "$1" == *hm* || "$1" == *retailrocket* ]]; then
    if [[ ! -f "${HOME}/.kaggle/kaggle.json" && -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
      echo "Kaggle credentials missing (needed for H&M and RetailRocket). Put kaggle.json in ~/.kaggle/ or export KAGGLE_USERNAME and KAGGLE_KEY." >&2
      exit 1
    fi
  fi
}

check_disk() {
  local need_gb="$1"
  mkdir -p "$DATA_DIR"
  local free_gb
  free_gb="$(df -Pk "$DATA_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
  if [[ "$free_gb" -lt "$need_gb" ]]; then
    echo "Need at least ${need_gb} GB free under $DATA_DIR (have ${free_gb} GB)." >&2
    exit 1
  fi
}
