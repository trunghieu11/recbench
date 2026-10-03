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
  cap_threads
  bash "$ROOT/scripts/fetch_third_party.sh"
}

# OpenBLAS aborts (iALS exit -11) when it starts more threads than it was built for; 192-core rented
# boxes exceed the usual limit of 128. Cap BLAS/OpenMP threads unless the caller already set them.
cap_threads() {
  local cores
  cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  local cap=$(( cores < 32 ? cores : 32 ))
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$cap}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cap}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$cap}"
}

# Kaggle CLI 2.x reads an access token from ~/.kaggle/access_token. A new-style key (starting with KGAT)
# stored in kaggle.json is copied there, so `kaggle competitions download` works without a manual step.
ensure_kaggle_token() {
  local json="${HOME}/.kaggle/kaggle.json" token="${HOME}/.kaggle/access_token" key
  [[ -f "$json" && ! -f "$token" ]] || return 0
  key="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("key", ""))' "$json" 2>/dev/null || true)"
  if [[ "$key" == KGAT* ]]; then
    (umask 077 && printf '%s' "$key" > "$token")
    echo "Wrote the Kaggle access token for the 2.x CLI to $token"
  fi
}

check_kaggle() {
  if [[ "$1" == *hm* || "$1" == *retailrocket* ]]; then
    if [[ ! -f "${HOME}/.kaggle/kaggle.json" && -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
      echo "Kaggle credentials missing (needed for H&M and RetailRocket). Put kaggle.json in ~/.kaggle/ or export KAGGLE_USERNAME and KAGGLE_KEY." >&2
      exit 1
    fi
    ensure_kaggle_token
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
