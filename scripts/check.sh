#!/usr/bin/env bash
# The checks to run before you open a pull request (docs/handbook/test.md).
#   ./scripts/check.sh           # everything: notebooks, fast tests (including the docs build); a few minutes
#   ./scripts/check.sh --quick   # notebooks and the lab's own tests only; about a minute
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
step() { printf '\n== %s\n' "$1"; }

step "notebooks are saved without outputs"
python - <<'PY'
import json, pathlib, sys
bad = [str(nb) for nb in sorted(pathlib.Path("labs").glob("*/*.ipynb"))
       if any(c.get("outputs") or c.get("execution_count") for c in json.loads(nb.read_text())["cells"] if c.get("cell_type") == "code")]
if bad:
    sys.exit("these notebooks still have outputs (in VS Code: the ... menu > Clear All Outputs, then save):\n  " + "\n  ".join(bad))
print("ok")
PY

if [[ "${1:-}" == "--quick" ]]; then
  shopt -s nullglob
  variants=(tests/test_*_variant.py)
  step "the lab's tests and your variant tests"
  pytest -q -p no:warnings tests/test_lab.py tests/test_compare.py tests/test_lab_defaults.py tests/test_labs.py "${variants[@]}"
else
  step "fast tests, including the strict docs build (a few minutes)"
  pytest -q -p no:warnings -m "not slow"
fi
printf '\nAll checks passed: ready for a pull request.\n'
