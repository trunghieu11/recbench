"""The default behaviour of the lab's deterministic methods must not change by accident (docs/handbook/test.md).

A variant adds a setting whose default keeps today's behaviour, so the baseline and every earlier result stay valid.
This test pins that behaviour: at default settings on the toy data, each method's 10 best scores (and their items) for
the first 30 users must equal those stored in tests/golden/lab_defaults.json.

If you change a default on purpose (a promotion), bump the method's impl_version in its MethodSpec, then regenerate:

    RECBENCH_UPDATE_GOLDEN=1 pytest tests/test_lab_defaults.py

The methods run in one child process that loads no PyTorch, because on macOS SANSA (SuiteSparse) cannot share a
process with it (see recbench.methods).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recbench.lab.checks import compare_golden

GOLDEN = Path(__file__).parent / "golden" / "lab_defaults.json"
METHODS = ("itemknn", "rp3beta", "ease", "slim", "sansa", "puresvd", "gfcf", "vsknn")  # the deterministic lab methods
SCRIPT = """
import json, sys
from recbench.data import TrainView
from recbench.lab.checks import golden_record
view = TrainView(sys.argv[1])
out = {}
for method in sys.argv[2].split(","):
    try:
        out[method] = golden_record(method, view)
    except Exception as exc:
        out[method] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def records(toy_split) -> dict:
    methods = [m for m in METHODS if m != "sansa" or importlib.util.find_spec("sansa")]
    env = {**os.environ, "RECBENCH_METHOD_MODULES": "baselines,neighbourhood,linear,graph_filters"}
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(toy_split), ",".join(methods)], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    found = json.loads(out.stdout.strip().splitlines()[-1])
    if os.environ.get("RECBENCH_UPDATE_GOLDEN"):
        stored = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({**stored, **{m: r for m, r in found.items() if "error" not in r}}, indent=1, sort_keys=True) + "\n")
    return found


def test_every_deterministic_lab_method_is_pinned(records):  # `records` first: with RECBENCH_UPDATE_GOLDEN it writes the file
    from recbench.registry import ensure_loaded

    registry = ensure_loaded().methods
    assert all(registry[m].spec.deterministic for m in METHODS)
    assert set(json.loads(GOLDEN.read_text())) >= set(METHODS) - {"sansa"}


@pytest.mark.parametrize("method", METHODS)
def test_default_behaviour_is_unchanged(records, method):
    if method not in records:
        pytest.skip(f"{method}'s package is not installed")
    record = records[method]
    assert "error" not in record, record.get("error")
    golden = json.loads(GOLDEN.read_text()).get(method)
    assert golden, f"{method} is not in {GOLDEN.name}: run RECBENCH_UPDATE_GOLDEN=1 pytest {Path(__file__).name}"
    if golden["impl_version"] != record["impl_version"]:
        pytest.fail(f"{method}'s impl_version changed ({golden['impl_version']} -> {record['impl_version']}). If its default behaviour "
                    f"changed on purpose, regenerate: RECBENCH_UPDATE_GOLDEN=1 pytest tests/{Path(__file__).name}")
    problems = compare_golden(record, golden)
    assert not problems, (f"{method}'s default behaviour changed:\n  " + "\n  ".join(problems[:5]) +
                          "\nMake your new setting's default keep the old behaviour. If the change is meant (a promotion), bump the "
                          "method's impl_version and regenerate the golden file (see the top of this test).")
