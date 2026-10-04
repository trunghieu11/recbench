"""The lab notebooks (labs/<nn>-<method>/<method>.ipynb): well-formed, saved without outputs, and runnable.

The slow test runs every notebook from top to bottom on the toy data, after a tiny toy baseline (2 settings), so a
broken cell is found before a learner does. Cells tagged `needs-experiments` compare experiments that only exist after
box runs; they are skipped there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from recbench.lab.experiments import lab_folders

ROOT = Path(__file__).resolve().parents[1]
LABS = [(method, folder) for method, folder in lab_folders(ROOT).items()]


def _notebook(folder: Path, method: str) -> dict:
    return json.loads((folder / f"{method}.ipynb").read_text())


@pytest.mark.parametrize("method,folder", LABS, ids=[m for m, _ in LABS])
def test_each_lab_has_a_clean_well_formed_notebook(method, folder):
    import nbformat

    path = folder / f"{method}.ipynb"
    assert path.exists(), f"{path.relative_to(ROOT)} is missing"
    nbformat.validate(nbformat.read(path, as_version=4))
    cells = _notebook(folder, method)["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    assert not any(c.get("outputs") or c.get("execution_count") for c in code), "save the notebook without outputs (Clear All Outputs)"
    setup = "".join(code[0]["source"])
    assert "lab.setup()" in setup and f'METHOD = "{method}"' in setup, "the first code cell sets up the lab for this method"
    headings = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    for level in ("Level 1", "Level 2", "Level 3", "Level 4"):
        assert re.search(rf"^## {level}", headings, re.M), f"missing a '## {level}' section"
    for number, cell in enumerate(code):
        compile("".join(cell["source"]), f"{path.name} cell {number}", "exec")


def test_the_notebook_api_names_exist():
    from recbench import lab

    used = set()
    for method, folder in LABS:
        for cell in _notebook(folder, method)["cells"]:
            used |= set(re.findall(r"\blab\.([a-z_]+)\(", "".join(cell["source"])))
    assert used <= set(lab.__all__), f"notebooks call lab functions that do not exist: {sorted(used - set(lab.__all__))}"


@pytest.mark.slow
@pytest.mark.parametrize("method,folder", LABS, ids=[m for m, _ in LABS])
def test_each_notebook_runs_on_the_toy_data(lab, method, folder):
    """Run the notebook top to bottom on the toy lab (tiny fits only; nothing touches the real datasets)."""
    import nbformat
    from nbclient import NotebookClient

    from recbench.lab import runs
    from recbench.tuning.job import run_job

    if method == "sansa":
        pytest.importorskip("sansa")
    _, resolved, spaces, settings = runs.lab_config()
    baseline = run_job("toy", method, resolved, settings, spaces.get(method))  # child processes: safe for SANSA on macOS
    if baseline.get("status") != "finished":
        pytest.skip(f"{method}'s toy baseline ended {baseline.get('status')}: {baseline.get('reason')}")
    path = lab / "labs" / Path(folder).name / f"{method}.ipynb"
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if "needs-experiments" not in c.get("metadata", {}).get("tags", [])]
    nb.cells[1].source = nb.cells[1].source.replace('DATASET = "movielens-25m"', 'DATASET = "toy"')
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
