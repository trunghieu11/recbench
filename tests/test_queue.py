"""The quick-tier queue: dataset-by-dataset order with backfill, dependencies, resume, status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from conftest import FAST_CFG
from recbench.pipeline.materialize import materialize
from recbench.pipeline.toy import write_toy_clean
from recbench.queue import Queue, print_status

ROOT = Path(__file__).resolve().parents[1]


def write_config(path: Path, datasets: list[str], methods: list[dict], *, confirm_top: int = 0, trials: int = 2) -> Path:
    config = {
        "tier": "full", "preset": "cpu", "hardware": str(ROOT / "configs" / "hardware" / "local-cpu.yaml"), "seed": 42,
        "resume": True, "continue_on_error": True, "export_bundles": False, "datasets": datasets,
        "method_params": {}, "max_steps": FAST_CFG["max_steps"],
        "eval": {k: v for k, v in FAST_CFG.items() if k not in {"seed", "device"}},
        "tuning": {"spaces": str(ROOT / "configs" / "tuning" / "quick.yaml"), "trials": trials, "search_users": 50, "cap_minutes": 20},
        "queue": {"methods": methods, "jobs_per_gpu": 2},
        "confirm": {"top": confirm_top, "tier": "full", "seeds": [1, 2]},
    }
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    for name in ("toy", "toy2"):
        write_toy_clean(tmp_path / name / "clean", seed=0 if name == "toy" else 1)
        splits = tmp_path / "data" / "splits" / name
        for fold in (None, "valid"):
            materialize(tmp_path / name / "clean", splits / ("full-val" if fold else "full"), dataset=name, tier="full",
                        tier_overrides={"min_eval_users": 1}, fold=fold)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    return tmp_path


def test_jobs_run_dataset_by_dataset_and_backfill_only_when_blocked(workspace):
    methods = [{"name": "random", "resource": "cpu"}, {"name": "ease", "resource": "cpu"},
               {"name": "itemknn", "resource": "cpu", "after": ["ease"]}, {"name": "sasrec", "resource": "gpu"}]
    queue = Queue(write_config(workspace / "q.yaml", ["toy", "toy2"], methods), isolate=False, gpus=1, cpu_workers=2)
    first = queue._next_job("cpu")
    assert (first.dataset, first.method) == ("toy", "random")
    first.status = "running"
    second = queue._next_job("cpu")
    assert (second.dataset, second.method) == ("toy", "ease")  # itemknn waits for ease
    second.status = "running"
    backfill = queue._next_job("cpu")
    assert (backfill.dataset, backfill.method) == ("toy2", "random")  # toy has nothing runnable for a CPU worker
    assert queue._next_job("gpu").method == "sasrec" and queue._next_job("gpu").dataset == "toy"
    queue.stop_after_dataset, queue.active_dataset = True, "toy"
    assert queue._next_job("cpu") is None  # no backfill into toy2 when told to stop after toy


def test_queue_runs_all_jobs_and_resumes(workspace, capsys):
    methods = [{"name": "random", "resource": "cpu"}, {"name": "most_popular", "resource": "cpu"},
               {"name": "itemknn", "resource": "cpu", "after": ["most_popular"]}]
    config = write_config(workspace / "q.yaml", ["toy", "toy2"], methods)
    jobs = Queue(config, isolate=False, gpus=0, cpu_workers=2).run()
    assert {j.status for j in jobs} == {"finished"} and len(jobs) == 6
    state = json.loads((workspace / "runs" / "queue" / "full.json").read_text())
    assert all(j["status"] == "finished" for j in state["jobs"])
    again = Queue(config, isolate=False, gpus=0, cpu_workers=2)
    assert all(j.status == "finished" for j in again.jobs)  # resumed from the summaries: nothing to run
    print_status(config)
    out = capsys.readouterr().out
    assert "== toy: 3/3 jobs done" in out and "== toy2: 3/3 jobs done" in out


def test_confirmation_rechecks_the_size_setting_on_full_data(tmp_path, monkeypatch):
    write_toy_clean(tmp_path / "clean")
    splits = tmp_path / "data" / "splits" / "toy"
    for tier in ("smoke", "full"):
        for fold in (None, "valid"):
            materialize(tmp_path / "clean", splits / (f"{tier}-val" if fold else tier), dataset="toy", tier=tier,
                        tier_overrides={"min_eval_users": 1}, fold=fold)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    from recbench.tuning import JobSettings, load_spaces, run_job
    from recbench.tuning.job import run_confirm

    spaces, _ = load_spaces(ROOT / "configs" / "tuning" / "quick.yaml")
    resolved = {**FAST_CFG, "tier": "smoke", "preset": "cpu", "hardware_name": "test", "resume": True, "continue_on_error": True,
                "timeout_minutes": 5, "managed_services": False, "export_bundles": False, "datasets": [], "methods": [], "method_params": {}}
    settings = JobSettings(tier="smoke", trials=2, search_users=50, cap_minutes=10)
    quick = run_job("toy", "ease", resolved, settings, spaces["ease"], isolate=False)
    assert quick["status"] == "finished"
    confirmed = run_confirm("toy", "ease", resolved, settings, spaces["ease"], tier="full", seeds=[1, 2, 3], isolate=False)
    assert confirmed["status"] == "finished" and len(confirmed["checks"]) == 3
    lambdas = sorted(c["params"]["ease_lambda"] for c in confirmed["checks"])
    assert lambdas[1] == pytest.approx(2 * lambdas[0]) and lambdas[2] == pytest.approx(2 * lambdas[1])
    assert len(confirmed["finals"]) == 1  # EASE is deterministic: one seed is enough
