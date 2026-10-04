"""The improvement lab: its own workspace, experiments, labels, seeds, re-runs, and the one-setting commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from recbench.lab import experiments as ex
from recbench.lab import runs
from recbench.paths import WORKSPACE_ENV, reports_dir, runs_dir, tracking_uri, use_workspace
from recbench.tuning.spaces import MethodSpace

ROOT = Path(__file__).resolve().parents[1]
COURSE = ["itemknn", "rp3beta", "ease", "slim", "sansa", "puresvd", "gfcf", "ials", "bpr_mf", "vsknn", "lgbm_rerank"]

# ----- the workspace ------------------------------------------------------------------------------------------------------

def test_a_workspace_moves_every_output(tmp_path, monkeypatch):
    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "elsewhere").as_uri())
    monkeypatch.setenv(WORKSPACE_ENV, "")
    use_workspace(None)
    assert runs_dir() == tmp_path / "runs" and reports_dir() == tmp_path / "reports"
    assert tracking_uri() == (tmp_path / "elsewhere").as_uri()  # the bake-off honours MLFLOW_TRACKING_URI
    use_workspace("lab")
    assert runs_dir() == tmp_path / "runs" / "lab" and reports_dir() == tmp_path / "reports" / "lab"
    assert tracking_uri() == (tmp_path / "runs" / "lab" / "mlflow").as_uri()  # a lab never writes to another store
    assert runs_dir("") == tmp_path / "runs"  # reading the bake-off from the lab


def test_the_lab_baseline_writes_only_under_runs_lab(lab):
    from recbench.queue import Queue

    Queue(lab / "configs" / "benchmarks" / "lab.yaml", isolate=False).run()
    tuning = lab / "runs" / "lab" / "tuning" / "full" / "toy"
    assert {p.name for p in tuning.glob("*.json")} == {"most_popular.json", "itemknn.json", "bpr_mf.json"}
    assert json.loads((tuning / "itemknn.json").read_text())["status"] == "finished"
    assert (lab / "runs" / "lab" / "mlflow").exists() and (lab / "runs" / "lab" / "queue" / "full.json").exists()
    assert (lab / "reports" / "lab" / "scoreboard.md").exists() and (lab / "reports" / "lab" / "full-tuned").exists()
    assert not (lab / "runs" / "tuning").exists() and not (lab / "runs" / "mlflow").exists() and not (lab / "reports" / "full-tuned").exists()


def test_random_methods_are_tested_once_per_seed_and_averaged(lab):
    from recbench.tuning.job import run_job

    _, resolved, spaces, settings = runs.lab_config()
    random = run_job("toy", "bpr_mf", resolved, settings, spaces["bpr_mf"], isolate=False)
    assert [r["seed"] for r in random["final_runs"]] == [42, 43]
    values = [r["value"] for r in random["final_runs"]]
    assert random["test"]["ndcg_at_10"] == pytest.approx(np.mean(values))
    assert "ndcg_at_10_seed_sd" in random["test"]
    fixed = run_job("toy", "itemknn", resolved, settings, spaces["itemknn"], isolate=False)
    assert len(fixed["final_runs"]) == 1  # deterministic: one test run is enough


# ----- experiments ----------------------------------------------------------------------------------------------------------

def test_the_course_order_is_the_folder_order():
    assert ex.lab_methods(ROOT) == COURSE
    from recbench.registry import ensure_loaded

    assert set(COURSE) <= set(ensure_loaded().methods)
    queued = [e["name"] for e in ex.lab_benchmark(ROOT)["queue"]["methods"]]
    assert sorted(queued) == sorted([ex.REFERENCE, *COURSE])


@pytest.mark.parametrize("method", COURSE)
def test_every_experiment_file_loads_and_samples(method):
    for label in ex.load_experiments(method, ROOT):
        experiment = ex.get(method, label, ROOT)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        experiment.space.sample(study.ask())
        if label != ex.BASELINE:  # the ready-to-run experiments need no code change
            assert ex.unread_settings(method, set(experiment.space.params) | set(experiment.space.fixed)) == []


def test_an_experiment_changes_the_baseline_space():
    base = MethodSpace("m", params={"a": {"type": "choice", "values": [1, 2]}, "b": {"type": "float", "low": 0, "high": 1}},
                       fixed={"c": 1}, trials=None)
    space = ex.build_space("m", {"remove": ["a"], "params": {"d": {"type": "choice", "values": ["x"]}}, "fixed": {"b": 0.5}}, base)
    assert set(space.params) == {"d"} and space.fixed == {"c": 1, "b": 0.5} and space.trials is None
    with pytest.raises(ex.ExperimentError, match="does not search"):
        ex.build_space("m", {"remove": ["zzz"]}, base)
    with pytest.raises(ex.ExperimentError, match="write it as in"):
        ex.build_space("m", {"params": {"d": [1, 2]}}, base)


def test_the_baseline_cannot_be_changed_and_labels_are_checked(tmp_path, monkeypatch):
    folder = tmp_path / "labs" / "01-itemknn"
    folder.mkdir(parents=True)
    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    (folder / "experiments.yaml").write_text("experiments:\n  baseline: {trials: 3}\n")
    with pytest.raises(ex.ExperimentError, match="unchanged"):
        ex.load_experiments("itemknn", tmp_path)
    (folder / "experiments.yaml").write_text("experiments:\n  My Idea: {}\n")
    with pytest.raises(ex.ExperimentError, match="lower-case"):
        ex.load_experiments("itemknn", tmp_path)


def test_settings_no_code_reads_are_caught():
    assert ex.unread_settings("ease", {"ease_lambda", "ease_variant", "train_window_days", "decay_half_life_days"}) == ["ease_variant"]


def test_track_code_puts_the_source_into_the_run_identity():
    from recbench.config import config_hash
    from recbench.registry import ensure_loaded
    from recbench.runner import method_version, run_hash_for

    plain = method_version("itemknn", {})
    assert plain == ensure_loaded().methods["itemknn"].spec.impl_version
    assert method_version("itemknn", {"track_code": True}).startswith(plain + "+src.")
    cfg = {"seed": 42, "knn_neighbors": 10}
    assert run_hash_for(cfg, "itemknn", "s") != run_hash_for({**cfg, "track_code": True}, "itemknn", "s")
    assert config_hash({**cfg, "save_topk": True}, "itemknn", "s") == config_hash(cfg, "itemknn", "s")  # bookkeeping only


def test_an_experiment_runs_with_its_own_label_and_reruns_when_it_changes(lab):
    results = runs.run_experiment("itemknn", "no-time-knobs", ["toy"], log=lambda _: None)
    tuning = lab / "runs" / "lab" / "tuning" / "full" / "toy"
    summary = json.loads((tuning / "itemknn@no-time-knobs.json").read_text())
    assert results[0]["status"] == summary["status"] == "finished" and summary["label"] == "no-time-knobs"
    assert {"decay_half_life_days", "train_window_days"}.isdisjoint(summary["space"]["params"])
    assert "+src." in summary["impl_version"] and summary["fingerprint"]
    assert runs.run_experiment("itemknn", "no-time-knobs", ["toy"], log=lambda _: None) == []  # done: skipped
    experiments_file = lab / "labs" / "01-itemknn" / "experiments.yaml"
    experiments_file.write_text(experiments_file.read_text().replace("remove: [decay_half_life_days, train_window_days]",
                                                                     "remove: [decay_half_life_days]"))
    again = runs.run_experiment("itemknn", "no-time-knobs", ["toy"], log=lambda _: None)
    assert again and again[0]["attempt"] == 1  # the definition changed: a fresh study
    assert list((tuning / "archive").glob("itemknn@no-time-knobs.*.json"))


def test_prepare_rerun_archives_and_starts_a_fresh_attempt(lab):
    from recbench.tuning.job import prepare_rerun, read_summary, summary_path

    use_workspace("lab")
    assert prepare_rerun("full", "toy", "itemknn") is False  # nothing to re-run
    path = summary_path("full", "toy", "itemknn")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tier": "full", "dataset": "toy", "method": "itemknn", "status": "finished", "attempt": 2}))
    assert prepare_rerun("full", "toy", "itemknn") is True
    stub = read_summary("full", "toy", "itemknn")
    assert stub["status"] == "pending" and stub["attempt"] == 3
    assert len(list((path.parent / "archive").glob("itemknn.*.json"))) == 1


# ----- one setting on the validation fold -------------------------------------------------------------------------------

def test_once_uses_the_validation_fold_and_reproduces_the_baseline(lab, monkeypatch):
    from recbench import runner
    from recbench.tuning.job import run_job

    _, resolved, spaces, settings = runs.lab_config()
    baseline = run_job("toy", "itemknn", resolved, settings, spaces["itemknn"], isolate=False)
    seen = []
    real = runner.run_pair
    monkeypatch.setattr(runner, "run_pair", lambda split, *a, **k: seen.append(Path(split).name) or real(split, *a, **k))
    again = runs.once("itemknn", "toy", best=True)
    assert seen == ["full-val"]  # never the test split
    assert again["metrics"]["ndcg_at_10"] == pytest.approx(baseline["best_val"])  # the best trial, reproduced
    other = runs.once("itemknn", "toy", {"knn_neighbors": 3}, best=True)
    assert other["settings"]["knn_neighbors"] == 3
    with pytest.raises(ex.ExperimentError, match="never reads"):
        runs.once("itemknn", "toy", {"knn_neighbours": 3})  # a typo is caught before anything runs


def test_the_notebook_api_gives_the_same_numbers_as_once(lab):
    from recbench.lab import api
    from recbench.tuning.job import run_job

    _, resolved, spaces, settings = runs.lab_config()
    baseline = run_job("toy", "itemknn", resolved, settings, spaces["itemknn"], isolate=False)
    data, split = api.load("toy")
    model = api.fit("itemknn", data, **baseline["best_params"])
    result = api.evaluate(model, data, split, users=settings.search_users)
    assert result.metrics["ndcg_at_10"] == pytest.approx(baseline["best_val"])
    assert len(api.neighbours(model, data, 1, k=3)) <= 3
    frame = runs.sweep("itemknn", "toy", "knn_neighbors", [2, 5], log=lambda _: None)
    assert list(frame["knn_neighbors"]) == [2, 5] and frame["ndcg_at_10"].notna().all()
    assert (lab / "reports" / "lab" / "sweeps" / "itemknn" / "toy-knn_neighbors.csv").exists()


def test_from_baseline_searches_only_the_new_settings(lab):
    from recbench.tuning.job import run_job

    experiments = lab / "labs" / "01-itemknn" / "experiments.yaml"
    experiments.write_text(experiments.read_text() + "\n  focused:\n    from_baseline: true\n    params:\n"
                           "      train_window_keep_last: {type: choice, values: [10, 50]}\n")
    focused = ex.get("itemknn", "focused", lab)
    with pytest.raises(ex.ExperimentError, match="no finished baseline"):
        focused.space_for(None)
    space = focused.space_for({"knn_neighbors": 7, "knn_shrink": 0, "train_window_keep_last": 99})
    assert list(space.params) == ["train_window_keep_last"] and space.fixed == {"knn_neighbors": 7, "knn_shrink": 0}

    _, resolved, spaces, settings = runs.lab_config()
    baseline = run_job("toy", "itemknn", resolved, settings, spaces["itemknn"], isolate=False)
    result = runs.run_experiment("itemknn", "focused", ["toy"], log=lambda _: None)[0]
    assert list(result["space"]["params"]) == ["train_window_keep_last"]
    held = {k: v for k, v in baseline["best_params"].items() if k != "train_window_keep_last"}
    assert {k: result["best_params"][k] for k in held} == held  # every other setting at the baseline's best
