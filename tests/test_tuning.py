"""Quick-tier tuning jobs: equal budgets, validation fold for search, one test run, resume, time cap."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import optuna
import pytest

from conftest import FAST_CFG
from recbench.pipeline.materialize import materialize
from recbench.pipeline.toy import write_toy_clean
from recbench.tuning import JobSettings, load_spaces, read_summary, run_job

ROOT = Path(__file__).resolve().parents[1]
RESOLVED = {**FAST_CFG, "tier": "full", "preset": "cpu", "hardware_name": "test", "resume": True, "continue_on_error": True,
            "timeout_minutes": 5, "managed_services": False, "export_bundles": False, "datasets": [], "methods": [],
            "method_params": {}}


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A data folder with the toy dataset's test split and validation fold, plus private MLflow and tuning stores."""
    write_toy_clean(tmp_path / "clean")
    splits = tmp_path / "data" / "splits" / "toy"
    materialize(tmp_path / "clean", splits / "full", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1})
    materialize(tmp_path / "clean", splits / "full-val", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1}, fold="valid")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    return tmp_path


def test_every_search_space_samples(tmp_path):
    spaces, settings = load_spaces(ROOT / "configs" / "tuning" / "quick.yaml")
    assert settings["version"] >= 1 and "ease" in spaces
    for name, space in spaces.items():  # one study per method, as in real jobs
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        params = space.sample(study.ask())
        assert set(space.params) <= set(params), name


def test_job_searches_the_fold_then_tests_once_and_resumes(workspace):
    spaces, _ = load_spaces(ROOT / "configs" / "tuning" / "quick.yaml")
    settings = JobSettings(tier="full", trials=3, search_users=50, cap_minutes=10)
    summary = run_job("toy", "most_popular", dict(RESOLVED), settings, spaces["most_popular"], isolate=False)
    assert summary["status"] == "finished", summary
    assert len(summary["trials"]) == 3 and set(summary["best_params"]) >= {"pop_window_days", "pop_half_life_days"}
    assert 0.0 <= summary["test"]["ndcg_at_10"] <= 1.0
    runs = mlflow.search_runs(experiment_names=["recbench"])
    search, final = runs[runs["tags.stage"] == "search"], runs[runs["tags.stage"] == "final"]
    assert set(search["tags.tier"]) == {"full-val"} and list(final["tags.tier"]) == ["full"]  # test split used once
    assert len(final) == 1 and set(runs["tags.tuning"]) == {"tuned"}
    again = run_job("toy", "most_popular", dict(RESOLVED), settings, spaces["most_popular"], isolate=False)
    assert again["status"] == "finished" and len(again["trials"]) == 3  # resumed: no new trials
    assert again["test"]["ndcg_at_10"] == pytest.approx(summary["test"]["ndcg_at_10"])
    assert len(mlflow.search_runs(experiment_names=["recbench"])) == len(runs)  # nothing re-ran
    assert read_summary("full", "toy", "most_popular")["status"] == "finished"


def test_retry_starts_a_fresh_attempt_after_failed_trials(workspace):
    from recbench.tuning.spaces import MethodSpace

    spaces, _ = load_spaces(ROOT / "configs" / "tuning" / "quick.yaml")
    settings = JobSettings(tier="full", trials=2, search_users=50, cap_minutes=10)
    good = spaces["most_popular"]
    broken = MethodSpace("most_popular", params=good.params, fixed={"train_window_days": "not a number"})  # every trial fails
    assert run_job("toy", "most_popular", dict(RESOLVED), settings, broken, isolate=False)["status"] == "failed"
    again = run_job("toy", "most_popular", dict(RESOLVED), settings, good, isolate=False)
    assert again["status"] == "failed" and again["attempt"] == 0  # without retry, the failed trials use up the budget
    fixed = run_job("toy", "most_popular", dict(RESOLVED), settings, good, isolate=False, retry=True)
    assert fixed["status"] == "finished" and fixed["attempt"] == 1 and len(fixed["trials"]) == 2
    resumed = run_job("toy", "most_popular", dict(RESOLVED), settings, good, isolate=False)
    assert resumed["attempt"] == 1  # later runs keep the new attempt
    assert resumed["test"]["ndcg_at_10"] == pytest.approx(fixed["test"]["ndcg_at_10"]) and set(fixed["test"]) <= set(resumed["test"])


def test_job_whose_trials_are_all_unsupported_is_unsupported(workspace):
    # The toy data has no item images, so the multimodal tower refuses to run: that is "unsupported", not "failed".
    summary = run_job("toy", "multimodal_tower", dict(RESOLVED), JobSettings(tier="full", trials=2), None, isolate=False)
    assert summary["status"] == "unsupported" and "images" in summary["reason"]


def test_job_without_time_for_one_trial_is_over_budget(workspace):
    spaces, _ = load_spaces(ROOT / "configs" / "tuning" / "quick.yaml")
    settings = JobSettings(tier="full", trials=3, search_users=50, cap_minutes=0.5)
    summary = run_job("toy", "ease", dict(RESOLVED), settings, spaces["ease"], isolate=False)
    assert summary["status"] == "over_budget" and summary["search_stopped"] == "time"


def test_missing_splits_are_reported(workspace):
    summary = run_job("nope", "ease", dict(RESOLVED), JobSettings(tier="full"), None, isolate=False)
    assert summary["status"] == "missing_split"


def test_a_reranker_job_pins_its_generators_settings(tmp_path, monkeypatch):
    from recbench.tuning.job import _write_summary, pin_generators

    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    _write_summary({"tier": "quick", "dataset": "d", "method": "ease", "status": "finished",
                    "best_params": {"ease_lambda": 100.0, "train_window_days": 90}})
    _write_summary({"tier": "quick", "dataset": "d", "method": "itemknn", "status": "finished", "best_params": {"knn_neighbors": 50}})
    cfg = {"rerank_itemknn": {"knn_neighbors": 20}}  # a setting given explicitly wins
    pinned = pin_generators(cfg, "lgbm_rerank", "full", "d")
    assert pinned == {"rerank_ease": {"ease_lambda": 100.0}, "rerank_itemknn": {"knn_neighbors": 20}}  # no training window
    _write_summary({"tier": "full", "dataset": "d", "method": "ease", "stage": "confirm", "status": "finished",
                    "best_params": {"ease_lambda": 400.0}})
    assert pin_generators({}, "lgbm_rerank", "full", "d")["rerank_ease"] == {"ease_lambda": 400.0}  # confirmed on full
    assert pin_generators({}, "lgbm_rerank", "quick", "d")["rerank_ease"] == {"ease_lambda": 100.0}  # the quick tier's own
    assert pin_generators({}, "ease", "full", "d") == {}  # other methods are left alone


def test_quick_tier_generator_settings_are_scaled_to_the_full_data(tmp_path, monkeypatch):
    from recbench.tuning import job
    from recbench.tuning.spaces import MethodSpace

    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    monkeypatch.setattr(job, "size_scale", lambda dataset, small, big: 20.0)  # the full fold has 20x the users
    job._write_summary({"tier": "quick", "dataset": "d", "method": "ease", "status": "finished", "best_params": {"ease_lambda": 100.0}})
    spaces = {"ease": MethodSpace("ease", confirm={"param": "ease_lambda", "scale_with": "users"}), "itemknn": MethodSpace("itemknn")}
    # No confirmation of EASE on full data: its quick-tier λ grows with the data, as a confirmation would start from.
    assert job.pin_generators({}, "lgbm_rerank", "full", "d", spaces)["rerank_ease"] == {"ease_lambda": 2000.0}
    assert job.pin_generators({}, "lgbm_rerank", "quick", "d", spaces)["rerank_ease"] == {"ease_lambda": 100.0}  # same tier
