"""Runner (MLflow logging, resume, isolation), serving bundles, and the API."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FAST_CFG
from recbench.data import TrainView
from recbench.evaluation import EvalSplit, Evaluator
from recbench.protocol import K_MAX
from recbench.registry import ensure_loaded
from recbench.runner import run_pair
from recbench.serving.bundle import Bundle, export_bundle

RESOLVED = {**FAST_CFG, "tier": "full", "preset": "cpu", "hardware_name": "test", "resume": True, "continue_on_error": False,
            "timeout_minutes": 5, "managed_services": False, "export_bundles": False, "datasets": [], "methods": [], "method_params": {}}


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return uri


def _runs(uri):
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("recbench")
    return mlflow.search_runs(experiment_ids=[exp.experiment_id])


def test_runner_logs_applicable_metrics_and_resumes(toy_split, tracking):
    assert run_pair(toy_split, "itemknn", RESOLVED, isolate=False)["status"] == "finished"
    runs = _runs(tracking)
    row = runs.iloc[0]
    assert row["tags.protocol_version"] == "2" and row["tags.config_hash"]
    assert row["metrics.ndcg_at_10"] >= 0
    assert "metrics.sampled_logloss" not in runs or np.isnan(row.get("metrics.sampled_logloss", np.nan))
    assert run_pair(toy_split, "itemknn", RESOLVED, isolate=False)["status"] == "skipped_existing"
    changed = {**RESOLVED, "knn_neighbors": 5}
    assert run_pair(toy_split, "itemknn", changed, isolate=False)["status"] == "finished"


def test_unsupported_methods_are_recorded_not_failed(toy_split, tracking):
    outcome = run_pair(toy_split, "multimodal_tower", RESOLVED, isolate=False)
    assert outcome["status"] == "unsupported"


def test_runs_from_another_machine_are_imported_once(toy_split, tracking, tmp_path, monkeypatch):
    from recbench.import_runs import import_runs
    from recbench.results import load_runs

    other = tmp_path / "gpu-mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", other.as_uri())
    assert run_pair(toy_split, "most_popular", RESOLVED, isolate=False)["status"] == "finished"
    source = mlflow.MlflowClient(other.as_uri())
    run_id = source.search_runs([source.get_experiment_by_name("recbench").experiment_id])[0].info.run_id
    for epoch, loss in enumerate([0.9, 0.5, 0.4], start=1):  # a learning curve, as the epoch-trained methods log it
        source.log_metric(run_id, "curve/loss", loss, step=epoch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    assert run_pair(toy_split, "random", RESOLVED, isolate=False)["status"] == "finished"
    assert import_runs(other)["imported"] == 1
    assert import_runs(other) == {"imported": 0, "already_imported": 1, "still_running": 0}
    frame = load_runs("full")
    assert set(frame["tags.method"]) == {"most_popular", "random"}
    imported = frame[frame["tags.method"] == "most_popular"].iloc[0]
    assert imported["metrics.ndcg_at_10"] >= 0 and imported["params.seed"] == "42"
    assert mlflow.MlflowClient(tracking).list_artifacts(imported["run_id"])
    curve = mlflow.MlflowClient(tracking).get_metric_history(imported["run_id"], "curve/loss")
    assert [(m.step, m.value) for m in curve] == [(1, 0.9), (2, 0.5), (3, 0.4)]  # every epoch, not only the last


@pytest.mark.slow
def test_child_process_records_peak_memory(toy_split, tracking):
    assert run_pair(toy_split, "most_popular", {**RESOLVED, "resume": False}, isolate=True)["status"] == "finished"
    runs = _runs(tracking)
    assert runs["metrics.peak_rss_mb"].iloc[0] > 10


def test_bundle_matches_the_evaluator_and_falls_back_for_unknown_users(toy_split, tmp_path):
    view, split = TrainView(toy_split), EvalSplit(toy_split)
    method = ensure_loaded().create_method("ease")
    method.fit(view, {"ease_lambda": 10.0})
    folder = export_bundle(method, view, tmp_path / "bundles" / "toy" / "full" / "ease", k=10, max_users=10_000)
    bundle = Bundle(folder)
    users = split.users_of(warm=True)
    topk, _ = Evaluator(split, view, {"seq_len": 10}).rank(method, users, ["exclude_seen"])
    for row, user in enumerate(users[:20]):
        served = [r["item_id"] for r in bundle.recommend(view.user_ids[user], 10)["recommendations"]]
        expected = [view.item_ids[i] for i in topk["exclude_seen"][row][:10] if i > 0]
        assert served == expected
    unknown = bundle.recommend("nobody", 5)
    assert unknown["fallback"] and len(unknown["recommendations"]) == 5
    assert K_MAX >= 10


def test_api_serves_bundles(toy_split, tmp_path, monkeypatch):
    view = TrainView(toy_split)
    method = ensure_loaded().create_method("most_popular")
    method.fit(view, {})
    export_bundle(method, view, tmp_path / "bundles" / "toy" / "smoke" / "most_popular", k=10)
    monkeypatch.setenv("RECBENCH_BUNDLES", str(tmp_path / "bundles"))
    monkeypatch.setenv("RECBENCH_TIER", "smoke")
    from recbench.serving import app as app_module

    app_module.load_bundle.cache_clear()
    client = TestClient(app_module.app)
    assert client.get("/health").json()["bundles"] == 1
    reply = client.post("/recommend", json={"dataset": "toy", "method": "most_popular", "user_id": str(view.user_ids[1]), "k": 3})
    assert reply.status_code == 200 and len(reply.json()["recommendations"]) == 3
    assert client.post("/recommend", json={"dataset": "toy", "method": "ease", "user_id": "u1"}).status_code == 404


def test_stats_and_monitor_report_freshness_quality_and_traffic(toy_split, tmp_path, monkeypatch, capsys):
    view = TrainView(toy_split)
    method = ensure_loaded().create_method("most_popular")
    method.fit(view, {})
    folder = tmp_path / "bundles" / "toy" / "smoke" / "most_popular"
    export_bundle(method, view, folder, k=10, extra={"run_id": "abc", "offline": {"coverage_at_10": 0.05}})
    monkeypatch.setenv("RECBENCH_BUNDLES", str(tmp_path / "bundles"))
    monkeypatch.setenv("RECBENCH_TIER", "smoke")
    from recbench.serving import app as app_module
    from recbench.serving.monitor import run_checks

    app_module.load_bundle.cache_clear()
    app_module._BUNDLE_STATS.clear()
    app_module.TRAFFIC = app_module.Traffic()
    client = TestClient(app_module.app)
    client.post("/recommend", json={"dataset": "toy", "method": "most_popular", "user_id": str(view.user_ids[1]), "k": 3})
    client.post("/recommend", json={"dataset": "toy", "method": "most_popular", "user_id": "nobody", "k": 3})
    log = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert [e["fallback"] for e in log] == [False, True] and all(e["status"] == 200 for e in log)
    stats = client.get("/stats").json()
    assert stats["traffic"]["requests"] == 2 and stats["traffic"]["fallback_share"] == 0.5
    bundle = stats["bundles"][0]
    assert bundle["run_id"] == "abc" and 0 < bundle["served"]["coverage_at_10"] <= 1
    assert 0 < bundle["served"]["popularity_percentile_at_10"] <= 1 and bundle["export_age_days"] < 1
    checks = {c.name: c.status for c in run_checks(client)}
    assert checks["health"] == "PASS" and checks["errors"] == "PASS"
    assert checks["unknown users"] == "PASS"  # 50% fallback is at the default limit, not above it
    assert {c.name: c.status for c in run_checks(client, max_fallback_share=0.2)}["unknown users"] == "WARN"
    assert checks["toy/most_popular: freshness"] == "PASS"
    # a bundle exported long ago is reported as stale
    manifest = json.loads((folder / "manifest.json").read_text())
    (folder / "manifest.json").write_text(json.dumps({**manifest, "exported_at": "2020-01-01T00:00:00Z"}))
    app_module._BUNDLE_STATS.clear()
    stale = {c.name: c.status for c in run_checks(client, max_export_age_days=30)}
    assert stale["toy/most_popular: freshness"] == "WARN"


def test_api_without_bundles_or_results_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("RECBENCH_BUNDLES", str(tmp_path / "empty"))
    monkeypatch.setenv("RECBENCH_TIER", "full")
    from recbench import results
    from recbench.serving import app as app_module

    def no_mlflow(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'mlflow'")  # as in the serving image

    monkeypatch.setattr(results, "load_runs", no_mlflow)
    client = TestClient(app_module.app)
    health = client.get("/health")
    assert health.status_code == 503 and health.json() == {**health.json(), "ok": False, "bundles": 0, "tier": "full"}
    assert client.get("/").json()["service"] == "recbench"  # an index, not a redirect to a page that may fail
    assert client.get("/dashboard").status_code == 404


def test_serving_does_not_import_torch():
    code = "import sys, recbench.serving.app; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert out.stdout.strip() == "False", out.stderr
