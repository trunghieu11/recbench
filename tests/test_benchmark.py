"""Protocol, models, report, and API."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from recbench.pipeline.materialize import materialize
from recbench.pipeline.toy import write_toy_clean
from recbench.registry import ensure_loaded


@pytest.fixture()
def store(tmp_path: Path):
    clean = tmp_path / "clean"
    write_toy_clean(clean)
    out = tmp_path / "splits" / "toy" / "smoke"
    return materialize(clean, out, dataset="toy", tier="smoke", split_mode="quantile")


def test_histories_stop_at_cutoff(store):
    train = pq.read_table(store.train_path).to_pandas()
    valid = pq.read_table(store.valid_path).to_pandas()
    test = pq.read_table(store.test_path).to_pandas()
    assert train["timestamp"].max() < test["timestamp"].min()
    pre = pd.concat([train, valid], ignore_index=True)
    allowed = set(zip(pre["user_idx"].astype(int), pre["item_idx"].astype(int)))
    sequences = pq.read_table(store.sequences_path).to_pandas()
    for user, hist, target in zip(sequences["user_idx"], sequences["history"], sequences["target_item"]):
        history = [] if hist is None else list(hist)
        for item in history:
            assert (int(user), int(item)) in allowed
        assert int(target) in set(test.loc[test.user_idx == user, "item_idx"].astype(int))


def test_one_positive_per_user(store):
    candidates = pq.read_table(store.candidates_path).to_pandas()
    positives = candidates[candidates["label"] == 1].groupby("user_idx").size()
    assert (positives == 1).all()
    assert candidates["label"].isin([0, 1]).all()


def test_methods_score_candidates(store, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlflow'}")
    os.environ["DATA_DIR"] = str(tmp_path)
    # Point the store root's artifacts path: runner uses store.root.parents.
    reg = ensure_loaded()
    resolved = {
        "tier": "smoke",
        "preset": "cpu",
        "max_steps": 1,
        "hardware_name": "local-cpu",
        "model_config": "smoke_cpu",
        "dim": 8,
        "layers": 1,
        "seq_len": 8,
        "batch_size": 16,
        "lr": 1e-3,
        "managed_services": False,
        "continue_on_error": True,
        "resume": False,
    }
    from recbench.runner import run_pair

    for name in ["xsimgcl", "bert4rec", "s3rec", "dcnv2", "din", "tiger", "hstu", "generative_llm"]:
        status = run_pair(store, name, resolved, resume=False)
        assert status == "finished", name
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name("recbench")
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    finished = frame[frame["tags.method"] == "xsimgcl"]
    assert finished["metrics.ndcg_at_10"].notna().any()
    assert finished["metrics.sanity_popularity_recall_at_10"].notna().any()
    status = run_pair(store, "multimodal_tower", resolved, resume=False)
    assert status == "unsupported"
    status = run_pair(store, "personalize", resolved, resume=False)
    assert status == "skipped"


def test_multimodal_with_images(store, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlflow2'}")
    items = pq.read_table(store.items_path).to_pandas()
    image_dir = tmp_path / "images"
    paths = []
    for item_id in items["item_id"].astype(str):
        path = image_dir / f"{item_id}.ppm"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"P6\n8 8\n255\n" + bytes((1, 2, 3)) * 64)
        paths.append(str(path))
    items["image_path"] = paths
    items.to_parquet(store.items_path, index=False)
    from recbench.runner import run_pair

    status = run_pair(
        store,
        "multimodal_tower",
        {
            "tier": "smoke",
            "preset": "cpu",
            "max_steps": 1,
            "hardware_name": "local-cpu",
            "model_config": "smoke_cpu",
            "dim": 8,
            "layers": 1,
            "seq_len": 8,
            "batch_size": 16,
            "lr": 1e-3,
            "managed_services": False,
            "continue_on_error": True,
            "resume": False,
        },
        resume=False,
    )
    assert status == "finished"


def test_dictionary_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlflow'}")
    from recbench.dictionary.build import build
    from recbench.report.build import build_report

    docs = build(Path(__file__).resolve().parents[1])
    assert (docs / "methods" / "bert4rec.md").exists()
    assert "not paper SOTA" in (docs / "index.md").read_text()
    out = build_report(tmp_path / "report", tier="full", preset="cpu", repo_root=Path(__file__).resolve().parents[1])
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()


def test_dashboard_pages():
    from fastapi.testclient import TestClient
    from recbench.serving.app import app

    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True
    assert client.get("/dashboard/wiring").status_code == 200
    assert client.get("/dashboard/topn").status_code == 200
    assert "not paper SOTA" in client.get("/dashboard/topn").text
