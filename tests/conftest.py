"""Shared fixtures: small synthetic splits that build in about a second."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from recbench.data import TrainView
from recbench.evaluation import EvalSplit
from recbench.pipeline.materialize import materialize
from recbench.pipeline.toy import write_copy_task, write_toy_clean

FAST_CFG = {
    "seed": 42, "dim": 16, "layers": 1, "heads": 2, "seq_len": 10, "batch_size": 32, "max_steps": 30, "lr": 1e-2,
    "device": "cpu", "ease_lambda": 10.0, "knn_shrink": 1.0, "ials_iterations": 5, "bpr_iterations": 20, "graph_batch_size": 256,
    "max_epochs": 8,
}


def fake_encode_texts(texts, model_name, device, batch_size=256):
    """A tiny deterministic stand-in for a sentence-transformer: hashed bag of words, unit length."""
    import zlib

    import numpy as np

    out = np.zeros((len(texts), 32), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in str(text).lower().replace(",", " ").split():
            out[row, zlib.crc32(word.encode()) % 32] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms > 0, norms, 1.0)


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch, request):
    """Tests never download a text encoder, except the one test marked to use the real model."""
    if "real_text_encoder" not in request.keywords:
        monkeypatch.setattr("recbench.methods.text_knn.encode_texts", fake_encode_texts)


def build_toy(root: Path, **kwargs) -> Path:
    write_toy_clean(root / "clean", **kwargs)
    return materialize(root / "clean", root / "split", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1})


@pytest.fixture(scope="session")
def toy_split(tmp_path_factory) -> Path:
    return build_toy(tmp_path_factory.mktemp("toy"))


@pytest.fixture()
def toy(toy_split):
    return TrainView(toy_split), EvalSplit(toy_split)


@pytest.fixture(scope="session")
def toy_fold(tmp_path_factory) -> Path:
    """The toy data's validation fold: no real test events; the validation window is its test window."""
    root = tmp_path_factory.mktemp("toyfold")
    write_toy_clean(root / "clean")
    return materialize(root / "clean", root / "split", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1}, fold="valid")


@pytest.fixture(scope="session")
def copy_split(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("copy")
    write_copy_task(root / "clean")
    return materialize(root / "clean", root / "split", dataset="copy", tier="full",
                       repeat_policies=("allow_repeats",), tier_overrides={"min_eval_users": 1})


# ----- the improvement lab (tests/test_lab.py, tests/test_compare.py) -------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
TOY_LAB = """
workspace: lab
tier: full
preset: quick
hardware: configs/hardware/local-cpu.yaml
seed: 42
resume: true
continue_on_error: true
export_bundles: false
write_docs: false
track_code: true
overrides: {ease_backend: numpy, dim: 16, threads: 1}
eval: {save_topk: true}
datasets: [toy]
tuning: {spaces: configs/tuning/quick.yaml, trials: 2, search_users: 500, cap_minutes: 10, final_seeds: [42, 43]}
confirm: {top: 0}
queue:
  cpu_workers: 1
  methods: [{name: most_popular}, {name: itemknn}, {name: bpr_mf}]
"""


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    """A lab on the toy data: test split and validation fold, the real search spaces and experiment files, a small
    lab.yaml (2 settings per job), and everything in-process."""
    from recbench.lab import runs
    from recbench.paths import WORKSPACE_ENV

    write_toy_clean(tmp_path / "clean")
    splits = tmp_path / "data" / "splits" / "toy"
    materialize(tmp_path / "clean", splits / "full", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1})
    materialize(tmp_path / "clean", splits / "full-val", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1}, fold="valid")
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    shutil.copytree(REPO / "labs", tmp_path / "labs")
    shutil.copytree(REPO / "dictionary", tmp_path / "dictionary")  # the method catalog, for the overall comparison
    (tmp_path / "configs" / "benchmarks" / "lab.yaml").write_text(TOY_LAB)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECBENCH_ROOT", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv(WORKSPACE_ENV, "")  # restored after the test, whatever the code under test sets
    monkeypatch.setattr(runs, "ISOLATE", False)
    runs.lab_config.cache_clear()
    yield tmp_path
    runs.lab_config.cache_clear()
