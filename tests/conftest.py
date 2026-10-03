"""Shared fixtures: small synthetic splits that build in about a second."""

from __future__ import annotations

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
