"""Leakage guards: models can only ever see events before the test cutoff."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recbench.data import TrainView
from recbench.methods._torch import edge_batches
from recbench.methods.seq_trainer import sequence_windows

FORBIDDEN = ("test.parquet", "test_items", "eval_users", "candidates")


def test_train_view_has_no_route_to_test_files(toy_split):
    view = TrainView(toy_split)
    text = " ".join(str(v) for v in vars(view).values())
    assert not any(name in text for name in FORBIDDEN)
    assert "n_test" not in view.meta and "n_eval_warm" not in view.meta


def _pretest_pairs(view: TrainView) -> set[tuple[int, int]]:
    events = view.events()
    return set(zip(events["user_idx"], events["item_idx"]))


def _test_pairs(split) -> set[tuple[int, int]]:
    test = pd.read_parquet(split / "test_items.parquet")
    return set(zip(test["user_idx"], test["item_idx"]))


def test_sequence_targets_come_from_pretest_history(toy_split):
    view = TrainView(toy_split)
    allowed = _pretest_pairs(view)
    batches = sequence_windows(view, seq_len=8, batch_size=64, seed=0)
    for _ in range(20):
        batch = next(batches)
        for user, row in zip(batch["users"], batch["targets"]):
            for item in row[row > 0]:
                assert (int(user), int(item)) in allowed


def test_pairwise_positives_come_from_pretest_history(toy_split):
    view = TrainView(toy_split)
    allowed = _pretest_pairs(view)
    batch = next(edge_batches(view, batch_size=512, n_neg=4, seed=0))
    assert all((int(u), int(i)) in allowed for u, i in zip(batch["users"], batch["pos"]))
    seen = view.seen
    # negatives are re-drawn when seen; almost none should remain
    hits = np.asarray(seen[np.repeat(batch["users"], 4), batch["neg"].ravel()]).ravel()
    assert hits.mean() < 0.05


def test_new_test_items_are_absent_from_training_windows(toy_split):
    view = TrainView(toy_split)
    new_test = {pair for pair in _test_pairs(toy_split) if pair not in _pretest_pairs(view)}
    batches = sequence_windows(view, seq_len=8, batch_size=128, seed=1)
    for _ in range(10):
        batch = next(batches)
        for user, row in zip(batch["users"], batch["inputs"]):
            assert not any((int(user), int(item)) in new_test for item in row[row > 0])
