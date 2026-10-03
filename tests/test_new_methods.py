"""Correctness checks for the low-budget methods added for the quick tier (reference computations on toy data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from conftest import FAST_CFG
from recbench.data import TrainView
from recbench.methods.baselines import EASE
from recbench.methods.linear import SANSA, SLIM, PureSVD
from recbench.methods.neighbourhood import RP3beta, VSKNN, rp3beta_similarity, sessions_from_events
from recbench.pipeline.materialize import materialize


def naive_rp3beta(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    p_ui = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    binary = (x > 0).astype(float).T
    pop = binary.sum(axis=1)
    p_iu = binary / np.maximum(pop[:, None], 1e-12)
    w = (p_iu**alpha) @ (p_ui**alpha)
    w = w * np.where(pop > 0, pop ** (-beta) if beta else 1.0, 0.0)[None, :]
    np.fill_diagonal(w, 0.0)
    return w


@pytest.mark.parametrize("beta", [0.0, 0.6])
def test_rp3beta_matches_the_dense_definition(toy, beta):
    data, _ = toy
    x = data.seen
    fast = rp3beta_similarity(x, alpha=0.8, beta=beta, k=10_000).toarray()
    assert np.allclose(fast, naive_rp3beta(x.toarray(), 0.8, beta), atol=1e-6)


def test_puresvd_projects_onto_the_top_singular_vectors(toy):
    data, _ = toy
    method = PureSVD()
    method.fit(data, {**FAST_CFG, "svd_factors": 5})
    _, _, vt = np.linalg.svd(data.seen.toarray(), full_matrices=False)
    reference = vt[:5].T @ vt[:5]
    assert np.allclose(method.v @ method.v.T, reference, atol=1e-3)


def test_slim_weights_are_non_negative_without_self_loops(toy):
    data, _ = toy
    method = SLIM()
    method.fit(data, {**FAST_CFG, "slim_alpha": 1e-4, "slim_l1_ratio": 0.1, "slim_neighbors": 20, "threads": 1})
    w = method.weights
    assert w.nnz > 0 and (w.data >= 0).all() and np.allclose(w.diagonal(), 0.0)


def test_vsknn_recommends_items_from_similar_sessions(tmp_path):
    day = pd.Timestamp("2020-01-01")
    rows = [  # (user, item, minutes after day start)
        ("a", "i1", 0), ("a", "i2", 1),                      # a's only session: i1, i2
        ("b", "i1", 0), ("b", "i2", 2), ("b", "i3", 3),      # b had i1, i2 -> i3
        ("c", "i4", 0), ("c", "i5", 1),                      # unrelated
        ("a", "i9", 60 * 24 * 40), ("b", "i9", 60 * 24 * 40), ("c", "i9", 60 * 24 * 40),  # test window
    ]
    clean = tmp_path / "clean"
    clean.mkdir()
    pd.DataFrame({"user_id": [r[0] for r in rows], "item_id": [r[1] for r in rows],
                  "timestamp": [day + pd.Timedelta(minutes=r[2]) for r in rows], "session_id": "",
                  "feedback_type": "implicit", "value": 1.0}).to_parquet(clean / "interactions.parquet")
    pd.DataFrame({"item_id": [f"i{i}" for i in (1, 2, 3, 4, 5, 9)], "text": "", "category": "", "image_path": None}).to_parquet(clean / "items.parquet")
    pd.DataFrame({"user_id": ["a"], "attributes": "", "group_label": None}).to_parquet(clean / "users.parquet")
    split = materialize(clean, tmp_path / "split", dataset="tiny", tier="full", split_rule="last_days", test_days=7,
                        tier_overrides={"min_eval_users": 0})
    data = TrainView(split)
    sessions, _, owner = sessions_from_events(data)
    assert sessions.shape[0] == 3 and sorted(np.bincount(owner)[1:].tolist()) == [1, 1, 1]
    method = VSKNN()
    method.fit(data, {"vsknn_k": 10, "vsknn_sample": 100})
    user_a = int(np.flatnonzero(data.user_ids == "a")[0])
    scores = method.score_users(np.array([user_a]), data.history_batch(np.array([user_a]), 10))[0]
    item = {iid: idx for idx, iid in enumerate(data.item_ids)}
    assert scores[item["i3"]] > 0 and scores[item["i4"]] == 0 and scores[item["i5"]] == 0


def test_sansa_ranks_like_exact_ease(toy):
    pytest.importorskip("sansa")
    data, _ = toy
    exact, approx = EASE(), SANSA()
    exact.fit(data, {"ease_lambda": 50.0})
    approx.fit(data, {"sansa_lambda": 50.0, "sansa_density": 1.0})
    users = data.warm_users()[:30]
    hist = data.history_batch(users, 10)
    a, b = exact.score_users(users, hist), approx.score_users(users, hist)
    unseen = (data.seen[users].toarray() == 0)
    unseen[:, 0] = False
    for row in range(len(users)):  # the two differ only on already-seen items (EASE zeroes its diagonal)
        mask = unseen[row] & np.isfinite(a[row])
        assert np.corrcoef(a[row][mask], b[row][mask])[0, 1] > 0.99
