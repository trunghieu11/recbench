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


def _dense_norm(x: np.ndarray, a: float) -> np.ndarray:
    rows, cols = x.sum(axis=1), x.sum(axis=0)
    d_u = np.where(rows > 0, np.power(np.maximum(rows, 1e-12), -a), 0.0)
    d_i = np.where(cols > 0, np.power(np.maximum(cols, 1e-12), a - 1.0), 0.0)
    return d_u[:, None] * x * d_i[None, :]


def test_gfcf_matches_the_dense_formula(toy):
    from recbench.methods.graph_filters import GFCF

    data, _ = toy
    method = GFCF()
    method.fit(data, {"gfcf_alpha": 0.3, "gfcf_k": 6})
    x = data.seen.toarray()
    norm = _dense_norm(x, 0.5)
    _, _, vt = np.linalg.svd(norm, full_matrices=False)
    cols = x.sum(axis=0)
    d_inv, d = np.where(cols > 0, cols ** -0.5, 0), np.where(cols > 0, cols**0.5, 0)
    users = data.warm_users()[:20]
    reference = x[users] @ norm.T @ norm + 0.3 * ((x[users] * d_inv) @ vt[:6].T @ vt[:6]) * d
    got = method.score_users(users, data.history_batch(users, 10))
    assert np.allclose(got[:, 1:], reference[:, 1:], atol=1e-3)  # randomized vs exact SVD, float32


@pytest.mark.parametrize("order", [1, 2, 3])
def test_turbocf_matches_the_dense_formula(toy, order):
    from recbench.methods.graph_filters import TurboCF

    data, _ = toy
    method = TurboCF()
    method.fit(data, {"turbocf_alpha": 0.6, "turbocf_power": 0.8, "turbocf_filter": order, "device": "cpu"})
    keep = method.kept
    x = data.seen.toarray()[:, keep]
    norm = _dense_norm(x, 0.6)
    p = (norm.T @ norm) ** 0.8
    f = {1: p, 2: 2 * p - p @ p, 3: p + 0.01 * (-p @ p @ p + 10 * p @ p - 29 * p)}[order]
    users = data.warm_users()[:20]
    got = method.score_users(users, data.history_batch(users, 10))[:, keep]
    assert np.allclose(got, x[users] @ f, atol=1e-3)


def test_ease_torch_backend_matches_numpy(toy):
    data, _ = toy
    users = data.warm_users()[:20]
    hist = data.history_batch(users, 10)
    a, b = EASE(), EASE()
    a.fit(data, {"ease_lambda": 20.0, "ease_backend": "numpy"})
    b.fit(data, {"ease_lambda": 20.0, "ease_backend": "torch", "device": "cpu"})
    sa, sb = a.score_users(users, hist), b.score_users(users, hist)
    finite = np.isfinite(sa)
    assert np.allclose(sa[finite], sb[finite], atol=1e-4)
    assert b.explain(users[:2], np.array([[1, 2], [3, 4]]), hist)  # explanations work on the torch path too


@pytest.mark.parametrize("name,extra", [
    ("simplex", {}),
    ("directau", {}),
    ("ultragcn", {"ultragcn_negatives": 10, "ultragcn_neg_weight": 10}),  # the paper's 200 negatives exceed the toy's 40 items
])
def test_learned_embedding_methods_beat_random_on_toy_data(toy, name, extra):
    from recbench.evaluation import Evaluator
    from recbench.registry import ensure_loaded

    data, split = toy
    cfg = {**FAST_CFG, "max_epochs": 30, "lr": 1e-2, "batch_size": 256, **extra}
    scores = {}
    for method_name in ("random", name):
        method = ensure_loaded().create_method(method_name)
        method.fit(data, cfg)
        scores[method_name] = Evaluator(split, data, cfg).run(method).metrics["ndcg_at_10"]
    assert scores[name] > scores["random"] + 0.1, scores


def test_alignment_and_uniformity():
    import torch

    from recbench.methods.mf_losses import alignment, uniformity

    same = torch.nn.functional.normalize(torch.ones(4, 3), dim=-1)
    spread = torch.eye(3)
    assert alignment(same, same).item() == pytest.approx(0.0)
    assert uniformity(same).item() == pytest.approx(0.0, abs=1e-6)  # all vectors in one spot: the worst case
    assert uniformity(spread).item() < -3.0  # orthogonal unit vectors: squared distance 2, log(exp(-4)) = -4


def test_ultragcn_neighbours_match_the_dense_definition(toy):
    from recbench.methods.ultragcn import item_neighbours

    data, _ = toy
    x = (data.seen.toarray() > 0).astype(float)
    a = x.T @ x
    g = a.sum(axis=1)
    left = np.where(g > 0, np.sqrt(g + 1) / np.maximum(g, 1e-12), 0)
    omega = a * left[:, None] * (1 / np.sqrt(g + 1))[None, :]
    ids, weights = item_neighbours(data.seen, k=5)
    for item in range(1, a.shape[0]):
        best = np.sort(omega[item])[::-1][:5]
        assert np.allclose(np.sort(weights[item])[::-1], best, atol=1e-5)
