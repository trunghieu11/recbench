"""Metric formulas (worked examples shared with the docs) and evaluator behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from recbench.evaluation import Evaluator, bootstrap_ci
from recbench.metrics.catalog import average_precision, gini, hitrate, ndcg, precision, recall, reciprocal_rank
from recbench.protocol import MethodSpec, Recommender, Task

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "assets" / "metric_examples.yaml"


def _hits(ranked: list[int], relevant: list[int]) -> np.ndarray:
    return np.array([[item in relevant for item in ranked]])


@pytest.mark.parametrize("case", yaml.safe_load(EXAMPLES.read_text())["cases"], ids=lambda c: c["name"])
def test_worked_examples_from_the_docs(case):
    hits = _hits(case["ranked"], case["relevant"])
    n_rel = np.array([len(case["relevant"])])
    k = case["k"]
    expected = case["expected"]
    computed = {
        "hitrate": hitrate(hits, k)[0],
        "precision": precision(hits, k)[0],
        "recall": recall(hits, n_rel, k)[0],
        "ndcg": ndcg(hits, n_rel, k)[0],
        "map": average_precision(hits, n_rel, k)[0],
        "mrr": reciprocal_rank(hits[:, :k])[0],
    }
    for name, value in expected.items():
        assert computed[name] == pytest.approx(value, abs=1e-3), name


def test_gini_extremes():
    assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 12])) == pytest.approx(0.75)


def test_bootstrap_interval_brackets_the_mean():
    values = np.random.default_rng(0).random(500)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high


class Oracle(Recommender):
    """Scores each user's relevant test items highest: every accuracy metric must be 1."""

    spec = MethodSpec("oracle", {Task.topn, Task.sequential}, scores_cold_items=True)

    def __init__(self, split):
        self.split = split

    def fit(self, data, cfg):
        self.bind(data)

    def score_users(self, users, hist):
        scores = np.zeros((len(users), self.n_items + 1), dtype=np.float32)
        for row, relevant in enumerate(self.split.relevant(users, self.split.primary)):
            scores[row, relevant] = 10.0
        nxt = self.split.next_items(users, self.split.primary)
        scores[np.arange(len(users)), nxt] = 20.0
        return scores


def test_oracle_scores_perfectly(toy):
    view, split = toy
    oracle = Oracle(split)
    oracle.fit(view, {})
    metrics = Evaluator(split, view, {"seq_len": 10}).run(oracle).metrics
    for name in ("hitrate_at_10", "ndcg_at_10", "recall_at_10", "next_hitrate_at_10", "mrr_at_50", "sampled_hitrate_at_10"):
        assert metrics[name] == pytest.approx(1.0), name


class SeenLover(Recommender):
    """Puts already-seen items and padding on top; the evaluator must remove them."""

    spec = MethodSpec("seen_lover", {Task.topn})

    def fit(self, data, cfg):
        self.bind(data)
        self.seen = data.seen

    def score_users(self, users, hist):
        scores = np.asarray(self.seen[users].todense(), dtype=np.float32) * 100.0
        scores[:, 0] = 1000.0
        return scores


def test_evaluator_masks_padding_seen_and_cold_items(toy):
    view, split = toy
    method = SeenLover()
    method.fit(view, {})
    evaluator = Evaluator(split, view, {"seq_len": 10})
    users = split.users_of(warm=True)
    topk, _ = evaluator.rank(method, users, ["exclude_seen"])
    lists = topk["exclude_seen"]
    cold = ~view.warm_item_mask()
    cold[0] = False
    for row, user in enumerate(users):
        listed = lists[row][lists[row] > 0]
        assert not set(listed) & set(view.user_items(int(user)))
        assert not cold[listed].any()
