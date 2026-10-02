"""Recombee adapter with a fake client: budget checks, uploads, guards. Live calls only with RECBENCH_LIVE=1."""

from __future__ import annotations

import os

import numpy as np
import pytest

from recbench.data import TrainView
from recbench.evaluation import EvalSplit, Evaluator
from recbench.methods.recombee import RecombeeMethod
from recbench.protocol import Unsupported

api = pytest.importorskip("recombee_api_client.api_requests")


class FakeRecombee:
    """Remembers uploads; recommends the most viewed items the user has not viewed."""

    def __init__(self, existing_items: int = 0):
        self.items: dict[str, dict] = {f"old{i}": {} for i in range(existing_items)}
        self.views: list[tuple[str, str, int]] = []
        self.batches = 0
        self.resets = 0

    def send(self, request):
        name = type(request).__name__
        if name == "Batch":
            self.batches += 1
            return [self.send(r) for r in request.requests]
        if name == "ListItems":
            return list(self.items)[:1]
        if name == "ResetDatabase":
            self.resets += 1
            self.items.clear()
            self.views.clear()
            return "ok"
        if name == "AddItemProperty":
            return "ok"
        if name == "SetItemValues":
            self.items[request.item_id] = request.values
            return "ok"
        if name == "AddDetailView":
            self.views.append((request.user_id, request.item_id, request.timestamp))
            return "ok"
        if name == "RecommendItemsToUser":
            seen = {i for u, i, _ in self.views if u == request.user_id}
            counts: dict[str, int] = {}
            for _, item, _ in self.views:
                counts[item] = counts.get(item, 0) + 1
            ranked = [i for i, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if i not in seen]
            return {"recomms": [{"id": i} for i in ranked[: request.count]]}
        raise AssertionError(name)


CFG = {"max_eval_users": 50, "recombee_max_polls": 2, "recombee_poll_seconds": 0, "recombee_reset_wait_seconds": 0}


def test_upload_evaluate_and_budget(toy_split):
    view, split = TrainView(toy_split), EvalSplit(toy_split)
    fake = FakeRecombee()
    method = RecombeeMethod(client=fake)
    method.fit(view, CFG)
    assert len(fake.items) == view.n_items
    assert len(fake.views) == int(view.user_lengths.sum())
    assert all(isinstance(ts, int) and ts > 1_500_000_000 for _, _, ts in fake.views[:10])  # seconds, not microseconds
    result = Evaluator(split, view, {"seq_len": 10, **CFG}).run(method)
    assert 0 < result.metrics["ndcg_at_10"] <= 1
    stats = method.finish()
    assert stats["recombee_requests_used"] <= 90_000 and "served_p50_ms" in stats


def test_refuses_a_non_empty_database_without_permission(toy_split, monkeypatch):
    monkeypatch.delenv("RECBENCH_RECOMBEE_ALLOW_RESET", raising=False)
    with pytest.raises(Unsupported, match="not empty"):
        RecombeeMethod(client=FakeRecombee(existing_items=3)).fit(TrainView(toy_split), CFG)


def test_refuses_runs_that_would_exceed_the_budget(toy_split):
    with pytest.raises(Unsupported, match="requests"):
        RecombeeMethod(client=FakeRecombee()).fit(TrainView(toy_split), {**CFG, "recombee_max_requests": 100})


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RECBENCH_LIVE") != "1", reason="live Recombee test; set RECBENCH_LIVE=1 and credentials")
def test_live_recombee(toy_split):
    method = RecombeeMethod()
    method.fit(TrainView(toy_split), {**CFG, "recombee_poll_seconds": 30, "recombee_max_polls": 20})
    items, _ = method.topk(np.array([1]), None, 5)
    assert (items > 0).any()
    method.finish()
