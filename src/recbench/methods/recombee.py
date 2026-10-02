"""Recombee, a managed recommendation API (SaaS), benchmarked like any other method.

How a run works:
1. Check credentials and the request budget BEFORE sending anything:
   items + pre-test interactions + eval users + polling must fit `recombee_max_requests`
   (default 90,000). Recombee's docs say a batch is "equivalent as if [the requests]
   were executed one-by-one", so every sub-request is counted.
2. Upload items (title, category) and every pre-test interaction (as a detail view
   with its original timestamp) to a DEDICATED, empty database.
3. Poll until the service returns recommendations, then request top-K lists for the
   eval users. Ranking metrics are computed by recbench exactly as for local models.
4. Optionally reset the database afterwards (RECBENCH_RECOMBEE_ALLOW_RESET=1).

Environment variables:
    RECBENCH_RECOMBEE_DB            database id (create a free one in the Recombee admin UI)
    RECBENCH_RECOMBEE_TOKEN         its PRIVATE token (keep it secret; never commit it)
    RECBENCH_RECOMBEE_REGION        ap-se | ca-east | eu-west | us-west (default eu-west)
    RECBENCH_RECOMBEE_ALLOW_RESET   1 to allow wiping the database before/after a run
Use the "slice" tier (configs/benchmarks/recombee-slice.yaml) to stay inside the free plan.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from recbench.data import HistoryBatch, TrainView
from recbench.protocol import MethodSpec, Recommender, Task, Unsupported
from recbench.registry import register_method

REGIONS = {"ap-se": "AP_SE", "ca-east": "CA_EAST", "eu-west": "EU_WEST", "us-west": "US_WEST"}
BATCH_SIZE = 5_000


class RequestBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self, n: int) -> None:
        if self.used + n > self.limit:
            raise Unsupported(f"Recombee request budget exhausted ({self.used} + {n} > {self.limit})")
        self.used += n


def _sdk():
    try:
        from recombee_api_client import api_requests
        from recombee_api_client.api_client import RecombeeClient, Region
    except ImportError as exc:
        raise Unsupported("recombee-api-client is not installed (pip install 'recbench[managed]')") from exc
    return RecombeeClient, Region, api_requests


@register_method
class RecombeeMethod(Recommender):
    spec = MethodSpec(
        name="recombee",
        tasks={Task.topn},
        output="list",
        managed=True,
        scores_cold_items=True,
        upstream="Recombee API via recombee-api-client 6.3.1",
        cost_band="free plan (paid tiers from ~$99/month)",
    )

    def __init__(self, client: Any = None):
        self._client = client  # tests inject a fake client

    def _connect(self):
        if self._client is not None:
            return self._client
        db, token = os.environ.get("RECBENCH_RECOMBEE_DB"), os.environ.get("RECBENCH_RECOMBEE_TOKEN")
        if not db or not token:
            raise Unsupported("set RECBENCH_RECOMBEE_DB and RECBENCH_RECOMBEE_TOKEN to benchmark Recombee")
        client_cls, region_enum, _ = _sdk()
        region = REGIONS.get(os.environ.get("RECBENCH_RECOMBEE_REGION", "eu-west").lower(), "EU_WEST")
        return client_cls(db, token, region=getattr(region_enum, region))

    def _send_batch(self, requests: list, budget: RequestBudget) -> list:
        _, _, api = _sdk()
        out = []
        for start in range(0, len(requests), BATCH_SIZE):
            chunk = requests[start : start + BATCH_SIZE]
            budget.spend(len(chunk))
            out.extend(self.client.send(api.Batch(chunk)))
        return out

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        _, _, api = _sdk()
        frame = data.export_frame()
        n_eval = int(cfg.get("max_eval_users") or 2_000)
        polls = int(cfg.get("recombee_max_polls", 20))
        planned = 2 + data.n_items + len(frame) + n_eval + polls
        self.budget = RequestBudget(int(cfg.get("recombee_max_requests", 90_000)))
        if planned > self.budget.limit:
            raise Unsupported(
                f"Recombee would need ~{planned:,} requests (> {self.budget.limit:,}); use the 'slice' tier or raise recombee_max_requests"
            )
        self.client = self._connect()
        allow_reset = os.environ.get("RECBENCH_RECOMBEE_ALLOW_RESET") == "1"
        self.budget.spend(1)
        existing = self.client.send(api.ListItems(count=1))
        if existing:
            if not allow_reset:
                raise Unsupported("the Recombee database is not empty; use a dedicated database and set RECBENCH_RECOMBEE_ALLOW_RESET=1")
            self.budget.spend(1)
            self.client.send(api.ResetDatabase())
            time.sleep(float(cfg.get("recombee_reset_wait_seconds", 30)))
        self._send_batch([api.AddItemProperty("title", "string"), api.AddItemProperty("category", "string")], self.budget)
        items = [
            api.SetItemValues(str(data.item_ids[i]), {"title": str(data.item_text[i])[:500], "category": str(data.item_category[i])}, cascade_create=True)
            for i in range(1, data.n_items + 1)
        ]
        self._send_batch(items, self.budget)
        views = [
            api.AddDetailView(str(u), str(i), timestamp=int(ts) // 1_000_000, cascade_create=True)
            for u, i, ts in zip(frame["user_id"], frame["item_id"], frame["ts_us"])
        ]
        self._send_batch(views, self.budget)
        self._wait_until_ready(frame["user_id"].iloc[-1], polls, float(cfg.get("recombee_poll_seconds", 30)))
        self.index_of = {str(item_id): idx for idx, item_id in enumerate(data.item_ids) if idx > 0}
        self.user_ids = data.user_ids
        self.scenario = cfg.get("recombee_scenario")
        self.latencies_ms: list[float] = []
        self.fit_info = {"requests_used_fit": self.budget.used, "uploaded_interactions": len(frame)}

    def _wait_until_ready(self, user_id: str, polls: int, seconds: float) -> None:
        _, _, api = _sdk()
        for _ in range(polls):
            self.budget.spend(1)
            reply = self.client.send(api.RecommendItemsToUser(str(user_id), 5))
            if reply.get("recomms"):
                return
            time.sleep(seconds)
        raise Unsupported("Recombee did not return recommendations in time; try again later")

    def topk(self, users: np.ndarray, hist: HistoryBatch, k: int, exclude: Any = None) -> tuple[np.ndarray, np.ndarray]:
        _, _, api = _sdk()
        out = np.zeros((len(users), k), dtype=np.int64)
        for row, user in enumerate(users):
            request = api.RecommendItemsToUser(str(self.user_ids[user]), int(k), cascade_create=False, return_properties=False,
                                               **({"scenario": self.scenario} if self.scenario else {}))
            self.budget.spend(1)
            began = time.perf_counter()
            reply = self.client.send(request)
            self.latencies_ms.append((time.perf_counter() - began) * 1000)
            ids = [self.index_of.get(str(r["id"]), 0) for r in reply.get("recomms", [])][:k]
            out[row, : len(ids)] = ids
        return out, np.zeros_like(out, dtype=np.float32)

    def finish(self) -> dict[str, float]:
        """Latency of the live API calls and requests spent; reset the database when allowed."""
        stats: dict[str, float] = {"recombee_requests_used": float(self.budget.used)}
        if self.latencies_ms:
            values = np.asarray(self.latencies_ms)
            stats.update({f"served_p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)})
        if os.environ.get("RECBENCH_RECOMBEE_ALLOW_RESET") == "1" and os.environ.get("RECBENCH_RECOMBEE_KEEP") != "1":
            _, _, api = _sdk()
            self.client.send(api.ResetDatabase())
        return stats
