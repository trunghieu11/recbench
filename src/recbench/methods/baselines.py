"""Non-neural baselines: Random, MostPopular, ItemKNN, EASE.

They train in seconds to minutes on a CPU and are surprisingly hard to beat;
every advanced method must be compared against them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recbench.data import HistoryBatch, TrainView
from recbench.methods._explain import contribution_explanations
from recbench.protocol import NEG_INF, Explanation, MethodSpec, Recommender, Task
from recbench.registry import register_method

DAY_US = 86_400_000_000
BASELINE_TASKS = {Task.topn, Task.sequential, Task.similar_items}


@register_method
class RandomRec(Recommender):
    """Uniformly random scores. The floor: anything that does not beat it learned nothing."""

    spec = MethodSpec(
        name="random",
        tasks={Task.topn, Task.sequential},
        scores_cold_items=True,
        handles_cold_users=True,
        upstream="numpy",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.seed = int(cfg.get("seed", 42))

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        out = np.empty((len(users), self.n_items + 1), dtype=np.float32)
        for row, user in enumerate(users):
            out[row] = np.random.default_rng((self.seed, int(user))).random(self.n_items + 1, dtype=np.float32)
        return out

    def explain(self, users, items, hist):
        return [[Explanation("random", "Picked at random.") for _ in row] for row in items]


@register_method
class MostPopular(Recommender):
    """Everyone gets the items with the most interactions in the last `pop_window_days` before the cutoff."""

    spec = MethodSpec(
        name="most_popular",
        tasks=BASELINE_TASKS,
        scores_cold_items=True,
        handles_cold_users=True,
        upstream="numpy",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.window_days = int(cfg.get("pop_window_days", 28))
        events = data.events()
        start = int(data.meta["test_start_us"]) - self.window_days * DAY_US
        recent = np.bincount(events.loc[events["ts_us"] >= start, "item_idx"], minlength=self.n_items + 1)
        overall = data.item_pop.astype(np.float64)
        # Recent count first; all-time count only breaks ties (scaled below 1).
        self.recent = recent.astype(np.int64)
        self.scores = (recent + overall / (overall.max() + 1.0)).astype(np.float32)
        self.scores[0] = NEG_INF

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        return np.broadcast_to(self.scores, (len(users), self.n_items + 1)).copy()

    def explain(self, users, items, hist):
        return [
            [
                Explanation(
                    "popularity",
                    f"Popular now: {int(self.recent[i])} interactions in the last {self.window_days} days.",
                    [{"recent_count": int(self.recent[i])}],
                )
                for i in row
            ]
            for row in items
        ]


@register_method
class ItemKNN(Recommender):
    """Item-based collaborative filtering: recommend items similar to what the user already interacted with.

    sim(i, j) = |users(i) & users(j)| / (sqrt(|users(i)|) * sqrt(|users(j)|) + shrink), keeping the top-k
    neighbours of each item. score(u, j) = sum of sim(i, j) over the items i in u's history.
    """

    spec = MethodSpec(
        name="itemknn",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="scipy.sparse (in-repo)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.k = int(cfg.get("knn_neighbors", 100))
        self.shrink = float(cfg.get("knn_shrink", 10.0))
        self.seen = data.seen
        self.sim = item_cosine_topk(self.seen, self.k, self.shrink)

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        return np.asarray((self.seen[users] @ self.sim).todense(), dtype=np.float32)

    def explain(self, users, items, hist):
        return contribution_explanations(
            users,
            items,
            lambda u: self.seen[u].indices,
            lambda history, j: self.sim[history, j].toarray(),
            self.item_ids,
            "ItemKNN",
        )


def item_cosine_topk(matrix: sp.csr_matrix, k: int, shrink: float, block: int = 2048) -> sp.csr_matrix:
    """Sparse item x item cosine similarity with shrinkage; each row keeps its k largest entries."""
    x = matrix.tocsc().astype(np.float32)
    n_items = x.shape[1]
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel())
    xt = x.T.tocsr()
    rows, cols, vals = [], [], []
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        co = (xt[start:stop] @ x).tocsr()
        co.sum_duplicates()
        for r in range(stop - start):
            item = start + r
            lo, hi = co.indptr[r], co.indptr[r + 1]
            if hi == lo:
                continue
            idx = co.indices[lo:hi]
            val = co.data[lo:hi] / (norms[item] * norms[idx] + shrink)
            keep = idx != item
            idx, val = idx[keep], val[keep]
            if len(val) > k:
                top = np.argpartition(-val, k - 1)[:k]
                idx, val = idx[top], val[top]
            rows.append(np.full(len(idx), item, dtype=np.int64))
            cols.append(idx.astype(np.int64))
            vals.append(val.astype(np.float32))
    if not rows:
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items), dtype=np.float32
    )


@register_method
class EASE(Recommender):
    """EASE^R (Steck, 2019): a linear item-to-item model with a closed-form solution.

    B = I - P / diag(P), with P = (X^T X + lambda I)^-1 and diag(B) = 0. score(u) = x_u B.
    The item x item matrix is dense, so the catalog is capped at `ease_max_items`
    (most recently popular items); items outside the cap are never recommended.
    """

    spec = MethodSpec(
        name="ease",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="numpy (in-repo closed form)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.lam = float(cfg.get("ease_lambda", 500.0))
        cap = int(cfg.get("ease_max_items", 30_000))
        warm = np.flatnonzero(data.item_pop > 0)
        warm = warm[warm > 0]
        if len(warm) > cap:
            order = np.lexsort((-data.item_pop[warm], -data.item_recent_pop[warm]))
            warm = np.sort(warm[order[:cap]])
        self.kept = warm.astype(np.int64)
        self.seen = data.seen
        x = self.seen[:, self.kept].astype(np.float32)
        # float32 keeps the dense item x item matrices at 4 bytes per entry (30K items ~ 3.6 GB each).
        gram = np.asarray((x.T @ x).todense(), dtype=np.float32)
        gram[np.diag_indices_from(gram)] += self.lam
        inverse = np.linalg.inv(gram)
        del gram
        diag = np.diag(inverse).copy()
        inverse /= -diag[None, :]
        inverse[np.diag_indices_from(inverse)] = 0.0
        self.weights = inverse
        self.position = np.full(self.n_items + 1, -1, dtype=np.int64)
        self.position[self.kept] = np.arange(len(self.kept))
        total = max(int(data.item_pop[1:].sum()), 1)
        self.fit_info = {"ease_items": len(self.kept), "item_cap_coverage": float(data.item_pop[self.kept].sum() / total)}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        x = self.seen[users][:, self.kept]
        partial = np.asarray(x @ self.weights, dtype=np.float32)
        out = np.full((len(users), self.n_items + 1), NEG_INF, dtype=np.float32)
        out[:, self.kept] = partial
        return out

    def explain(self, users, items, hist):
        def weight(history: np.ndarray, j: int) -> np.ndarray:
            pos_j = self.position[j]
            pos_h = self.position[history]
            out = np.zeros(len(history))
            ok = (pos_h >= 0) & (pos_j >= 0)
            out[ok] = self.weights[pos_h[ok], pos_j]
            return out

        return contribution_explanations(users, items, lambda u: self.seen[u].indices, weight, self.item_ids, "EASE")
