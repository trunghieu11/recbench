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
from recbench.methods._memory import dense_item_cap
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
    """Everyone gets the items with the most interactions in the last `pop_window_days` before the cutoff.

    With `pop_half_life_days`, each interaction in the window counts 2^(-age / half_life) instead of 1, so
    yesterday's interactions weigh more than last month's.
    """

    spec = MethodSpec(
        name="most_popular",
        tasks=BASELINE_TASKS,
        scores_cold_items=True,
        handles_cold_users=True,
        upstream="numpy",
        cost_band="low",
        impl_version="2",  # optional time decay
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.window_days = int(cfg.get("pop_window_days") or 28)
        half_life = cfg.get("pop_half_life_days")
        events = data.events()
        in_window = (events["ts_us"] >= int(data.meta["test_start_us"]) - self.window_days * DAY_US).to_numpy()
        items = events["item_idx"].to_numpy()[in_window]
        weights = data.event_weights(half_life)[in_window] if half_life else None
        recent = np.bincount(items, weights=weights, minlength=self.n_items + 1).astype(np.float64)
        overall = data.item_pop.astype(np.float64)
        # Recent (possibly decayed) count first; the all-time count only breaks ties (scaled below the smallest step).
        self.recent = np.bincount(items, minlength=self.n_items + 1).astype(np.int64)
        tie_break = overall / (overall.max() + 1.0) * (recent[recent > 0].min() if (recent > 0).any() else 1.0)
        self.scores = (recent + tie_break).astype(np.float32)
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
    Settings: `knn_weighting` (none / tfidf / bm25) down-weights very active users when measuring
    similarity; `decay_half_life_days` makes recent interactions count more, in the similarity and in the
    user's profile.
    """

    spec = MethodSpec(
        name="itemknn",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="scipy.sparse (in-repo)",
        cost_band="low",
        impl_version="2",  # weighting and time decay
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.k = int(cfg.get("knn_neighbors", 100))
        self.shrink = float(cfg.get("knn_shrink", 10.0))
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days"))
        self.sim = item_cosine_topk(weight_users(self.seen, str(cfg.get("knn_weighting") or "none")), self.k, self.shrink)

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


def weight_users(x: sp.csr_matrix, scheme: str = "none", k1: float = 1.2, b: float = 0.75) -> sp.csr_matrix:
    """Re-weight a user x item matrix before computing item-item similarity.

    Items play the role of documents and users the role of words, as in text retrieval:
    - tfidf: x_ui * idf_u with idf_u = log(n_items / (1 + items of u)); a user who touched everything says little.
    - bm25: like tfidf, but repeated (or large) weights saturate (k1) and long items (many users) are
      normalised (b), as in the BM25 ranking function.
    """
    if scheme in (None, "", "none"):
        return x
    x = sp.csr_matrix(x, dtype=np.float32, copy=True)
    n_items = x.shape[1]
    per_user = np.diff(x.indptr)
    idf = np.log(n_items / (1.0 + per_user)).astype(np.float32)
    idf = np.maximum(idf, 0.0)
    rows = np.repeat(np.arange(x.shape[0]), per_user)
    if scheme == "tfidf":
        x.data *= idf[rows]
        return x
    if scheme == "bm25":
        item_len = np.asarray((x > 0).sum(axis=0)).ravel().astype(np.float32)
        norm = k1 * (1.0 - b + b * item_len / max(item_len.mean(), 1e-9))
        x.data = x.data * (k1 + 1.0) / (x.data + norm[x.indices]) * idf[rows]
        return x
    raise ValueError(f"unknown weighting {scheme}; use none, tfidf, or bm25")


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
    On a CUDA GPU (or with ease_backend: torch) the inverse and the scoring run in PyTorch, and the cap
    is set by the free GPU memory instead (about 55,000 items on a 48 GB card).
    """

    spec = MethodSpec(
        name="ease",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="numpy or PyTorch (in-repo closed form)",
        cost_band="low",
        impl_version="3",  # time decay; GPU backend
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.lam = float(cfg.get("ease_lambda", 500.0))
        self.device = _ease_device(cfg)
        cap = dense_item_cap(int(cfg.get("ease_max_items", 30_000)), self.device, n_matrices=3)
        warm = np.flatnonzero(data.item_pop > 0)
        warm = warm[warm > 0]
        if len(warm) > cap:
            order = np.lexsort((-data.item_pop[warm], -data.item_recent_pop[warm]))
            warm = np.sort(warm[order[:cap]])
        self.kept = warm.astype(np.int64)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days"))
        x = self.seen[:, self.kept].astype(np.float32)
        # float32 keeps the dense item x item matrices at 4 bytes per entry (30K items ~ 3.6 GB each).
        gram = np.asarray((x.T @ x).todense(), dtype=np.float32)
        if self.device is None:
            gram[np.diag_indices_from(gram)] += self.lam
            inverse = np.linalg.inv(gram)
            del gram
            diag = np.diag(inverse).copy()
            inverse /= -diag[None, :]
            inverse[np.diag_indices_from(inverse)] = 0.0
            self.weights = inverse
        else:
            import torch

            g = torch.as_tensor(gram, device=self.device)
            del gram
            g.diagonal().add_(self.lam)
            inverse = torch.linalg.inv(g)
            del g
            inverse /= -torch.diagonal(inverse).clone()[None, :]
            inverse.fill_diagonal_(0.0)
            self.weights = inverse  # a torch tensor on the GPU
        self.position = np.full(self.n_items + 1, -1, dtype=np.int64)
        self.position[self.kept] = np.arange(len(self.kept))
        total = max(int(data.item_pop[1:].sum()), 1)
        self.fit_info = {"ease_items": len(self.kept), "item_cap_coverage": float(data.item_pop[self.kept].sum() / total)}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        x = self.seen[users][:, self.kept]
        if self.device is None:
            partial = np.asarray(x @ self.weights, dtype=np.float32)
        else:
            import torch

            partial = (torch.as_tensor(x.toarray(), device=self.device) @ self.weights).float().cpu().numpy()
        out = np.full((len(users), self.n_items + 1), NEG_INF, dtype=np.float32)
        out[:, self.kept] = partial
        return out

    def explain(self, users, items, hist):
        def weight(history: np.ndarray, j: int) -> np.ndarray:
            pos_j = self.position[j]
            pos_h = self.position[history]
            out = np.zeros(len(history))
            ok = (pos_h >= 0) & (pos_j >= 0)
            if ok.any():
                if self.device is None:
                    out[ok] = self.weights[pos_h[ok], pos_j]
                else:
                    import torch

                    rows = torch.as_tensor(pos_h[ok], device=self.device)
                    out[ok] = self.weights[rows, int(pos_j)].float().cpu().numpy()
            return out

        return contribution_explanations(users, items, lambda u: self.seen[u].indices, weight, self.item_ids, "EASE")


def _ease_device(cfg: dict[str, Any]):
    """The torch device for EASE, or None for the numpy path (no CUDA, or torch not installed)."""
    backend = str(cfg.get("ease_backend", "auto"))
    if backend == "numpy":
        return None
    try:
        from recbench.methods._torch import resolve_device
    except ImportError:  # the bench extra has no torch
        return None
    device = resolve_device(cfg)
    return device if (backend == "torch" or device.type == "cuda") else None
