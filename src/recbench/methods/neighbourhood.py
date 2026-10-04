"""Neighbourhood methods beyond ItemKNN: RP3beta (graph random walks) and V-SKNN (session neighbours).

Both are cheap to train (sparse matrix products on a CPU), have no learned parameters to tune with SGD, and
are known to be strong baselines in published re-evaluations (Ferrari Dacrema et al. 2019; Ludewig &
Jannach 2018).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recbench.data import HistoryBatch, TrainView
from recbench.methods._explain import contribution_explanations
from recbench.protocol import NEG_INF, MethodSpec, Recommender, Task
from recbench.registry import register_method

SESSION_GAP_US = 30 * 60 * 1_000_000


def _row_normalise(matrix: sp.csr_matrix) -> sp.csr_matrix:
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1.0, sums, out=np.zeros_like(sums, dtype=np.float64), where=sums > 0)
    return sp.diags(scale.astype(np.float32)) @ matrix


def _keep_top_k(cols: np.ndarray, vals: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(vals) > k:
        top = np.argpartition(-vals, k - 1)[:k]
        return cols[top], vals[top]
    return cols, vals


def rp3beta_similarity(x: sp.csr_matrix, alpha: float, beta: float, k: int, block: int = 2048) -> sp.csr_matrix:
    """Item x item weights of a 3-step random walk item -> user -> item, with a popularity penalty.

    W[i, j] = sum_u P(u | i)^alpha * P(j | u)^alpha / pop(j)^beta, keeping each row's k largest entries.
    P(u | i): from item i, step to one of its users uniformly. P(j | u): from user u, step to one of u's items
    (weighted by x). beta = 0 gives P3alpha; larger beta pushes less popular items up.
    """
    x = sp.csr_matrix(x, dtype=np.float32)
    p_ui = _row_normalise(x).tocsr()
    p_ui.data **= alpha
    binary_t = x.T.tocsr()
    binary_t.data[:] = 1.0
    p_iu = _row_normalise(binary_t).tocsr()
    p_iu.data **= alpha
    pop = np.asarray(binary_t.sum(axis=1)).ravel()
    penalty = np.divide(1.0, np.power(pop, beta), out=np.zeros_like(pop, dtype=np.float64), where=pop > 0).astype(np.float32)
    n_items = x.shape[1]
    rows, cols, vals = [], [], []
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        walk = (p_iu[start:stop] @ p_ui).tocsr()
        for r in range(stop - start):
            item = start + r
            lo, hi = walk.indptr[r], walk.indptr[r + 1]
            if hi == lo:
                continue
            idx = walk.indices[lo:hi]
            val = walk.data[lo:hi] * penalty[idx]
            keep = (idx != item) & (val > 0)
            idx, val = _keep_top_k(idx[keep], val[keep], k)
            rows.append(np.full(len(idx), item, dtype=np.int64))
            cols.append(idx.astype(np.int64))
            vals.append(val.astype(np.float32))
    if not rows:
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items), dtype=np.float32)


@register_method
class RP3beta(Recommender):
    """RP3beta (Paudel et al. 2016): item-to-item random walks with a popularity penalty; beta = 0 is P3alpha."""

    spec = MethodSpec(
        name="rp3beta",
        tasks={Task.topn, Task.sequential, Task.similar_items},
        uses_history=True,
        upstream="scipy.sparse (in-repo), after Ferrari Dacrema et al.'s RP3betaRecommender",
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days"))
        self.sim = rp3beta_similarity(self.seen, float(cfg.get("rp3_alpha", 1.0)), float(cfg.get("rp3_beta", 0.5)),
                                      int(cfg.get("rp3_neighbors", 200)))

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        out = np.asarray((self.seen[users] @ self.sim).todense(), dtype=np.float32)
        out[:, 0] = NEG_INF
        return out

    def explain(self, users, items, hist):
        return contribution_explanations(users, items, lambda u: self.seen[u].indices,
                                         lambda history, j: self.sim[history, j].toarray(), self.item_ids, "RP3beta")


def sessions_from_events(data: TrainView, gap_us: int = SESSION_GAP_US) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Rebuild sessions from pre-test events: a gap of more than 30 minutes starts a new session.

    Returns (session x item binary matrix, session end time, session owner).
    """
    offsets = np.asarray(data._offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    owner = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    ts = np.asarray(data._ts, dtype=np.int64)
    items = np.asarray(data._items, dtype=np.int64)
    new = np.ones(len(ts), dtype=bool)
    new[1:] = (owner[1:] != owner[:-1]) | (ts[1:] - ts[:-1] > gap_us)
    session = np.cumsum(new) - 1
    n_sessions = int(session[-1]) + 1 if len(session) else 0
    matrix = sp.csr_matrix((np.ones(len(items), dtype=np.float32), (session, items)), shape=(n_sessions, data.n_items + 1))
    matrix.data[:] = 1.0
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    end = np.zeros(n_sessions, dtype=np.int64)
    np.maximum.at(end, session, ts)
    session_owner = np.zeros(n_sessions, dtype=np.int64)
    session_owner[session] = owner
    return matrix, end, session_owner


@register_method
class VSKNN(Recommender):
    """V-SKNN (Ludewig & Jannach 2018): score items by the past sessions most similar to the user's recent items.

    1. The user's profile is their last session (or last N items), weighted so recent items count more.
    2. Among past sessions sharing an item with the profile, keep the `sample_size` most recent, then the
       `k` most similar (cosine).
    3. score(item) = sum of the similarities of the neighbour sessions that contain it (optionally times the
       item's IDF, so rare co-occurrences count more).
    """

    spec = MethodSpec(
        name="vsknn",
        sequence_aware=True,
        tasks={Task.topn, Task.sequential, Task.session},
        uses_history=True,
        upstream="in-repo numpy/scipy, after Ludewig & Jannach's session-rec VSKNN",
        fidelity="simplified",  # see docs/dictionary/algorithms/vsknn.md: no dwell time, user-level next item
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.k = int(cfg.get("vsknn_k", 100))
        self.sample_size = int(cfg.get("vsknn_sample", 1000))
        self.weighting = str(cfg.get("vsknn_weighting", "div"))
        self.last_n = cfg.get("vsknn_last_n")
        self.use_idf = bool(cfg.get("vsknn_idf", False))
        self.sessions, self.session_end, self.session_owner = sessions_from_events(data)
        self.session_norm = np.sqrt(np.asarray(self.sessions.sum(axis=1)).ravel()).astype(np.float32)
        self.sessions_t = self.sessions.T.tocsr()
        # The user's own latest session supplies the profile, so it must not count as one of its neighbours.
        self.last_session = np.full(self.n_users + 1, -1, dtype=np.int64)
        self.last_session[self.session_owner] = np.arange(len(self.session_owner))
        n_sessions = max(self.sessions.shape[0], 1)
        df = np.asarray(self.sessions.sum(axis=0)).ravel()
        self.idf = np.log(n_sessions / np.maximum(df, 1.0)).astype(np.float32)
        self.data = data
        self.fit_info = {"sessions": int(self.sessions.shape[0])}

    def _profile(self, user: int) -> tuple[np.ndarray, np.ndarray]:
        """Items of the user's last session (or last N items) and their position weights (latest = 1)."""
        items = self.data.user_items(user)
        times = self.data.user_times(user)
        if len(items) == 0:
            return items, np.zeros(0, dtype=np.float32)
        if self.last_n:
            start = max(0, len(items) - int(self.last_n))
        else:
            gaps = np.flatnonzero(np.diff(times) > SESSION_GAP_US)
            start = int(gaps[-1]) + 1 if len(gaps) else 0
        recent = items[start:][::-1]  # latest first
        position = np.arange(1, len(recent) + 1, dtype=np.float32)
        if self.weighting == "div":
            weights = 1.0 / position
        elif self.weighting == "linear":
            weights = np.maximum(1.0 - 0.1 * (position - 1), 0.1)
        else:  # "same"
            weights = np.ones_like(position)
        out: dict[int, float] = {}
        for item, w in zip(recent.tolist(), weights.tolist()):
            out[item] = max(out.get(item, 0.0), w)  # an item seen twice keeps its most recent weight
        return np.fromiter(out, dtype=np.int64), np.fromiter(out.values(), dtype=np.float32)

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        out = np.zeros((len(users), self.n_items + 1), dtype=np.float32)
        for row, user in enumerate(users):
            items, weights = self._profile(int(user))
            if len(items) == 0:
                continue
            candidates = self.sessions_t[items]  # item -> sessions containing it
            sims = np.asarray(candidates.T @ weights).ravel() if candidates.nnz else np.zeros(0)
            if len(sims) and self.last_session[user] >= 0:
                sims[self.last_session[user]] = 0.0
            touched = np.flatnonzero(sims > 0)
            if len(touched) == 0:
                continue
            if len(touched) > self.sample_size:  # the most recent sessions sharing an item
                touched = touched[np.argpartition(-self.session_end[touched], self.sample_size - 1)[: self.sample_size]]
            sim = sims[touched] / (np.linalg.norm(weights) * self.session_norm[touched] + 1e-9)  # cosine
            if len(touched) > self.k:
                keep = np.argpartition(-sim, self.k - 1)[: self.k]
                touched, sim = touched[keep], sim[keep]
            scores = np.asarray(self.sessions[touched].T @ sim).ravel()
            out[row] = scores * self.idf if self.use_idf else scores
        out[:, 0] = NEG_INF
        return out
