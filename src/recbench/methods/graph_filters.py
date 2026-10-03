"""Training-free graph filters: GF-CF (Shen et al. 2021) and Turbo-CF (Park et al. 2024).

Graph neural networks such as LightGCN smooth embeddings over the user-item graph by gradient descent. These
methods skip the training: they apply a fixed "low-pass filter" to the normalised item-item graph and read
scores off directly.

    R~ = D_U^(-a) R D_I^(a-1)        normalised interactions (a = 1/2: the symmetric normalisation)
    P  = R~^T R~                      item-item graph (a "linear filter")

GF-CF:    score(u) = r_u P + alpha * r_u D_I^(-1/2) V_k V_k^T D_I^(1/2)   (V_k: top-k singular vectors of R~)
Turbo-CF: score(u) = r_u f(P^s)   with f(P) = P, 2P - P^2, or P + 0.01(-P^3 + 10P^2 - 29P), and s an
          element-wise power. No decomposition, so it runs fast with dense matrix products on a GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recbench.data import HistoryBatch, TrainView
from recbench.methods._memory import dense_item_cap
from recbench.methods.baselines import BASELINE_TASKS
from recbench.protocol import NEG_INF, MethodSpec, Recommender
from recbench.registry import register_method


def normalise(x: sp.csr_matrix, a: float = 0.5) -> sp.csr_matrix:
    """D_U^(-a) X D_I^(a-1); users and items without interactions get 0 instead of an infinite weight."""
    rows = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    cols = np.asarray(x.sum(axis=0)).ravel().astype(np.float64)
    d_u = np.power(rows, -a, out=np.zeros_like(rows), where=rows > 0)
    d_i = np.power(cols, a - 1.0, out=np.zeros_like(cols), where=cols > 0)
    return (sp.diags(d_u.astype(np.float32)) @ x @ sp.diags(d_i.astype(np.float32))).tocsr()


@register_method
class GFCF(Recommender):
    """GF-CF: a linear item-item filter plus an "ideal low-pass" part from the top singular vectors."""

    spec = MethodSpec(
        name="gfcf",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="in-repo scipy implementation of Shen et al. 2021 (GF-CF)",
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        from sklearn.utils.extmath import randomized_svd

        self.bind(data)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days")).astype(np.float32)
        norm = normalise(self.seen, 0.5)
        self.linear = (norm.T @ norm).tocsr()  # the linear filter P = R~^T R~ (sparse item x item)
        self.alpha = float(cfg.get("gfcf_alpha", 0.3))
        self.vt = None
        if self.alpha > 0:
            k = max(1, min(int(cfg.get("gfcf_k", 256)), min(norm.shape) - 1))
            _, _, vt = randomized_svd(norm, n_components=k, n_iter=5, random_state=int(cfg.get("seed", 42)))
            cols = np.asarray(self.seen.sum(axis=0)).ravel().astype(np.float64)
            self.d_inv = np.power(cols, -0.5, out=np.zeros_like(cols), where=cols > 0).astype(np.float32)
            self.d = np.power(cols, 0.5, out=np.zeros_like(cols), where=cols > 0).astype(np.float32)
            self.vt = vt.astype(np.float32)  # k x items
        self.fit_info = {"linear_nonzeros": int(self.linear.nnz)}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        x = self.seen[users]
        out = np.asarray((x @ self.linear).todense(), dtype=np.float32)
        if self.vt is not None:  # ideal low-pass: x D^(-1/2) V V^T D^(1/2)
            low = np.asarray(x.multiply(self.d_inv[None, :]) @ self.vt.T, dtype=np.float32)
            out += self.alpha * ((low @ self.vt) * self.d[None, :])
        out[:, 0] = NEG_INF
        return out


@register_method
class TurboCF(Recommender):
    """Turbo-CF: a polynomial low-pass filter of the dense item-item graph; matrix products only (GPU-friendly)."""

    spec = MethodSpec(
        name="turbocf",
        tasks=BASELINE_TASKS,
        uses_history=True,
        needs_torch=True,
        upstream="in-repo PyTorch, following the authors' code (github.com/jindeok/Turbo-CF, MIT)",
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        import torch

        from recbench.methods._torch import resolve_device

        self.bind(data)
        self.device = resolve_device(cfg)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days")).astype(np.float32).tocsr()
        order = int(cfg.get("turbocf_filter", 1))
        # The dense item-item matrix (and its powers for filters 2 and 3) must fit in memory: keep the
        # most recently popular items, as EASE does.
        cap = dense_item_cap(int(cfg.get("turbocf_max_items", 30_000)), self.device, n_matrices=1 + order)
        warm = np.flatnonzero(data.item_pop > 0)
        warm = warm[warm > 0]
        if len(warm) > cap:
            ranked = np.lexsort((-data.item_pop[warm], -data.item_recent_pop[warm]))
            warm = np.sort(warm[ranked[:cap]])
        self.kept = warm.astype(np.int64)
        x = self.seen[:, self.kept]
        norm = normalise(x, float(cfg.get("turbocf_alpha", 0.5)))
        p = torch.as_tensor((norm.T @ norm).toarray(), device=self.device)
        p = p.pow(float(cfg.get("turbocf_power", 1.0)))
        if order == 2:
            p = 2 * p - p @ p
        elif order == 3:
            p2 = p @ p
            p = p + 0.01 * (-(p2 @ p) + 10 * p2 - 29 * p)
        self.filter = p
        total = max(int(data.item_pop[1:].sum()), 1)
        self.fit_info = {"items": len(self.kept), "item_cap_coverage": float(data.item_pop[self.kept].sum() / total)}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        import torch

        x = torch.as_tensor(self.seen[users][:, self.kept].toarray(), device=self.device)
        partial = (x @ self.filter).float().cpu().numpy()
        out = np.full((len(users), self.n_items + 1), NEG_INF, dtype=np.float32)
        out[:, self.kept] = partial
        return out

