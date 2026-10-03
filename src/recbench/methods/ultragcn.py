"""UltraGCN (Mao et al. 2021): the effect of infinitely many graph-convolution layers, without message passing.

LightGCN smooths embeddings by repeatedly averaging over graph neighbours, which costs a full pass over the
graph at every training step. UltraGCN derives what that smoothing converges to and turns it into extra
weights in an ordinary matrix-factorisation loss:

    L = L_O + gamma * ||theta||^2 / 2 + lambda * L_I
    L_O: binary cross-entropy on (user, item) pairs and sampled negatives, each weighted by
         w1 + w2 * beta_ui with beta_ui = sqrt(d_u + 1) / d_u * 1 / sqrt(d_i + 1)  (the "constraint" weights)
    L_I: pulls a user towards the top-K item-graph neighbours of their positive items, weighted by omega_ij.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from recbench.data import HistoryBatch, TrainView
from recbench.methods._torch import EmbeddingRecommender, edge_batches, resolve_device, steps_per_epoch, train_epochs
from recbench.methods.mf_losses import MF_TASKS
from recbench.methods.seq_trainer import to_tensor
from recbench.protocol import MethodSpec
from recbench.registry import register_method


def item_neighbours(x: sp.csr_matrix, k: int, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-k item-item neighbours with UltraGCN's omega weights.

    omega_ij = A_ij * sqrt(g_i + 1) / g_i * 1 / sqrt(g_j + 1), with A = X^T X the co-occurrence graph and g its
    degrees (row sums). Returns (neighbour ids [items, k], weights [items, k]); padding has weight 0.
    """
    x = sp.csr_matrix(x, dtype=np.float32)
    x.data[:] = 1.0
    a = (x.T @ x).tocsr()
    g = np.asarray(a.sum(axis=1)).ravel().astype(np.float64)
    left = np.divide(np.sqrt(g + 1), g, out=np.zeros_like(g), where=g > 0).astype(np.float32)
    right = (1.0 / np.sqrt(g + 1)).astype(np.float32)
    n_items = a.shape[0]
    ids = np.zeros((n_items, k), dtype=np.int64)
    weights = np.zeros((n_items, k), dtype=np.float32)
    for start in range(0, n_items, block):
        rows = a[start : min(start + block, n_items)].tocsr()
        for r in range(rows.shape[0]):
            lo, hi = rows.indptr[r], rows.indptr[r + 1]
            if hi == lo:
                continue
            idx = rows.indices[lo:hi]
            val = rows.data[lo:hi] * left[start + r] * right[idx]
            take = min(k, len(val))
            top = np.argpartition(-val, take - 1)[:take]
            ids[start + r, :take] = idx[top]
            weights[start + r, :take] = val[top]
    return ids, weights


class UltraGCNNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, init_std: float):
        super().__init__()
        self.user = nn.Embedding(n_users + 1, dim)
        self.item = nn.Embedding(n_items + 1, dim)
        nn.init.normal_(self.user.weight, std=init_std)
        nn.init.normal_(self.item.weight, std=init_std)


@register_method
class UltraGCN(EmbeddingRecommender):
    """UltraGCN: matrix factorisation whose loss weights encode infinite-layer graph convolution."""

    spec = MethodSpec(
        name="ultragcn",
        tasks=MF_TASKS,
        needs_torch=True,
        upstream="in-repo PyTorch, following the authors' code (reczoo/RecZoo UltraGCN, Apache-2.0)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.net = UltraGCNNet(self.n_users, self.n_items, int(cfg.get("dim", 64)), float(cfg.get("ultragcn_init", 1e-3))).to(self.device)
        seen = data.seen
        d_u = np.asarray(seen.sum(axis=1)).ravel().astype(np.float64)
        d_i = np.asarray(seen.sum(axis=0)).ravel().astype(np.float64)
        beta_u = torch.as_tensor(np.divide(np.sqrt(d_u + 1), d_u, out=np.zeros_like(d_u), where=d_u > 0), dtype=torch.float32, device=self.device)
        beta_i = torch.as_tensor(1.0 / np.sqrt(d_i + 1), dtype=torch.float32, device=self.device)
        ids, weights = item_neighbours(seen, int(cfg.get("ultragcn_neighbors", 10)))
        nb_ids, nb_w = torch.as_tensor(ids, device=self.device), torch.as_tensor(weights, device=self.device)
        w1, w2, w3, w4 = (float(cfg.get(f"ultragcn_w{n}", v)) for n, v in ((1, 1e-7), (2, 1.0), (3, 1e-7), (4, 1.0)))
        neg_weight, gamma, lam = (float(cfg.get("ultragcn_neg_weight", 200.0)), float(cfg.get("ultragcn_gamma", 1e-4)),
                                  float(cfg.get("ultragcn_lambda", 1e-3)))
        batch_size = int(cfg.get("batch_size", 1024))
        batches = edge_batches(data, batch_size, int(cfg.get("ultragcn_negatives", 200)), int(cfg.get("seed", 42)))

        def loss(batch):
            users, pos, neg = (to_tensor(batch[k], self.device) for k in ("users", "pos", "neg"))
            e_u, e_pos, e_neg = self.net.user(users), self.net.item(pos), self.net.item(neg)
            pos_score = (e_u * e_pos).sum(-1)
            neg_score = torch.einsum("bd,bnd->bn", e_u, e_neg)
            pos_w = w1 + w2 * beta_u[users] * beta_i[pos]
            neg_w = w3 + w4 * beta_u[users].unsqueeze(1) * beta_i[neg]
            pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score), weight=pos_w, reduction="none")
            neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score), weight=neg_w, reduction="none").mean(-1)
            l_o = (pos_loss + neg_weight * neg_loss).sum()
            neighbour_score = torch.einsum("bd,bkd->bk", e_u, self.net.item(nb_ids[pos]))
            l_i = -(nb_w[pos] * F.logsigmoid(neighbour_score)).sum()
            norm = sum((p**2).sum() for p in self.net.parameters()) / 2
            return (l_o + gamma * norm + lam * l_i) / len(users)

        per_epoch = steps_per_epoch(len(data._items), batch_size)
        self.fit_info = train_epochs(self.net, batches, per_epoch, loss, cfg, owner=self)

    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        return self.net.user.weight[to_tensor(users, self.device)]

    def item_matrix(self) -> torch.Tensor:
        return self.net.item.weight
