"""Matrix factorisation with modern losses: SimpleX (Mao et al. 2021) and DirectAU (Wang et al. 2022).

Both learn a vector per user and per item, like BPR-MF, but with better training objectives:
- SimpleX: a cosine contrastive loss with many negatives and a margin, and a user vector that mixes the
  user's own embedding with the average of their history items.
- DirectAU: no negatives at all. It pulls each user towards the items they interacted with ("alignment")
  and spreads all users and all items evenly over the unit sphere ("uniformity").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recbench.data import HistoryBatch, TrainView
from recbench.methods._torch import EmbeddingRecommender, edge_batches, resolve_device, steps_per_epoch, train_epochs
from recbench.methods.seq_trainer import to_tensor
from recbench.protocol import MethodSpec, Task
from recbench.registry import register_method

MF_TASKS = {Task.topn, Task.sequential, Task.similar_items}


class SimpleXNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, gamma: float, dropout: float):
        super().__init__()
        self.user = nn.Embedding(n_users + 1, dim)
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.gamma = gamma
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.user.weight, std=1e-4)  # the authors' initialiser
        nn.init.normal_(self.item.weight, std=1e-4)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def user_vector(self, users: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """gamma * e_u + (1 - gamma) * W_v(mean of the history embeddings); padding (item 0) is ignored."""
        emb = self.item(history)
        mask = (history > 0).unsqueeze(-1).to(emb.dtype)
        mean = (emb * mask).sum(1) / (mask.sum(1) + 1e-9)
        return F.normalize(self.gamma * self.user(users) + (1 - self.gamma) * self.w_v(mean), dim=-1)


@register_method
class SimpleX(EmbeddingRecommender):
    """SimpleX: cosine similarity, a cosine contrastive loss (CCL), and history-aware user vectors."""

    spec = MethodSpec(
        name="simplex",
        tasks=MF_TASKS,
        uses_history=True,
        needs_torch=True,
        upstream="in-repo PyTorch, following the authors' code (reczoo/RecZoo SimpleX, Apache-2.0)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.history_len = int(cfg.get("simplex_history", 50))
        self.net = SimpleXNet(self.n_users, self.n_items, int(cfg.get("dim", 64)), float(cfg.get("simplex_gamma", 0.5)),
                              float(cfg.get("dropout", 0.1))).to(self.device)
        histories = data.history_batch(np.arange(self.n_users + 1), self.history_len).items
        self.histories = torch.as_tensor(histories, device=self.device)
        n_neg, margin = int(cfg.get("simplex_negatives", 100)), float(cfg.get("simplex_margin", 0.8))
        neg_weight = float(cfg.get("simplex_neg_weight", 150.0))
        batch_size = int(cfg.get("batch_size", 512))
        batches = edge_batches(data, batch_size, n_neg, int(cfg.get("seed", 42)))

        def loss(batch):
            users = to_tensor(batch["users"], self.device)
            u = self.net.drop(self.net.user_vector(users, self.histories[users]))
            pos = F.normalize(self.net.item(to_tensor(batch["pos"], self.device)), dim=-1)
            neg = F.normalize(self.net.item(to_tensor(batch["neg"], self.device)), dim=-1)
            pos_score = (u * pos).sum(-1)
            neg_score = torch.einsum("bd,bnd->bn", u, neg)
            # CCL: pull positives to cosine 1; push negatives below the margin (weighted average over negatives).
            return (F.relu(1 - pos_score) + neg_weight * F.relu(neg_score - margin).mean(-1)).mean()

        per_epoch = steps_per_epoch(len(data._items), batch_size)
        self.fit_info = train_epochs(self.net, batches, per_epoch, loss, cfg, owner=self,
                                     weight_decay=float(cfg.get("simplex_l2", 0.0)))

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        items = to_tensor(hist.items[:, -self.history_len :], self.device)
        return self.net.user_vector(to_tensor(users, self.device), items)

    def item_matrix(self) -> torch.Tensor:
        return F.normalize(self.net.item.weight, dim=-1)


class DirectAUNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user = nn.Embedding(n_users + 1, dim)
        self.item = nn.Embedding(n_items + 1, dim)
        nn.init.xavier_normal_(self.user.weight)  # RecBole's xavier_normal_initialization, as in the authors' code
        nn.init.xavier_normal_(self.item.weight)


def alignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between matched (user, item) pairs of unit vectors."""
    return (x - y).norm(p=2, dim=1).pow(2).mean()


def uniformity(x: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """log E[exp(-t * ||x_a - x_b||^2)] over all pairs in the batch: low when vectors are spread evenly."""
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


@register_method
class DirectAU(EmbeddingRecommender):
    """DirectAU: matrix factorisation trained with alignment + gamma * uniformity, and no negative sampling."""

    spec = MethodSpec(
        name="directau",
        tasks=MF_TASKS,
        needs_torch=True,
        upstream="in-repo PyTorch, following the authors' code (THUwangcy/DirectAU, MIT)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.net = DirectAUNet(self.n_users, self.n_items, int(cfg.get("dim", 64))).to(self.device)
        gamma = float(cfg.get("directau_gamma", 1.0))
        batch_size = int(cfg.get("batch_size", 256))
        batches = edge_batches(data, batch_size, 0, int(cfg.get("seed", 42)))

        def loss(batch):
            u = F.normalize(self.net.user(to_tensor(batch["users"], self.device)), dim=-1)
            i = F.normalize(self.net.item(to_tensor(batch["pos"], self.device)), dim=-1)
            return alignment(u, i) + gamma * (uniformity(u) + uniformity(i)) / 2

        per_epoch = steps_per_epoch(len(data._items), batch_size)
        self.fit_info = train_epochs(self.net, batches, per_epoch, loss, cfg, owner=self,
                                     weight_decay=float(cfg.get("directau_l2", 1e-6)))

    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        return self.net.user.weight[to_tensor(users, self.device)]  # the authors score with the raw dot product

    def item_matrix(self) -> torch.Tensor:
        return self.net.item.weight
