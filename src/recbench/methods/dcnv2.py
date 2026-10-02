"""DCN-V2 (Wang et al., WWW 2021): a CTR-style ranker that learns explicit feature crosses.

Input features: user id, item id, and the item's category tokens (averaged
embeddings). A cross network (FuxiCTR's CrossNetV2) models feature
interactions x_{l+1} = x_0 * (W_l x_l + b_l) + x_l, a deep MLP runs in parallel,
and a linear head outputs the logit of "this user interacts with this item".

Training: binary cross-entropy on pre-test interactions (label 1) and randomly
sampled unseen items (label 0), 4 negatives per positive. It never reads the
evaluation candidates. As a pointwise ranker it must score every (user, item)
pair for full ranking, which is why it is slower to evaluate than embedding models.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recbench.data import HistoryBatch, TrainView
from recbench.methods._explain import embedding_explanations
from recbench.methods._torch import edge_batches, resolve_device, train_steps
from recbench.protocol import MethodSpec, Recommender, Task, Unsupported
from recbench.registry import register_method

CATEGORY_BUCKETS = 4096
MAX_TOKENS = 8


def category_index(categories: np.ndarray) -> np.ndarray:
    """[n_items + 1, MAX_TOKENS] hashed category tokens per item; 0 = padding."""
    out = np.zeros((len(categories), MAX_TOKENS), dtype=np.int64)
    for row, cat in enumerate(categories):
        tokens = [t for t in str(cat).split("|") if t][:MAX_TOKENS]
        out[row, : len(tokens)] = [1 + zlib.crc32(t.encode()) % (CATEGORY_BUCKETS - 1) for t in tokens]
    return out


class DCNNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int, categories: np.ndarray):
        super().__init__()
        from fuxictr.pytorch.layers.interactions.cross_net import CrossNetV2

        self.user = nn.Embedding(n_users + 1, dim)
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.category = nn.Embedding(CATEGORY_BUCKETS, dim, padding_idx=0)
        self.register_buffer("cat_index", torch.as_tensor(categories))
        width = 3 * dim
        self.cross = CrossNetV2(width, max(layers, 1))
        self.deep = nn.Sequential(nn.Linear(width, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim), nn.ReLU())
        self.head = nn.Linear(width + dim, 1)

    def logits(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        tokens = self.cat_index[items]
        present = (tokens > 0).unsqueeze(-1).float()
        cat = (self.category(tokens) * present).sum(-2) / present.sum(-2).clamp(min=1.0)
        x0 = torch.cat([self.user(users), self.item(items), cat], dim=-1)
        return self.head(torch.cat([self.cross(x0), self.deep(x0)], dim=-1)).squeeze(-1)


@register_method
class DCNV2(Recommender):
    spec = MethodSpec(
        name="dcnv2",
        tasks={Task.topn, Task.ctr},
        output="pairs",
        outputs_probability=True,
        needs_torch=True,
        upstream="FuxiCTR 2.3 CrossNetV2 inside an in-repo DCN-V2 (ids + category tokens)",
        cost_band="medium",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        try:
            self.net = DCNNet(self.n_users, self.n_items, int(cfg.get("dim", 32)), int(cfg.get("layers", 2)), category_index(data.item_category))
        except ImportError as exc:
            raise Unsupported("fuxictr is not installed") from exc
        self.net = self.net.to(self.device)
        n_neg = int(cfg.get("dcn_negatives", 4))
        batches = edge_batches(data, int(cfg.get("batch_size", 128)) * 4, n_neg, int(cfg.get("seed", 42)))
        as_t = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)  # noqa: E731

        def loss(batch):
            users = as_t(batch["users"])
            pos = self.net.logits(users, as_t(batch["pos"]))
            neg = self.net.logits(users[:, None].expand(-1, n_neg), as_t(batch["neg"]))
            return F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + F.binary_cross_entropy_with_logits(
                neg, torch.zeros_like(neg)
            ) * n_neg / (n_neg + 1)

        self.fit_info = train_steps(self.net, batches, loss, cfg)

    @torch.no_grad()
    def score_pairs(self, users: np.ndarray, items: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        u = torch.as_tensor(users, dtype=torch.long, device=self.device)[:, None].expand(-1, items.shape[1])
        i = torch.as_tensor(items, dtype=torch.long, device=self.device)
        return self.net.logits(u, i).float().cpu().numpy()

    def item_embeddings(self) -> np.ndarray:
        return self.net.item.weight.detach().float().cpu().numpy()

    def explain(self, users, items, hist):
        return embedding_explanations(self.item_embeddings(), users, items, hist, self.item_ids, "DCN-V2")
