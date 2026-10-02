"""DCN-V2 cross network plus a deep tower. Trains on the shared candidate labels.

Reference: FuxiCTR DCN-V2.
"""

from __future__ import annotations

import torch
from fuxictr.pytorch.layers.interactions.cross_net import CrossNetV2
from torch import nn
from torch.nn import functional as F

from recbench.methods._common import TorchMethod
from recbench.protocol import Explanation, MethodSpec, Task
from recbench.registry import register_method


class _Net(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int):
        super().__init__()
        self.user = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        width = dim * 2
        self.cross = CrossNetV2(width, max(layers, 1))
        self.deep = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, dim))
        self.head = nn.Linear(width + dim, 1)

    def features(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.user(users), self.item(items)], dim=-1)

    def logits(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        x0 = self.features(users, items)
        crossed = self.cross(x0)
        deep = self.deep(x0)
        return self.head(torch.cat([crossed, deep], dim=-1)).squeeze(-1)

    def bce(self, users: torch.Tensor, items: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.logits(users, items), labels)


@register_method
class DCNV2(TorchMethod):
    spec = MethodSpec(
        name="dcnv2",
        tasks={Task.topn, Task.ctr, Task.rating},
        feedback={"implicit", "explicit"},
        requires_side_features=True,
        cost_band="medium",
        upstream="FuxiCTR CrossNetV2 inside DCNv2, trained on the shared candidate file",
    )
    loss = "bce"

    def build_model(self, n_users: int, n_items: int, cfg):
        return _Net(n_users, n_items, cfg.dim, cfg.layers)

    def loss_batch(self, model, batch, cfg) -> torch.Tensor:
        return model.bce(batch["users"], batch["items"], batch["labels"])

    def score_batch(self, model, users, items, history) -> torch.Tensor:
        return model.logits(users, items)

    def _explain(self, user_id: str, item_id: str, cold: bool = False) -> Explanation:
        return Explanation(
            "local",
            f"DCN-V2 crosses the user and item features of {user_id} for {item_id}.",
            [{"feature": "user x item", "item_id": item_id}],
        )
