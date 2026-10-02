"""Two-tower scorer over hashed text and the mean color of the article image.

Runs only when the split has image files. H&M smoke writes those files for the slice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from recbench.methods._common import TorchMethod
from recbench.protocol import Explanation, MethodSpec, Task, Unsupported
from recbench.registry import register_method


def _color(path: str) -> tuple[float, float, float]:
    file = Path(path)
    if not file.is_file():
        return 0.0, 0.0, 0.0
    raw = file.read_bytes()
    if raw.startswith(b"P6"):
        body = raw.split(b"\n", 3)[-1]
        if len(body) >= 3:
            return body[0] / 255, body[1] / 255, body[2] / 255
    return 0.0, 0.0, 0.0


def _hash_vec(text: str, dim: int) -> np.ndarray:
    import hashlib

    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split() or ["empty"]:
        digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[digest % dim] += 1.0
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm


class _Net(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, content: torch.Tensor):
        super().__init__()
        self.user = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.register_buffer("content", content)
        self.proj = nn.Linear(dim + 3, dim)

    def item_vec(self, items: torch.Tensor) -> torch.Tensor:
        return self.item(items) + self.proj(self.content[items])

    def bpr(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        user = self.user(users)
        pos_s = (user * self.item_vec(pos)).sum(-1)
        neg_s = (user * self.item_vec(neg)).sum(-1)
        return -F.logsigmoid(pos_s - neg_s).mean()

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item_vec(items)).sum(-1)


@register_method
class MultimodalTower(TorchMethod):
    spec = MethodSpec(
        name="multimodal_tower",
        tasks={Task.topn, Task.similar_items},
        feedback={"implicit", "explicit"},
        requires_images=True,
        requires_side_features=True,
        cost_band="medium",
        upstream="In-repo two-tower over text hashes and image color",
    )
    loss = "bpr"

    def build_model(self, n_users: int, n_items: int, cfg):
        content = torch.zeros(n_items + 1, cfg.dim + 3)
        if getattr(self, "store", None) is not None:
            items = pq.read_table(self.store.items_path, columns=["item_idx", "text", "image_path"]).to_pandas()
            for idx, text, path in zip(items["item_idx"], items["text"], items["image_path"]):
                vec = _hash_vec(text, cfg.dim)
                color = _color("" if path is None else str(path))
                content[int(idx), : cfg.dim] = torch.tensor(vec)
                content[int(idx), cfg.dim :] = torch.tensor(color)
        return _Net(n_users, n_items, cfg.dim, content)

    def fit(self, store, config):
        self.store = store
        if not store.images_available():
            raise Unsupported("multimodal_tower needs image files")
        super().fit(store, config)

    def loss_batch(self, model, batch, cfg) -> torch.Tensor:
        return model.bpr(batch["users"], batch["pos"], batch["neg"])

    def score_batch(self, model, users, items, history) -> torch.Tensor:
        return model.score(users, items)

    def _explain(self, user_id: str, item_id: str, cold: bool = False) -> Explanation:
        return Explanation(
            "local",
            f"The image and text tower for {item_id} matches the history embedding of {user_id}.",
            [{"modality": "image+text", "item_id": item_id}],
        )
