"""TIGER residual quantization plus a score for every candidate id.

The author repository URLs that were tried returned 404, so this module follows
the paper's RQ-VAE (Rajput et al., NeurIPS 2023) and scores candidate ids.
"""

from __future__ import annotations

import torch
from torch import nn

from recbench.methods._common import TorchMethod, sampled_softmax
from recbench.protocol import Explanation, MethodSpec, Task
from recbench.registry import register_method


class _Net(nn.Module):
    def __init__(self, n_items: int, dim: int, seq_len: int):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.books = nn.Parameter(torch.randn(4, 16, dim) * 0.02)
        self.n_items = n_items
        self.seq_len = seq_len

    def quantize(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = vectors
        recon = torch.zeros_like(vectors)
        codes = []
        for book in range(self.books.shape[0]):
            table = self.books[book]
            distance = torch.cdist(residual, table)
            index = distance.argmin(dim=-1)
            chosen = table[index]
            recon = recon + chosen
            residual = residual - chosen
            codes.append(index)
        return recon, torch.stack(codes, dim=-1)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        emb = self.item(history)
        out, _ = self.gru(emb)
        lengths = (history != 0).sum(dim=1).clamp(min=1) - 1
        return out[torch.arange(out.shape[0], device=history.device), lengths]

    def sequence_loss(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(history)
        ce = sampled_softmax(hidden, self.item.weight, target, self.n_items)
        recon, _ = self.quantize(self.item(target))
        rq = torch.nn.functional.mse_loss(recon, self.item(target).detach())
        return ce + 0.1 * rq

    def score(self, history: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(history)
        return (hidden * self.item(items)).sum(-1)


@register_method
class TIGER(TorchMethod):
    spec = MethodSpec(
        name="tiger",
        tasks={Task.topn, Task.sequential},
        feedback={"implicit", "explicit"},
        cost_band="medium",
        upstream="TIGER RQ-VAE (Rajput et al., NeurIPS 2023). Author git URLs returned 404.",
    )
    loss = "seq"

    def build_model(self, n_users: int, n_items: int, cfg):
        return _Net(n_items, cfg.dim, cfg.seq_len)

    def loss_batch(self, model, batch, cfg) -> torch.Tensor:
        return model.sequence_loss(batch["history"], batch["items"])

    def score_batch(self, model, users, items, history) -> torch.Tensor:
        return model.score(history, items)

    def _explain(self, user_id: str, item_id: str, cold: bool = False) -> Explanation:
        return Explanation(
            "local",
            f"Semantic codes for {item_id} are the nearest residual-quantized path for {user_id}.",
            [{"semantic_id_item": item_id}],
        )
