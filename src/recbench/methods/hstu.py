"""HSTU attention from Meta's generative-recommenders PyTorch op.

Commit: third_party/generative-recommenders ea7b85f16647766cbd7188a54bb447b20cee14bd
CPU stands in for the fbgemm jagged kernels so the official attention math still runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from recbench.methods._common import TorchMethod, sampled_softmax
from recbench.protocol import Explanation, MethodSpec, Task
from recbench.registry import register_method

_REPO = Path(__file__).resolve().parents[3] / "third_party" / "generative-recommenders"


def _install_fbgemm_shim() -> None:
    def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        offset = offsets[0]
        length = int(max_lengths[0])
        batch = offset.numel() - 1
        out = values.new_full((batch, length, *values.shape[1:]), padding_value)
        for row in range(batch):
            start, stop = int(offset[row]), int(offset[row + 1])
            out[row, : stop - start] = values[start:stop]
        return out

    def dense_to_jagged(dense, offsets, total):
        offset = offsets[0]
        pieces = []
        for row in range(offset.numel() - 1):
            start, stop = int(offset[row]), int(offset[row + 1])
            pieces.append(dense[row, : stop - start])
        packed = torch.cat(pieces, dim=0) if pieces else dense.new_zeros((0, dense.shape[-1]))
        return (packed,)

    torch.ops.fbgemm.jagged_to_padded_dense = jagged_to_padded_dense
    torch.ops.fbgemm.dense_to_jagged = dense_to_jagged


def _hstu_attention():
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    _install_fbgemm_shim()
    from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_mha

    return pytorch_hstu_mha


class _Net(nn.Module):
    def __init__(self, n_items: int, dim: int, layers: int, seq_len: int):
        super().__init__()
        heads = 2 if dim % 2 == 0 else 1
        self.heads = heads
        self.head_dim = dim // heads
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.layers = max(layers, 1)
        self.seq_len = seq_len
        self.n_items = n_items
        self.attend = _hstu_attention()

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.item(history)
        batch, length, dim = hidden.shape
        flat = hidden.reshape(batch * length, self.heads, self.head_dim)
        offsets = torch.arange(0, (batch + 1) * length, length, device=history.device)
        for _ in range(self.layers):
            query, key, value = self.q(hidden), self.k(hidden), self.v(hidden)
            query = query.reshape(batch * length, self.heads, self.head_dim)
            key = key.reshape(batch * length, self.heads, self.head_dim)
            value = value.reshape(batch * length, self.heads, self.head_dim)
            mixed = self.attend(length, 1.0, query, key, value, offsets, causal=True, training=self.training)
            mixed = self.out(mixed.reshape(batch, length, dim))
            hidden = hidden + mixed
            flat = hidden.reshape(batch * length, self.heads, self.head_dim)
        lengths = (history != 0).sum(dim=1).clamp(min=1) - 1
        return hidden[torch.arange(batch, device=history.device), lengths]

    def sequence_loss(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sampled_softmax(self.encode(history), self.item.weight, target, self.n_items)

    def score(self, history: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.encode(history) * self.item(items)).sum(-1)


@register_method
class HSTU(TorchMethod):
    spec = MethodSpec(
        name="hstu",
        tasks={Task.topn, Task.sequential, Task.session},
        feedback={"implicit", "explicit"},
        cost_band="medium",
        upstream="Meta generative-recommenders pytorch_hstu_mha @ ea7b85f",
    )
    loss = "seq"

    def build_model(self, n_users: int, n_items: int, cfg):
        return _Net(n_items, cfg.dim, cfg.layers, cfg.seq_len)

    def loss_batch(self, model, batch, cfg) -> torch.Tensor:
        return model.sequence_loss(batch["history"], batch["items"])

    def score_batch(self, model, users, items, history) -> torch.Tensor:
        return model.score(history, items)

    def _explain(self, user_id: str, item_id: str, cold: bool = False) -> Explanation:
        return Explanation("local", f"HSTU attention over {user_id}'s history scores {item_id}.", [{"item_id": item_id}])
