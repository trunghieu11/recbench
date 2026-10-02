"""HSTU, the Hierarchical Sequential Transduction Unit (Zhai et al., ICML 2024, "Actions Speak Louder than Words").

One HSTU layer, following the paper's equations:
    U, V, Q, K = split(SiLU(f1(LayerNorm(X))))
    A          = SiLU(Q K^T + rab(positions, timestamps)) / n      (pointwise, NOT softmax)
    Y          = f2(LayerNorm(A V) * U)                            (U gates the attention output)
    X'         = X + Dropout(Y)
rab is a learned relative attention bias over position distance and log time gaps.

The attention core matches Meta's reference op `pytorch_hstu_mha`
(third_party/generative-recommenders, commit ea7b85f); tests/test_methods.py
checks the two agree. This is a small single-GPU model, not Meta's production
system (no jagged kernels, no multi-task heads, plain dot-product scoring).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recbench.data import HistoryBatch, TrainView
from recbench.methods._torch import EmbeddingRecommender, resolve_device, train_steps
from recbench.methods.seq_trainer import next_item_loss, sequence_windows, to_tensor
from recbench.methods.sasrec import SEQ_TASKS
from recbench.protocol import MethodSpec
from recbench.registry import register_method


def hstu_attention(q, k, v, valid_keys, rab=None):
    """Pointwise SiLU attention. q, k, v: [B, H, L, d]; valid_keys: [B, L] bool; rab: [B, L, L] or None."""
    length = q.shape[2]
    scores = q @ k.transpose(-1, -2)
    if rab is not None:
        scores = scores + rab[:, None]
    weights = F.silu(scores) / length
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    mask = causal[None, None] & valid_keys[:, None, None, :]
    return (weights * mask) @ v


class RelativeBias(nn.Module):
    """Learned bias for (position distance, log2 time gap) of every query-key pair."""

    def __init__(self, seq_len: int, n_time_buckets: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.n_time_buckets = n_time_buckets
        self.position = nn.Parameter(torch.zeros(2 * seq_len - 1))
        self.time = nn.Parameter(torch.zeros(n_time_buckets + 1))

    def forward(self, times_us: torch.Tensor) -> torch.Tensor:
        length = times_us.shape[1]
        idx = torch.arange(length, device=times_us.device)
        position = self.position[(idx[:, None] - idx[None, :]) + self.seq_len - 1]
        gap_seconds = (times_us[:, :, None] - times_us[:, None, :]).abs().float() / 1e6
        bucket = (torch.log2(gap_seconds.clamp(min=1.0))).long().clamp(0, self.n_time_buckets)
        return position[None] + self.time[bucket]


class HSTULayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm_in = nn.LayerNorm(dim)
        self.uvqk = nn.Linear(dim, 4 * dim)
        self.norm_attn = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, rab: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        u, v, q, k = F.silu(self.uvqk(self.norm_in(x))).split(dim, dim=-1)
        heads = lambda t: t.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)  # noqa: E731
        attended = hstu_attention(heads(q), heads(k), heads(v), valid, rab)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return x + self.drop(self.out(self.norm_attn(attended) * u))


class HSTUNet(nn.Module):
    def __init__(self, n_items: int, dim: int, layers: int, heads: int, seq_len: int, dropout: float):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.bias = RelativeBias(seq_len)
        self.layers = nn.ModuleList(HSTULayer(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        valid = seq > 0
        keep = valid.unsqueeze(-1).to(self.item.weight.dtype)
        x = self.drop(self.item(seq)) * keep
        rab = self.bias(times)
        for layer in self.layers:
            x = layer(x, valid, rab) * keep
        return self.norm(x) * keep


@register_method
class HSTU(EmbeddingRecommender):
    spec = MethodSpec(
        name="hstu",
        tasks=SEQ_TASKS,
        uses_history=True,
        needs_torch=True,
        fidelity="simplified",
        upstream="in-repo dense HSTU; attention verified against Meta generative-recommenders pytorch_hstu_mha @ ea7b85f",
        cost_band="medium",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.seq_len = int(cfg.get("seq_len", 50))
        self.net = HSTUNet(
            self.n_items,
            int(cfg.get("dim", 64)),
            int(cfg.get("layers", 2)),
            int(cfg.get("heads", 2)),
            self.seq_len,
            float(cfg.get("dropout", 0.2)),
        ).to(self.device)
        windows = sequence_windows(data, self.seq_len, int(cfg.get("batch_size", 128)), int(cfg.get("seed", 42)))

        def loss(batch):
            hidden = self.net(to_tensor(batch["inputs"], self.device), to_tensor(batch["times"], self.device))
            return next_item_loss(hidden, self.net.item.weight, to_tensor(batch["targets"], self.device))

        self.fit_info = train_steps(self.net, windows, loss, cfg)

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        items = to_tensor(hist.items[:, -self.seq_len :], self.device)
        times = to_tensor(hist.times[:, -self.seq_len :], self.device)
        return self.net(items, times)[:, -1]

    def item_matrix(self) -> torch.Tensor:
        return self.net.item.weight
