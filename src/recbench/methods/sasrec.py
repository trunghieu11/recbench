"""SASRec (Kang & McAuley, 2018): a causal self-attention Transformer over the user's item sequence.

Trained with full (or sampled) softmax cross-entropy at every position, which
Klenitskiy & Vasilev (2023) showed is much stronger than the original
binary cross-entropy with one negative. The user representation is the hidden
state at the last real position of the right-aligned history.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from recbench.data import HistoryBatch, TrainView
from recbench.methods._torch import EmbeddingRecommender, resolve_device, steps_per_epoch, train_epochs
from recbench.methods.seq_trainer import next_item_loss, sequence_windows, to_tensor
from recbench.protocol import MethodSpec, Task
from recbench.registry import register_method

SEQ_TASKS = {Task.topn, Task.sequential, Task.session, Task.similar_items}


def causal_padding_mask(seq: torch.Tensor, heads: int) -> torch.Tensor:
    """Boolean attention mask (True = blocked): no looking ahead, no attending to padding keys.

    Each position may always attend to itself, so no row is fully blocked (that would produce NaN).
    """
    _, length = seq.shape
    causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=seq.device), diagonal=1)
    blocked = causal[None] | (seq == 0)[:, None, :]
    blocked &= ~torch.eye(length, dtype=torch.bool, device=seq.device)[None]
    return blocked.repeat_interleave(heads, dim=0)


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, dim: int, layers: int, heads: int, seq_len: int, dropout: float):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = dim**0.5

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        keep = (seq > 0).unsqueeze(-1).to(self.item.weight.dtype)
        x = self.item(seq) * self.scale + self.pos.weight[: seq.shape[1]]
        x = self.drop(x) * keep
        mask = causal_padding_mask(seq, self.heads)
        for layer in self.layers:
            x = layer(x, src_mask=mask) * keep
        return self.norm(x) * keep


@register_method
class SASRec(EmbeddingRecommender):
    spec = MethodSpec(
        name="sasrec",
        sequence_aware=True,
        tasks=SEQ_TASKS,
        uses_history=True,
        needs_torch=True,
        upstream="in-repo PyTorch, following Kang & McAuley 2018 with full cross-entropy (Klenitskiy & Vasilev 2023)",
        cost_band="medium",
        impl_version="4",  # 2: epochs with early stopping, bf16, loss setting; 3: epochs sized by events; 4: a final run that blows up keeps its best weights
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.seq_len = int(cfg.get("seq_len", 50))
        self.net = SASRecNet(
            self.n_items,
            int(cfg.get("dim", 64)),
            int(cfg.get("layers", 2)),
            int(cfg.get("heads", 2)),
            self.seq_len,
            float(cfg.get("dropout", 0.2)),
        ).to(self.device)
        batch_size = int(cfg.get("batch_size", 128))
        windows = sequence_windows(data, self.seq_len, batch_size, int(cfg.get("seed", 42)))
        mode, n_neg = str(cfg.get("sasrec_loss", "auto")), int(cfg.get("n_negatives", 1024))

        def loss(batch):
            hidden = self.net(to_tensor(batch["inputs"], self.device))
            return next_item_loss(hidden, self.net.item.weight, to_tensor(batch["targets"], self.device), mode=mode, n_negatives=n_neg)

        # One epoch = enough random windows for every event to be a target about once (each window has seq_len
        # targets), and at least one window per user with two or more events. Sizing by users alone gave tiny
        # epochs on datasets with few, long histories (MovieLens quick: 14 batches; Last.fm: 4).
        n_users = int((data.user_lengths >= 2).sum())
        per_epoch = max(steps_per_epoch(n_users, batch_size), steps_per_epoch(len(data._items), batch_size * self.seq_len))
        self.fit_info = train_epochs(self.net, windows, per_epoch, loss, cfg, owner=self)

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        items = hist.items[:, -self.seq_len :]
        return self.net(to_tensor(items, self.device))[:, -1]

    def item_matrix(self) -> torch.Tensor:
        return self.net.item.weight
