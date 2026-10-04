"""TIGER-lite: a GRU next-item model plus TIGER-style residual-quantised "semantic IDs". Experimental, unranked.

What real TIGER does (Rajput et al., NeurIPS 2023): an RQ-VAE turns each item's
content embedding into a short tuple of codes (a "semantic ID"); a Transformer
encoder-decoder then *generates* the next item's code tuple token by token with
beam search. No official code was released.

What this module does: learns item embeddings and a GRU over the history
(trained with next-item cross-entropy), and in parallel learns a 3-level
residual codebook over those item embeddings. Scores are dot products, so the
codes do not drive ranking; they are used to explain recommendations ("shares
the semantic-ID prefix of items you liked"). The docs page explains the gap.
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
from recbench.protocol import Explanation, MethodSpec, Task
from recbench.registry import register_method


class ResidualQuantizer(nn.Module):
    """levels x codes codebook. Code k at level l approximates what levels < l left unexplained."""

    def __init__(self, dim: int, levels: int = 3, codes: int = 64):
        super().__init__()
        self.books = nn.Parameter(torch.randn(levels, codes, dim) * 0.1)

    def forward(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = vectors
        recon = torch.zeros_like(vectors)
        codes = []
        for book in self.books:
            index = torch.cdist(residual, book).argmin(dim=-1)
            chosen = book[index]
            recon = recon + chosen
            residual = residual - chosen.detach()
            codes.append(index)
        return recon, torch.stack(codes, dim=-1)


class TigerLiteNet(nn.Module):
    def __init__(self, n_items: int, dim: int):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.rq = ResidualQuantizer(dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.item(seq))
        return out


@register_method
class TigerLite(EmbeddingRecommender):
    spec = MethodSpec(
        name="tiger_lite",
        sequence_aware=True,
        tasks={Task.topn, Task.sequential},
        uses_history=True,
        needs_torch=True,
        ranked=False,
        fidelity="simplified",
        upstream="in-repo GRU + residual codebook inspired by TIGER (no official code); experimental",
        cost_band="medium",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.seq_len = int(cfg.get("seq_len", 50))
        self.net = TigerLiteNet(self.n_items, int(cfg.get("dim", 64))).to(self.device)
        windows = sequence_windows(data, self.seq_len, int(cfg.get("batch_size", 128)), int(cfg.get("seed", 42)))

        def loss(batch):
            hidden = self.net(to_tensor(batch["inputs"], self.device))
            targets = to_tensor(batch["targets"], self.device)
            ce = next_item_loss(hidden, self.net.item.weight, targets)
            vectors = self.net.item(targets[targets > 0]).detach()
            recon, _ = self.net.rq(vectors)
            return ce + 0.1 * F.mse_loss(recon, vectors)

        self.fit_info = train_steps(self.net, windows, loss, cfg)
        with torch.no_grad():
            _, codes = self.net.rq(self.net.item.weight)
        self.codes = codes.cpu().numpy()

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        return self.net(to_tensor(hist.items[:, -self.seq_len :], self.device))[:, -1]

    def item_matrix(self) -> torch.Tensor:
        return self.net.item.weight

    def explain(self, users, items, hist):
        out = []
        for b, row in enumerate(items):
            history = np.unique(hist.items[b][hist.items[b] > 0])
            exps = []
            for item in row:
                if item <= 0 or len(history) == 0:
                    exps.append(Explanation("none", "No history to explain from."))
                    continue
                target = self.codes[item]
                prefix = np.array([np.argmin(np.append(self.codes[h] == target, False)) for h in history])
                best = history[np.argsort(-prefix)[:2]]
                best = [int(h) for h in best if prefix[list(history).index(h)] > 0]
                if not best:
                    exps.append(Explanation("none", f"Semantic ID {tuple(target)} shares no prefix with your history."))
                    continue
                exps.append(
                    Explanation(
                        "personal",
                        f"Semantic ID {tuple(int(c) for c in target)} shares a code prefix with {', '.join(self.item_ids[h] for h in best)}.",
                        [{"history_item": str(self.item_ids[h]), "semantic_id": [int(c) for c in self.codes[h]]} for h in best],
                    )
                )
            out.append(exps)
        return out
