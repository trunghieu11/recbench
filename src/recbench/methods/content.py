"""Content-based two-tower models: item vectors are computed from item content, not item ids.

text_hash_tower: item tower = linear layer over a hashed bag of words (title,
description, category tokens). Placeholder for LLM text encoders: same idea
(understand items from their text), far smaller model. Can score brand-new items.

multimodal_tower: the same text features plus simple image features (mean
colour and a 2x2 colour grid, read with Pillow) plus a zero-initialised item-id
residual. Runs only when real images are on disk (H&M with the image archive).

User tower (both): the mean of the item vectors of the user's recent history,
followed by a linear layer. Trained with next-item cross-entropy at every
position of pre-test windows (cumulative-mean user vectors).
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from recbench.data import HistoryBatch, TrainView
from recbench.methods._torch import EmbeddingRecommender, resolve_device, train_steps
from recbench.methods.seq_trainer import next_item_loss, sequence_windows, to_tensor
from recbench.protocol import MethodSpec, Task, Unsupported
from recbench.registry import register_method

TOKEN = re.compile(r"[a-z0-9]+")
CONTENT_TASKS = {Task.topn, Task.sequential, Task.similar_items}


def hashed_text_features(texts: np.ndarray, categories: np.ndarray, n_features: int = 2048) -> sp.csr_matrix:
    """Rows = items, columns = hash buckets of words and "cat:" tokens; L2-normalised term counts."""
    rows, cols, vals = [], [], []
    for row, (text, category) in enumerate(zip(texts, categories)):
        tokens = TOKEN.findall(str(text).lower()) + [f"cat:{c.strip().lower()}" for c in str(category).split("|") if c.strip()]
        if not tokens:
            continue
        buckets = np.array([zlib.crc32(t.encode()) % n_features for t in tokens])
        unique, counts = np.unique(buckets, return_counts=True)
        rows.append(np.full(len(unique), row))
        cols.append(unique)
        vals.append(counts / np.linalg.norm(counts))
    if not rows:
        return sp.csr_matrix((len(texts), n_features), dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(len(texts), n_features)
    )


def image_features(paths: np.ndarray, cache: Path) -> np.ndarray:
    """15 numbers per item: mean RGB + mean RGB of each 2x2 quadrant, in [0, 1]. Zeros when no image."""
    if cache.exists():
        return np.load(cache)
    from PIL import Image

    out = np.zeros((len(paths), 15), dtype=np.float32)
    for row, path in enumerate(paths):
        if not path or not Path(path).is_file():
            continue
        try:
            with Image.open(path) as img:
                pixels = np.asarray(img.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
        except OSError:
            continue
        quads = [pixels[:16, :16], pixels[:16, 16:], pixels[16:, :16], pixels[16:, 16:]]
        out[row] = np.concatenate([pixels.mean(axis=(0, 1))] + [q.mean(axis=(0, 1)) for q in quads])
    np.save(cache, out)
    return out


def _sparse_tensor(matrix: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    index = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.as_tensor(coo.data), coo.shape, device=device).coalesce()


class ContentTowerNet(nn.Module):
    def __init__(self, text: torch.Tensor, dense: torch.Tensor | None, n_items: int, dim: int, id_residual: bool):
        super().__init__()
        self.register_buffer("text", text)
        self.register_buffer("dense", dense if dense is not None else torch.zeros(n_items + 1, 0))
        self.text_proj = nn.Linear(text.shape[1], dim)
        self.dense_proj = nn.Linear(self.dense.shape[1], dim, bias=False) if self.dense.shape[1] else None
        self.id = nn.Embedding(n_items + 1, dim, padding_idx=0) if id_residual else None
        if self.id is not None:
            nn.init.zeros_(self.id.weight)
        self.user_proj = nn.Linear(dim, dim)

    def item_matrix(self) -> torch.Tensor:
        vectors = torch.sparse.mm(self.text, self.text_proj.weight.T) + self.text_proj.bias
        if self.dense_proj is not None:
            vectors = vectors + self.dense_proj(self.dense)
        if self.id is not None:
            vectors = vectors + self.id.weight
        return vectors

    def users(self, seq: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Cumulative mean of history item vectors at every position, then a linear layer: [B, L, d]."""
        keep = (seq > 0).unsqueeze(-1).to(items.dtype)
        summed = torch.cumsum(items[seq] * keep, dim=1)
        count = torch.cumsum(keep, dim=1).clamp(min=1.0)
        return self.user_proj(summed / count) * keep


class _ContentTower(EmbeddingRecommender):
    use_images = False

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.device = resolve_device(cfg)
        self.seq_len = int(cfg.get("seq_len", 50))
        text = hashed_text_features(data.item_text, data.item_category, int(cfg.get("hash_features", 2048)))
        if text.nnz == 0:
            raise Unsupported(f"{self.spec.name}: items have no text or categories")
        dense = None
        if self.use_images:
            feats = image_features(data.item_image_path, data.cache_dir / "image_features.npy")
            if not feats.any():
                raise Unsupported("multimodal_tower needs real item images on disk")
            dense = torch.as_tensor(feats)
        self.net = ContentTowerNet(_sparse_tensor(text, self.device), dense, self.n_items, int(cfg.get("dim", 64)), self.use_images).to(self.device)
        windows = sequence_windows(data, self.seq_len, int(cfg.get("batch_size", 128)), int(cfg.get("seed", 42)))

        def loss(batch):
            items = self.net.item_matrix()
            hidden = self.net.users(to_tensor(batch["inputs"], self.device), items)
            return next_item_loss(hidden, items, to_tensor(batch["targets"], self.device))

        self.fit_info = train_steps(self.net, windows, loss, cfg)
        with torch.no_grad():
            self._items = self.net.item_matrix().detach()

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        seq = to_tensor(hist.items[:, -self.seq_len :], self.device)
        return self.net.users(seq, self._items)[:, -1]

    def item_matrix(self) -> torch.Tensor:
        return self._items


@register_method
class TextHashTower(_ContentTower):
    spec = MethodSpec(
        name="text_hash_tower",
        tasks=CONTENT_TASKS,
        uses_history=True,
        scores_cold_items=True,
        requires_side_features=True,
        needs_torch=True,
        fidelity="simplified",
        upstream="in-repo content two-tower over hashed bag-of-words (stand-in for LLM text encoders)",
        cost_band="low",
    )


@register_method
class MultimodalTower(_ContentTower):
    use_images = True
    spec = MethodSpec(
        name="multimodal_tower",
        tasks=CONTENT_TASKS,
        uses_history=True,
        scores_cold_items=True,
        requires_side_features=True,
        requires_images=True,
        needs_torch=True,
        fidelity="simplified",
        upstream="in-repo two-tower over hashed text + colour image features + id residual",
        cost_band="medium",
    )
