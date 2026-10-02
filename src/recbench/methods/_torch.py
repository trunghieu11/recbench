"""Shared PyTorch helpers: device choice, negative sampling, a step-budgeted training loop."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator

import numpy as np
import scipy.sparse as sp
import torch

from recbench.data import HistoryBatch, TrainView
from recbench.methods._explain import embedding_explanations
from recbench.protocol import NEG_INF, Recommender


def resolve_device(cfg: dict[str, Any]) -> torch.device:
    """'auto' -> CUDA when available, else CPU. 'mps' (Apple GPU) only when asked for explicitly."""
    choice = str(cfg.get("device", "auto"))
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if choice == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(choice)


def warm_items(data: TrainView) -> np.ndarray:
    items = np.flatnonzero(data.item_pop > 0)
    return items[items > 0]


def sample_negatives(
    seen: sp.csr_matrix,
    users: np.ndarray,
    n: int,
    pool: np.ndarray,
    rng: np.random.Generator,
    rounds: int = 3,
) -> np.ndarray:
    """n negatives per user drawn uniformly from `pool`, re-drawn (a few times) when the user has seen the item."""
    negs = pool[rng.integers(0, len(pool), size=(len(users), n))]
    for _ in range(rounds):
        hit = np.asarray(seen[np.repeat(users, n), negs.ravel()]).reshape(len(users), n) > 0
        if not hit.any():
            break
        negs[hit] = pool[rng.integers(0, len(pool), size=int(hit.sum()))]
    return negs


def edge_batches(data: TrainView, batch_size: int, n_neg: int, seed: int) -> Iterator[dict[str, np.ndarray]]:
    """Endless (user, positive, negatives) batches from pre-test events, sampled uniformly over events."""
    events = data.events()
    users = events["user_idx"].to_numpy()
    items = events["item_idx"].to_numpy()
    pool = warm_items(data)
    seen = data.seen
    rng = np.random.default_rng(seed)
    while True:
        pick = rng.integers(0, len(users), size=batch_size)
        batch_users = users[pick]
        yield {
            "users": batch_users,
            "pos": items[pick],
            "neg": sample_negatives(seen, batch_users, n_neg, pool, rng),
        }


def train_steps(
    model: torch.nn.Module,
    batches: Iterator[Any],
    loss_fn: Callable[[Any], torch.Tensor],
    cfg: dict[str, Any],
    lr: float | None = None,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    """Adam for cfg['max_steps'] steps. Returns a small training log for MLflow params."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr or float(cfg.get("lr", 1e-3)), weight_decay=weight_decay)
    steps = int(cfg.get("max_steps", 400))
    model.train()
    began = time.perf_counter()
    first = last = None
    for step, batch in zip(range(steps), batches):
        loss = loss_fn(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        first = value if first is None else first
        last = value
    model.eval()
    return {"steps": steps, "first_loss": first or 0.0, "last_loss": last or 0.0, "train_loop_seconds": time.perf_counter() - began}


class EmbeddingRecommender(Recommender):
    """Base for models that end with score(u, j) = <user vector, item vector>."""

    device: torch.device = torch.device("cpu")

    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        raise NotImplementedError

    def item_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        scores = self.user_vectors(users, hist) @ self.item_matrix().T
        out = scores.float().cpu().numpy()
        out[:, 0] = NEG_INF
        return out

    @torch.no_grad()
    def item_embeddings(self) -> np.ndarray:
        return self.item_matrix().float().cpu().numpy()

    def explain(self, users, items, hist):
        return embedding_explanations(self.item_embeddings(), users, items, hist, self.item_ids, self.spec.name)
