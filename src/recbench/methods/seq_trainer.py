"""Training data and loss for sequence models (SASRec, HSTU, TIGER-lite, two-tower text models).

Leak-free by construction: windows are cut only from each user's PRE-TEST
history (TrainView). For a window s_1..s_n the model reads s_1..s_{n-1} and
must predict s_2..s_n at every position ("shifted targets"), exactly like
next-word prediction in a language model.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from recbench.data import TrainView


def sequence_windows(data: TrainView, seq_len: int, batch_size: int, seed: int) -> Iterator[dict[str, np.ndarray]]:
    """Endless batches of right-aligned windows.

    Users with at least 2 pre-test events are sampled uniformly; for each, a random
    end point t is chosen so that long histories contribute windows from their whole
    span (not only the most recent tail).
    """
    lengths = data.user_lengths
    eligible = np.flatnonzero(lengths >= 2)
    if len(eligible) == 0:
        raise ValueError("No user has two or more pre-test events")
    offsets = np.asarray(data._offsets)
    flat_items = np.asarray(data._items)
    flat_times = np.asarray(data._ts)
    rng = np.random.default_rng(seed)
    cols = np.arange(seq_len + 1)[None, :]
    while True:
        users = eligible[rng.integers(0, len(eligible), size=batch_size)]
        n = lengths[users]
        end = 2 + (rng.random(batch_size) * (n - 1)).astype(np.int64)  # window ends after event `end`, end in [2, n]
        width = np.minimum(end, seq_len + 1)
        start_abs = offsets[users] + end - width
        pad = seq_len + 1 - width
        index = start_abs[:, None] + cols - pad[:, None]
        valid = cols >= pad[:, None]
        safe = np.clip(index, 0, len(flat_items) - 1)
        window = np.where(valid, flat_items[safe], 0).astype(np.int64)
        times = np.where(valid, flat_times[safe], 0).astype(np.int64)
        yield {
            "users": users,
            "inputs": window[:, :-1],
            "targets": window[:, 1:],
            "times": times[:, :-1],
            "target_times": times[:, 1:],
        }


LOSSES = ("auto", "sampled", "bce")


def next_item_loss(
    hidden: torch.Tensor,
    item_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str = "auto",
    logits_budget: float = 2.0e8,
    n_negatives: int = 1024,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Loss for next-item prediction at every non-padding position.

    hidden: [B, L, d], item_weight: [N + 1, d] (row 0 = padding), targets: [B, L] (0 = ignore).
    mode="auto": full softmax over the catalog when B * L * N fits the budget, else sampled softmax.
    mode="sampled": softmax over the target plus `n_negatives` uniform negatives shared by the batch.
    mode="bce": the original SASRec loss, binary cross-entropy with one random negative per position.
    Accidental hits (a negative equal to the target) are masked.
    """
    if mode not in LOSSES:
        raise ValueError(f"unknown loss {mode}; choose from {LOSSES}")
    mask = targets > 0
    if not mask.any():
        return hidden.sum() * 0.0
    h = hidden[mask]  # [P, d]
    t = targets[mask]  # [P]
    n_items = item_weight.shape[0] - 1
    if mode == "bce":
        negatives = torch.randint(1, n_items + 1, (len(t),), device=h.device, generator=generator)
        pos = (h * item_weight[t]).sum(-1)
        neg = (h * item_weight[negatives]).sum(-1)
        keep = (negatives != t).to(h.dtype)
        return (F.softplus(-pos) + keep * F.softplus(neg)).mean()
    if mode == "auto" and h.shape[0] * (n_items + 1) <= logits_budget:
        logits = h @ item_weight.T
        logits[:, 0] = float("-inf")
        return F.cross_entropy(logits, t)
    negatives = torch.randint(1, n_items + 1, (n_negatives,), device=h.device, generator=generator)
    pos = (h * item_weight[t]).sum(-1, keepdim=True)
    neg = h @ item_weight[negatives].T
    neg = neg.masked_fill(negatives[None, :] == t[:, None], float("-inf"))
    logits = torch.cat([pos, neg], dim=1)
    return F.cross_entropy(logits, torch.zeros(len(t), dtype=torch.long, device=h.device))


def to_tensor(array: np.ndarray, device: torch.device, dtype=torch.long) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)
