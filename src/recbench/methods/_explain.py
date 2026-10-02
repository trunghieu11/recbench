"""Grounded explanations: each one cites history items that actually drove the score."""

from __future__ import annotations

from typing import Callable

import numpy as np

from recbench.data import HistoryBatch
from recbench.protocol import Explanation


def _names(item_ids: np.ndarray, items: list[int]) -> str:
    return ", ".join(str(item_ids[i]) for i in items)


def contribution_explanations(
    users: np.ndarray,
    items: np.ndarray,
    history_of: Callable[[int], np.ndarray],
    weight: Callable[[np.ndarray, int], np.ndarray],
    item_ids: np.ndarray,
    label: str,
    top: int = 2,
) -> list[list[Explanation]]:
    """For additive models (ItemKNN, EASE): score(u, j) = sum over history items i of weight(i, j).

    The explanation cites the history items with the largest weight(i, j).
    """
    out: list[list[Explanation]] = []
    for user, row in zip(users, items):
        history = np.unique(history_of(int(user)))
        exps = []
        for item in row:
            if item <= 0 or len(history) == 0:
                exps.append(Explanation("none", "No history to explain from."))
                continue
            contrib = np.asarray(weight(history, int(item)), dtype=np.float64).ravel()
            order = np.argsort(-contrib)[:top]
            order = [o for o in order if contrib[o] > 0]
            if not order:
                exps.append(Explanation("none", f"{label}: no history item contributes to this item."))
                continue
            cited = [int(history[o]) for o in order]
            exps.append(
                Explanation(
                    "personal",
                    f"{label}: because you interacted with {_names(item_ids, cited)}.",
                    [{"history_item": str(item_ids[i]), "weight": float(contrib[o])} for i, o in zip(cited, order)],
                )
            )
        out.append(exps)
    return out


def embedding_explanations(
    item_vecs: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    hist: HistoryBatch,
    item_ids: np.ndarray,
    label: str,
    top: int = 2,
) -> list[list[Explanation]]:
    """For embedding models: cite the history items whose vectors are closest (cosine) to the recommended item."""
    vecs = np.asarray(item_vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.where(norms > 0, norms, 1.0)
    out: list[list[Explanation]] = []
    for b, row in enumerate(items):
        history = np.unique(hist.items[b][hist.items[b] > 0])
        exps = []
        for item in row:
            if item <= 0 or len(history) == 0:
                exps.append(Explanation("none", "No history to explain from."))
                continue
            sims = unit[history] @ unit[int(item)]
            order = np.argsort(-sims)[:top]
            cited = [int(history[o]) for o in order]
            exps.append(
                Explanation(
                    "personal",
                    f"{label}: similar (in the model's embedding space) to {_names(item_ids, cited)}, which you interacted with.",
                    [{"history_item": str(item_ids[i]), "similarity": float(sims[o])} for i, o in zip(cited, order)],
                )
            )
        out.append(exps)
    return out
