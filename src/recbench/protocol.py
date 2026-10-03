"""Shared types for methods, metrics, and datasets (protocol v2).

A method declares what it can do in its MethodSpec; the evaluator enforces the
protocol (what data a method may see, how it is ranked, which metrics apply).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from recbench.data import HistoryBatch, TrainView

PROTOCOL_VERSION = "2"
# Bump when the evaluator changes how numbers are computed (metrics, masks, user sampling).
EVAL_VERSION = "1"
PROTOCOL_NOTE = (
    "Protocol v2: one global time cutoff (UTC); models are fitted only on events before the test cutoff; "
    "warm test users are ranked against the full catalog (sampled 1+100 is kept as a secondary check); "
    "training is budget-capped and untuned, so these numbers are not paper SOTA."
)

N_NEGATIVES = 100
K_VALUES = (10, 20, 50)
K_MAX = max(K_VALUES)
NEG_INF = np.float32(-np.inf)


class Task(str, Enum):
    topn = "topn"
    sequential = "sequential"
    session = "session"
    ctr = "ctr"
    rating = "rating"
    similar_items = "similar_items"


class Unsupported(Exception):
    """The method cannot run this call or on this dataset (a skip, not a failure)."""


@dataclass
class Explanation:
    """Why an item was recommended. Evidence entries that cite a history item make it personal."""

    kind: str
    text: str
    evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def personal(self) -> bool:
        return any("history_item" in entry for entry in self.evidence)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "evidence": self.evidence, "personal": self.personal}


@dataclass
class MethodSpec:
    name: str
    tasks: set[Task]
    feedback: set[str] = field(default_factory=lambda: {"implicit", "explicit"})
    # How the evaluator gets scores: "scores" = score_users over the catalog,
    # "pairs" = score_pairs in item chunks (pointwise models), "list" = topk only (remote APIs).
    output: str = "scores"
    uses_history: bool = False
    scores_cold_items: bool = False
    handles_cold_users: bool = False
    requires_side_features: bool = False
    requires_images: bool = False
    outputs_probability: bool = False
    managed: bool = False
    ranked: bool = True
    needs_torch: bool = False
    upstream: str = ""
    fidelity: str = "faithful"
    cost_band: str = "low"
    # Bump when a change to the method's code changes its results; old runs then stop counting as done.
    impl_version: str = "1"
    # True when fitting involves no randomness (closed forms, counting): one seed is enough when confirming.
    deterministic: bool = False


class Recommender:
    """Base class. Subclasses implement fit plus score_users, score_pairs, or topk."""

    spec: MethodSpec

    # Filled by bind(); every method needs the catalog size and id maps.
    n_users: int = 0
    n_items: int = 0
    # Max (user, item) pairs per score_pairs call; models with big per-pair tensors lower it.
    pair_budget: int = 1 << 20

    def bind(self, data: TrainView) -> None:
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.item_ids = data.item_ids
        self.item_pop = data.item_pop

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        """Return float32 scores of shape [len(users), n_items + 1]. Column 0 is padding."""
        raise Unsupported(f"{self.spec.name} does not score the whole catalog")

    def score_pairs(self, users: np.ndarray, items: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        """Score item matrix items[b, c] for user users[b]. Returns float32 [B, C]."""
        raise Unsupported(f"{self.spec.name} does not score (user, item) pairs")

    def full_scores(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        if self.spec.output == "scores":
            return np.asarray(self.score_users(users, hist), dtype=np.float32)
        if self.spec.output == "pairs":
            item_chunk = max(16, self.pair_budget // max(len(users), 1))
            out = np.empty((len(users), self.n_items + 1), dtype=np.float32)
            out[:, 0] = NEG_INF
            for start in range(1, self.n_items + 1, item_chunk):
                stop = min(start + item_chunk, self.n_items + 1)
                items = np.broadcast_to(np.arange(start, stop), (len(users), stop - start))
                out[:, start:stop] = self.score_pairs(users, np.ascontiguousarray(items), hist)
            return out
        raise Unsupported(f"{self.spec.name} only returns ranked lists")

    def topk(
        self,
        users: np.ndarray,
        hist: HistoryBatch,
        k: int,
        exclude: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k item indices and scores per user. exclude is a boolean mask [B, n_items + 1]."""
        scores = self.full_scores(users, hist)
        if exclude is not None:
            scores = np.where(exclude, NEG_INF, scores)
        scores[:, 0] = NEG_INF
        return top_k(scores, k)

    def explain(self, users: np.ndarray, items: np.ndarray, hist: HistoryBatch) -> list[list[Explanation]]:
        """One explanation list per user, aligned with items[b]. Default: none."""
        return [[] for _ in users]

    def item_embeddings(self) -> np.ndarray | None:
        """Item vectors [n_items + 1, d] when the model has them (used for explanations)."""
        return None


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k of a dense score matrix, best first. -inf entries come last and map to item 0."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    items = np.take_along_axis(part, order, axis=1)
    values = np.take_along_axis(part_scores, order, axis=1)
    items = np.where(np.isneginf(values), 0, items)
    return items.astype(np.int64), values.astype(np.float32)


@dataclass
class MetricSpec:
    name: str
    tasks: set[Task]
    kind: str = "accuracy"  # accuracy | beyond | slice | sampled | efficiency | explainability | diagnostic
    higher_is_better: bool = True
    per_user: bool = False
    leaderboard: bool = True
    description: str = ""


class Metric:
    spec: MetricSpec

    def compute(self, ctx: Any) -> float | np.ndarray | None:
        raise NotImplementedError


@dataclass
class DatasetSpec:
    name: str
    domain: str
    feedback: set[str]
    has_images: bool = False
    split_rule: str = "quantile"  # "quantile" (last 10% of events = test) or "last_days"
    test_days: int = 7
    repeat_policies: tuple[str, ...] = ("exclude_seen",)
    description: str = ""
