"""Shared types for methods, metrics, and datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


PROTOCOL_NOTE = (
    "These numbers follow the recbench protocol: a temporal cutoff, "
    "histories that stop at that cutoff, and one shared file of the held-out "
    "item plus 100 uniform negatives. Training is budget-capped. "
    "They are not paper SOTA."
)

N_NEGATIVES = 100
MAX_HISTORY = 200
SMOKE_STEPS = 200
PRESET_STEPS = {"24gb": 10_000, "48gb": 30_000, "cpu": SMOKE_STEPS}


class Task(str, Enum):
    topn = "topn"
    rating = "rating"
    ctr = "ctr"
    sequential = "sequential"
    session = "session"
    similar_items = "similar_items"


class Unsupported(Exception):
    """The method does not implement this call or this dataset."""


@dataclass
class Explanation:
    kind: str
    text: str
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "evidence": self.evidence}


@dataclass
class MethodSpec:
    name: str
    tasks: set[Task]
    feedback: set[str]
    requires_side_features: bool = False
    requires_images: bool = False
    can_score_candidates: bool = True
    managed: bool = False
    text_only: bool = False
    cost_band: str = "medium"
    upstream: str = ""


class Recommender:
    """Fit on a split store and score the shared candidate file."""

    spec: MethodSpec

    def fit(self, store: Any, config: dict[str, Any]) -> None:
        raise NotImplementedError

    def score_candidates(self, store: Any, candidates: Any) -> Any:
        raise Unsupported(f"{self.spec.name} cannot score candidates")

    def recommend(self, user_id: str, k: int = 10) -> list[dict[str, Any]]:
        raise Unsupported(f"{self.spec.name} cannot recommend")

    def predict_rating(self, frame: Any) -> Any:
        raise Unsupported(f"{self.spec.name} has no rating head")

    def predict_ctr(self, frame: Any) -> Any:
        raise Unsupported(f"{self.spec.name} has no CTR head")

    def similar_items(self, item_id: str, k: int = 10) -> list[dict[str, Any]]:
        raise Unsupported(f"{self.spec.name} has no similar-items head")

    def explain_global(self) -> Explanation | None:
        return None

    def explain_local(self, user_id: str, item_ids: list[str]) -> list[Explanation]:
        return []

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


@dataclass
class MetricSpec:
    name: str
    tasks: set[Task]
    leaderboard: bool = True
    description: str = ""


class Metric:
    spec: MetricSpec

    def compute(self, scores: Any, store: Any, context: dict[str, Any]) -> float | None:
        raise NotImplementedError


@dataclass
class DatasetSpec:
    name: str
    domain: str
    feedback: set[str]
    has_images: bool = False
    official_split: str = "quantile"
    description: str = ""
