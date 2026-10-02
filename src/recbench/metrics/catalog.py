"""Metrics on the shared candidate file. Ranking is per task; smoke runs do not sort these."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recbench.protocol import Metric, MetricSpec, Task
from recbench.registry import register_metric


def _ranked(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.sort_values(["user_idx", "score", "item_idx"], ascending=[True, False, True], kind="mergesort")
    frame = frame.copy()
    frame["rank"] = frame.groupby("user_idx").cumcount() + 1
    return frame


def _positive_ranks(scores: pd.DataFrame, mask: pd.Series | None = None) -> pd.Series:
    ranked = _ranked(scores)
    hits = ranked[ranked["label"] == 1]
    if mask is not None:
        hits = hits[mask.reindex(hits.index).fillna(False)]
    return hits["rank"]


def _recall_at(scores: pd.DataFrame, k: int, user_mask: pd.Series | None = None) -> float | None:
    ranked = _ranked(scores)
    hits = ranked[ranked["label"] == 1]
    if user_mask is not None:
        allowed = set(scores.loc[user_mask, "user_idx"].unique())
        hits = hits[hits["user_idx"].isin(allowed)]
    if hits.empty:
        return None
    return float((hits["rank"] <= k).mean())


def _top(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    ranked = _ranked(scores)
    return ranked[ranked["rank"] <= k]


@register_metric
class NDCG(Metric):
    spec = MetricSpec("ndcg_at_10", {Task.topn, Task.sequential, Task.session}, description="NDCG@10 on the shared candidate file.")

    def compute(self, scores, store, context) -> float | None:
        hits = _ranked(scores)
        hits = hits[hits["label"] == 1]
        if hits.empty:
            return None
        gain = np.where(hits["rank"] <= 10, 1.0 / np.log2(hits["rank"] + 1), 0.0)
        return float(np.mean(gain))


@register_metric
class Recall(Metric):
    spec = MetricSpec("recall_at_10", {Task.topn, Task.sequential, Task.session}, description="Recall@10 with one held-out item.")

    def compute(self, scores, store, context) -> float | None:
        return _recall_at(scores, 10)


@register_metric
class MAP(Metric):
    spec = MetricSpec("map_at_10", {Task.topn, Task.sequential, Task.session}, description="MAP@10 for a single positive.")

    def compute(self, scores, store, context) -> float | None:
        hits = _ranked(scores)
        hits = hits[hits["label"] == 1]
        if hits.empty:
            return None
        ap = np.where(hits["rank"] <= 10, 1.0 / hits["rank"], 0.0)
        return float(np.mean(ap))


@register_metric
class MRR(Metric):
    spec = MetricSpec("mrr", {Task.topn, Task.sequential, Task.session}, description="Mean reciprocal rank of the held-out item.")

    def compute(self, scores, store, context) -> float | None:
        hits = _ranked(scores)
        hits = hits[hits["label"] == 1]
        if hits.empty:
            return None
        return float(np.mean(1.0 / hits["rank"]))


@register_metric
class HitRate(Metric):
    spec = MetricSpec("hitrate_at_10", {Task.topn, Task.sequential, Task.session}, description="Hit rate@10.")

    def compute(self, scores, store, context) -> float | None:
        return _recall_at(scores, 10)


@register_metric
class PopularityRecall(Metric):
    spec = MetricSpec(
        "sanity_popularity_recall_at_10",
        {Task.topn},
        leaderboard=False,
        description="Popularity Recall@10 on the same candidate file. Not ranked.",
    )

    def compute(self, scores, store, context) -> float | None:
        counts = context.get("item_counts")
        if counts is None:
            return None
        frame = scores.copy()
        frame["score"] = frame["item_idx"].map(lambda i: counts.get(int(i), 0)).astype(float)
        return _recall_at(frame, 10)


@register_metric
class RMSE(Metric):
    spec = MetricSpec("rmse", {Task.rating}, description="RMSE on explicit test rows when the method has a rating head.")

    def compute(self, scores, store, context) -> float | None:
        pred = context.get("rating_pred")
        truth = context.get("rating_truth")
        if pred is None or truth is None or len(pred) == 0:
            return None
        return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


@register_metric
class MAE(Metric):
    spec = MetricSpec("mae", {Task.rating}, description="MAE on explicit test rows.")

    def compute(self, scores, store, context) -> float | None:
        pred = context.get("rating_pred")
        truth = context.get("rating_truth")
        if pred is None or truth is None or len(pred) == 0:
            return None
        return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


@register_metric
class AUC(Metric):
    spec = MetricSpec("auc", {Task.ctr}, description="AUC on the shared candidate labels.")

    def compute(self, scores, store, context) -> float | None:
        labels = scores["label"].to_numpy()
        pred = 1 / (1 + np.exp(-scores["score"].to_numpy()))
        order = np.argsort(pred, kind="mergesort")
        ranks = np.empty(len(pred), dtype=np.float64)
        ranks[order] = np.arange(1, len(pred) + 1)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        sum_pos = float(ranks[labels == 1].sum())
        return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@register_metric
class LogLoss(Metric):
    spec = MetricSpec("logloss", {Task.ctr}, description="Log loss on the shared candidate labels.")

    def compute(self, scores, store, context) -> float | None:
        labels = scores["label"].to_numpy().astype(float)
        pred = 1 / (1 + np.exp(-np.clip(scores["score"].to_numpy(), -20, 20)))
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        return float(-np.mean(labels * np.log(pred) + (1 - labels) * np.log(1 - pred)))


@register_metric
class Coverage(Metric):
    spec = MetricSpec("coverage_at_10", {Task.topn}, description="Fraction of the catalog that appears in any top-10.")

    def compute(self, scores, store, context) -> float | None:
        n_items = int(store.meta["n_items"])
        if n_items == 0:
            return None
        top = _top(scores, 10)
        return float(top["item_idx"].nunique() / n_items)


@register_metric
class Diversity(Metric):
    spec = MetricSpec("intra_list_diversity", {Task.topn}, description="Average pairwise category distance inside each top-10.")

    def compute(self, scores, store, context) -> float | None:
        cats = store.item_categories()
        top = _top(scores, 10)
        distances = []
        for _, group in top.groupby("user_idx"):
            labels = [set(str(cats.get(int(i), "")).split("|")) for i in group["item_idx"]]
            if len(labels) < 2:
                continue
            acc = []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    union = labels[i] | labels[j]
                    inter = labels[i] & labels[j]
                    acc.append(1 - (len(inter) / len(union) if union else 0))
            if acc:
                distances.append(float(np.mean(acc)))
        if not distances:
            return None
        return float(np.mean(distances))


@register_metric
class Novelty(Metric):
    spec = MetricSpec("novelty", {Task.topn}, description="Mean self-information of top-10 items.")

    def compute(self, scores, store, context) -> float | None:
        counts = context.get("item_counts") or {}
        total = sum(counts.values()) or 1
        top = _top(scores, 10)
        info = []
        for item in top["item_idx"]:
            prob = (counts.get(int(item), 0) + 1) / (total + 1)
            info.append(-np.log2(prob))
        if not info:
            return None
        return float(np.mean(info))


@register_metric
class Serendipity(Metric):
    spec = MetricSpec("serendipity", {Task.topn}, description="Hits that are outside the user profile and outside the global top 5 percent.")

    def compute(self, scores, store, context) -> float | None:
        counts = context.get("item_counts") or {}
        if not counts:
            return None
        threshold = np.quantile(list(counts.values()), 0.95)
        popular = {item for item, count in counts.items() if count >= threshold}
        cats = store.item_categories()
        ranked = _ranked(scores)
        hits = ranked[(ranked["label"] == 1) & (ranked["rank"] <= 10)]
        if hits.empty:
            return 0.0
        profiles = context.get("user_categories") or {}
        flags = []
        for user, item in zip(hits["user_idx"], hits["item_idx"]):
            profile = profiles.get(int(user), set())
            category = str(cats.get(int(item), ""))
            unexpected = category not in profile and int(item) not in popular
            flags.append(unexpected)
        return float(np.mean(flags))


@register_metric
class ExposureGini(Metric):
    spec = MetricSpec("exposure_gini", {Task.topn}, description="Gini of how often each catalog item is recommended.")

    def compute(self, scores, store, context) -> float | None:
        n_items = int(store.meta["n_items"])
        counts = np.zeros(n_items + 1, dtype=np.float64)
        top = _top(scores, 10)
        for item in top["item_idx"]:
            counts[int(item)] += 1
        values = np.sort(counts[1:])
        if values.sum() == 0:
            return None
        n = len(values)
        index = np.arange(1, n + 1)
        return float((2 * (index * values).sum()) / (n * values.sum()) - (n + 1) / n)


@register_metric
class MeanPopularity(Metric):
    spec = MetricSpec("mean_popularity_percentile", {Task.topn}, description="Mean train-popularity percentile of top-10 items.")

    def compute(self, scores, store, context) -> float | None:
        counts = context.get("item_counts") or {}
        if not counts:
            return None
        ordered = np.array(sorted(counts.values()))
        top = _top(scores, 10)
        percentiles = []
        for item in top["item_idx"]:
            count = counts.get(int(item), 0)
            percentiles.append(float(np.searchsorted(ordered, count, side="right") / max(len(ordered), 1)))
        return float(np.mean(percentiles)) if percentiles else None


@register_metric
class GroupExposure(Metric):
    spec = MetricSpec(
        "group_exposure_gap",
        {Task.topn},
        description="Absolute gap in mean recommendation popularity between user groups. No-ops without group labels.",
    )

    def compute(self, scores, store, context) -> float | None:
        users = pq.read_table(store.users_path, columns=["user_idx", "group_label"]).to_pandas()
        labeled = users.dropna(subset=["group_label"])
        if labeled.empty:
            return None
        top = _top(scores, 10)
        merged = top.merge(labeled, on="user_idx", how="inner")
        if merged["group_label"].nunique() < 2:
            return None
        means = merged.groupby("group_label")["score"].mean()
        return float(means.max() - means.min())


@register_metric
class UserColdRecall(Metric):
    spec = MetricSpec("user_cold_recall_at_10", {Task.topn}, description="Recall@10 for users whose first event is in the test window.")

    def compute(self, scores, store, context) -> float | None:
        cold_users = scores.loc[scores["is_cold_user"].astype(bool), "user_idx"]
        if cold_users.empty:
            return None
        mask = scores["user_idx"].isin(set(cold_users))
        return _recall_at(scores, 10, mask)


@register_metric
class ItemColdRecall(Metric):
    spec = MetricSpec("item_cold_recall_at_10", {Task.topn}, description="Recall@10 when the held-out item is cold.")

    def compute(self, scores, store, context) -> float | None:
        positives = scores[(scores["label"] == 1) & scores["is_cold_item"].astype(bool)]
        if positives.empty:
            return None
        mask = scores["user_idx"].isin(set(positives["user_idx"]))
        return _recall_at(scores, 10, mask)


def _context_metric(name: str, description: str, key: str):
    @register_metric
    class _Metric(Metric):
        spec = MetricSpec(name, {Task.topn}, description=description)

        def compute(self, scores, store, context) -> float | None:
            value = context.get(key)
            return None if value is None else float(value)

    _Metric.__name__ = name
    return _Metric


TrainWall = _context_metric("train_wall_seconds", "Training wall time on the active hardware profile.", "train_wall_s")
PeakRSS = _context_metric("peak_rss_mb", "Peak resident memory during the run.", "peak_rss_mb")
BatchInfer = _context_metric("batch_infer_seconds", "Wall time to score a fixed 1000-user slice.", "batch_infer_s")
ServedP50 = _context_metric("served_p50_ms", "Served latency p50 against the HTTP endpoint.", "served_p50_ms")
ServedP95 = _context_metric("served_p95_ms", "Served latency p95 against the HTTP endpoint.", "served_p95_ms")
Throughput = _context_metric("served_rps", "Served requests per second.", "served_rps")
ExplainCov = _context_metric(
    "local_explanation_coverage",
    "Share of sampled recommendations with a non-empty local reason.",
    "explanation_coverage",
)


def item_counts(store: Any) -> dict[int, int]:
    frame = pq.read_table(store.train_path, columns=["item_idx"]).to_pandas()
    return {int(k): int(v) for k, v in frame["item_idx"].value_counts().to_dict().items()}


def user_categories(store: Any) -> dict[int, set[str]]:
    train = pq.read_table(store.train_path, columns=["user_idx", "item_idx"]).to_pandas()
    cats = store.item_categories()
    profiles: dict[int, set[str]] = {}
    for user, item in zip(train["user_idx"], train["item_idx"]):
        profiles.setdefault(int(user), set()).add(str(cats.get(int(item), "")))
    return profiles
