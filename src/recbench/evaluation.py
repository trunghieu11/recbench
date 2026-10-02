"""Evaluate a fitted method on one split.

Primary protocol: every warm eval user (has pre-test history and at least one
relevant test item) is ranked against the WHOLE catalog. Padding is never
recommended; already-seen items are removed under the "exclude_seen" repeat
policy; items with no pre-test history are removed for methods that cannot
score them (pure ID models).

Secondary protocol: the next test item is ranked against 100 random unseen
items (the sampled candidates file). It is cheaper but known to be unreliable,
so it is reported in separate "sampled_*" columns only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recbench.data import HistoryBatch, SplitError, TrainView
from recbench.protocol import K_MAX, NEG_INF, Explanation, MethodSpec, Recommender, Unsupported, top_k
from recbench.registry import ensure_loaded

CI_METRICS = ("ndcg_at_10", "recall_at_10", "hitrate_at_10", "next_ndcg_at_10")
N_BOOTSTRAP = 1000
N_EXPLAIN_USERS = 50


class EvalSplit:
    """The evaluator's side of a split: eval users, relevance sets, candidates."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            raise SplitError(f"No split at {self.root}")
        self.meta = json.loads(meta_path.read_text())
        self.policies: list[str] = list(self.meta["repeat_policies"])
        self.primary: str = self.meta["primary_policy"]
        self.eval_users = pq.read_table(self.root / "eval_users.parquet").to_pandas()
        self.test_items = pq.read_table(self.root / "test_items.parquet").to_pandas()
        self.candidates = pq.read_table(self.root / "candidates.parquet").to_pandas()
        self.users = pq.read_table(self.root / "users.parquet", columns=["user_idx", "group_label", "n_pretest"]).to_pandas()

    def users_of(self, warm: bool) -> np.ndarray:
        return self.eval_users.loc[self.eval_users["is_warm"] == warm, "user_idx"].to_numpy(dtype=np.int64)

    def next_items(self, users: np.ndarray, policy: str) -> np.ndarray:
        column = "next_item" if policy == self.primary else f"next_item__{policy}"
        lookup = self.eval_users.set_index("user_idx")[column]
        return lookup.reindex(users).fillna(0).to_numpy(dtype=np.int64)

    def relevant(self, users: np.ndarray, policy: str) -> list[np.ndarray]:
        frame = self.test_items if policy == "allow_repeats" else self.test_items[~self.test_items["is_repeat"]]
        grouped = frame.groupby("user_idx")["item_idx"].apply(lambda s: s.to_numpy(dtype=np.int64))
        empty = np.zeros(0, dtype=np.int64)
        return [grouped.get(int(u), empty) for u in users]

    def groups(self, users: np.ndarray) -> np.ndarray:
        return self.users.set_index("user_idx")["group_label"].reindex(users).fillna("cold").to_numpy(dtype=object)

    def candidate_matrix(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Candidate items per user, positive in column 0, padded with 0. Returns (items [U, C], valid [U])."""
        grouped = self.candidates.sort_values(["user_idx", "label"], ascending=[True, False]).groupby("user_idx")["item_idx"]
        lists = {int(u): g.to_numpy(dtype=np.int64) for u, g in grouped}
        width = max((len(v) for v in lists.values()), default=1)
        out = np.zeros((len(users), width), dtype=np.int64)
        valid = np.zeros(len(users), dtype=bool)
        for row, user in enumerate(users):
            items = lists.get(int(user))
            if items is not None and len(items) > 1:
                out[row, : len(items)] = items
                valid[row] = True
        return out, valid


@dataclass
class MetricContext:
    """Everything a metric needs for one (method, repeat policy, user population)."""

    method_spec: MethodSpec
    policy: str
    users: np.ndarray
    topk: np.ndarray
    relevant: list[np.ndarray]
    next_item: np.ndarray
    n_items: int
    item_pop: np.ndarray
    item_categories: list[frozenset]
    profile_fn: Any
    user_groups: np.ndarray
    cold_items: np.ndarray
    sampled_rank: np.ndarray | None = None
    sampled_auc: np.ndarray | None = None
    sampled_logloss: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def relevant_sets(self) -> list[set]:
        return [set(r.tolist()) for r in self.relevant]

    @cached_property
    def hits(self) -> np.ndarray:
        return np.stack([np.isin(row, rel) & (row > 0) for row, rel in zip(self.topk, self.relevant)]) if len(self.topk) else np.zeros((0, K_MAX), bool)

    @cached_property
    def n_rel(self) -> np.ndarray:
        return np.array([len(r) for r in self.relevant], dtype=np.int64)

    @cached_property
    def next_rank(self) -> np.ndarray:
        """1-based rank of the next item in the list; K_MAX + 1 when absent."""
        match = (self.topk == self.next_item[:, None]) & (self.next_item[:, None] > 0)
        return np.where(match.any(axis=1), match.argmax(axis=1) + 1, K_MAX + 1)

    @cached_property
    def has_categories(self) -> bool:
        return any(self.item_categories)

    @cached_property
    def head_items(self) -> np.ndarray:
        return _head_mask(self.item_pop, 0.2)

    @cached_property
    def head_items_5pct(self) -> np.ndarray:
        return _head_mask(self.item_pop, 0.05)

    @cached_property
    def _profiles(self) -> dict[int, dict[str, float]]:
        return {}

    def user_profile(self, u: int) -> dict[str, float]:
        """Category mix of user u's pre-test history (computed on first use)."""
        if u not in self._profiles:
            self._profiles[u] = self.profile_fn(int(self.users[u]))
        return self._profiles[u]


def _head_mask(pop: np.ndarray, share: float) -> np.ndarray:
    """True for the `share` most popular items (by pre-test count)."""
    mask = np.zeros(len(pop), dtype=bool)
    order = np.argsort(-pop[1:], kind="stable") + 1
    mask[order[: max(1, int(np.ceil(share * (len(pop) - 1))))]] = True
    return mask


@dataclass
class EvalResult:
    metrics: dict[str, float]
    per_user: dict[str, np.ndarray]
    explanations: list[dict[str, Any]]
    errors: dict[str, str]
    n_users: int


class Evaluator:
    def __init__(self, split: EvalSplit, data: TrainView, cfg: dict[str, Any]):
        self.split = split
        self.data = data
        self.cfg = cfg
        self.seq_len = int(cfg.get("seq_len", 50))
        budget = float(cfg.get("eval_score_budget", 2.5e8))  # floats held at once (~1 GB)
        self.batch_users = int(cfg.get("eval_batch_users") or max(1, min(2048, budget // (data.n_items + 1))))
        self.item_categories = [frozenset(t for t in str(c).split("|") if t) for c in data.item_category]
        self.has_categories = any(self.item_categories)
        self.cold_items = ~data.warm_item_mask()
        self.cold_items[0] = False

    # ----- ranking -----
    def _masks(self, method: Recommender, users: np.ndarray, policy: str, n_cols: int) -> np.ndarray:
        mask = np.zeros((len(users), n_cols), dtype=bool)
        mask[:, 0] = True
        if not method.spec.scores_cold_items:
            mask[:, self.cold_items] = True
        if policy == "exclude_seen":
            seen = self.data.seen[users]
            rows, cols = seen.nonzero()
            mask[rows, cols] = True
        return mask

    def rank(self, method: Recommender, users: np.ndarray, policies: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Top-K_MAX lists per policy plus scores for the sampled candidates."""
        topk = {p: np.zeros((len(users), K_MAX), dtype=np.int64) for p in policies}
        cand_items, cand_valid = self.split.candidate_matrix(users)
        cand_scores = np.full(cand_items.shape, np.nan, dtype=np.float64)
        score_seconds = 0.0
        for start in range(0, len(users), self.batch_users):
            batch = users[start : start + self.batch_users]
            hist = self.data.history_batch(batch, self.seq_len)
            rows = slice(start, start + len(batch))
            if method.spec.output == "list":
                began = time.perf_counter()
                listed, _ = method.topk(batch, hist, K_MAX + 50)
                score_seconds += time.perf_counter() - began
                for policy in policies:
                    mask = self._masks(method, batch, policy, self.data.n_items + 1)
                    topk[policy][rows] = _filter_lists(listed, mask, K_MAX)
                continue
            began = time.perf_counter()
            scores = method.full_scores(batch, hist)
            score_seconds += time.perf_counter() - began
            if scores.shape != (len(batch), self.data.n_items + 1):
                raise ValueError(f"{method.spec.name} returned scores of shape {scores.shape}")
            cand = cand_items[rows]
            gathered = np.take_along_axis(scores, cand, axis=1).astype(np.float64)
            if not method.spec.scores_cold_items:
                gathered = np.where(self.cold_items[cand], -np.inf, gathered)
            cand_scores[rows] = np.where(cand > 0, gathered, np.nan)
            for policy in policies:
                masked = np.where(self._masks(method, batch, policy, scores.shape[1]), NEG_INF, scores)
                items, _ = top_k(masked, K_MAX)  # fewer than K_MAX columns when the catalog is tiny
                topk[policy][rows, : items.shape[1]] = items
        info = {
            "score_seconds": score_seconds,
            "cand_scores": cand_scores,
            "cand_valid": cand_valid & (method.spec.output != "list"),
        }
        return topk, info

    # ----- metrics -----
    def _context(self, method, policy, users, topk, sampled, extra) -> MetricContext:
        return MetricContext(
            method_spec=method.spec,
            policy=policy,
            users=users,
            topk=topk,
            relevant=self.split.relevant(users, policy),
            next_item=self.split.next_items(users, policy),
            n_items=self.data.n_items,
            item_pop=self.data.item_pop,
            item_categories=self.item_categories,
            profile_fn=self._profile,
            user_groups=self.split.groups(users),
            cold_items=self.cold_items,
            sampled_rank=sampled.get("rank"),
            sampled_auc=sampled.get("auc"),
            sampled_logloss=sampled.get("logloss"),
            extra=extra,
        )

    def _profile(self, user: int) -> dict[str, float]:
        if not self.has_categories:
            return {}
        weights: dict[str, float] = {}
        total = 0.0
        for item in self.data.user_items(int(user)):
            cats = self.item_categories[int(item)]
            for cat in cats:
                weights[cat] = weights.get(cat, 0.0) + 1.0 / len(cats)
            total += 1.0 if cats else 0.0
        return {cat: w / total for cat, w in weights.items()} if total else {}

    def run(self, method: Recommender, extra: dict[str, Any] | None = None) -> EvalResult:
        reg = ensure_loaded()
        extra = dict(extra or {})
        users = self.split.users_of(warm=True)
        if len(users) == 0:
            raise SplitError(f"{self.data.dataset}: no warm eval users")
        cap = int(self.cfg.get("max_eval_users") or 0)
        if cap and len(users) > cap:
            # Same seeded subsample for every method, so methods stay comparable.
            users = np.sort(np.random.default_rng(int(self.cfg.get("seed", 42))).permutation(users)[:cap])
        policies = self.split.policies
        topk, info = self.rank(method, users, policies)
        extra["score_seconds_per_1k_users"] = 1000.0 * info["score_seconds"] / len(users)
        sampled = _sampled_stats(info["cand_scores"], info["cand_valid"])
        explanations, rate = self._explanations(method, users, topk[self.split.primary])
        extra["personal_explanation_rate"] = rate
        metrics: dict[str, float] = {}
        per_user: dict[str, np.ndarray] = {}
        errors: dict[str, str] = {}
        for policy in policies:
            prefix = "" if policy == self.split.primary else f"{policy}/"
            ctx = self._context(method, policy, users, topk[policy], sampled if not prefix else {}, extra if not prefix else {})
            for name, metric_cls in reg.metrics.items():
                metric = metric_cls()
                if not (metric.spec.tasks & method.spec.tasks):
                    continue
                if prefix and metric.spec.kind not in ("accuracy",):
                    continue
                try:
                    value = metric.compute(ctx)
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    errors[prefix + name] = f"{type(exc).__name__}: {exc}"
                    continue
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        continue
                    per_user[prefix + name] = value
                    value = float(np.nanmean(value))
                metrics[prefix + name] = float(value)
        for name in CI_METRICS:
            if name in per_user:
                low, high = bootstrap_ci(per_user[name])
                metrics[f"{name}_ci_low"], metrics[f"{name}_ci_high"] = low, high
        if method.spec.handles_cold_users:
            metrics.update(self._cold_users(method))
        metrics["n_eval_users"] = float(len(users))
        return EvalResult(metrics, per_user, explanations, errors, len(users))

    def _cold_users(self, method: Recommender) -> dict[str, float]:
        users = self.split.users_of(warm=False)
        if len(users) == 0:
            return {}
        topk, _ = self.rank(method, users, [self.split.primary])
        lists = topk[self.split.primary]
        relevant = self.split.relevant(users, "allow_repeats")
        hits = np.stack([np.isin(row, rel) & (row > 0) for row, rel in zip(lists, relevant)])
        n_rel = np.array([len(r) for r in relevant])
        from recbench.metrics.catalog import ndcg, recall

        return {
            "cold_users/ndcg_at_10": float(ndcg(hits, n_rel, 10).mean()),
            "cold_users/recall_at_10": float(recall(hits, n_rel, 10).mean()),
            "cold_users/n_users": float(len(users)),
        }

    def _explanations(self, method: Recommender, users: np.ndarray, lists: np.ndarray) -> tuple[list[dict], float | None]:
        rng = np.random.default_rng(7)
        pick = np.sort(rng.permutation(len(users))[: min(N_EXPLAIN_USERS, len(users))])
        sample_users = users[pick]
        items = lists[pick, :3]
        hist = self.data.history_batch(sample_users, self.seq_len)
        try:
            explained = method.explain(sample_users, items, hist)
        except Unsupported:
            return [], None
        records, personal, total = [], 0, 0
        for user, row, exps in zip(sample_users, items, explained):
            for item, exp in zip(row, exps):
                if item <= 0 or not isinstance(exp, Explanation):
                    continue
                total += 1
                personal += int(exp.personal)
                if len(records) < 30:
                    records.append({"user_id": self.data.user_ids[user], "item_id": self.data.item_ids[item], **exp.as_dict()})
        return records, (personal / total if total else None)


def _filter_lists(listed: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Drop masked items from ranked lists (e.g., a remote API's output) and keep the first k."""
    out = np.zeros((len(listed), k), dtype=np.int64)
    for row, items in enumerate(listed):
        kept = [int(i) for i in items if i > 0 and not mask[row, int(i)]][:k]
        out[row, : len(kept)] = kept
    return out


def _sampled_stats(scores: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    if not valid.any():
        return {}
    scores = scores[valid]
    pos = scores[:, 0:1]
    neg = scores[:, 1:]
    present = ~np.isnan(neg)
    greater = ((neg > pos) & present).sum(axis=1)
    ties = ((neg == pos) & present).sum(axis=1)
    n_neg = np.maximum(present.sum(axis=1), 1)
    rank = 1.0 + greater + 0.5 * ties
    auc = 1.0 - (greater + 0.5 * ties) / n_neg
    prob = 1.0 / (1.0 + np.exp(-np.clip(np.nan_to_num(scores, nan=0.0), -30, 30)))
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    loss = -(np.log(prob[:, 0]) + np.where(present, np.log(1 - prob[:, 1:]), 0.0).sum(axis=1)) / (1 + present.sum(axis=1))
    return {"rank": rank, "auc": auc, "logloss": loss}


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean over users."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def save_result(result: EvalResult, out_dir: Path) -> list[Path]:
    """Write per-user arrays, explanations, and metric errors next to the run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if result.per_user:
        path = out_dir / "per_user_metrics.npz"
        np.savez_compressed(path, **{k.replace("/", "__"): v for k, v in result.per_user.items()})
        paths.append(path)
    if result.explanations:
        path = out_dir / "explanations.json"
        path.write_text(json.dumps(result.explanations, indent=2, default=str))
        paths.append(path)
    if result.errors:
        path = out_dir / "metric_errors.json"
        path.write_text(json.dumps(result.errors, indent=2))
        paths.append(path)
    return paths


def per_user_frame(result: EvalResult) -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in result.per_user.items() if np.ndim(v) == 1})
