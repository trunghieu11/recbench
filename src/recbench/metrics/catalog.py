"""Metric definitions (protocol v2).

Every metric reads a MetricContext built by recbench.evaluation:
- ctx.topk[u, r]: the item at rank r+1 for eval user u (0 = empty slot), full-catalog ranking.
- ctx.relevant[u]: the user's relevant test items under the active repeat policy.
- ctx.next_item[u]: the first relevant test item (next-item task).

Per-user metrics return one value per user (the evaluator averages them and adds
bootstrap confidence intervals); list-level metrics return a single number.
The worked examples in docs/dictionary/metrics use these exact functions.
"""

from __future__ import annotations

import numpy as np

from recbench.protocol import K_VALUES, Metric, MetricSpec, Task
from recbench.registry import register_metric

RANKING_TASKS = {Task.topn, Task.sequential, Task.session, Task.similar_items}
TOPN = {Task.topn}


def _discount(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def hitrate(hits: np.ndarray, k: int) -> np.ndarray:
    """1 if any of the top-k items is relevant."""
    return hits[:, :k].any(axis=1).astype(np.float64)


def precision(hits: np.ndarray, k: int) -> np.ndarray:
    """Share of the k recommended slots that are relevant."""
    return hits[:, :k].sum(axis=1) / k


def recall(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """Relevant items found in the top-k, divided by min(k, number of relevant items)."""
    return hits[:, :k].sum(axis=1) / np.maximum(np.minimum(n_rel, k), 1)


def ndcg(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """DCG of the top-k list divided by the best achievable DCG (IDCG) for this user."""
    disc = _discount(k)
    dcg = (hits[:, :k] * disc).sum(axis=1)
    ideal_cum = np.concatenate([[0.0], np.cumsum(disc)])
    idcg = ideal_cum[np.minimum(n_rel, k)]
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def average_precision(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """Mean of precision@r over the ranks r <= k that hold a relevant item, normalised by min(k, |rel|)."""
    h = hits[:, :k].astype(np.float64)
    precision_at_r = np.cumsum(h, axis=1) / np.arange(1, k + 1)
    return (precision_at_r * h).sum(axis=1) / np.maximum(np.minimum(n_rel, k), 1)


def reciprocal_rank(hits: np.ndarray) -> np.ndarray:
    """1 / rank of the first relevant item, 0 when none is in the list."""
    any_hit = hits.any(axis=1)
    first = hits.argmax(axis=1) + 1
    return np.where(any_hit, 1.0 / first, 0.0)


def gini(counts: np.ndarray) -> float | None:
    """0 = every item recommended equally often, 1 = one item gets all exposure."""
    values = np.sort(np.asarray(counts, dtype=np.float64))
    total = values.sum()
    if total == 0 or len(values) == 0:
        return None
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * (index * values).sum()) / (n * total) - (n + 1) / n)


def _accuracy(name: str, k: int, fn, description: str, tasks=TOPN):
    class _M(Metric):
        spec = MetricSpec(name, set(tasks), kind="accuracy", per_user=True, description=description)

        def compute(self, ctx):
            return fn(ctx, k)

    _M.__name__ = name
    return register_metric(_M)


for _k in K_VALUES:
    _accuracy(f"hitrate_at_{_k}", _k, lambda c, k: hitrate(c.hits, k), f"Share of users with at least one relevant item in the top {_k}.")
    _accuracy(f"recall_at_{_k}", _k, lambda c, k: recall(c.hits, c.n_rel, k), f"Relevant items found in the top {_k} / min({_k}, #relevant).")
    _accuracy(f"ndcg_at_{_k}", _k, lambda c, k: ndcg(c.hits, c.n_rel, k), f"Position-aware ranking quality of the top {_k} (1 = perfect order).")
_accuracy("precision_at_10", 10, lambda c, k: precision(c.hits, k), "Share of the 10 recommended slots that are relevant.")
_accuracy("map_at_10", 10, lambda c, k: average_precision(c.hits, c.n_rel, k), "Mean average precision of the top 10.")
_accuracy("mrr_at_50", 50, lambda c, k: reciprocal_rank(c.hits[:, :k]), "1 / rank of the first relevant item within the top 50.")
_accuracy(
    "next_hitrate_at_10",
    10,
    lambda c, k: (c.next_rank <= k).astype(np.float64),
    "Next-item task: the very next item the user interacted with is in the top 10.",
    tasks=RANKING_TASKS,
)
_accuracy(
    "next_ndcg_at_10",
    10,
    lambda c, k: np.where(c.next_rank <= k, 1.0 / np.log2(c.next_rank + 1), 0.0),
    "Next-item task: NDCG@10 with the next item as the only relevant item.",
    tasks=RANKING_TASKS,
)


def _list_metric(name: str, fn, description: str, *, kind="beyond", higher=True, per_user=False, tasks=TOPN):
    class _M(Metric):
        spec = MetricSpec(name, set(tasks), kind=kind, higher_is_better=higher, per_user=per_user, description=description)

        def compute(self, ctx):
            return fn(ctx)

    _M.__name__ = name
    return register_metric(_M)


def _top10(ctx) -> np.ndarray:
    return ctx.topk[:, :10]


def _coverage(ctx):
    items = _top10(ctx)
    return float(len(np.unique(items[items > 0])) / max(ctx.n_items, 1))


def _gini(ctx):
    items = _top10(ctx)
    counts = np.bincount(items[items > 0].ravel(), minlength=ctx.n_items + 1)[1:]
    return gini(counts)


def _pop_percentile(ctx):
    items = _top10(ctx)
    flat = items[items > 0]
    if len(flat) == 0:
        return None
    ordered = np.sort(ctx.item_pop[1:])
    return float(np.mean(np.searchsorted(ordered, ctx.item_pop[flat], side="right") / len(ordered)))


def _long_tail(ctx):
    items = _top10(ctx)
    flat = items[items > 0]
    if len(flat) == 0:
        return None
    return float(np.mean(~ctx.head_items[flat]))


def _novelty(ctx):
    items = _top10(ctx)
    flat = items[items > 0]
    if len(flat) == 0:
        return None
    prob = (ctx.item_pop[flat] + 1) / (ctx.item_pop.sum() + ctx.n_items)
    return float(np.mean(-np.log2(prob)))


def _ild(ctx):
    if not ctx.has_categories:
        return None
    values = []
    for row in _top10(ctx):
        cats = [ctx.item_categories[i] for i in row if i > 0 and ctx.item_categories[i]]
        if len(cats) < 2:
            continue
        dist = [1 - len(a & b) / len(a | b) for x, a in enumerate(cats) for b in cats[x + 1 :]]
        values.append(np.mean(dist))
    return float(np.mean(values)) if values else None


def _serendipity(ctx):
    out = np.zeros(len(ctx.users))
    for u, row in enumerate(_top10(ctx)):
        rel = ctx.relevant_sets[u]
        profile = ctx.user_profile(u)
        good = 0
        for item in row:
            if item <= 0 or item not in rel or ctx.head_items_5pct[item]:
                continue
            cats = ctx.item_categories[item]
            if ctx.has_categories and cats and cats & set(profile):
                continue
            good += 1
        out[u] = good / 10
    return out


def _calibration(ctx):
    if not ctx.has_categories:
        return None
    alpha = 0.01
    values = []
    for u, row in enumerate(_top10(ctx)):
        p = ctx.user_profile(u)
        if not p:
            continue
        q: dict[str, float] = {}
        n = 0
        for item in row:
            cats = ctx.item_categories[item] if item > 0 else frozenset()
            if not cats:
                continue
            n += 1
            for cat in cats:
                q[cat] = q.get(cat, 0.0) + 1.0 / len(cats)
        if n == 0:
            continue
        q = {cat: weight / n for cat, weight in q.items()}
        kl = 0.0
        for cat, p_c in p.items():
            q_c = (1 - alpha) * q.get(cat, 0.0) + alpha * p_c
            kl += p_c * np.log2(p_c / q_c)
        values.append(kl)
    return float(np.mean(values)) if values else None


def _group_gap(ctx):
    per_user = ndcg(ctx.hits, ctx.n_rel, 10)
    means = [per_user[ctx.user_groups == group].mean() for group in ("light", "medium", "heavy") if (ctx.user_groups == group).any()]
    if len(means) < 2:
        return None
    return float(max(means) - min(means))


def _item_cold_recall(ctx):
    values = []
    for u, row in enumerate(_top10(ctx)):
        cold_rel = [i for i in ctx.relevant[u] if ctx.cold_items[i]]
        if not cold_rel:
            continue
        found = len(set(row.tolist()) & set(cold_rel))
        values.append(found / min(10, len(cold_rel)))
    return float(np.mean(values)) if values else None


_list_metric("coverage_at_10", _coverage, "Share of the catalog that appears in at least one top-10 list.")
_list_metric("gini_at_10", _gini, "Inequality of exposure across catalog items in top-10 lists (0 = equal).", higher=False)
_list_metric("popularity_percentile_at_10", _pop_percentile, "Average popularity percentile of recommended items (1 = most popular).", higher=False)
_list_metric("long_tail_share_at_10", _long_tail, "Share of recommended items outside the 20% most popular items.")
_list_metric("novelty_at_10", _novelty, "Mean self-information -log2 p(item) of recommended items (higher = less obvious).")
_list_metric("ild_at_10", _ild, "Intra-list diversity: mean pairwise category (Jaccard) distance inside a top-10.")
_list_metric("serendipity_at_10", _serendipity, "Relevant AND unexpected (not top-5% popular, new category for the user) share of a top-10.", per_user=True)
_list_metric("calibration_kl_at_10", _calibration, "KL divergence between the user's category mix and the list's mix (Steck 2018).", higher=False)
_list_metric("user_group_ndcg_gap_at_10", _group_gap, "Max - min mean NDCG@10 across light/medium/heavy user groups.", kind="beyond", higher=False)
_list_metric("item_cold_recall_at_10", _item_cold_recall, "Recall@10 counting only relevant items with no pre-test history.", kind="slice")


def _sampled(name: str, fn, description: str, *, kind="sampled", higher=True, tasks=RANKING_TASKS, needs_probability=False):
    class _M(Metric):
        spec = MetricSpec(name, set(tasks), kind=kind, higher_is_better=higher, per_user=True, description=description)

        def compute(self, ctx):
            if ctx.sampled_rank is None:
                return None
            if needs_probability and not ctx.method_spec.outputs_probability:
                return None
            return fn(ctx)

    _M.__name__ = name
    return register_metric(_M)


_sampled("sampled_hitrate_at_10", lambda c: (c.sampled_rank <= 10).astype(np.float64), "Next item ranked in the top 10 among itself + 100 random unseen items.")
_sampled(
    "sampled_ndcg_at_10",
    lambda c: np.where(c.sampled_rank <= 10, 1.0 / np.log2(c.sampled_rank + 1), 0.0),
    "NDCG@10 of the next item among itself + 100 random unseen items.",
)
_sampled("sampled_auc", lambda c: c.sampled_auc, "Per-user AUC (GAUC) of the next item vs 100 random unseen items.", kind="diagnostic", tasks={Task.ctr, Task.topn})
_sampled(
    "sampled_logloss",
    lambda c: c.sampled_logloss,
    "Log loss on the sampled candidates; only for models that output probabilities.",
    kind="diagnostic",
    higher=False,
    tasks={Task.ctr},
    needs_probability=True,
)


def _context_metric(name: str, key: str, description: str, *, kind="efficiency", higher=False):
    class _M(Metric):
        spec = MetricSpec(name, set(RANKING_TASKS) | {Task.ctr}, kind=kind, higher_is_better=higher, description=description)

        def compute(self, ctx):
            value = ctx.extra.get(key)
            return None if value is None else float(value)

    _M.__name__ = name
    return register_metric(_M)


_context_metric("train_seconds", "train_seconds", "Wall time of fit() on the active hardware.")
_context_metric("score_seconds_per_1k_users", "score_seconds_per_1k_users", "Batch scoring time for 1,000 users against the whole catalog.")
_context_metric("peak_rss_mb", "peak_rss_mb", "Peak resident memory of the run's process.")
_context_metric("peak_gpu_mb", "peak_gpu_mb", "Peak GPU memory allocated by torch (0 on CPU).")
_context_metric(
    "personal_explanation_rate",
    "personal_explanation_rate",
    "Share of sampled recommendations whose explanation cites one of the user's own history items.",
    kind="explainability",
    higher=True,
)
