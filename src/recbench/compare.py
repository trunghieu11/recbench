"""Is B really better than A? A paired comparison of two lab results, user by user.

    python -m recbench.compare itemknn itemknn:no-time-knobs                  # an experiment against the baseline
    python -m recbench.compare most_popular itemknn --datasets movielens-25m   # two methods' baselines
    python -m recbench.compare ease ease:edlae --segments                     # plus where it got better or worse

A and B are `method[:experiment]`; the experiment defaults to `baseline`. For each dataset the command reads both
final test runs, pairs the users (the same test users, matched by id) and reports B - A in test NDCG@10 with a 95%
bootstrap interval of the paired differences (2,000 resamples of the users):

- better: the whole interval is above 0;
- worse: the whole interval is below 0;
- otherwise: no clear difference.

Each side's setting was chosen on the validation fold, and the test split was used once per setting. Random methods
(iALS, BPR-MF, the re-ranker) are tested with 3 seeds in the lab; their per-user results are averaged first.
`--segments` breaks the difference down by user activity, recency and taste for popular items, and shows recall by
the popularity of the items to find. The rules and how to read the output: docs/handbook/compare-results.md.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recbench.paths import repo_root, tracking_uri, use_workspace

BASELINE = "baseline"
N_RESAMPLES = 2000
DAY_US = 86_400_000_000
RUN_METRICS = ("train_seconds", "score_seconds_per_1k_users", "coverage_at_10", "popularity_percentile_at_10", "peak_rss_mb")


class Missing(Exception):
    """A side has no finished result on a dataset (the message says how to produce it)."""


@dataclass(frozen=True)
class Ref:
    method: str
    label: str = BASELINE

    @classmethod
    def parse(cls, text: str) -> "Ref":
        method, _, label = text.partition(":")
        return cls(method.strip(), (label.strip() or BASELINE))

    def __str__(self) -> str:
        return self.method if self.label == BASELINE else f"{self.method}:{self.label}"


@dataclass
class Side:
    ref: Ref
    dataset: str
    summary: dict[str, Any]
    users: np.ndarray  # user_idx, sorted
    values: dict[str, np.ndarray]  # per-user arrays aligned with `users`, averaged over seeds
    top10: np.ndarray | None  # [users, 10] from the first seed (None if the run did not save its lists)
    metrics: dict[str, float] = field(default_factory=dict)  # run-level metrics, averaged over seeds
    seeds: int = 1


# ----- reading results ------------------------------------------------------------------------------------------------

def _experiment_id(workspace: str) -> tuple[Any, str | None]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(workspace))
    experiment = mlflow.get_experiment_by_name("recbench")
    return mlflow, (experiment.experiment_id if experiment else None)


def _runs(workspace: str, filter_string: str, n: int = 1) -> pd.DataFrame:
    mlflow, exp_id = _experiment_id(workspace)
    if exp_id is None:
        return pd.DataFrame()
    return mlflow.search_runs([exp_id], filter_string=filter_string, order_by=["attributes.start_time DESC"], max_results=n)


def _per_user(artifact_uri: str, run_id: str, workspace: str) -> dict[str, np.ndarray]:
    path = None
    if artifact_uri.startswith("file:"):
        from urllib.parse import unquote, urlparse

        path = Path(unquote(urlparse(artifact_uri).path)) / "per_user_metrics.npz"
    if path is None or not path.exists():
        from mlflow.tracking import MlflowClient

        path = Path(MlflowClient(tracking_uri=tracking_uri(workspace)).download_artifacts(run_id, "per_user_metrics.npz"))
    with np.load(path, allow_pickle=False) as data:
        return {k.replace("__", "/"): data[k] for k in data.files}


def _final_runs(summary: dict[str, Any], ref: Ref, dataset: str, workspace: str) -> pd.DataFrame:
    """The final test run(s) of a job: by the identities its summary recorded, or (older summaries) by tags."""
    frames = []
    for run in summary.get("final_runs") or []:
        if run.get("config_hash") and run.get("value") is not None:
            frames.append(_runs(workspace, f"tags.config_hash = '{run['config_hash']}' and tags.status = 'finished'"))
    if not frames:  # bake-off summaries written before final_runs existed
        frames.append(_runs(workspace, f"tags.dataset = '{dataset}' and tags.method = '{ref.method}' and tags.tier = '{summary['tier']}' "
                                       "and tags.stage = 'final' and tags.status = 'finished'"))
    frame = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    return frame


def load_side(ref: Ref, dataset: str, *, workspace: str = "lab", tier: str = "quick") -> Side:
    """One side of a comparison: its job summary, per-user test results (seed-averaged) and run metrics."""
    from recbench.tuning.job import read_summary

    summary = read_summary(tier, dataset, ref.method, label=ref.label, workspace=workspace)
    if not summary:
        how = (f"python -m recbench.lab baseline --methods {ref.method} --datasets {dataset}" if ref.label == BASELINE else
               f"python -m recbench.lab run --method {ref.method} --experiment {ref.label} --datasets {dataset}")
        raise Missing(f"{ref} has no result on {dataset} yet ({how})")
    if summary.get("status") != "finished":
        raise Missing(f"{ref} on {dataset}: {summary.get('status')} {summary.get('reason') or ''}".strip())
    runs = _final_runs(summary, ref, dataset, workspace)
    if runs.empty:
        raise Missing(f"{ref} on {dataset}: its final run is not in the {workspace or 'bake-off'} MLflow store")
    per_seed = [_per_user(row["artifact_uri"], row["run_id"], workspace) for _, row in runs.iterrows()]
    users = np.unique(per_seed[0]["user_idx"])
    for arrays in per_seed[1:]:
        users = np.intersect1d(users, arrays["user_idx"])
    order = [_positions(a["user_idx"], users) for a in per_seed]  # each seed's rows, in the order of `users`
    n_rows = len(per_seed[0]["user_idx"])
    names = [k for k, v in per_seed[0].items() if np.ndim(v) == 1 and len(v) == n_rows and k != "user_idx" and not k.startswith("sampled_")]
    values = {}
    for name in names:
        stacked = [a[name][o].astype(np.float64) for a, o in zip(per_seed, order) if name in a]
        if len(stacked) == len(per_seed):
            with np.errstate(all="ignore"):
                values[name] = np.nanmean(np.stack(stacked), axis=0) if len(stacked) > 1 else stacked[0]
    top10 = per_seed[0]["top10"][order[0]] if "top10" in per_seed[0] else None
    metrics = {}
    for name in RUN_METRICS:
        column = f"metrics.{name}"
        if column in runs and runs[column].notna().any():
            metrics[name] = float(runs[column].mean())
    return Side(ref, dataset, summary, users, values, top10, metrics, seeds=len(per_seed))


def _positions(ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Row of each wanted user id in `ids`."""
    sorter = np.argsort(ids, kind="stable")
    return sorter[np.searchsorted(ids, wanted, sorter=sorter)]


# ----- the paired test --------------------------------------------------------------------------------------------------

def paired_difference(x: np.ndarray, y: np.ndarray, *, resamples: int = N_RESAMPLES, seed: int = 0) -> dict[str, float]:
    """B - A over the same users: mean difference, its 95% bootstrap interval and the share of users each way.
    Users with no value on either side are left out."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = ~np.isnan(x) & ~np.isnan(y)
    d = y[ok] - x[ok]
    if len(d) == 0:
        return {"users": 0, "a": np.nan, "b": np.nan, "diff": np.nan, "low": np.nan, "high": np.nan, "better_users": np.nan, "worse_users": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 250):  # in chunks, so 10,000 users x 2,000 resamples stays small in memory
        stop = min(start + 250, resamples)
        means[start:stop] = d[rng.integers(0, len(d), size=(stop - start, len(d)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"users": int(len(d)), "a": float(x[ok].mean()), "b": float(y[ok].mean()), "diff": float(d.mean()), "low": float(low),
            "high": float(high), "better_users": float((d > 0).mean()), "worse_users": float((d < 0).mean())}


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "no clear difference" if low == low else "-"


def compare_side(a: Side, b: Side, metric: str = "ndcg_at_10") -> dict[str, Any]:
    if metric not in a.values or metric not in b.values:
        raise Missing(f"{metric} is not among the per-user results of both runs")
    _, ia, ib = np.intersect1d(a.users, b.users, return_indices=True)
    row = paired_difference(a.values[metric][ia], b.values[metric][ib])
    row["verdict"] = verdict(row["low"], row["high"])
    row["relative"] = row["diff"] / row["a"] if row["a"] else np.nan
    for side, key in ((a, "a"), (b, "b")):
        row[f"{key}_val"] = side.summary.get("best_val")
        row[f"{key}_trials"] = len(side.summary.get("trials") or [])
        row[f"{key}_seeds"] = side.seeds
        for name in RUN_METRICS:
            row[f"{key}_{name}"] = side.metrics.get(name)
    return row


def compare(a: Ref | str, b: Ref | str, datasets: list[str] | None = None, *, metric: str = "ndcg_at_10", workspace: str = "lab",
            tier: str = "quick") -> pd.DataFrame:
    """One row per dataset: the paired difference, its verdict, and cost and side effects of each side."""
    a, b = (Ref.parse(a) if isinstance(a, str) else a), (Ref.parse(b) if isinstance(b, str) else b)
    rows = []
    for dataset in datasets or default_datasets(workspace):
        try:
            rows.append({"dataset": dataset, **compare_side(load_side(a, dataset, workspace=workspace, tier=tier),
                                                            load_side(b, dataset, workspace=workspace, tier=tier), metric)})
        except Missing as exc:
            rows.append({"dataset": dataset, "verdict": "-", "note": str(exc)})
    return pd.DataFrame(rows)


def promotion(frame: pd.DataFrame, n_datasets: int = 5) -> str:
    """The lab's promotion rule applied to a comparison (B is the candidate)."""
    verdicts = list(frame.get("verdict", pd.Series(dtype=str)))
    better, worse = verdicts.count("better"), verdicts.count("worse")
    compared = sum(v in ("better", "worse", "no clear difference") for v in verdicts)
    head = f"better on {better} of {compared} datasets, worse on {worse}"
    if compared < n_datasets:
        return f"{head}. Compare on all {n_datasets} datasets before deciding."
    if worse:
        return f"{head}: do not promote (record what you learned)."
    if better == compared:
        return f"{head}: promote as the new default (bump impl_version) and add it to the search space."
    if better:
        return f"{head}: promote as a search-space option (the default stays)."
    return f"{head}: no evidence of a difference; not promoted."


def default_datasets(workspace: str = "lab") -> list[str]:
    from recbench.config import load_benchmark_yaml

    name = "lab.yaml" if workspace == "lab" else "quick.yaml"
    return list(load_benchmark_yaml(repo_root() / "configs" / "benchmarks" / name, repo_root()).get("datasets") or [])


# ----- error analysis ---------------------------------------------------------------------------------------------------

def segments(a: Ref | str, b: Ref | str, dataset: str, *, metric: str = "ndcg_at_10", workspace: str = "lab",
             tier: str = "quick") -> dict[str, pd.DataFrame]:
    """Where B gained or lost against A on one dataset's test users:

    - activity: users grouped by how many events they had before the test period (quartiles);
    - recency: by days since their last event (quartiles);
    - taste: by how popular their test items are (quartiles of the items' popularity percentile);
    - items: recall@10 for test items of the popular head (top 20% of items), the long tail, and new items;
    - gained / lost: the test items B finds and A misses most often, and the reverse."""
    from recbench.data import TrainView
    from recbench.evaluation import EvalSplit, _head_mask
    from recbench.runner import data_root

    a, b = (Ref.parse(a) if isinstance(a, str) else a), (Ref.parse(b) if isinstance(b, str) else b)
    side_a, side_b = load_side(a, dataset, workspace=workspace, tier=tier), load_side(b, dataset, workspace=workspace, tier=tier)
    users, ia, ib = np.intersect1d(side_a.users, side_b.users, return_indices=True)
    x, y = side_a.values[metric][ia], side_b.values[metric][ib]
    root = data_root() / "splits" / dataset / tier
    view, split = TrainView(root), EvalSplit(root)
    offsets, ts = np.asarray(view._offsets), np.asarray(view._ts)
    last = ts[np.maximum(offsets[users + 1] - 1, 0)]
    recency = (int(view.meta["test_start_us"]) - last) / DAY_US
    relevant = split.relevant(users, split.primary)
    warm_pop = np.sort(view.item_pop[1:][view.item_pop[1:] > 0])
    percentile = np.searchsorted(warm_pop, view.item_pop, side="right") / max(len(warm_pop), 1)
    percentile[view.item_pop == 0] = 0.0
    taste = np.array([percentile[r].mean() if len(r) else np.nan for r in relevant])
    out = {
        "activity": _by_quartile(view.user_lengths[users].astype(float), x, y, "events before the test", "{:.0f}"),
        "recency": _by_quartile(recency, x, y, "days since the last event", "{:.1f}"),
        "taste": _by_quartile(taste * 100, x, y, "popularity percentile of the test items", "{:.0f}"),
    }
    if side_a.top10 is not None and side_b.top10 is not None:
        out.update(_items(view, relevant, side_a.top10[ia], side_b.top10[ib], _head_mask(view.item_pop, 0.2)))
    return out


def _by_quartile(values: np.ndarray, x: np.ndarray, y: np.ndarray, what: str, fmt: str) -> pd.DataFrame:
    keep = ~np.isnan(values)
    try:
        bins = pd.qcut(values[keep], 4, duplicates="drop")
    except ValueError:  # too few distinct values for quartiles
        bins = pd.cut(values[keep], 1)
    rows = []
    for interval in bins.categories:
        mask = np.zeros(len(values), dtype=bool)
        mask[np.flatnonzero(keep)[np.asarray(bins == interval)]] = True
        stats = paired_difference(x[mask], y[mask], resamples=1000)
        low, high = max(interval.left, np.nanmin(values[keep])), interval.right
        rows.append({"group": f"{fmt.format(low)}-{fmt.format(high)} {what}", **stats, "verdict": verdict(stats["low"], stats["high"])})
    return pd.DataFrame(rows)


def _items(view, relevant: list[np.ndarray], top_a: np.ndarray, top_b: np.ndarray, head: np.ndarray) -> dict[str, pd.DataFrame]:
    buckets = {"popular head (top 20% of items)": [0, 0, 0], "long tail": [0, 0, 0], "new (no history before the test)": [0, 0, 0]}
    gained, lost = Counter(), Counter()
    for rel, la, lb in zip(relevant, top_a, top_b):
        if not len(rel):
            continue
        hit_a, hit_b = np.isin(rel, la), np.isin(rel, lb)
        for item, ha, hb in zip(rel.tolist(), hit_a, hit_b):
            key = ("new (no history before the test)" if view.item_pop[item] == 0 else
                   "popular head (top 20% of items)" if head[item] else "long tail")
            buckets[key][0] += 1
            buckets[key][1] += int(ha)
            buckets[key][2] += int(hb)
            if hb and not ha:
                gained[item] += 1
            elif ha and not hb:
                lost[item] += 1
    items = pd.DataFrame([{"test items": name, "pairs": n, "recall A": a / n if n else np.nan, "recall B": b / n if n else np.nan,
                           "difference": (b - a) / n if n else np.nan} for name, (n, a, b) in buckets.items()])

    def named(counter: Counter) -> pd.DataFrame:
        return pd.DataFrame([{"item": _item_name(view, i), "users": c, "popularity": int(view.item_pop[i])} for i, c in counter.most_common(10)])

    return {"items": items, "gained": named(gained), "lost": named(lost)}


def _item_name(view, item: int) -> str:
    text = str(view.item_text[item] or "").split(". ")[0]
    return f"{text[:60]} ({view.item_ids[item]})" if text else str(view.item_ids[item])


# ----- command line -----------------------------------------------------------------------------------------------------

def _num(value: Any, digits: int = 4, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def render(frame: pd.DataFrame, a: Ref, b: Ref, metric: str) -> str:
    lines = [f"A = {a}   B = {b}   test {metric}, paired over the same users (95% interval of B - A)", ""]
    header = f"{'dataset':15s} {'users':>6s} {'A':>8s} {'B':>8s} {'B - A':>8s}  {'95% interval':19s} {'users +/-':>11s}  verdict"
    lines.append(header)
    for _, r in frame.iterrows():
        if r.get("verdict") == "-" and isinstance(r.get("note"), str):
            lines.append(f"{r['dataset']:15s} {r['note']}")
            continue
        lines.append(f"{r['dataset']:15s} {int(r['users']):6d} {_num(r['a']):>8s} {_num(r['b']):>8s} {_num(r['diff'], sign=True):>8s}  "
                     f"[{_num(r['low'], sign=True)}, {_num(r['high'], sign=True)}] {r['better_users']:5.0%}/{r['worse_users']:<5.0%}  {r['verdict']}")
    lines += ["", f"{'dataset':15s} {'validation A -> B':>19s} {'train s A -> B':>17s} {'score s/1k A -> B':>18s} {'coverage A -> B':>16s} {'seeds':>6s}"]
    for _, r in frame.iterrows():
        if r.get("verdict") == "-" and isinstance(r.get("note"), str):
            continue
        lines.append(f"{r['dataset']:15s} {_num(r['a_val'])} -> {_num(r['b_val']):6s} {_num(r['a_train_seconds'], 1):>7s} -> {_num(r['b_train_seconds'], 1):6s} "
                     f"{_num(r['a_score_seconds_per_1k_users'], 2):>8s} -> {_num(r['b_score_seconds_per_1k_users'], 2):6s} "
                     f"{_num(r['a_coverage_at_10'], 3):>6s} -> {_num(r['b_coverage_at_10'], 3):6s} {int(r['a_seeds'])}/{int(r['b_seeds'])}")
    lines += ["", "Promotion rule: " + promotion(frame)]
    return "\n".join(lines)


def render_segments(parts: dict[str, pd.DataFrame], dataset: str) -> str:
    lines = [f"\n== {dataset}: where B - A comes from"]
    for name in ("activity", "recency", "taste"):
        lines.append("")
        for _, r in parts[name].iterrows():
            lines.append(f"  {r['group']:52s} {int(r['users']):6d} users  A {_num(r['a'])}  B {_num(r['b'])}  "
                         f"B - A {_num(r['diff'], sign=True)} [{_num(r['low'], sign=True)}, {_num(r['high'], sign=True)}]  {r['verdict']}")
    if "items" in parts:
        lines.append("")
        for _, r in parts["items"].iterrows():
            lines.append(f"  recall@10, {r['test items']:34s} {int(r['pairs']):7d} pairs  A {_num(r['recall A'])}  B {_num(r['recall B'])}  "
                         f"B - A {_num(r['difference'], sign=True)}")
        for title, key in (("found by B, missed by A", "gained"), ("found by A, missed by B", "lost")):
            if len(parts[key]):
                lines.append(f"\n  most often {title}:")
                lines += [f"    {r['users']:4d} users  {r['item']} (popularity {r['popularity']})" for _, r in parts[key].iterrows()]
    else:
        lines.append("\n  (item-level analysis needs runs that saved their top-10 lists: the lab's eval.save_topk)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m recbench.compare", description="Paired comparison of two lab results (B against A).")
    parser.add_argument("a", help="method[:experiment], the reference (for example itemknn)")
    parser.add_argument("b", help="method[:experiment], the candidate (for example itemknn:no-time-knobs)")
    parser.add_argument("--datasets", default="all", help="comma-separated, default all five")
    parser.add_argument("--metric", default="ndcg_at_10", help="a per-user metric (default ndcg_at_10)")
    parser.add_argument("--segments", action="store_true", help="also break the difference down (error analysis)")
    parser.add_argument("--bake-off", action="store_true", help="compare the bake-off's quick-tier results instead of the lab's")
    args = parser.parse_args(argv)
    workspace = "" if args.bake_off else "lab"
    use_workspace(workspace)
    a, b = Ref.parse(args.a), Ref.parse(args.b)
    datasets = None if args.datasets in ("", "all") else [d.strip() for d in args.datasets.split(",") if d.strip()]
    frame = compare(a, b, datasets, metric=args.metric, workspace=workspace)
    print(render(frame, a, b, args.metric))
    if args.segments:
        for dataset in frame["dataset"]:
            try:
                print(render_segments(segments(a, b, dataset, metric=args.metric, workspace=workspace), dataset))
            except Missing as exc:
                print(f"\n== {dataset}: {exc}")
    if frame.get("verdict", pd.Series(dtype=str)).eq("-").all():
        sys.exit(1)


if __name__ == "__main__":
    main()
