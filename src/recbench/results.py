"""Read benchmark results back from MLflow (protocol v2 runs only).

Reports, the dashboard, and the generated docs all use these helpers, so they
always agree on which run counts: the most recent finished run per
(dataset, tier, method).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from recbench.protocol import PROTOCOL_VERSION

# Leaderboard metric per task. CTR is diagnostic (no real impressions in these datasets).
TASK_BOARDS = {
    "topn": ("ndcg_at_10", "Top-N for warm users (full ranking)"),
    "sequential": ("next_ndcg_at_10", "Next-item prediction (full ranking)"),
    "ctr": ("sampled_auc", "CTR-style ranking vs. random negatives (diagnostic)"),
}
LOWER_IS_BETTER = {"gini_at_10", "popularity_percentile_at_10", "calibration_kl_at_10", "user_group_ndcg_gap_at_10",
                   "train_seconds", "score_seconds_per_1k_users", "peak_rss_mb", "peak_gpu_mb", "sampled_logloss",
                   "served_p50_ms", "served_p95_ms", "served_p99_ms"}


def tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    root = Path(os.environ.get("RECBENCH_ROOT", Path.cwd())).resolve()
    return (root / "runs" / "mlflow").as_uri()


def load_runs(tier: str | None = None, *, include_unfinished: bool = False, tuning: str | None = None) -> pd.DataFrame:
    """One row per (dataset, tier, method): tags.* and metrics.* columns, protocol v2 only.

    tuning: "defaults" or "tuned" keeps only runs with that tag (runs from before the tag count as defaults).
    When the latest finished run of a method was repeated with other seeds (same tags.config_group), its
    metrics are averaged over those seeds; metrics.n_seeds and metrics.ndcg_at_10_seed_sd say how many and
    how much they varied.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    experiment = mlflow.get_experiment_by_name("recbench")
    if experiment is None:
        return pd.DataFrame()
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=50_000)
    if frame.empty:
        return frame
    frame = frame[frame.get("tags.protocol_version", pd.Series(dtype=str)) == PROTOCOL_VERSION]
    if tier:
        frame = frame[frame["tags.tier"] == tier]
    if tuning:
        tags = frame["tags.tuning"].fillna("defaults") if "tags.tuning" in frame else pd.Series("defaults", index=frame.index)
        frame = frame[tags == tuning]
    if not include_unfinished:
        frame = frame[frame["tags.status"] == "finished"]
    frame = frame.sort_values("start_time")
    latest = frame.drop_duplicates(["tags.dataset", "tags.tier", "tags.method"], keep="last").copy()
    if not include_unfinished and "tags.config_group" in frame and not latest.empty:
        metric_cols = [c for c in frame.columns if c.startswith("metrics.")]
        for idx, row in latest.iterrows():
            group = row.get("tags.config_group")
            if not isinstance(group, str) or not group:
                continue
            same = frame[(frame["tags.config_group"] == group) & (frame["tags.dataset"] == row["tags.dataset"])
                         & (frame["tags.tier"] == row["tags.tier"]) & (frame["tags.method"] == row["tags.method"])]
            if len(same) > 1:
                latest.loc[idx, metric_cols] = same[metric_cols].mean(numeric_only=True)
                latest.loc[idx, "metrics.ndcg_at_10_seed_sd"] = float(same["metrics.ndcg_at_10"].std(ddof=0))
            latest.loc[idx, "metrics.n_seeds"] = float(len(same))
    return latest.reset_index(drop=True)


def latest_status(tier: str | None = None) -> pd.DataFrame:
    """The latest run of every (dataset, method), whatever its status, with the reason when it did not finish."""
    frame = load_runs(tier, include_unfinished=True)
    if frame.empty:
        return frame
    cols = ["tags.dataset", "tags.method", "tags.status", "tags.reason"]
    for col in cols:
        if col not in frame:
            frame[col] = ""
    return frame[cols].rename(columns=lambda c: c.split(".", 1)[1]).fillna("")


def leaderboard(frame: pd.DataFrame, dataset: str, metric: str, *, ranked_only: bool = True) -> pd.DataFrame:
    """Methods for one dataset sorted by `metric`, with CI columns and a 'tied with best' flag."""
    column = f"metrics.{metric}"
    if frame.empty or column not in frame:
        return pd.DataFrame()
    rows = frame[(frame["tags.dataset"] == dataset) & frame[column].notna()].copy()
    if ranked_only and "tags.ranked" in rows:
        rows = rows[rows["tags.ranked"] != "false"]
    if rows.empty:
        return rows
    ascending = metric in LOWER_IS_BETTER
    rows = rows.sort_values(column, ascending=ascending).reset_index(drop=True)
    low, high = f"metrics.{metric}_ci_low", f"metrics.{metric}_ci_high"
    if low in rows and high in rows and rows[low].notna().any():
        best_low, best_high = rows.loc[0, low], rows.loc[0, high]
        rows["tied_with_best"] = (rows[high] >= best_low) & (rows[low] <= best_high)
    else:
        rows["tied_with_best"] = np.nan
    rows.insert(0, "rank", np.arange(1, len(rows) + 1))
    return rows


def fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}"
    return str(value)
