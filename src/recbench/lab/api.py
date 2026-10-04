"""Notebook helpers for the labs (labs/<nn>-<method>/<method>.ipynb).

Call `setup()` first. It loads only the light method modules: none of them imports PyTorch, so SANSA and LightGBM can
share one notebook kernel on macOS. It also switches to the lab workspace and prints a health check.

`fit` + `evaluate` run in the notebook itself, so you can look inside a fitted model; they give the same numbers as
`python -m recbench.lab once` (same training window, same seeded validation users). The other helpers call the lab
commands: they run in child processes and store their results in the lab's MLflow store.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recbench.lab import experiments as ex
from recbench.lab import runs
from recbench.paths import use_workspace

LIGHT_MODULES = ("baselines", "neighbourhood", "linear", "graph_filters", "implicit_mf", "rerank")


def _find_repo(start: Path) -> Path | None:
    for folder in [start, *start.parents]:
        project = folder / "pyproject.toml"
        if project.exists() and 'name = "recbench"' in project.read_text():
            return folder
    return None


def setup(verbose: bool = True) -> None:
    """Prepare this notebook: the repository root (the notebook runs in labs/<nn>-<method>/), the light method modules, the
    lab workspace. Prints which splits and baselines exist."""
    root = _find_repo(Path.cwd())
    if root is None and "RECBENCH_ROOT" not in os.environ:
        raise RuntimeError("cannot find the recbench repository above this folder; set RECBENCH_ROOT")
    if root is not None:
        os.environ["RECBENCH_ROOT"] = str(root)
    if "recbench.methods" in sys.modules and not os.environ.get("RECBENCH_METHOD_MODULES"):
        print("note: every method module is already loaded (PyTorch may be too). If SANSA or LightGBM crashes, restart the kernel "
              "and run lab.setup() first.")
    os.environ["RECBENCH_METHOD_MODULES"] = ",".join(LIGHT_MODULES)
    use_workspace("lab")
    runs.lab_config.cache_clear()
    runs.lab_config()
    if not verbose:
        return
    datasets = ex.lab_datasets()
    print(f"recbench repository: {os.environ.get('RECBENCH_ROOT')}; workspace: lab (runs/lab, reports/lab)")
    ready = [d for d in datasets if _has_split(d, "val") and _has_split(d, "test")]
    print(f"splits ready: {', '.join(ready) or 'none'}" + (f"; missing: {', '.join(sorted(set(datasets) - set(ready)))}" if set(datasets) - set(ready) else ""))
    done = {m: sum((runs.summary(m, d) or {}).get("status") == "finished" for d in datasets) for m in [ex.REFERENCE, *ex.lab_methods()]}
    print("baselines finished (datasets): " + ", ".join(f"{m} {n}/{len(datasets)}" for m, n in done.items()))


def _has_split(dataset: str, fold: str) -> bool:
    try:
        runs.split_dir(dataset, fold)
        return True
    except FileNotFoundError:
        return False


def load(dataset: str, fold: str = "val"):
    """(TrainView, EvalSplit) of the quick tier's validation fold (fold="val", the default) or test split (fold="test").
    Tune and compare settings on the validation fold only; the test split is for looking, not for choosing."""
    from recbench.data import TrainView
    from recbench.evaluation import EvalSplit

    path = runs.split_dir(dataset, fold)
    if fold != "val":
        print("This is the TEST split: look at it, but never choose a setting by its score here.")
    return TrainView(path), EvalSplit(path)


def config(method: str, **settings: Any) -> dict[str, Any]:
    """The settings a lab run of `method` gets (lab defaults plus yours)."""
    from recbench.config import method_config

    _, resolved, _, _ = runs.lab_config()
    return {**method_config(resolved, method), "ease_backend": "numpy", **settings}


def fit(method: str, data, **settings: Any):
    """Fit a method in this notebook, as a lab run does (same training window). Returns the fitted model; its fit time is
    `model.lab_seconds`."""
    from recbench.registry import ensure_loaded

    unread = ex.unread_settings(method, set(settings))
    if unread:
        print(f"warning: {method}'s code never reads {', '.join(unread)} (a typo, or a variant not written yet)")
    cfg = config(method, **settings)
    np.random.seed(int(cfg.get("seed", 42)))
    fit_data = data.restrict(cfg.get("train_window_days"), int(cfg.get("train_window_keep_last", 10)))
    model = ensure_loaded().create_method(method)
    began = time.perf_counter()
    model.fit(fit_data, cfg)
    model.lab_seconds, model.lab_cfg = time.perf_counter() - began, cfg
    print(f"fitted {method} in {model.lab_seconds:.1f} s")
    return model


@dataclass
class Evaluation:
    metrics: pd.Series  # every metric (ndcg_at_10, its interval, recall_at_10, coverage_at_10, ...)
    per_user: pd.DataFrame  # one row per scored user: user_idx and the per-user metrics

    def __repr__(self) -> str:
        m = self.metrics
        return (f"NDCG@10 {m['ndcg_at_10']:.4f} [{m.get('ndcg_at_10_ci_low', float('nan')):.4f}, {m.get('ndcg_at_10_ci_high', float('nan')):.4f}] "
                f"on {int(m['n_eval_users'])} users; recall@10 {m.get('recall_at_10', float('nan')):.4f}; "
                f"coverage@10 {m.get('coverage_at_10', float('nan')):.3f}")


def evaluate(model, data, split, users: int = 3000) -> Evaluation:
    """Score a model fitted with `fit` on the split's users (the same seeded sample of `users` as the lab commands)."""
    from recbench.evaluation import Evaluator, per_user_frame

    cfg = {**model.lab_cfg, "max_eval_users": int(users)}
    result = Evaluator(split, data, cfg).run(model, extra={"train_seconds": getattr(model, "lab_seconds", float("nan"))})
    return Evaluation(pd.Series(result.metrics), per_user_frame(result))


def baseline(method: str, dataset: str | None = None):
    """The laptop baseline: per dataset its status, validation and test NDCG@10, best settings and job time."""
    rows = []
    for d in [dataset] if dataset else ex.lab_datasets():
        found = runs.summary(method, d) or {}
        test = found.get("test") or {}
        rows.append({"dataset": d, "status": found.get("status", "not run"), "validation": found.get("best_val"),
                     "test": test.get("ndcg_at_10"), "test_seed_sd": test.get("ndcg_at_10_seed_sd"), "trials": len(found.get("trials") or []),
                     "minutes": (found["ended"] - found["started"]) / 60 if found.get("ended") and found.get("started") else None,
                     "best_settings": found.get("best_params")})
    frame = pd.DataFrame(rows).set_index("dataset")
    return frame.loc[dataset] if dataset else frame


def trials(method: str, dataset: str, experiment: str = ex.BASELINE) -> pd.DataFrame:
    """The settings the tuner tried for a job, with their validation NDCG@10 and seconds."""
    found = runs.summary(method, dataset, None if experiment == ex.BASELINE else experiment) or {}
    rows = [{"trial": t["number"], "validation": t.get("value"), "seconds": t.get("seconds"), "status": t.get("status"), **(t.get("params") or {})}
            for t in found.get("trials") or []]
    return pd.DataFrame(rows).sort_values("validation", ascending=False) if rows else pd.DataFrame()


def once(method: str, dataset: str, best: bool = False, users: int | None = None, fresh: bool = False, **settings: Any) -> dict[str, Any]:
    """`python -m recbench.lab once`: one setting, scored on the validation fold (best=True starts from the baseline's best
    setting on this dataset; your settings change it)."""
    result = runs.once(method, dataset, settings, best=best, users=users, fresh=fresh)
    m = result.get("metrics") or {}
    if m:
        print(f"{method} on {dataset}: validation NDCG@10 {m.get('ndcg_at_10', float('nan')):.4f} "
              f"[{m.get('ndcg_at_10_ci_low', float('nan')):.4f}, {m.get('ndcg_at_10_ci_high', float('nan')):.4f}] ({result['status']})")
    return result


def sweep(method: str, dataset: str, param: str, values: list[Any], start: str = "best", users: int | None = None, **settings: Any) -> pd.DataFrame:
    """`python -m recbench.lab sweep`: vary one setting on the validation fold; returns one row per value."""
    return runs.sweep(method, dataset, param, list(values), settings=settings, start=start, users=users)


def run(method: str, experiment: str, datasets: list[str] | None = None, workers: int = 2, rerun: bool = False) -> list[dict[str, Any]]:
    """`python -m recbench.lab run`: an experiment from the method's experiments.yaml, tuned and tested like the baseline.
    Long on the big datasets: for those, prefer the terminal command."""
    return runs.run_experiment(method, experiment, datasets, workers=workers, rerun=rerun)


def results(method: str) -> pd.DataFrame:
    """Every experiment of a method and its test NDCG@10 (or status) per dataset."""
    return runs.experiment_table(method)


def compare(a: str, b: str, datasets: list[str] | None = None, metric: str = "ndcg_at_10") -> pd.DataFrame:
    """`python -m recbench.compare A B`: the paired test per dataset. Prints the promotion rule's verdict."""
    from recbench.compare import compare as _compare
    from recbench.compare import promotion

    frame = _compare(a, b, datasets, metric=metric)
    print("Promotion rule: " + promotion(frame))
    columns = [c for c in ("dataset", "users", "a", "b", "diff", "low", "high", "better_users", "worse_users", "verdict", "note") if c in frame]
    return frame[columns]


def segments(a: str, b: str, dataset: str, metric: str = "ndcg_at_10") -> dict[str, pd.DataFrame]:
    """`python -m recbench.compare A B --segments` for one dataset: {"activity", "recency", "taste", "items", "gained", "lost"}."""
    from recbench.compare import segments as _segments

    return _segments(a, b, dataset, metric=metric)


def find_items(data, text: str, k: int = 10) -> pd.DataFrame:
    """Items whose title or description contains `text` (case-insensitive), most popular first."""
    names = pd.Series(data.item_text, dtype=object).astype(str)
    hits = np.flatnonzero(names.str.contains(text, case=False, regex=False).to_numpy())
    hits = hits[hits > 0]
    hits = hits[np.argsort(-data.item_pop[hits], kind="stable")][:k]
    return pd.DataFrame({"item": hits, "id": data.item_ids[hits], "title": names.to_numpy()[hits], "popularity": data.item_pop[hits]})


def neighbours(model, data, item: int | str, k: int = 10) -> pd.DataFrame:
    """The k items the model scores highest for someone whose whole history is `item` (an index, an item id, or part of a
    title). For item-to-item models (ItemKNN, RP3beta, EASE, SLIM, SANSA, PureSVD, GF-CF); for iALS and BPR-MF, the
    nearest item vectors (cosine)."""
    import scipy.sparse as sp

    idx = _item_index(data, item)
    if hasattr(model, "user_factors"):  # matrix factorisation: compare item vectors
        vectors = model.item_factors / np.maximum(np.linalg.norm(model.item_factors, axis=1, keepdims=True), 1e-9)
        scores = (vectors @ vectors[idx]).astype(np.float64)
    elif hasattr(model, "seen"):
        one = sp.csr_matrix((np.ones(1, dtype=np.float32), ([0], [idx])), shape=(1, data.n_items + 1))
        saved, model.seen = model.seen, one  # score a pretend user whose history is this one item
        try:
            scores = np.asarray(model.score_users(np.array([0]), None))[0].astype(np.float64)
        finally:
            model.seen = saved
    else:
        raise ValueError(f"{model.spec.name} has no item-to-item scores")
    scores[[0, idx]] = -np.inf
    top = np.argsort(-scores, kind="stable")[:k]
    top = top[np.isfinite(scores[top])]
    return pd.DataFrame({"item": top, "title": [_title(data, i) for i in top], "score": scores[top], "popularity": data.item_pop[top]})


def _item_index(data, item: int | str) -> int:
    if isinstance(item, (int, np.integer)):
        return int(item)
    matches = np.flatnonzero(data.item_ids == str(item))
    if len(matches):
        return int(matches[0])
    found = find_items(data, str(item), k=1)
    if found.empty:
        raise KeyError(f"no item with id or title {item!r}")
    return int(found["item"].iloc[0])


def _title(data, item: int) -> str:
    text = str(data.item_text[item] or "")
    return text[:70] if text else str(data.item_ids[item])


def plot_sweep(frame: pd.DataFrame, title: str | None = None):
    """NDCG@10 against the swept setting, with the 95% intervals; the baseline's value is marked."""
    import matplotlib.pyplot as plt

    param = frame.columns[0]
    x = np.arange(len(frame))
    y = frame["ndcg_at_10"].astype(float)
    err = np.vstack([y - frame["ci_low"].astype(float), frame["ci_high"].astype(float) - y])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.errorbar(x, y, yerr=err, fmt="o-", capsize=3)
    for i in np.flatnonzero(frame.get("is_baseline_value", pd.Series(False, index=frame.index)).to_numpy()):
        ax.plot(x[i], y.iloc[i], "s", color="tab:red", label="the baseline's value")
    ax.set_xticks(x, [str(v) for v in frame[param]])
    ax.set_xlabel(param)
    ax.set_ylabel("validation NDCG@10")
    ax.set_title(title or f"Sweep of {param}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_segments(parts: dict[str, pd.DataFrame], kind: str = "activity"):
    """B - A per user group (with 95% intervals) for one kind of segment from `segments`."""
    import matplotlib.pyplot as plt

    frame = parts[kind]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = frame["diff"].astype(float)
    err = np.vstack([y - frame["low"].astype(float), frame["high"].astype(float) - y])
    ax.bar(np.arange(len(frame)), y, yerr=err, capsize=3, color=["tab:green" if v > 0 else "tab:red" for v in y])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(frame)), [g.split(" ", 1)[0] for g in frame["group"]])
    ax.set_xlabel(frame["group"].iloc[0].split(" ", 1)[1] if len(frame) else kind)
    ax.set_ylabel("B - A (NDCG@10)")
    fig.tight_layout()
    return fig
