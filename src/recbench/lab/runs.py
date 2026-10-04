"""What the lab commands do. `python -m recbench.lab ...` and the notebooks call these same functions.

Every function here works in the lab workspace (runs/lab/, reports/lab/), so nothing touches the bake-off's results.
`once` and `sweep` only ever use a validation fold; `run_experiment` tunes on the validation fold and tests the chosen
setting once, exactly like the baseline.
"""

from __future__ import annotations

import concurrent.futures
import functools
import os
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from recbench.lab import experiments as ex
from recbench.paths import reports_dir, repo_root

FINISHED_OR_FINAL = {"finished", "over_budget", "failed", "unsupported", "missing_split", "skipped"}
ISOLATE = True  # each fit in its own child process, as in the bake-off (tests switch it off to run in-process)
Log = Callable[[str], None]


@functools.lru_cache(maxsize=1)
def lab_config() -> tuple[dict, dict, Any, Any]:
    """(benchmark, resolved config, search spaces, job settings) of configs/benchmarks/lab.yaml. Loading it switches
    this process (and the processes it starts) to the lab workspace. Callers copy `resolved` before changing it."""
    from recbench.tuning.__main__ import load_benchmark

    return load_benchmark(repo_root() / ex.LAB_CONFIG)


def tier() -> str:
    return lab_config()[3].tier


def split_dir(dataset: str, fold: str = "val") -> Path:
    """The quick tier's validation fold (fold="val") or its test split (fold="test")."""
    from recbench.runner import data_root

    path = data_root() / "splits" / dataset / (f"{tier()}-val" if fold == "val" else tier())
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no split at {path}: prepare it with python -m recbench.pipeline.prepare "
                                f"--config configs/benchmarks/quick.yaml --tier {tier()},{tier()}-val")
    return path


def parse_value(text: str) -> Any:
    """A command-line value as YAML reads it: 200 -> int, 0.5 -> float, null -> None, bm25 -> str."""
    return yaml.safe_load(text)


def parse_settings(items: list[str] | None) -> dict[str, Any]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ex.ExperimentError(f"--set expects name=value, got {item!r}")
        name, value = item.split("=", 1)
        out[name.strip()] = parse_value(value.strip())
    return out


def summary(method: str, dataset: str, label: str | None = None) -> dict[str, Any] | None:
    """A lab job's summary: the baseline (label None or "baseline") or an experiment."""
    from recbench.tuning.job import read_summary

    lab_config()
    return read_summary(tier(), dataset, method, label=label)


def _metrics_by_hash(run_hash: str) -> dict[str, float]:
    """All metrics of the latest finished run with this identity in the lab's MLflow store."""
    from recbench.runner import EXPERIMENT, _mlflow

    mlflow = _mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return {}
    frame = mlflow.search_runs([experiment.experiment_id], filter_string=f"tags.config_hash = '{run_hash}' and tags.status = 'finished'",
                               order_by=["attributes.start_time DESC"], max_results=1)
    if frame.empty:
        return {}
    row = frame.iloc[0]
    return {c[len("metrics."):]: float(row[c]) for c in frame.columns if c.startswith("metrics.") and row[c] == row[c]}


def once(method: str, dataset: str, settings: dict[str, Any] | None = None, *, best: bool = False, users: int | None = None,
         fresh: bool = False) -> dict[str, Any]:
    """Fit one setting and score it on the dataset's validation fold (never the test split).

    best=True starts from the baseline's best setting on this dataset; `settings` change it. The same call twice
    returns the stored result instantly, unless fresh=True. Returns {status, settings, metrics, seconds}."""
    from recbench.data import TrainView
    from recbench.runner import run_hash_for, run_pair
    from recbench.tuning.job import _base_cfg

    _, resolved, _, job = lab_config()
    val = split_dir(dataset, "val")
    params: dict[str, Any] = {}
    if best:
        found = summary(method, dataset)
        if not found or not found.get("best_params"):
            raise ex.ExperimentError(f"{method} has no baseline on {dataset} yet: python -m recbench.lab baseline --methods {method} "
                                     f"--datasets {dataset}")
        params = dict(found["best_params"])
    settings = dict(settings or {})
    unread = ex.unread_settings(method, set(settings))
    if unread:
        raise ex.ExperimentError(f"{method}'s code never reads {', '.join(unread)}: a typo, or a variant not written yet")
    cfg = {**_base_cfg(resolved, method, set(params) | set(settings)), **params, **settings, "stage": "once", "tuning": "lab",
           "export_bundles": False, "resume": not fresh, "max_eval_users": int(users or job.search_users),
           "timeout_minutes": job.cap_minutes}
    began = time.time()
    outcome = run_pair(val, method, cfg, isolate=ISOLATE)
    if outcome.get("status") not in ("finished", "skipped_existing"):
        return {"status": outcome.get("status"), "reason": outcome.get("reason", ""), "settings": {**params, **settings}, "metrics": {}}
    metrics = _metrics_by_hash(run_hash_for(cfg, method, TrainView(val).split_hash))
    return {"status": "cached" if outcome["status"] == "skipped_existing" else "finished", "settings": {**params, **settings},
            "metrics": metrics, "seconds": time.time() - began}


def sweep(method: str, dataset: str, param: str, values: list[Any], *, settings: dict[str, Any] | None = None, start: str = "best",
          users: int | None = None, log: Log = print) -> pd.DataFrame:
    """Change one setting and keep everything else at the baseline's best setting (start="best") or at the defaults
    (start="defaults"). Validation fold only. Also written to reports/lab/sweeps/<method>/<dataset>-<param>.csv."""
    rows = []
    best = (summary(method, dataset) or {}).get("best_params") or {} if start == "best" else {}
    for value in values:
        result = once(method, dataset, {**(settings or {}), param: value}, best=(start == "best"), users=users)
        m = result.get("metrics") or {}
        rows.append({param: value, "ndcg_at_10": m.get("ndcg_at_10"), "ci_low": m.get("ndcg_at_10_ci_low"),
                     "ci_high": m.get("ndcg_at_10_ci_high"), "recall_at_10": m.get("recall_at_10"), "coverage_at_10": m.get("coverage_at_10"),
                     "train_seconds": m.get("train_seconds"), "status": result["status"],
                     "is_baseline_value": param in best and best[param] == value})
        log(f"  {param}={value!s:>10}  NDCG@10={_fmt(m.get('ndcg_at_10'))} [{_fmt(m.get('ndcg_at_10_ci_low'))}, "
            f"{_fmt(m.get('ndcg_at_10_ci_high'))}]  train {_fmt(m.get('train_seconds'), 1)} s  ({result['status']})")
    frame = pd.DataFrame(rows)
    out = reports_dir() / "sweeps" / method / f"{dataset}-{param}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def run_experiment(method: str, label: str, datasets: list[str] | None = None, *, workers: int = 1, rerun: bool = False,
                   log: Log = print) -> list[dict[str, Any]]:
    """Run one experiment from the method's experiments.yaml on each dataset, with the baseline's budget: search on the
    validation fold, then test the chosen setting once. A dataset is skipped when it is already done with the same
    definition and code; it runs again when either changed, or with rerun=True."""
    from recbench.runner import method_version
    from recbench.tuning.job import prepare_rerun, run_job

    _, resolved, _, job = lab_config()
    experiment = ex.get(method, label)
    unread = ex.unread_settings(method, set(experiment.space.params) | set(experiment.space.fixed))
    if unread:
        raise ex.ExperimentError(f"{method}'s code never reads {', '.join(unread)}. Write the variant first (Level 4 of the lab), "
                                 "or fix the spelling.")
    datasets = datasets or ex.lab_datasets()
    fingerprint = None if experiment.is_baseline else ex.fingerprint(experiment, method_version(method, resolved))
    tag = None if experiment.is_baseline else label
    todo = []
    for dataset in datasets:
        found = summary(method, dataset, tag) or {}
        status = found.get("status")
        if status in FINISHED_OR_FINAL:
            changed = bool(fingerprint) and found.get("fingerprint") != fingerprint
            if rerun or changed:
                prepare_rerun(tier(), dataset, method, label=tag)
                log(f"{dataset}: running again ({'--rerun' if rerun else 'the experiment or its code changed since it ran'})")
            elif status == "finished":
                log(f"{dataset}: already done (test NDCG@10 {_fmt((found.get('test') or {}).get('ndcg_at_10'))}); --rerun runs it again")
                continue
            elif status != "failed":  # failed jobs start a new attempt on their own (retry below)
                log(f"{dataset}: {status} ({found.get('reason', '')}); --rerun runs it again")
                continue
        if method == "lgbm_rerank":
            missing = [g for g in ("ease", "itemknn") if (summary(g, dataset) or {}).get("status") != "finished"]
            if missing:
                log(f"{dataset}: note: no lab baseline for {', '.join(missing)} yet, so the re-ranker uses their default settings")
        todo.append(dataset)
    if not todo:
        return []
    workers = max(1, min(int(workers), len(todo)))
    settings = {**resolved, "threads": max(1, min(32, (os.cpu_count() or 1) // workers))}
    log(f"{method}:{label} on {', '.join(todo)} ({workers} at a time; each job: {job.trials if experiment.space.trials is None else experiment.space.trials} "
        f"settings on {tier()}-val, then the best once on {tier()})")

    def one(dataset: str) -> dict[str, Any]:
        result = run_job(dataset, method, settings, job, experiment.space, child_env={"CUDA_VISIBLE_DEVICES": ""}, retry=True,
                         isolate=ISOLATE, label=tag, fingerprint=fingerprint)
        test = (result.get("test") or {}).get("ndcg_at_10")
        log(f"{dataset}: {result.get('status')}" + (f", test NDCG@10 {test:.4f} (validation {result.get('best_val'):.4f})" if test is not None
                                                     else f" ({result.get('reason', '')})"))
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, todo))
    if not experiment.is_baseline:
        log(f"\nNext: python -m recbench.compare {method} {method}:{label}")
    return results


def experiment_table(method: str) -> pd.DataFrame:
    """One row per experiment of a method (from its experiments.yaml and the summaries on disk), one column per dataset."""
    from recbench.tuning.job import tuning_dir

    lab_config()
    labels = list(ex.load_experiments(method))
    for path in sorted(tuning_dir().glob(f"{tier()}/*/{method}@*.json")):
        label = path.stem.split("@", 1)[1]
        if label not in labels:
            labels.append(label)
    rows = []
    for label in labels:
        row: dict[str, Any] = {"experiment": label}
        for dataset in ex.lab_datasets():
            found = summary(method, dataset, None if label == ex.BASELINE else label) or {}
            test = (found.get("test") or {}).get("ndcg_at_10") if found.get("status") == "finished" else None
            row[dataset] = f"{test:.4f}" if test is not None else (found.get("status") or "not run")
        rows.append(row)
    return pd.DataFrame(rows)


def status(log: Log = print) -> None:
    """The baseline queue's progress, then each lab's experiments, with a warning where code changed after the baseline."""
    from recbench.queue import print_status
    from recbench.runner import method_version

    lab_config()
    print_status(repo_root() / ex.LAB_CONFIG)
    log("\n== experiments (test NDCG@10 per dataset)")
    for method in ex.lab_methods():
        log(f"\n{method}")
        log(experiment_table(method).to_string(index=False))
        current = method_version(method, {"track_code": True})
        ran = {(summary(method, d) or {}).get("impl_version") for d in ex.lab_datasets()} - {None}
        if ran and current not in ran:
            log(f"  note: {method}'s code changed after its baseline ran. That is fine while the default behaviour is unchanged "
                "(tests/test_lab_defaults.py checks it).")


def _fmt(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"
