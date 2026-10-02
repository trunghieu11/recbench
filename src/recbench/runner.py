"""Fit methods, score the shared candidate file, and log to MLflow."""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mlflow
import pyarrow.parquet as pq

from recbench.config import load_yaml, resolve_run_config
from recbench.metrics.catalog import item_counts, user_categories
from recbench.protocol import PROTOCOL_NOTE, Task, Unsupported
from recbench.registry import ensure_loaded
from recbench.store import SplitStore


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./runs/mlflow")


def _finished(dataset: str, method: str, tier: str, preset: str, max_steps: int) -> bool:
    experiment = mlflow.get_experiment_by_name("recbench")
    if experiment is None:
        return False
    frame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.dataset = '{dataset}' and tags.method = '{method}' and tags.tier = '{tier}' "
            f"and tags.preset = '{preset}' and tags.max_steps = '{max_steps}' and tags.status = 'finished'"
        ),
    )
    return not frame.empty


def run_pair(store: SplitStore, method_name: str, resolved: dict[str, Any], resume: bool) -> str:
    reg = ensure_loaded()
    method = reg.create_method(method_name)
    tier = resolved["tier"]
    preset = resolved["preset"]
    rankable = tier == "full" and not method.spec.managed
    if method.spec.managed and not resolved.get("managed_services", False):
        return _log_status(store, method_name, resolved, "skipped", rankable=False, reason="managed services off")
    if method.spec.requires_images and not store.images_available():
        return _log_status(store, method_name, resolved, "unsupported", rankable=False, reason="no images")
    if resume and _finished(store.dataset, method_name, tier, preset, resolved["max_steps"]):
        return "skipped_existing"
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("recbench")
    with mlflow.start_run(run_name=f"{store.dataset}-{method_name}-{tier}-{preset}"):
        mlflow.set_tags(
            {
                "dataset": store.dataset,
                "method": method_name,
                "tier": tier,
                "preset": preset,
                "max_steps": str(resolved["max_steps"]),
                "hardware": resolved["hardware_name"],
                "status": "running",
                "rankable": str(rankable).lower(),
                "managed": str(method.spec.managed).lower(),
                "protocol": PROTOCOL_NOTE,
            }
        )
        mlflow.log_params(
            {
                "model_config": resolved["model_config"],
                "dim": resolved["dim"],
                "layers": resolved["layers"],
                "seq_len": resolved["seq_len"],
                "batch_size": resolved["batch_size"],
                "n_negatives": store.meta.get("n_negatives", 100),
                "split_mode": store.meta.get("split_mode", ""),
            }
        )
        try:
            started = time.perf_counter()
            method.fit(store, resolved)
            train_s = time.perf_counter() - started
            candidates = pq.read_table(store.candidates_path).to_pandas()
            users = candidates["user_idx"].drop_duplicates().head(1000)
            subset = candidates[candidates["user_idx"].isin(set(users))]
            infer_started = time.perf_counter()
            scored_subset = method.score_candidates(store, subset) if method.spec.can_score_candidates else None
            infer_s = time.perf_counter() - infer_started
            if scored_subset is not None and len(users) < candidates["user_idx"].nunique():
                scored = method.score_candidates(store, candidates)
            else:
                scored = scored_subset
            context = _context(store, method, scored, train_s, infer_s)
            metrics = _evaluate(method, scored, store, context)
            mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})
            artifact = store.root.parent.parent.parent / "artifacts" / store.dataset / tier / preset
            artifact.mkdir(parents=True, exist_ok=True)
            ckpt = artifact / f"{method_name}.pt"
            try:
                method.save(str(ckpt))
                if ckpt.exists() and ckpt.stat().st_size > 0:
                    mlflow.log_artifact(str(ckpt))
            except NotImplementedError:
                pass
            mlflow.set_tag("status", "finished")
            return "finished"
        except Unsupported as exc:
            mlflow.set_tag("status", "unsupported")
            mlflow.set_tag("reason", str(exc))
            return "unsupported"
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "error.txt")
            mlflow.set_tag("reason", str(exc))
            if not resolved.get("continue_on_error", True):
                raise
            return "failed"


def _context(store, method, scored, train_s: float, infer_s: float) -> dict[str, Any]:
    context: dict[str, Any] = {
        "train_wall_s": train_s,
        "batch_infer_s": infer_s,
        "peak_rss_mb": _rss_mb(),
        "item_counts": item_counts(store),
        "user_categories": user_categories(store),
    }
    if scored is not None and Task.rating in method.spec.tasks:
        test = pq.read_table(store.test_path).to_pandas()
        explicit = test[test["feedback_type"] == "explicit"]
        if not explicit.empty:
            sample = explicit.head(2000)
            try:
                context["rating_pred"] = method.predict_rating(sample)
                context["rating_truth"] = sample["value"].to_numpy()
            except Unsupported:
                pass
    explanations = []
    if scored is not None:
        sample_users = scored["user_idx"].drop_duplicates().head(20)
        idx_user = store.id_maps()[1]
        idx_item = store.id_maps()[3]
        for user in sample_users:
            item = int(scored[scored["user_idx"] == user].iloc[0]["item_idx"])
            local = method.explain_local(idx_user.get(int(user), str(user)), [idx_item.get(item, str(item))])
            explanations.append(bool(local and local[0].text))
    context["explanation_coverage"] = float(np_mean(explanations))
    return context


def np_mean(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _evaluate(method, scored, store, context) -> dict[str, float]:
    reg = ensure_loaded()
    found: dict[str, float] = {}
    if scored is None:
        return found
    for metric_cls in reg.metrics.values():
        metric = metric_cls()
        if metric.spec.name in {"rmse", "mae"} and Task.rating not in method.spec.tasks:
            continue
        try:
            value = metric.compute(scored, store, context)
        except Exception:
            value = None
        if value is not None:
            found[metric.spec.name] = float(value)
    return found


def _log_status(store, method_name, resolved, status: str, rankable: bool, reason: str) -> str:
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("recbench")
    with mlflow.start_run(run_name=f"{store.dataset}-{method_name}-{resolved['tier']}-{status}"):
        mlflow.set_tags(
            {
                "dataset": store.dataset,
                "method": method_name,
                "tier": resolved["tier"],
                "preset": resolved["preset"],
                "max_steps": str(resolved["max_steps"]),
                "status": status,
                "rankable": str(rankable).lower(),
                "reason": reason,
            }
        )
    return status


def run_matrix(config_path: Path, datasets: list[str], methods: list[str], preset: str | None, repo_root: Path) -> None:
    benchmark = load_yaml(config_path)
    resolved = resolve_run_config(benchmark, preset=preset, repo_root=repo_root)
    root = Path(os.environ.get("DATA_DIR", repo_root / "data"))
    reg = ensure_loaded()
    chosen_methods = methods or resolved["methods"]
    for dataset in datasets:
        split = root / "splits" / dataset / resolved["tier"]
        if not (split / "meta.json").exists():
            print(f"missing split for {dataset}, skipping")
            continue
        store = SplitStore(split, dataset, resolved["tier"])
        for method_name in chosen_methods:
            if method_name not in reg.methods:
                print(f"unknown method {method_name}")
                continue
            status = run_pair(store, method_name, resolved, resolved["resume"])
            print(f"{dataset} {method_name}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recommendation benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--preset", type=str, default=None)
    args = parser.parse_args()
    datasets = [part for part in args.datasets.split(",") if part]
    methods = [part for part in args.methods.split(",") if part]
    if not datasets:
        datasets = list(load_yaml(args.config).get("datasets") or [])
    run_matrix(args.config, datasets, methods, args.preset, Path.cwd())


if __name__ == "__main__":
    main()
