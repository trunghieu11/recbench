"""Copy runs from another machine's MLflow folder (e.g. the GPU machine's) into this machine's store.

Why not just copy the folder into runs/mlflow? Each machine's file store has its own "recbench"
experiment with a random id and absolute artifact paths. After copying, MLflow sees two experiments
with the same name and reads only one of them, so half the results silently disappear from reports.
Instead, copy the other folder anywhere and import it:

    rsync -a gpu-box:recommendation_benchmark/runs/mlflow/ runs/mlflow-gpu/
    python -m recbench.import_runs runs/mlflow-gpu

Each run is re-created here with the same tags, params, metrics, times, and artifact files, plus the tag
imported_from=<original run id>. Importing again skips runs already imported and runs still in progress.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

from recbench.paths import use_workspace
from recbench.runner import EXPERIMENT, _mlflow, tracking_uri

PARAMS_PER_BATCH = 100  # MLflow's limit per log_batch call
METRICS_PER_BATCH = 1000


def _all_runs(client: Any, experiment_id: str) -> Iterator[Any]:
    token = None
    while True:
        page = client.search_runs([experiment_id], max_results=1000, page_token=token)
        yield from page
        token = page.token
        if not token:
            return


def import_runs(source: str | Path) -> dict[str, int]:
    """Re-create the runs of a copied MLflow file store in this machine's store. Returns counts."""
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    root = Path(source).resolve()
    src = MlflowClient(tracking_uri=root.as_uri())
    src_exp = src.get_experiment_by_name(EXPERIMENT)
    if src_exp is None:
        raise FileNotFoundError(f"no '{EXPERIMENT}' experiment in {root}")
    dst_exp = _mlflow().get_experiment_by_name(EXPERIMENT).experiment_id
    dst = MlflowClient(tracking_uri=tracking_uri())
    done = {run.data.tags.get("imported_from") for run in _all_runs(dst, dst_exp)}
    counts = {"imported": 0, "already_imported": 0, "still_running": 0}
    for run in _all_runs(src, src_exp.experiment_id):
        info = run.info
        if info.run_id in done:
            counts["already_imported"] += 1
            continue
        if info.status == "RUNNING":
            counts["still_running"] += 1
            continue
        tags = {k: v for k, v in run.data.tags.items() if k != "mlflow.runName"}
        new_id = dst.create_run(dst_exp, start_time=info.start_time, tags={**tags, "imported_from": info.run_id},
                                run_name=info.run_name).info.run_id
        params = [Param(k, v) for k, v in run.data.params.items()]
        metrics = [Metric(k, v, info.end_time or info.start_time, 0) for k, v in run.data.metrics.items()]
        for start in range(0, len(params), PARAMS_PER_BATCH):
            dst.log_batch(new_id, params=params[start : start + PARAMS_PER_BATCH])
        for start in range(0, len(metrics), METRICS_PER_BATCH):
            dst.log_batch(new_id, metrics=metrics[start : start + METRICS_PER_BATCH])
        # The copy's artifact paths point at the other machine, so read the files from the copied folder.
        artifacts = root / src_exp.experiment_id / info.run_id / "artifacts"
        if artifacts.is_dir() and any(artifacts.iterdir()):
            dst.log_artifacts(new_id, str(artifacts))
        dst.set_terminated(new_id, status=info.status, end_time=info.end_time)
        counts["imported"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Import runs from another machine's copied runs/mlflow folder.")
    parser.add_argument("source", type=Path, help="the copied folder, e.g. runs/mlflow-gpu")
    parser.add_argument("--workspace", default="", help="import into this workspace's store, e.g. lab (runs/lab/mlflow)")
    args = parser.parse_args()
    use_workspace(args.workspace or None)
    counts = import_runs(args.source)
    print(f"{counts} -> {tracking_uri()}")


if __name__ == "__main__":
    main()
