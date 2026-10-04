"""Fit and evaluate (dataset x method) pairs and log everything to MLflow.

Each pair runs in its own child process by default, so that:
- peak memory is measured per method (not accumulated across methods),
- one crashing or hanging method cannot take down the whole benchmark (timeout),
- GPU memory is released between methods.

Usage:
    python -m recbench.runner --config configs/benchmarks/smoke-cpu.yaml
    python -m recbench.runner --config ... --datasets movielens-25m --methods ease,itemknn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from recbench import __version__
from recbench.config import config_hash, load_benchmark_yaml, method_config, resolve_run_config
from recbench.data import SplitError, TrainView
from recbench.paths import repo_root, reports_dir, runs_dir, tracking_uri, workspace  # noqa: F401 - re-exported
from recbench.protocol import EVAL_VERSION, PROTOCOL_NOTE, PROTOCOL_VERSION, Unsupported
from recbench.registry import ensure_loaded

EXPERIMENT = "recbench"
# OpenBLAS aborts (exit -11) when asked for more threads than it was built for (128 on the rented box),
# so every child process gets an explicit cap.
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
MAX_THREADS = 32


def source_fingerprint(method_name: str) -> str:
    """A short hash of the code behind a method: its module, the recbench.methods modules that module imports, and
    recbench/data.py. Editing any of them changes the hash."""
    module = sys.modules[ensure_loaded().methods[method_name].__module__]
    path = Path(module.__file__)
    files = {path, Path(__file__).with_name("data.py")}
    files |= {path.with_name(f"{name}.py") for name in re.findall(r"recbench\.methods\.(\w+)", path.read_text())}
    digest = hashlib.sha256()
    for file in sorted(f for f in files if f.exists()):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:8]


def method_version(method_name: str, cfg: dict[str, Any]) -> str:
    """The implementation version that goes into a run's identity: the method's impl_version, plus a hash of its
    source when `track_code` is on (as in the lab). Then a run is computed again after a code edit, instead of
    returning the cached result of the old code."""
    version = ensure_loaded().methods[method_name].spec.impl_version
    return f"{version}+src.{source_fingerprint(method_name)}" if cfg.get("track_code") else version


def data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", repo_root() / "data")).resolve()


def thread_env(threads: int | None = None) -> dict[str, str]:
    """BLAS/OpenMP thread caps for a child process: `threads` if given, else the current value or min(cores, 32)."""
    if threads:
        return {var: str(int(threads)) for var in THREAD_VARS}
    default = str(min(os.cpu_count() or 1, MAX_THREADS))
    return {var: os.environ.get(var, default) for var in THREAD_VARS}


def git_state() -> tuple[str, bool]:
    """(short commit id, has uncommitted changes); ("", False) outside a git checkout."""
    try:
        run = lambda *args: subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
        return run("rev-parse", "--short=12", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "", False


def run_hash_for(resolved: dict[str, Any], method_name: str, split_hash: str) -> str:
    cfg = method_config(resolved, method_name)
    return config_hash(cfg, method_name, split_hash, method_version(method_name, cfg))


def _mlflow():
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def already_finished(dataset: str, method: str, run_hash: str) -> bool:
    mlflow = _mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return False
    frame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.dataset = '{dataset}' and tags.method = '{method}' "
            f"and tags.config_hash = '{run_hash}' and tags.status = 'finished'"
        ),
        max_results=1,
    )
    return not frame.empty


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _peak_gpu_mb() -> float:
    if "torch" not in sys.modules:
        return 0.0
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)


def run_single(split_dir: Path, method_name: str, resolved: dict[str, Any]) -> dict[str, Any]:
    """Fit + evaluate one pair in THIS process and log it. Returns a status record."""
    from recbench.evaluation import EvalSplit, Evaluator, ValidationMonitor, save_result

    reg = ensure_loaded()
    method = reg.create_method(method_name)
    data = TrainView(split_dir)
    cfg = method_config(resolved, method_name)
    version = method_version(method_name, cfg)
    run_hash = config_hash(cfg, method_name, data.split_hash, version)
    # Runs that differ only in their seed share a config group; leaderboards average them.
    group = config_hash({k: v for k, v in cfg.items() if k != "seed"}, method_name, data.split_hash, version)
    sha, dirty = (resolved["git_sha"], bool(resolved.get("git_dirty"))) if "git_sha" in resolved else git_state()
    tags = {
        "dataset": data.dataset,
        "method": method_name,
        "tier": data.tier,
        "preset": str(resolved.get("preset")),
        "hardware": str(resolved.get("hardware_name")),
        "protocol_version": PROTOCOL_VERSION,
        "config_hash": run_hash,
        "config_group": group,
        "split_hash": data.split_hash,
        "managed": str(method.spec.managed).lower(),
        "ranked": str(method.spec.ranked).lower(),
        "fidelity": method.spec.fidelity,
        "tasks": ",".join(sorted(t.value for t in method.spec.tasks)),
        "protocol": PROTOCOL_NOTE,
        # stage: benchmark | search | final | confirm; tuning: defaults | tuned (set by recbench.tuning)
        "stage": str(resolved.get("stage", "benchmark")),
        "tuning": str(resolved.get("tuning", "defaults")),
        "impl_version": version,
        "eval_version": EVAL_VERSION,
        "recbench_version": __version__,
        "git_sha": sha,
        "git_dirty": str(dirty).lower(),
    }
    skip = _skip_reason(method, data, resolved)
    mlflow = _mlflow()
    with mlflow.start_run(run_name=f"{data.dataset}-{method_name}-{data.tier}"):
        mlflow.set_tags({**tags, "status": "running"})
        mlflow.log_params({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))})
        if skip:
            mlflow.set_tags({"status": "unsupported", "reason": skip})
            return {"status": "unsupported", "reason": skip}
        try:
            _seed_everything(int(cfg.get("seed", 42)))
            split = EvalSplit(split_dir)
            # Early stopping may look at a validation fold only (ValidationMonitor refuses test splits).
            if split.meta.get("fold") == "valid" and cfg.get("early_stopping", True) and not cfg.get("epochs"):
                method.monitor = ValidationMonitor(split, data, cfg)
            # The training window only changes what the method learns from; the evaluator keeps the full history.
            fit_data = data.restrict(cfg.get("train_window_days"), int(cfg.get("train_window_keep_last", 10)))
            began = time.perf_counter()
            method.fit(fit_data, cfg)
            train_seconds = time.perf_counter() - began
            fit_info = getattr(method, "fit_info", None) or {}
            for key, value in fit_info.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    mlflow.log_param(f"fit.{key}", value)
            for point in getattr(method, "fit_curve", None) or []:  # learning curves, one point per epoch
                mlflow.log_metrics({f"curve/{k}": float(v) for k, v in point.items() if k != "epoch" and v == v}, step=int(point["epoch"]))
            result = Evaluator(split, data, cfg).run(method, extra={"train_seconds": train_seconds})
            result.metrics["peak_rss_mb"] = _peak_rss_mb()
            result.metrics["peak_gpu_mb"] = _peak_gpu_mb()
            finish = getattr(method, "finish", None)
            if callable(finish):  # managed services report live latency and request usage
                result.metrics.update(finish())
            mlflow.log_metrics({k: v for k, v in result.metrics.items() if np.isfinite(v)})
            with tempfile.TemporaryDirectory() as tmp:
                for path in save_result(result, Path(tmp)):
                    mlflow.log_artifact(str(path))
            bundle = None
            if resolved.get("export_bundles") and method.spec.ranked and not method.spec.managed:
                bundle = _export_bundle(method, data, split, cfg, run_id=mlflow.active_run().info.run_id, run_hash=run_hash,
                                        offline=result.metrics)
                if bundle is not None:
                    mlflow.set_tag("bundle", str(bundle))
            mlflow.set_tag("status", "finished")
            summary = {k: result.metrics[k] for k in ("ndcg_at_10", "recall_at_10") if k in result.metrics}
            outcome = {"status": "finished", "metrics": summary, "train_seconds": train_seconds,
                       "fit": {k: v for k, v in fit_info.items() if isinstance(v, (int, float, str, bool))}}
            if bundle is not None:
                outcome["bundle"] = str(bundle)
            return outcome
        except Unsupported as exc:
            mlflow.set_tags({"status": "unsupported", "reason": str(exc)[:500]})
            return {"status": "unsupported", "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001 - logged with traceback, then re-raised if asked
            mlflow.set_tags({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500]})
            mlflow.log_text(traceback.format_exc(), "error.txt")
            if not resolved.get("continue_on_error", True):
                raise
            return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _skip_reason(method, data: TrainView, resolved: dict[str, Any]) -> str | None:
    spec = method.spec
    if spec.managed and not resolved.get("managed_services", False):
        return "managed services are off in this config"
    if spec.requires_images and not any(path and Path(path).is_file() for path in data.item_image_path[1:200]):
        return "no item images on disk"
    if spec.requires_side_features and not (any(data.item_text[1:]) or any(data.item_category[1:])):
        return "dataset has no item text or categories"
    return None


def _export_bundle(method, data: TrainView, split, cfg: dict[str, Any], *, run_id: str = "", run_hash: str = "",
                   offline: dict[str, float] | None = None) -> Path | None:
    try:
        from recbench.serving.bundle import export_bundle
    except ImportError:
        return None
    out = data_root() / "bundles" / data.dataset / data.tier / method.spec.name
    extra = {"run_id": run_id, "config_hash": run_hash, "stage": str(cfg.get("stage", "benchmark")), "tuning": str(cfg.get("tuning", "defaults")),
             # the offline results of the same run, so the serving monitor can compare what is served with them
             "offline": {k: float(v) for k, v in (offline or {}).items()
                         if k in ("ndcg_at_10", "coverage_at_10", "popularity_percentile_at_10") and v == v}}
    return export_bundle(method, data, out, k=int(cfg.get("bundle_k", 100)), max_users=int(cfg.get("bundle_users", 20_000)), extra=extra)


def run_pair(split_dir: Path, method_name: str, resolved: dict[str, Any], *, isolate: bool = True) -> dict[str, Any]:
    """Run one pair, by default in a child process with a timeout."""
    data = TrainView(split_dir)
    run_hash = run_hash_for(resolved, method_name, data.split_hash)
    if resolved.get("resume", True) and already_finished(data.dataset, method_name, run_hash):
        return {"status": "skipped_existing"}
    if not isolate:
        return run_single(split_dir, method_name, resolved)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "resolved.json"
        out_path = Path(tmp) / "result.json"
        cfg_path.write_text(json.dumps(resolved))
        command = [sys.executable, "-m", "recbench.runner", "--single", str(split_dir), method_name, str(cfg_path), str(out_path)]
        module = ensure_loaded().methods[method_name].__module__.rsplit(".", 1)[-1]
        env = {**os.environ, **thread_env(resolved.get("threads")), **(resolved.get("child_env") or {}),
               "MLFLOW_TRACKING_URI": tracking_uri(), "RECBENCH_ROOT": str(repo_root()),
               "RECBENCH_METHOD_MODULES": module}  # the child imports only this method's module (see recbench.methods)
        limit = float(resolved.get("timeout_minutes", 240)) * 60
        proc = subprocess.Popen(command, env=env)
        began = time.time()  # wall clock: keeps counting while a laptop sleeps (time.monotonic does not on macOS)
        while proc.poll() is None:
            if time.time() - began > limit:
                proc.kill()
                proc.wait()
                _log_outcome(data, method_name, run_hash, "timeout", f"exceeded {resolved.get('timeout_minutes')} minutes (wall clock)",
                             str(resolved.get("stage", "benchmark")))
                return {"status": "timeout"}
            time.sleep(1.0)
        if out_path.exists():
            return json.loads(out_path.read_text())
        reason = f"child process exited with code {proc.returncode} before reporting a result"
        _log_outcome(data, method_name, run_hash, "failed", reason, str(resolved.get("stage", "benchmark")))
        return {"status": "failed", "reason": reason}


def _log_outcome(data: TrainView, method_name: str, run_hash: str, status: str, reason: str, stage: str = "benchmark") -> None:
    """Record a run that ended without the child logging it (timeout, crash, out-of-memory kill)."""
    mlflow = _mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    # The child may have opened a run and died mid-way: close it instead of leaving it 'running'.
    stale = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.config_hash = '{run_hash}' and tags.status = 'running'",
    )
    client = mlflow.MlflowClient()
    for run_id in stale.get("run_id", []):
        client.set_tag(run_id, "status", status)
        client.set_tag(run_id, "reason", reason)
        client.set_terminated(run_id, status="KILLED" if status == "timeout" else "FAILED")
    if not stale.empty:
        return
    with mlflow.start_run(run_name=f"{data.dataset}-{method_name}-{data.tier}-{status}"):
        mlflow.set_tags(
            {
                "dataset": data.dataset,
                "method": method_name,
                "tier": data.tier,
                "config_hash": run_hash,
                "protocol_version": PROTOCOL_VERSION,
                "stage": stage,
                "status": status,
                "reason": reason,
            }
        )


def run_matrix(config_path: Path, datasets: list[str], methods: list[str], preset: str | None) -> list[dict[str, Any]]:
    resolved = resolve_run_config(load_benchmark_yaml(config_path, repo_root()), preset=preset, repo_root=repo_root())
    resolved["git_sha"], resolved["git_dirty"] = git_state()
    reg = ensure_loaded()
    results = []
    for dataset in datasets or resolved["datasets"]:
        split_dir = data_root() / "splits" / dataset / resolved["tier"]
        try:
            TrainView(split_dir)
        except SplitError as exc:
            print(f"{dataset}: {exc}; skipping")
            continue
        for method_name in methods or resolved["methods"]:
            if method_name not in reg.methods:
                print(f"unknown method {method_name}; registered: {sorted(reg.methods)}")
                continue
            began = time.perf_counter()
            outcome = run_pair(split_dir, method_name, resolved)
            outcome.update({"dataset": dataset, "method": method_name, "seconds": round(time.perf_counter() - began, 1)})
            print(json.dumps(outcome))
            results.append(outcome)
    return results


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        split_dir, method_name, cfg_path, out_path = sys.argv[2:6]
        resolved = json.loads(Path(cfg_path).read_text())
        outcome = run_single(Path(split_dir), method_name, resolved)
        Path(out_path).write_text(json.dumps(outcome, default=str))
        return
    parser = argparse.ArgumentParser(description="Run the recommendation benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--preset", type=str, default=None)
    args = parser.parse_args()
    run_matrix(
        args.config,
        [d for d in args.datasets.split(",") if d],
        [m for m in args.methods.split(",") if m],
        args.preset,
    )


if __name__ == "__main__":
    main()
