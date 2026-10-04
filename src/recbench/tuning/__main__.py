"""Run one tuning job (one method on one dataset) from a quick-tier config.

    python -m recbench.tuning --config configs/benchmarks/quick.yaml --dataset movielens-25m --method ease
    python -m recbench.tuning --config ... --dataset hm --method itemknn --hardware configs/hardware/local-cpu.yaml

The queue (python -m recbench.queue) runs many such jobs in parallel; this command is for one job,
for example to debug a method or to repeat a lab exercise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recbench.config import load_benchmark_yaml, resolve_run_config
from recbench.runner import git_state, repo_root
from recbench.tuning.job import JobSettings, run_job
from recbench.tuning.spaces import load_spaces


def job_settings(benchmark: dict[str, Any], space_settings: dict[str, Any]) -> JobSettings:
    tuning = benchmark.get("tuning") or {}
    return JobSettings(
        tier=str(benchmark.get("tier", "quick")),
        trials=int(tuning.get("trials", 10)),
        search_users=int(tuning.get("search_users", 3000)),
        cap_minutes=float(tuning.get("cap_minutes", 180)),
        seed=int(benchmark.get("seed", 42)),
        space_version=int(space_settings.get("version", 1)),
    )


def load_benchmark(path: Path, hardware: str | None = None, preset: str | None = None) -> tuple[dict, dict, Any, JobSettings]:
    """(raw benchmark yaml, resolved run config, method spaces, job settings)."""
    benchmark = load_benchmark_yaml(path, repo_root())
    if hardware:
        benchmark["hardware"] = hardware
    resolved = resolve_run_config(benchmark, preset=preset, repo_root=repo_root())
    resolved["git_sha"], resolved["git_dirty"] = git_state()
    spaces, space_settings = load_spaces(repo_root() / (benchmark.get("tuning") or {}).get("spaces", "configs/tuning/quick.yaml"))
    return benchmark, resolved, spaces, job_settings(benchmark, space_settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune one method on one dataset, then evaluate it once on test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--hardware", default=None, help="override the config's hardware profile")
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()
    _, resolved, spaces, settings = load_benchmark(args.config, args.hardware, args.preset)
    summary = run_job(args.dataset, args.method, resolved, settings, spaces.get(args.method))
    print(json.dumps({k: summary.get(k) for k in ("dataset", "method", "status", "best_val", "best_params", "test", "reason")}, indent=2, default=str))


if __name__ == "__main__":
    main()
