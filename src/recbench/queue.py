"""Run quick-tier jobs dataset by dataset, in parallel, each within its time cap.

    python -m recbench.queue run --config configs/benchmarks/quick.yaml
    python -m recbench.queue run --config ... --datasets movielens-25m --stop-after-dataset --deadline-hours 3
    python -m recbench.queue status --config configs/benchmarks/quick.yaml

How jobs are scheduled:
- A job is one method on one dataset: a tuning job (recbench.tuning.run_job), or a confirmation job
  (run_confirm) that re-checks a dataset's top methods on full data.
- Jobs are ordered by (dataset position, method position). A free worker always takes the first job it can
  run, so every job of a dataset comes before the next dataset's jobs. A worker starts the next dataset
  early ("backfill") only when the current one has nothing it can run, for example while the re-rankers
  wait for their candidate generators (`after:` in the config).
- CPU jobs run on CPU workers, each with a thread cap. GPU jobs run in GPU slots (`jobs_per_gpu` per GPU),
  with CUDA_VISIBLE_DEVICES set per job. Without CUDA, GPU jobs run on CPU workers.
- No job starts after --deadline-hours. --stop-after-dataset never starts a job of a later dataset.
- When a dataset's tuning jobs are all done, its report is written. If `confirm` is set, its top methods
  are queued for the full-data check right away, ahead of the next dataset.
Everything resumes: a finished job (its summary in runs/tuning/) is not run again.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recbench.runner import repo_root
from recbench.tuning.__main__ import load_benchmark
from recbench.tuning.job import JobSettings, read_summary, run_confirm, run_job

DONE = {"finished", "over_budget", "failed", "missing_split", "skipped"}


@dataclass
class Job:
    dataset: str
    method: str
    kind: str  # "tune" | "confirm"
    resource: str  # "cpu" | "gpu"
    priority: tuple[int, int, int]
    after: list[str] = field(default_factory=list)
    status: str = "pending"
    started: float | None = None
    ended: float | None = None
    result: dict[str, Any] | None = None
    slot: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.dataset}:{self.method}"


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:  # pragma: no cover - the bench extra has no torch
        return 0


def state_path(tier: str) -> Path:
    path = repo_root() / "runs" / "queue" / f"{tier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class Queue:
    def __init__(self, config: Path, *, datasets: list[str] | None = None, methods: list[str] | None = None,
                 hardware: str | None = None, deadline_hours: float | None = None, stop_after_dataset: bool = False,
                 retry_failed: bool = False, isolate: bool = True, gpus: int | None = None, cpu_workers: int | None = None):
        self.benchmark, self.resolved, self.spaces, self.settings = load_benchmark(config, hardware)
        queue_cfg = self.benchmark.get("queue") or {}
        self.confirm_cfg = self.benchmark.get("confirm") or {}
        self.datasets = datasets or list(self.benchmark.get("datasets") or [])
        entries = [e for e in queue_cfg.get("methods") or [] if not methods or e["name"] in methods]
        self.isolate = isolate
        self.retry_failed = retry_failed
        self.deadline = time.time() + deadline_hours * 3600 if deadline_hours else None
        self.stop_after_dataset = stop_after_dataset
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        cores = os.cpu_count() or 1
        max_threads = int(queue_cfg.get("max_threads_per_job", 32))
        self.n_gpus = gpus if gpus is not None else gpu_count()
        self.cpu_workers = cpu_workers or int(queue_cfg.get("cpu_workers") or max(1, min(8, cores // 16)))
        self.cpu_threads = max(1, min(max_threads, cores // self.cpu_workers))
        self.gpu_slots = [f"gpu{g}" for g in range(self.n_gpus) for _ in range(int(queue_cfg.get("jobs_per_gpu", 3)))]
        self.jobs: list[Job] = []
        self.resources = {e["name"]: e.get("resource", "cpu") for e in entries}
        for d, dataset in enumerate(self.datasets):
            for m, entry in enumerate(entries):
                self.jobs.append(Job(dataset, entry["name"], "tune", entry.get("resource", "cpu"), (d, 0, m), list(entry.get("after") or [])))
        for job in self.jobs:  # resume: finished jobs keep their summary
            self._load_done(job)
        self.active_dataset = next((j.dataset for j in sorted(self.jobs, key=lambda j: j.priority) if j.status == "pending"), None)

    # ----- bookkeeping -----
    def _load_done(self, job: Job) -> None:
        tier = self.settings.tier if job.kind == "tune" else str(self.confirm_cfg.get("tier", "full"))
        summary = read_summary(tier, job.dataset, job.method)
        if summary and summary.get("status") in DONE and not (self.retry_failed and summary["status"] == "failed"):
            if job.kind == "confirm" and summary.get("stage") != "confirm":
                return
            job.status, job.result = summary["status"], summary

    def _deps_done(self, job: Job) -> bool:
        return all(any(j.dataset == job.dataset and j.method == dep and j.kind == "tune" and j.status in DONE for j in self.jobs)
                   or not any(j.dataset == job.dataset and j.method == dep for j in self.jobs) for dep in job.after)

    def _allowed(self, job: Job) -> bool:
        if self.deadline and time.time() >= self.deadline:
            return False
        if self.stop_after_dataset and self.active_dataset and job.dataset != self.active_dataset:
            return False
        return job.status == "pending" and self._deps_done(job)

    def _next_job(self, slot_kind: str) -> Job | None:
        for job in sorted(self.jobs, key=lambda j: j.priority):
            fits = job.resource == slot_kind or (job.resource == "gpu" and slot_kind == "cpu" and not self.gpu_slots)
            if fits and self._allowed(job):
                return job
        return None

    def save_state(self) -> None:
        state = {
            "tier": self.settings.tier, "updated": time.time(), "deadline": self.deadline,
            "cpu_workers": self.cpu_workers, "cpu_threads": self.cpu_threads, "gpu_slots": len(self.gpu_slots),
            "jobs": [{"key": j.key, "dataset": j.dataset, "method": j.method, "kind": j.kind, "resource": j.resource,
                      "status": j.status, "started": j.started, "ended": j.ended, "slot": j.slot,
                      "test": (j.result or {}).get("test"), "reason": (j.result or {}).get("reason")} for j in self.jobs],
        }
        path = state_path(self.settings.tier)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(path)

    # ----- dataset completion -----
    def _dataset_done(self, dataset: str, kind: str) -> bool:
        jobs = [j for j in self.jobs if j.dataset == dataset and j.kind == kind]
        return bool(jobs) and all(j.status in DONE for j in jobs)

    def _on_job_end(self, job: Job) -> None:
        """Called with the lock held: queue confirmations and write reports when a dataset block completes."""
        if job.kind == "tune" and self._dataset_done(job.dataset, "tune"):
            self._report(job.dataset)
            top = int(self.confirm_cfg.get("top", 0))
            if top and not any(j.dataset == job.dataset and j.kind == "confirm" for j in self.jobs):
                ranked = sorted((j for j in self.jobs if j.dataset == job.dataset and j.kind == "tune" and j.status == "finished"
                                 and (j.result or {}).get("test", {}).get("ndcg_at_10") is not None and j.method != "random"),
                                key=lambda j: -j.result["test"]["ndcg_at_10"])
                for rank, winner in enumerate(ranked[:top]):
                    confirm = Job(job.dataset, winner.method, "confirm", winner.resource, (winner.priority[0], 1, rank))
                    self._load_done(confirm)
                    self.jobs.append(confirm)
        if job.kind == "confirm" and self._dataset_done(job.dataset, "confirm"):
            self._report(job.dataset, confirm=True)
        pending = [j for j in sorted(self.jobs, key=lambda j: j.priority) if j.status == "pending"]
        self.active_dataset = pending[0].dataset if pending and not self.stop_after_dataset else self.active_dataset

    def _report(self, dataset: str, confirm: bool = False) -> None:
        try:
            from recbench.report.build import build

            tier = str(self.confirm_cfg.get("tier", "full")) if confirm else self.settings.tier
            out = repo_root() / "reports" / f"{tier}-tuned"
            build(tier, out, docs_dir=repo_root() / "docs", tuning="tuned")
            print(f"[queue] {dataset}: {'confirmation' if confirm else 'quick tier'} done; report in {out}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a report failure must not stop the queue
            print(f"[queue] report for {dataset} failed: {type(exc).__name__}: {exc}", flush=True)

    # ----- running -----
    def _execute(self, job: Job, slot: str) -> dict[str, Any]:
        if slot.startswith("gpu"):
            env = {"CUDA_VISIBLE_DEVICES": slot[3:]}
            resolved = {**self.resolved, "threads": 8}
        else:
            env = {"CUDA_VISIBLE_DEVICES": ""} if job.resource == "cpu" else {}
            resolved = {**self.resolved, "threads": self.cpu_threads}
        if job.kind == "confirm":
            return run_confirm(job.dataset, job.method, resolved, self.settings, self.spaces.get(job.method),
                               tier=str(self.confirm_cfg.get("tier", "full")), seeds=list(self.confirm_cfg.get("seeds") or []),
                               child_env=env, isolate=self.isolate)
        return run_job(job.dataset, job.method, resolved, self.settings, self.spaces.get(job.method), child_env=env, isolate=self.isolate)

    def _worker(self, slot: str) -> None:
        kind = "gpu" if slot.startswith("gpu") else "cpu"
        while True:
            with self.wake:
                job = self._next_job(kind)
                while job is None:
                    if not any(j.status in ("pending", "running") for j in self.jobs if self._allowed(j) or j.status == "running"):
                        self.wake.notify_all()
                        return
                    self.wake.wait(timeout=5.0)
                    job = self._next_job(kind)
                job.status, job.started, job.slot = "running", time.time(), slot
                self.save_state()
            print(f"[queue] start {job.key} on {slot}", flush=True)
            try:
                result = self._execute(job, slot)
            except Exception as exc:  # noqa: BLE001 - recorded; the queue goes on
                result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            with self.wake:
                job.status, job.ended, job.result = result.get("status", "failed"), time.time(), result
                self._on_job_end(job)
                self.save_state()
                self.wake.notify_all()
            test = (result.get("test") or {}).get("ndcg_at_10") if isinstance(result.get("test"), dict) else None
            print(f"[queue] end {job.key}: {job.status}" + (f" ndcg@10={test:.4f}" if test is not None else ""), flush=True)

    def run(self) -> list[Job]:
        slots = [f"cpu{i}" for i in range(self.cpu_workers)] + self.gpu_slots
        print(f"[queue] {len(self.jobs)} jobs over {self.datasets}; {self.cpu_workers} CPU workers x {self.cpu_threads} threads, "
              f"{len(self.gpu_slots)} GPU slots on {self.n_gpus} GPU(s)", flush=True)
        with self.lock:
            for dataset in self.datasets:  # datasets finished in an earlier session: make sure their confirmations exist
                if self._dataset_done(dataset, "tune"):
                    tune_jobs = [j for j in self.jobs if j.dataset == dataset and j.kind == "tune"]
                    self._on_job_end(tune_jobs[-1])
            self.save_state()
        threads = [threading.Thread(target=self._worker, args=(slot,), daemon=True) for slot in slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        with self.lock:
            self.save_state()
        return self.jobs


def print_status(config: Path) -> None:
    """Jobs per dataset, and each dataset's provisional quick-tier leaderboard (best test NDCG@10 so far)."""
    benchmark, _, _, settings = load_benchmark(config)
    path = state_path(settings.tier)
    state = json.loads(path.read_text()) if path.exists() else {"jobs": []}
    running = {j["key"]: j for j in state["jobs"] if j["status"] == "running"}
    methods = [e["name"] for e in (benchmark.get("queue") or {}).get("methods") or []]
    for dataset in benchmark.get("datasets") or []:
        rows = []
        for method in methods:
            summary = read_summary(settings.tier, dataset, method)
            key = f"tune:{dataset}:{method}"
            status = "running" if key in running else (summary or {}).get("status", "pending")
            test = ((summary or {}).get("test") or {}).get("ndcg_at_10") if summary else None
            rows.append((method, status, test, (summary or {}).get("best_val")))
        done = sum(r[1] in DONE for r in rows)
        print(f"\n== {dataset}: {done}/{len(rows)} jobs done")
        for method, status, test, val in sorted(rows, key=lambda r: -(r[2] if r[2] is not None else -1)):
            print(f"  {method:20s} {status:12s} test NDCG@10={'%.4f' % test if test is not None else '   -  '}"
                  f"  val={'%.4f' % val if val is not None else '-'}")
    if state.get("updated"):
        print(f"\nstate updated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state['updated']))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or inspect the quick-tier job queue.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--datasets", default="")
    run.add_argument("--methods", default="")
    run.add_argument("--hardware", default=None)
    run.add_argument("--deadline-hours", type=float, default=None)
    run.add_argument("--stop-after-dataset", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--cpu-workers", type=int, default=None)
    status = sub.add_parser("status")
    status.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        print_status(args.config)
        return
    queue = Queue(args.config, datasets=[d for d in args.datasets.split(",") if d] or None,
                  methods=[m for m in args.methods.split(",") if m] or None, hardware=args.hardware,
                  deadline_hours=args.deadline_hours, stop_after_dataset=args.stop_after_dataset,
                  retry_failed=args.retry_failed, cpu_workers=args.cpu_workers)
    queue.run()
    print_status(args.config)


if __name__ == "__main__":
    main()
