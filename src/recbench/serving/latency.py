"""Load-test a running recbench API and log latency to the matching MLflow run.

    python -m recbench.serving.latency --base-url http://127.0.0.1:8080 --dataset movielens-25m --method ease

It sends `--requests` POST /recommend calls for real user ids taken from the
bundle (plus ~10% unknown users, which hit the popularity fallback), with
`--concurrency` requests in flight, after a short warm-up. Reported numbers:
p50/p95/p99 latency in milliseconds (successful requests), requests per second,
and the share of requests that failed. Latency measured from your laptop to
Cloud Run includes the network round trip.

The user ids come from the LOCAL bundle folder, so `--tier` must be the tier the
service serves (RECBENCH_TIER) and that bundle must exist on this machine. The
numbers are logged to the MLflow run that produced the bundle (its manifest
names it), or else to the latest default-settings run of that tier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def _bundle_dir(dataset: str, tier: str, method: str) -> Path:
    root = Path(os.environ.get("RECBENCH_BUNDLES", Path(os.environ.get("DATA_DIR", "data")) / "bundles"))
    return root / dataset / tier / method


def _user_ids(dataset: str, tier: str, method: str, n: int, seed: int) -> list[str]:
    users = pd.read_parquet(_bundle_dir(dataset, tier, method) / "users.parquet")["user_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    known = list(rng.choice(users, size=n, replace=len(users) < n))
    unknown = [f"unknown-{i}" for i in range(max(1, n // 10))]
    mixed = known + unknown
    rng.shuffle(mixed)
    return mixed[:n]


async def _run(base_url: str, payloads: list[dict], concurrency: int, headers: dict) -> tuple[list[float], int]:
    """(latencies of successful requests in ms, number of failed requests)."""
    latencies: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=30, headers=headers) as client:

        async def one(payload: dict) -> None:
            nonlocal errors
            async with semaphore:
                began = time.perf_counter()
                try:
                    response = await client.post("/recommend", json=payload)
                    ok = response.status_code < 400
                except httpx.HTTPError:
                    ok = False
                if ok:
                    latencies.append((time.perf_counter() - began) * 1000)
                else:
                    errors += 1

        await asyncio.gather(*(one(p) for p in payloads))
    return latencies, errors


def measure(base_url: str, dataset: str, method: str, *, tier: str = "smoke", requests: int = 300, concurrency: int = 8,
            warmup: int = 20, seed: int = 0, token: str | None = None) -> dict[str, float]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    users = _user_ids(dataset, tier, method, requests + warmup, seed)
    payloads = [{"dataset": dataset, "method": method, "user_id": u, "k": 10} for u in users]
    asyncio.run(_run(base_url, payloads[:warmup], concurrency, headers))
    began = time.perf_counter()
    latencies, errors = asyncio.run(_run(base_url, payloads[warmup:], concurrency, headers))
    elapsed = time.perf_counter() - began
    if not latencies:
        raise SystemExit(f"all {errors} requests failed: is the service up, is the token valid, does it serve {dataset}/{method} ({tier})?")
    values = np.asarray(latencies)
    return {
        "served_p50_ms": float(np.percentile(values, 50)),
        "served_p95_ms": float(np.percentile(values, 95)),
        "served_p99_ms": float(np.percentile(values, 99)),
        "served_rps": float(len(values) / elapsed) if elapsed else 0.0,
        "served_requests": float(len(values) + errors),
        "served_error_rate": float(errors / (len(values) + errors)),
        "served_concurrency": float(concurrency),
    }


def log_to_mlflow(dataset: str, tier: str, method: str, metrics: dict[str, float], base_url: str) -> bool:
    """Attach the numbers to the run that produced the served bundle (named in its manifest), else to the latest
    finished default-settings run of (dataset, tier, method)."""
    try:
        import mlflow

        from recbench.results import load_runs, tracking_uri
    except ImportError:
        return False
    manifest_path = _bundle_dir(dataset, tier, method) / "manifest.json"
    run_id = json.loads(manifest_path.read_text()).get("run_id") if manifest_path.exists() else None
    if not run_id:
        frame = load_runs(tier, tuning="defaults")
        match = frame[(frame["tags.dataset"] == dataset) & (frame["tags.method"] == method)] if not frame.empty else frame
        if match.empty:
            return False
        run_id = match.iloc[-1]["run_id"]
    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("served_from", base_url)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test a recbench API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--tier", default="smoke")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--token", default=os.environ.get("RECBENCH_ID_TOKEN"), help="identity token for a private Cloud Run service")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    metrics = measure(args.base_url, args.dataset, args.method, tier=args.tier, requests=args.requests,
                      concurrency=args.concurrency, token=args.token)
    if not args.no_log:
        metrics["logged"] = float(log_to_mlflow(args.dataset, args.tier, args.method, metrics, args.base_url))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
