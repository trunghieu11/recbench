"""Hit the running API. Do not call the recommender in-process."""

from __future__ import annotations

import argparse
import time

import httpx


def measure(base_url: str, method: str, dataset: str, user_id: str = "1", n: int = 30) -> dict[str, float]:
    latencies = []
    started = time.perf_counter()
    with httpx.Client(timeout=30) as client:
        for _ in range(n):
            t0 = time.perf_counter()
            response = client.post(
                f"{base_url}/recommend",
                params={"method": method},
                json={"user_id": user_id, "k": 10, "dataset": dataset},
            )
            response.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - started
    latencies.sort()
    return {
        "served_p50_ms": latencies[len(latencies) // 2],
        "served_p95_ms": latencies[max(int(len(latencies) * 0.95) - 1, 0)],
        "served_rps": n / elapsed if elapsed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--method", default="xsimgcl")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(measure(args.base_url, args.method, args.dataset))


if __name__ == "__main__":
    main()
