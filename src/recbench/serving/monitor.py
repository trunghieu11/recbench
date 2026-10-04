"""Check a running recbench API: is it up, how fresh and how good is what it serves, and how is it doing.

    python -m recbench.serving.monitor --base-url http://127.0.0.1:8080
    python -m recbench.serving.monitor --base-url "$URL" --token "$(gcloud auth print-identity-token)"

It reads GET /health and GET /stats and prints one line per check: PASS, WARN (look at it soon) or FAIL (act now).
The exit code is 1 when any check fails, so a scheduler such as cron can alert on it. Thresholds:

    --max-export-age-days 30   lists exported longer ago than this are stale                   WARN
    --min-coverage-ratio 0.5   served coverage@10 below this share of the offline value        WARN
    --max-fallback-share 0.5   more than this share of calls were for unknown users           WARN
    --max-error-share 0.05     more than this share of calls failed                            FAIL
    --max-p95-ms 500           the service's own 95th-percentile latency is slower than this   WARN

Traffic numbers count calls since the instance started; Cloud Run scales to zero, so after a quiet period there
may be none. docs/cloud/monitoring.md explains each check and what to do about it.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def run_checks(client: Any, *, max_export_age_days: float = 30, min_coverage_ratio: float = 0.5, max_fallback_share: float = 0.5,
               max_error_share: float = 0.05, max_p95_ms: float = 500.0) -> list[Check]:
    """`client` is anything with .get(path) returning a response with .status_code and .json() (httpx, TestClient)."""
    health = client.get("/health")
    try:
        body = health.json()
    except ValueError:
        body = {}
    up = health.status_code == 200
    checks = [Check("health", "PASS" if up else "FAIL",
                    f"HTTP {health.status_code}; {body.get('bundles', '?')} bundles for tier {body.get('tier', '?')}")]
    if not up:
        return checks
    stats = client.get("/stats").json()
    traffic = stats.get("traffic") or {}
    calls = int(traffic.get("requests") or 0)
    if calls:
        errors, fallbacks, p95 = traffic.get("error_share") or 0.0, traffic.get("fallback_share") or 0.0, traffic.get("p95_ms")
        checks.append(Check("errors", "FAIL" if errors > max_error_share else "PASS",
                            f"{errors:.1%} of {calls} calls failed (limit {max_error_share:.0%})"))
        checks.append(Check("unknown users", "WARN" if fallbacks > max_fallback_share else "PASS",
                            f"{fallbacks:.1%} of calls got the popularity fallback (limit {max_fallback_share:.0%})"))
        if p95 is not None:
            checks.append(Check("latency p95", "WARN" if p95 > max_p95_ms else "PASS",
                                f"{p95:.1f} ms inside the service, last {traffic.get('latency_window')} calls (limit {max_p95_ms:.0f} ms)"))
    else:
        checks.append(Check("traffic", "PASS", "no calls since this instance started (normal after scaling to zero)"))
    for b in stats.get("bundles") or []:
        name = f"{b['dataset']}/{b['method']}"
        age = b.get("export_age_days")
        if age is not None:
            checks.append(Check(f"{name}: freshness", "WARN" if age > max_export_age_days else "PASS",
                                f"exported {age:.1f} days ago (limit {max_export_age_days:g}); data up to {b.get('data_cutoff')}"))
        served, offline = (b.get("served") or {}), (b.get("offline") or {})
        coverage, offline_coverage = served.get("coverage_at_10"), offline.get("coverage_at_10")
        if coverage is not None and offline_coverage:
            ratio = coverage / offline_coverage
            checks.append(Check(f"{name}: coverage", "WARN" if ratio < min_coverage_ratio else "PASS",
                                f"served coverage@10 {coverage:.3f} vs {offline_coverage:.3f} offline ({ratio:.0%})"))
        pop, offline_pop = served.get("popularity_percentile_at_10"), offline.get("popularity_percentile_at_10")
        if pop is not None:
            checks.append(Check(f"{name}: popularity bias", "PASS",
                                f"popularity percentile {pop:.2f}" + (f" ({offline_pop:.2f} offline)" if offline_pop else "")
                                + "; 1.00 would mean only the most popular items"))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the health, freshness, quality and traffic of a recbench API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token", default=os.environ.get("RECBENCH_ID_TOKEN"), help="identity token for a private Cloud Run service")
    parser.add_argument("--max-export-age-days", type=float, default=30)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.5)
    parser.add_argument("--max-fallback-share", type=float, default=0.5)
    parser.add_argument("--max-error-share", type=float, default=0.05)
    parser.add_argument("--max-p95-ms", type=float, default=500.0)
    args = parser.parse_args()
    import httpx

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=60) as client:
        checks = run_checks(client, max_export_age_days=args.max_export_age_days, min_coverage_ratio=args.min_coverage_ratio,
                            max_fallback_share=args.max_fallback_share, max_error_share=args.max_error_share, max_p95_ms=args.max_p95_ms)
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"{c.status:4s}  {c.name:{width}s}  {c.detail}")
    failed = [c for c in checks if c.status == "FAIL"]
    warned = [c for c in checks if c.status == "WARN"]
    print(f"\n{len(checks)} checks: {len(failed)} FAIL, {len(warned)} WARN")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
