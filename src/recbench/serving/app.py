"""HTTP API: serve precomputed recommendation bundles, plus a small results dashboard.

Run locally:
    uvicorn recbench.serving.app:app --port 8080
    curl -X POST localhost:8080/recommend -H 'content-type: application/json' \\
         -d '{"dataset": "movielens-25m", "method": "ease", "user_id": "123", "k": 10}'

Environment:
    RECBENCH_BUNDLES        folder holding <dataset>/<tier>/<method>/ bundles (default: data/bundles)
    RECBENCH_TIER           tier to serve (default: smoke)
    RECBENCH_SERVE_METHODS  optional comma-separated allow-list of methods
    RECBENCH_REQUEST_LOG    "0" turns off the one-line JSON log of every /recommend call (default: on)

Monitoring: GET /health (ready or 503), GET /stats (what each bundle serves and how fresh it is, plus this
instance's traffic: requests, fallback share, errors, latency), and one JSON log line per request, which Cloud
Logging stores as structured fields (see docs/cloud/monitoring.md and python -m recbench.serving.monitor).

The API imports only numpy, pandas, and pyarrow, so the container needs no torch.
The dashboard pages additionally need MLflow (installed with the "bench" extra).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from recbench import __version__
from recbench.protocol import PROTOCOL_NOTE
from recbench.serving.bundle import Bundle

app = FastAPI(title="recbench", version=__version__, description="Recommendations from precomputed bundles.")
_TEMPLATES = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "dashboard" / "templates"),
    autoescape=select_autoescape(["html"]),
)


class RecommendRequest(BaseModel):
    user_id: str
    dataset: str
    method: str
    k: int = Field(10, ge=1, le=100)


def bundle_root() -> Path:
    default = Path(os.environ.get("DATA_DIR", Path.cwd() / "data")) / "bundles"
    return Path(os.environ.get("RECBENCH_BUNDLES", default))


def tier() -> str:
    return os.environ.get("RECBENCH_TIER", "smoke")


def _allowed(method: str) -> bool:
    raw = os.environ.get("RECBENCH_SERVE_METHODS", "")
    return not raw or method in {m.strip() for m in raw.split(",") if m.strip()}


@lru_cache(maxsize=64)
def load_bundle(dataset: str, tier_name: str, method: str) -> Bundle:
    folder = bundle_root() / dataset / tier_name / method
    if not (folder / "manifest.json").exists():
        raise FileNotFoundError(folder)
    return Bundle(folder)


def available() -> list[dict]:
    root = bundle_root()
    found = []
    for manifest in sorted(root.glob("*/*/*/manifest.json")):
        dataset, tier_name, method = manifest.parts[-4:-1]
        if tier_name == tier() and _allowed(method):
            found.append({"dataset": dataset, "tier": tier_name, "method": method})
    return found


class Traffic:
    """Counts since this instance started. Cloud Run starts and stops instances, so these reset; the request log
    (one JSON line per call) is the durable record."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = self.errors = self.fallbacks = 0
        self.latencies: deque[float] = deque(maxlen=1000)
        self.per_bundle: dict[tuple[str, str], list[int]] = {}

    def record(self, dataset: str, method: str, ms: float, *, error: bool, fallback: bool) -> None:
        with self.lock:
            self.requests += 1
            self.errors += int(error)
            self.fallbacks += int(fallback)
            self.latencies.append(ms)
            counts = self.per_bundle.setdefault((dataset, method), [0, 0])
            counts[0] += 1
            counts[1] += int(fallback)

    def summary(self) -> dict[str, Any]:
        with self.lock:
            values = sorted(self.latencies)
            pick = lambda q: round(values[min(len(values) - 1, int(q * len(values)))], 2) if values else None  # noqa: E731
            return {"since": _iso(self.started), "requests": self.requests,
                    "error_share": self.errors / self.requests if self.requests else None,
                    "fallback_share": self.fallbacks / self.requests if self.requests else None,
                    "p50_ms": pick(0.50), "p95_ms": pick(0.95), "latency_window": len(values)}

    def bundle(self, dataset: str, method: str) -> dict[str, Any]:
        with self.lock:
            requests, fallbacks = self.per_bundle.get((dataset, method), [0, 0])
        return {"requests": requests, "fallback_share": fallbacks / requests if requests else None}


TRAFFIC = Traffic()
_BUNDLE_STATS: dict[tuple[str, str, str], dict[str, Any]] = {}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(text: str | None) -> float | None:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(text), fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def _log(event: dict[str, Any]) -> None:
    if os.environ.get("RECBENCH_REQUEST_LOG", "1") != "0":
        print(json.dumps(event), flush=True)  # Cloud Logging turns a JSON line into structured fields


def bundle_stats(dataset: str, tier_name: str, method: str) -> dict[str, Any]:
    """Facts and quality of one bundle, computed once per instance."""
    key = (dataset, tier_name, method)
    if key not in _BUNDLE_STATS:
        bundle = Bundle(bundle_root() / dataset / tier_name / method)
        m = bundle.manifest
        cutoff = m.get("data_cutoff")
        if not cutoff:  # bundles before version 2 only have the sentence "lists reflect events before <time> UTC"
            found = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(m.get("freshness", "")))
            cutoff = found.group(0) if found else None
        _BUNDLE_STATS[key] = {"exported_at": m.get("exported_at"), "data_cutoff": cutoff,
                              "n_users": m.get("n_users"), "k": m.get("k"), "run_id": m.get("run_id"), "stage": m.get("stage"),
                              "served": bundle.quality(), "offline": m.get("offline") or {}}
    return _BUNDLE_STATS[key]


@app.get("/health")
def health():
    """Ready when at least one bundle can be served for this tier; otherwise 503, so Cloud Run's probe and an uptime
    check notice a deployment without bundles."""
    n = len(available())
    body = {"ok": n > 0, "version": __version__, "tier": tier(), "bundles": n}
    return body if n else JSONResponse(body, status_code=503)


@app.get("/methods")
def methods() -> list[dict]:
    return available()


@app.post("/recommend")
def recommend(body: RecommendRequest) -> dict:
    began, status, fallback = time.perf_counter(), 500, None
    try:
        if not _allowed(body.method):
            raise HTTPException(status_code=404, detail=f"{body.method} is not served here")
        try:
            bundle = load_bundle(body.dataset, tier(), body.method)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"No bundle for {body.method} on {body.dataset} ({tier()})") from None
        result = {"dataset": body.dataset, "method": body.method, **bundle.recommend(body.user_id, body.k)}
        status, fallback = 200, bool(result["fallback"])
        return result
    except HTTPException as exc:
        status = exc.status_code
        raise
    finally:
        ms = (time.perf_counter() - began) * 1000
        TRAFFIC.record(body.dataset, body.method, ms, error=status >= 400, fallback=bool(fallback))
        _log({"severity": "INFO" if status < 400 else "WARNING", "message": "recommend", "dataset": body.dataset,
              "method": body.method, "k": body.k, "status": status, "latency_ms": round(ms, 2),
              "fallback": fallback})  # fallback: an unknown user got the popularity list


@app.get("/stats")
def stats(dataset: str | None = None) -> dict:
    """Per bundle: when it was made, how old its data is, what it serves (coverage, popularity bias) next to the
    offline values of the run that made it, and its traffic. Plus this instance's traffic and latency."""
    now = time.time()
    bundles = []
    for entry in available():
        if dataset and entry["dataset"] != dataset:
            continue
        facts = bundle_stats(entry["dataset"], entry["tier"], entry["method"])
        exported, cutoff = _parse_time(facts["exported_at"]), _parse_time(facts["data_cutoff"])
        bundles.append({**entry, **facts,
                        "export_age_days": round((now - exported) / 86400, 2) if exported else None,
                        "data_age_days": round((now - cutoff) / 86400, 1) if cutoff else None,
                        "traffic": TRAFFIC.bundle(entry["dataset"], entry["method"])})
    return {"version": __version__, "tier": tier(), "started_at": _iso(TRAFFIC.started),
            "uptime_seconds": round(now - TRAFFIC.started, 1), "traffic": TRAFFIC.summary(), "bundles": bundles}


@app.get("/")
def root() -> dict:
    return {"service": "recbench", "version": __version__, "tier": tier(),
            "endpoints": {"GET /health": "ready check", "GET /methods": "what can be served", "POST /recommend": "top-K for one user",
                          "GET /stats": "freshness, quality and traffic of what is served",
                          "GET /dashboard": "leaderboards (only where MLflow results are available)"}}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(tier_name: str | None = None, tuning: str | None = None):
    """Leaderboards from MLflow. Needs MLflow and a results store, so it works locally but not in the serving image."""
    try:
        from recbench.results import TASK_BOARDS, latest_status, leaderboard, load_runs

        chosen = tier_name or tier()
        frame = load_runs(chosen, tuning=tuning)
    except Exception as exc:  # noqa: BLE001 - no MLflow in the image, or no results store
        raise HTTPException(status_code=404, detail=f"The dashboard needs MLflow results here ({type(exc).__name__})") from None
    boards = []
    if not frame.empty:
        for dataset in sorted(frame["tags.dataset"].unique()):
            for task, (metric, title) in TASK_BOARDS.items():
                board = leaderboard(frame, dataset, metric)
                if board.empty:
                    continue
                rows = [
                    {
                        "rank": int(r["rank"]),
                        "method": r["tags.method"],
                        "value": r[f"metrics.{metric}"],
                        "low": r.get(f"metrics.{metric}_ci_low"),
                        "high": r.get(f"metrics.{metric}_ci_high"),
                        "tied": bool(r["tied_with_best"]) if r["tied_with_best"] == r["tied_with_best"] else None,
                    }
                    for _, r in board.iterrows()
                ]
                boards.append({"dataset": dataset, "task": task, "title": title, "metric": metric, "rows": rows})
    status = latest_status(chosen, tuning=tuning)
    problems = [] if status.empty else status[status["status"] != "finished"].to_dict(orient="records")
    html = _TEMPLATES.get_template("leaderboard.html").render(
        title=f"recbench results — {chosen} tier", protocol=PROTOCOL_NOTE, boards=boards, problems=problems
    )
    return HTMLResponse(html)
