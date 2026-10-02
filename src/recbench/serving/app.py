"""HTTP API: serve precomputed recommendation bundles, plus a small results dashboard.

Run locally:
    uvicorn recbench.serving.app:app --port 8080
    curl -X POST localhost:8080/recommend -H 'content-type: application/json' \\
         -d '{"dataset": "movielens-25m", "method": "ease", "user_id": "123", "k": 10}'

Environment:
    RECBENCH_BUNDLES        folder holding <dataset>/<tier>/<method>/ bundles (default: data/bundles)
    RECBENCH_TIER           tier to serve (default: smoke)
    RECBENCH_SERVE_METHODS  optional comma-separated allow-list of methods

The API imports only numpy, pandas, and pyarrow, so the container needs no torch.
The dashboard pages additionally need MLflow (installed with the "bench" extra).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
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


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "bundles": len(available())}


@app.get("/methods")
def methods() -> list[dict]:
    return available()


@app.post("/recommend")
def recommend(body: RecommendRequest) -> dict:
    if not _allowed(body.method):
        raise HTTPException(status_code=404, detail=f"{body.method} is not served here")
    try:
        bundle = load_bundle(body.dataset, tier(), body.method)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No bundle for {body.method} on {body.dataset} ({tier()})") from None
    return {"dataset": body.dataset, "method": body.method, **bundle.recommend(body.user_id, body.k)}


@app.get("/")
def root():
    return RedirectResponse("/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(tier_name: str | None = None):
    try:
        from recbench.results import TASK_BOARDS, latest_status, leaderboard, load_runs
    except ImportError:
        raise HTTPException(status_code=404, detail="The dashboard needs MLflow (pip install 'recbench[bench]')") from None
    chosen = tier_name or tier()
    frame = load_runs(chosen)
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
    status = latest_status(chosen)
    problems = [] if status.empty else status[status["status"] != "finished"].to_dict(orient="records")
    html = _TEMPLATES.get_template("leaderboard.html").render(
        title=f"recbench results — {chosen} tier", protocol=PROTOCOL_NOTE, boards=boards, problems=problems
    )
    return HTMLResponse(html)
