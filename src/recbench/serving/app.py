"""HTTP API and the results dashboard. Latency is measured by calling this process over HTTP."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader, select_autoescape

from recbench.protocol import PROTOCOL_NOTE
from recbench.registry import ensure_loaded
from recbench.store import SplitStore

app = FastAPI(title="recbench")
_TEMPLATES = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "dashboard" / "templates"),
    autoescape=select_autoescape(["html"]),
)
_LOADED: dict[tuple[str, str], object] = {}

TASK_METRIC = {
    "topn": "ndcg_at_10",
    "sequential": "ndcg_at_10",
    "session": "ndcg_at_10",
    "ctr": "auc",
    "rating": "rmse",
}


class RecommendRequest(BaseModel):
    user_id: str
    k: int = 10
    dataset: str | None = None


def _data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", Path.cwd() / "data"))


def _allowed() -> set[str] | None:
    raw = os.environ.get("RECBENCH_SERVE_METHODS")
    if not raw:
        return None
    return {part for part in raw.split(",") if part}


def _load(method_name: str, dataset: str):
    key = (method_name, dataset)
    if key in _LOADED:
        return _LOADED[key]
    tier = os.environ.get("RECBENCH_TIER", "smoke")
    preset = os.environ.get("RECBENCH_PRESET", "cpu")
    root = _data_root()
    ckpt = root / "artifacts" / dataset / tier / preset / f"{method_name}.pt"
    split = root / "splits" / dataset / tier
    if not ckpt.exists() or not (split / "meta.json").exists():
        raise HTTPException(status_code=404, detail=f"No checkpoint for {method_name} on {dataset}")
    reg = ensure_loaded()
    allowed = _allowed()
    if allowed is not None and method_name not in allowed:
        raise HTTPException(status_code=404, detail=f"{method_name} is not loaded on this service")
    method = reg.create_method(method_name)
    method.load(str(ckpt))
    store = SplitStore(split, dataset, tier)
    method._bind(store)
    _LOADED[key] = method
    return method


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/recommend")
def recommend(method: str, body: RecommendRequest) -> dict:
    dataset = body.dataset or os.environ.get("RECBENCH_DATASET", "")
    if not dataset:
        raise HTTPException(status_code=400, detail="dataset is required")
    model = _load(method, dataset)
    rows = model.recommend(body.user_id, k=body.k)
    return {"method": method, "dataset": dataset, "recommendations": rows}


@app.get("/")
def root():
    return RedirectResponse("/dashboard/topn")


@app.get("/dashboard/wiring")
def wiring():
    rows = _mlflow_rows(rankable=False)
    html = _TEMPLATES.get_template("wiring.html").render(
        title="Smoke wiring",
        protocol=PROTOCOL_NOTE,
        rows=rows,
    )
    return HTML(html)


@app.get("/dashboard/{task}")
def leaderboard(task: str):
    if task not in TASK_METRIC:
        raise HTTPException(status_code=404, detail="unknown task")
    metric = TASK_METRIC[task]
    rows = _rank_rows(task, metric)
    html = _TEMPLATES.get_template("leaderboard.html").render(
        title=f"Full tier — {task}",
        protocol=PROTOCOL_NOTE,
        task=task,
        metric=metric,
        tasks=list(TASK_METRIC),
        rows=rows,
    )
    return HTML(html)


def HTML(content: str):
    from fastapi.responses import HTMLResponse

    return HTMLResponse(content)


def _mlflow_frame():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./runs/mlflow"))
    experiment = mlflow.get_experiment_by_name("recbench")
    if experiment is None:
        return []
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def _mlflow_rows(rankable: bool) -> list[dict]:
    rows = []
    for record in _mlflow_frame():
        if str(record.get("tags.tier")) == "full":
            continue
        rows.append(
            {
                "dataset": record.get("tags.dataset"),
                "method": record.get("tags.method"),
                "status": record.get("tags.status"),
            }
        )
    return rows


def _rank_rows(task: str, metric: str) -> list[dict]:
    column = f"metrics.{metric}"
    rows = []
    for record in _mlflow_frame():
        if str(record.get("tags.tier")) != "full":
            continue
        if str(record.get("tags.rankable", "")).lower() != "true":
            continue
        if str(record.get("tags.status")) != "finished":
            continue
        if record.get(column) in (None, ""):
            continue
        sanity = record.get("metrics.sanity_popularity_recall_at_10")
        rows.append(
            {
                "dataset": record.get("tags.dataset"),
                "method": record.get("tags.method"),
                "value": f"{float(record[column]):.4f}",
                "preset": record.get("tags.preset"),
                "steps": record.get("tags.max_steps"),
                "sanity": "" if sanity in (None, "") else f"{float(sanity):.4f}",
                "_sort": float(record[column]),
            }
        )
    reverse = metric not in {"rmse", "mae", "logloss"}
    rows.sort(key=lambda row: row["_sort"], reverse=reverse)
    return rows
