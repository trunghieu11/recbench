"""Export the full-tier run as Markdown and a self-contained HTML file."""

from __future__ import annotations

import argparse
import html
import os
import socket
from pathlib import Path

import mlflow
import yaml

from recbench.protocol import PROTOCOL_NOTE

TASK_METRIC = {
    "topn": "ndcg_at_10",
    "sequential": "ndcg_at_10",
    "session": "ndcg_at_10",
    "ctr": "auc",
    "rating": "rmse",
}
LOWER_BETTER = {"rmse", "mae", "logloss"}


def _runs(tier: str, preset: str | None):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./runs/mlflow"))
    experiment = mlflow.get_experiment_by_name("recbench")
    if experiment is None:
        return []
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if frame.empty:
        return []
    records = frame.to_dict(orient="records")
    kept = []
    for row in records:
        if str(row.get("tags.tier", "")) != tier:
            continue
        if preset and str(row.get("tags.preset", "")) not in {preset, "None"}:
            continue
        kept.append(row)
    return kept


def _catalog(root: Path) -> dict:
    path = root / "dictionary" / "catalog.yaml"
    if not path.exists():
        return {"methods": {}}
    with path.open() as handle:
        return yaml.safe_load(handle)


def build_report(out_dir: Path, tier: str = "full", preset: str | None = None, repo_root: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = repo_root or Path.cwd()
    catalog = _catalog(root).get("methods") or {}
    rows = _runs(tier, preset)
    sections: list[str] = [
        f"# Full-tier report ({preset or 'all presets'})",
        "",
        f"Host: {socket.gethostname()}",
        "",
        PROTOCOL_NOTE,
        "",
        "Methods are sorted only inside one dataset and one task. "
        "Popularity Recall@10 is a sanity line and is not a rank. "
        "Managed-service smokes are not in this sort.",
        "",
    ]
    failures = [row for row in rows if str(row.get("tags.status")) not in {"finished", "skipped_existing"}]
    rankable = [
        row
        for row in rows
        if str(row.get("tags.rankable", "")).lower() == "true" and str(row.get("tags.status")) == "finished"
    ]
    datasets = sorted({str(row.get("tags.dataset")) for row in rankable})
    for dataset in datasets:
        sections.append(f"## {dataset}")
        subset = [row for row in rankable if str(row.get("tags.dataset")) == dataset]
        for task, metric in TASK_METRIC.items():
            column = f"metrics.{metric}"
            scored = [row for row in subset if row.get(column) not in (None, "")]
            if not scored:
                continue
            reverse = metric not in LOWER_BETTER
            scored.sort(key=lambda row: float(row[column]), reverse=reverse)
            sections.append(f"### {task} by {metric}")
            for index, row in enumerate(scored, start=1):
                method = row.get("tags.method")
                extra = catalog.get(method, {})
                sanity = row.get("metrics.sanity_popularity_recall_at_10")
                sanity_text = f", sanity popularity Recall@10={float(sanity):.4f} (not ranked)" if sanity not in (None, "") else ""
                sections.append(
                    f"{index}. {method}: {metric}={float(row[column]):.4f}{sanity_text}; "
                    f"steps={row.get('tags.max_steps')}; cost={extra.get('cost_band', '')}; "
                    f"train_s={row.get('metrics.train_wall_seconds', '')}"
                )
            sections.append("")
    if failures:
        sections.append("## Not ranked")
        for row in failures:
            sections.append(
                f"- {row.get('tags.dataset')} / {row.get('tags.method')}: {row.get('tags.status')} {row.get('tags.reason', '')}"
            )
    text = "\n".join(sections) + "\n"
    (out_dir / "report.md").write_text(text)
    escaped = html.escape(text)
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Full-tier report</title>"
        "<style>body{font-family:Georgia,serif;max-width:860px;margin:2rem auto;line-height:1.45}"
        "pre{white-space:pre-wrap}</style></head><body><pre>"
        + escaped
        + "</pre></body></html>"
    )
    (out_dir / "report.html").write_text(page)
    return out_dir


def build_wiring_report(
    out_dir: Path,
    datasets: list[str],
    methods: list[str],
    *,
    data_root: Path | None = None,
) -> Path:
    """Pass/fail list. Methods are not sorted and scores are not treated as a ranking."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = data_root or Path(os.environ.get("DATA_DIR", Path.cwd() / "data"))
    rows = _runs("smoke", None)
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("tags.dataset")), str(row.get("tags.method")))
        previous = latest.get(key)
        if previous is None or str(row.get("start_time") or "") >= str(previous.get("start_time") or ""):
            latest[key] = row
    lines = [
        "# Smoke wiring report",
        "",
        "This is a pass/fail list. It does not rank methods.",
        "",
        PROTOCOL_NOTE,
        "",
    ]
    for dataset in datasets:
        unavailable = root / "unavailable" / f"{dataset}.txt"
        split = root / "splits" / dataset / "smoke" / "meta.json"
        lines.append(f"## {dataset}")
        if unavailable.exists() and not split.exists():
            reason = unavailable.read_text().strip().splitlines()[-1:] or ["download failed"]
            for method in methods:
                lines.append(f"- {method}: dataset_unavailable ({reason[0][:240]})")
            lines.append("")
            continue
        for method in methods:
            row = latest.get((dataset, method))
            if row is None:
                lines.append(f"- {method}: failed (no MLflow run)")
                continue
            status = str(row.get("tags.status") or "failed")
            if status == "skipped":
                status = "unsupported"
            reason = str(row.get("tags.reason") or "").strip()
            suffix = f" ({reason[:240]})" if reason and status != "finished" else ""
            lines.append(f"- {method}: {status}{suffix}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text)
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Smoke wiring report</title>"
        "<style>body{font-family:Georgia,serif;max-width:860px;margin:2rem auto;line-height:1.45}"
        "pre{white-space:pre-wrap}</style></head><body><pre>"
        + html.escape(text)
        + "</pre></body></html>"
    )
    (out_dir / "report.html").write_text(page)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the full-tier report.")
    parser.add_argument("--tracking-uri", default="")
    parser.add_argument("--tier", default="full")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wiring", action="store_true")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--methods", default="")
    args = parser.parse_args()
    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    if args.wiring:
        datasets = [part for part in args.datasets.split(",") if part]
        methods = [part for part in args.methods.split(",") if part]
        print(build_wiring_report(args.out, datasets, methods))
        return
    print(build_report(args.out, tier=args.tier, preset=args.preset))


if __name__ == "__main__":
    main()
