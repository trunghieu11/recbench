"""What the quick-tier queue is doing, as text (`python -m recbench.queue status`) and as an HTML page that refreshes
itself every 30 seconds (reports/queue/<tier>.html, written by the running queue).

Both are built from two sources:
- the queue's state file, runs/queue/<tier>.json: which jobs run where and since when. The running queue rewrites
  it when a job starts or ends and once a minute (a heartbeat), so its age tells whether the queue is alive;
- the job summaries in runs/tuning/<tier>/<dataset>/<method>[.confirm].json: results and reasons.
"""

from __future__ import annotations

import html
import json
import statistics
import time
from pathlib import Path
from typing import Any

from recbench.tuning.job import read_summary, summary_path

HEARTBEAT_SECONDS = 60
STALE_SECONDS = 180  # three missed heartbeats: the queue process has most likely stopped
PROBLEMS = {"failed", "over_budget", "unsupported", "missing_split", "skipped"}
FINAL = {"finished"} | PROBLEMS


def _minutes(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def collect(benchmark: dict[str, Any], tier: str, state: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    """Everything the text and HTML views show, as plain data."""
    now = now or time.time()
    confirm_tier = str((benchmark.get("confirm") or {}).get("tier", "full"))
    methods = [e["name"] for e in (benchmark.get("queue") or {}).get("methods") or []]
    datasets = list(benchmark.get("datasets") or state.get("datasets") or [])  # every dataset, not only this session's
    jobs = {j["key"]: j for j in state.get("jobs") or []}
    updated = state.get("updated")
    age = now - updated if updated else None
    alive = age is not None and age < STALE_SECONDS and state.get("finished") is not True
    view: dict[str, Any] = {"tier": tier, "now": now, "updated": updated, "age": age, "alive": alive,
                            "finished": bool(state.get("finished")), "host": state.get("host"), "datasets": []}

    durations, remaining = [], 0
    for dataset in datasets:
        tune = []
        for method in methods:
            key = f"tune:{dataset}:{method}"
            job, summary = jobs.get(key) or {}, read_summary(tier, dataset, method) or {}
            status = job.get("status") or summary.get("status") or "pending"
            if status == "pending" and summary.get("status") in FINAL:
                status = summary["status"]
            if status == "running" and not alive:
                status = "interrupted"  # it continues the next time the queue runs
            started, ended = job.get("started") or summary.get("started"), job.get("ended") or summary.get("ended")
            if status in FINAL and started and ended:
                durations.append(ended - started)
            if status not in FINAL:
                remaining += 1
            test = (summary.get("test") or {}).get("ndcg_at_10") if isinstance(summary.get("test"), dict) else None
            tune.append({"method": method, "status": status, "test": test, "val": summary.get("best_val"),
                         "elapsed": (now - started) if status == "running" and started else None,
                         "job_seconds": (ended - started) if status in FINAL and started and ended else None,
                         "trials": len(summary.get("trials") or []), "slot": job.get("slot", ""),
                         "reason": job.get("reason") or summary.get("reason") or ""})
        confirms = []
        confirm_jobs = {j["method"]: j for j in jobs.values() if j.get("kind") == "confirm" and j.get("dataset") == dataset}
        folder = summary_path(confirm_tier, dataset, "x", "confirm").parent  # confirmations from earlier sessions too
        for path in sorted(folder.glob("*.confirm.json")) if folder.exists() else []:
            confirm_jobs.setdefault(path.name[: -len(".confirm.json")], {})
        for method, job in confirm_jobs.items():
            summary = read_summary(confirm_tier, dataset, method, "confirm") or {}
            status = job.get("status") or summary.get("status") or "pending"
            if status == "running" and not alive:
                status = "interrupted"
            if status not in FINAL:
                remaining += 1
            test = summary.get("test") or {}
            confirms.append({"method": method, "status": status, "test": test.get("ndcg_at_10"),
                             "seed_sd": test.get("ndcg_at_10_seed_sd"), "bundle": summary.get("bundle", ""),
                             "elapsed": (now - job["started"]) if status == "running" and job.get("started") else None,
                             "slot": job.get("slot", ""), "reason": job.get("reason") or summary.get("reason") or ""})
        done = sum(r["status"] in FINAL for r in tune)
        view["datasets"].append({"name": dataset, "done": done, "total": len(tune), "tune": tune, "confirm": confirms})

    started = state.get("session_started")
    end = now if alive else (updated or now)
    hours = (end - started) / 3600 if started else None
    price = state.get("price_per_hour")
    slots = int(state.get("cpu_workers") or 1) + int(state.get("gpu_slots") or 0)
    eta = remaining * statistics.median(durations) / max(slots, 1) / 3600 if durations and remaining else None
    view.update(session_hours=hours, price=price, cost=hours * price if hours is not None and price else None,
                remaining=remaining, eta_hours=eta, slots=slots)
    return view


def render_text(view: dict[str, Any]) -> str:
    lines = []
    if view["updated"]:
        state = ("finished (the session ended normally)" if view["finished"] else
                 "running" if view["alive"] else "STOPPED? (no heartbeat for over 3 minutes)")
        lines.append(f"queue: {state}; state written {_minutes(view['age'])} ago"
                     + (f" on {view['host']}" if view.get("host") else ""))
    else:
        lines.append("queue: no state file here (results below come from the job summaries)")
    if view["session_hours"] is not None:
        cost = f", about ${view['cost']:.2f} at ${view['price']}/h" if view["cost"] is not None else ""
        eta = f"; roughly {view['eta_hours']:.1f} h of work left for {view['remaining']} jobs" if view["eta_hours"] else ""
        lines.append(f"session: {view['session_hours']:.2f} h{cost}{eta}")
    for d in view["datasets"]:
        lines.append(f"\n== {d['name']}: {d['done']}/{d['total']} jobs done")
        for r in sorted(d["tune"], key=lambda r: -(r["test"] if r["test"] is not None else -1)):
            test = f"{r['test']:.4f}" if r["test"] is not None else "   -  "
            val = f"{r['val']:.4f}" if r["val"] is not None else "-"
            extra = f"  {r['slot']} for {_minutes(r['elapsed'])}" if r["elapsed"] is not None else ""
            if r["status"] in PROBLEMS and r["reason"]:
                extra = f"  {r['reason'][:90]}"
            lines.append(f"  {r['method']:16s} {r['status']:12s} test NDCG@10={test}  val={val}{extra}")
        for c in d["confirm"]:
            test = f"{c['test']:.4f}" if c["test"] is not None else "-"
            extra = f"  {c['slot']} for {_minutes(c['elapsed'])}" if c["elapsed"] is not None else ""
            if c["status"] in PROBLEMS and c["reason"]:
                extra = f"  {c['reason'][:90]}"
            lines.append(f"  confirm {c['method']:8s} {c['status']:12s} full test NDCG@10={test}{extra}")
    return "\n".join(lines)


def render_html(view: dict[str, Any]) -> str:
    e = html.escape
    badge = ("finished", "#2e7d32") if view["finished"] else (("running", "#2e7d32") if view["alive"] else ("stopped?", "#c62828"))
    head = [f"<span class='badge' style='background:{badge[1]}'>{badge[0]}</span>"]
    if view["updated"]:
        head.append(f"state written {_minutes(view['age'])} ago")
    if view["session_hours"] is not None:
        head.append(f"session {view['session_hours']:.2f} h")
    if view["cost"] is not None:
        head.append(f"about ${view['cost']:.2f} so far")
    if view["eta_hours"]:
        head.append(f"roughly {view['eta_hours']:.1f} h left ({view['remaining']} jobs)")
    parts = [f"<h1>recbench queue: {e(view['tier'])} tier</h1><p>{' · '.join(head)}</p>",
             f"<p class='note'>Refreshes every 30 s. Generated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(view['now']))}.</p>"]
    for d in view["datasets"]:
        pct = 100 * d["done"] / max(d["total"], 1)
        parts.append(f"<h2>{e(d['name'])}: {d['done']}/{d['total']} jobs</h2><div class='bar'><div style='width:{pct:.0f}%'></div></div>")
        rows = []
        for r in sorted(d["tune"], key=lambda r: -(r["test"] if r["test"] is not None else -1)):
            when = _minutes(r["elapsed"]) + " so far" if r["elapsed"] is not None else _minutes(r["job_seconds"])
            note = e(r["reason"][:140]) if r["status"] in PROBLEMS else e(r["slot"]) if r["status"] == "running" else ""
            rows.append(f"<tr class='{e(r['status'])}'><td>{e(r['method'])}</td><td>{e(r['status'])}</td>"
                        f"<td>{'%.4f' % r['test'] if r['test'] is not None else ''}</td><td>{'%.4f' % r['val'] if r['val'] is not None else ''}</td>"
                        f"<td>{r['trials'] or ''}</td><td>{when}</td><td>{note}</td></tr>")
        parts.append("<table><tr><th>Method</th><th>Status</th><th>Test NDCG@10</th><th>Val NDCG@10</th><th>Trials</th>"
                     "<th>Time</th><th>Note</th></tr>" + "".join(rows) + "</table>")
        if d["confirm"]:
            rows = [f"<tr class='{e(c['status'])}'><td>{e(c['method'])}</td><td>{e(c['status'])}</td>"
                    f"<td>{'%.4f' % c['test'] if c['test'] is not None else ''}</td><td>{'yes' if c['bundle'] else ''}</td>"
                    f"<td>{e(c['reason'][:140]) if c['status'] in PROBLEMS else _minutes(c['elapsed']) if c['elapsed'] is not None else ''}</td></tr>"
                    for c in d["confirm"]]
            parts.append("<h3>Confirmed on full data</h3><table><tr><th>Method</th><th>Status</th><th>Full test NDCG@10</th>"
                         "<th>Bundle</th><th>Note</th></tr>" + "".join(rows) + "</table>")
    style = ("body{font:14px/1.45 -apple-system,system-ui,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 16px;color:#222}"
             "table{border-collapse:collapse;width:100%;margin:.4rem 0 1rem}th,td{border-bottom:1px solid #ddd;padding:.25rem .45rem;text-align:left}"
             ".badge{color:#fff;border-radius:4px;padding:.1rem .5rem}.bar{background:#eee;height:8px;border-radius:4px}"
             ".bar div{background:#1976d2;height:8px;border-radius:4px}.note{color:#666}"
             "tr.running td{background:#e3f2fd}tr.failed td,tr.over_budget td{background:#ffebee}tr.interrupted td{background:#fff8e1}")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='30'>"
            f"<title>recbench queue: {e(view['tier'])}</title><style>{style}</style></head><body>{''.join(parts)}</body></html>")


def write_html(view: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(render_html(view))
    tmp.replace(path)
    return path


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}
