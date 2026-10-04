"""One comparison of every method across every dataset, for docs/results/overall-comparison.md.

    python -m recbench.report.overall --docs                 # docs/generated/overall/*.md
    python -m recbench.report.overall --out reports/overall  # the same as one report.md

Leaderboards compare methods within one dataset. This module adds the view across datasets:

- a results matrix: NDCG@10, rank and ties ("≈") per dataset; wins; the mean rank over the methods that ran on
  every dataset; the mean relative score (a method's NDCG@10 divided by the dataset's best); and the critical
  difference of the mean ranks (Demšar 2006), which shows how few differences five datasets can prove;
- accuracy against cost: relative score against training time, peak memory, CPU or GPU, the Pareto front, and a
  chart;
- beyond accuracy: coverage, long-tail share, popularity bias, novelty, next-item and cold-item results;
- a status matrix: which pairs finished, failed (and why), or were never attempted;
- a qualitative scorecard of all registered methods, from dictionary/catalog.yaml and each MethodSpec.

Each source is shown on its own and never mixed: the tuned quick tier, the tuned full-data confirmations, the
untuned v0.2 full run, and the untuned smoke tier. Whatever exists is shown, so the page fills in as results arrive.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recbench.protocol import DATASET_ORDER, dataset_sort_key
from recbench.results import fmt, leaderboard

# key, label, tier, tuning
SOURCES = [
    ("quick", "Quick tier, tuned (the bake-off)", "quick", "tuned"),
    ("full-tuned", "Full data, confirmed (tuned)", "full", "tuned"),
    ("full-untuned", "Full data, untuned v0.2 defaults", "full", "defaults"),
    ("smoke", "Smoke tier, untuned (pipeline check only)", "smoke", "defaults"),
]
METRIC = "ndcg_at_10"
BEYOND = [  # (metric, column title, higher is better)
    ("coverage_at_10", "Coverage@10", True),
    ("long_tail_share_at_10", "Long-tail share", True),
    ("gini_at_10", "Gini (exposure)", False),
    ("popularity_percentile_at_10", "Popularity pct.", False),
    ("novelty_at_10", "Novelty", True),
    ("next_ndcg_at_10", "Next-item NDCG@10", True),
    ("item_cold_recall_at_10", "Cold-item recall@10", True),
    ("personal_explanation_rate", "Personal expl.", True),
]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _link(method: str) -> str:
    return f"[{method}](../dictionary/algorithms/{method.replace('_', '-')}.md)"


def _is_true(value: Any) -> bool:
    """True only for a real boolean True (a missing value, NaN, is not a tie)."""
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:g} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


# ----- results matrix -----------------------------------------------------------------------------------------------

def critical_difference(k: int, n: int, alpha: float = 0.05) -> float | None:
    """Nemenyi critical difference for mean ranks of k methods over n datasets (Demšar 2006): two methods whose mean
    ranks differ by less than this are not significantly different."""
    if k < 2 or n < 2:
        return None
    from scipy.stats import studentized_range

    q = float(studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2)
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def results_matrix(frame: pd.DataFrame, metric: str = METRIC) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per ranked method. Columns per dataset: value, rank, tied. Summary columns: wins, ties,
    datasets_run, mean_rank (over the methods that ran on every dataset), relative (mean value / dataset best)."""
    if frame.empty or f"metrics.{metric}" not in frame:
        return pd.DataFrame(), {"datasets": [], "common": [], "cd": None}
    datasets = sorted(set(frame["tags.dataset"]), key=dataset_sort_key)
    boards = {d: leaderboard(frame, d, metric) for d in datasets}
    boards = {d: b for d, b in boards.items() if not b.empty}
    datasets = list(boards)
    rows: dict[str, dict[str, Any]] = {}
    for dataset, board in boards.items():
        best = float(board.loc[0, f"metrics.{metric}"])
        for _, r in board.iterrows():
            row = rows.setdefault(r["tags.method"], {"method": r["tags.method"]})
            value = float(r[f"metrics.{metric}"])
            tied = r["tied_with_best"]
            row[f"{dataset}:value"] = value
            row[f"{dataset}:rank"] = int(r["rank"])
            row[f"{dataset}:tied"] = bool(tied) if tied == tied else False
            row[f"{dataset}:relative"] = value / best if best > 0 else np.nan
    table = pd.DataFrame(list(rows.values())).set_index("method")
    common = [m for m in table.index if all(pd.notna(table.loc[m].get(f"{d}:value")) for d in datasets)]
    table["datasets_run"] = [sum(pd.notna(table.loc[m].get(f"{d}:value")) for d in datasets) for m in table.index]
    table["wins"] = [sum(table.loc[m].get(f"{d}:rank") == 1 for d in datasets) for m in table.index]
    table["ties"] = [sum(_is_true(table.loc[m].get(f"{d}:tied")) for d in datasets) for m in table.index]
    table["relative"] = table[[f"{d}:relative" for d in datasets]].mean(axis=1, skipna=True)
    mean_rank = pd.Series(np.nan, index=table.index)
    if common and len(datasets) > 0:
        ranks = []
        for d in datasets:  # re-rank within the common set, so missing methods do not shift anyone's rank
            ranks.append(table.loc[common, f"{d}:value"].rank(ascending=False, method="average"))
        mean_rank.loc[common] = pd.concat(ranks, axis=1).mean(axis=1)
    table["mean_rank"] = mean_rank
    table = table.sort_values(["mean_rank", "relative"], ascending=[True, False], na_position="last")
    info = {"datasets": datasets, "common": common, "cd": critical_difference(len(common), len(datasets))}
    return table, info


def render_matrix(table: pd.DataFrame, info: dict[str, Any]) -> str:
    if table.empty:
        return "_No results yet for this source._\n"
    datasets = info["datasets"]
    header = ["Method"] + datasets + ["Wins", "Ties with best", "Mean rank", "Relative score"]
    rows = []
    for method, r in table.iterrows():
        cells = [_link(method)]
        for d in datasets:
            value = r.get(f"{d}:value")
            if pd.isna(value):
                cells.append("—")
            else:
                mark = " ≈" if _is_true(r.get(f"{d}:tied")) and r.get(f"{d}:rank") != 1 else ""
                best = " **1st**" if r.get(f"{d}:rank") == 1 else ""
                cells.append(f"{value:.4f} ({int(r[f'{d}:rank'])}){mark}{best}")
        cells += [str(int(r["wins"])), str(int(r["ties"])), fmt(r["mean_rank"], 1) if pd.notna(r["mean_rank"]) else "—",
                  fmt(r["relative"], 2)]
        rows.append(cells)
    text = _table(header, rows)
    common, cd = info["common"], info["cd"]
    text += (f"\nCells: NDCG@10 (rank on that dataset). **1st** = best; ≈ = tied with the best (95% confidence intervals "
             f"overlap). Mean rank: over the {len(common)} methods that ran on all {len(datasets)} datasets (lower is better). "
             "Relative score: NDCG@10 divided by the dataset's best, averaged over the datasets the method ran on.\n")
    if cd is not None:
        text += (f"\n**Critical difference: {cd:.1f} ranks.** With {len(common)} methods on {len(datasets)} datasets, two "
                 f"mean ranks closer than {cd:.1f} are not significantly different (Nemenyi test, 5% level).\n")
    return text


# ----- accuracy vs cost ------------------------------------------------------------------------------------------------

def pareto_front(points: dict[str, tuple[float, float]]) -> set[str]:
    """Methods that no other method beats on both: higher score AND lower cost. points: method -> (score, cost)."""
    front = set()
    for m, (score, cost) in points.items():
        dominated = any((s >= score and c <= cost) and (s > score or c < cost) for other, (s, c) in points.items() if other != m)
        if not dominated:
            front.add(m)
    return front


def cost_table(frame: pd.DataFrame, table: pd.DataFrame, resources: dict[str, str], job_minutes: dict[str, float]) -> pd.DataFrame:
    rows = []
    for method in table.index:
        mine = frame[frame["tags.method"] == method]
        rows.append({
            "method": method,
            "relative": table.loc[method, "relative"],
            "train_s": float(mine["metrics.train_seconds"].median()) if "metrics.train_seconds" in mine else np.nan,
            "score_s": float(mine["metrics.score_seconds_per_1k_users"].median()) if "metrics.score_seconds_per_1k_users" in mine else np.nan,
            "rss_mb": float(mine["metrics.peak_rss_mb"].max()) if "metrics.peak_rss_mb" in mine else np.nan,
            "gpu_mb": float(mine["metrics.peak_gpu_mb"].max()) if "metrics.peak_gpu_mb" in mine else np.nan,
            "runs_on": resources.get(method, "GPU" if ("metrics.peak_gpu_mb" in mine and (mine["metrics.peak_gpu_mb"] > 0).any()) else "CPU"),
            "job_min": job_minutes.get(method, np.nan),
        })
    cost = pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()
    if not cost.empty:
        usable = cost[cost["train_s"].notna() & cost["relative"].notna()]
        front = pareto_front({m: (r["relative"], max(r["train_s"], 1e-3)) for m, r in usable.iterrows()})
        cost["pareto"] = [m in front for m in cost.index]
    return cost


def svg_scatter(cost: pd.DataFrame, title: str) -> str:
    """Relative score (y) against training time (x, log scale); Pareto methods ringed; colour = CPU or GPU."""
    data = cost[cost["train_s"].notna() & cost["relative"].notna()]
    if data.empty:
        return ""
    width, height, left, right, top, bottom = 760, 440, 70, 30, 40, 60
    xs = np.log10(np.maximum(data["train_s"].to_numpy(dtype=float), 0.01))
    lo, hi = math.floor(xs.min()), math.ceil(xs.max()) if xs.max() > xs.min() else math.floor(xs.min()) + 1

    def x(v: float) -> float:
        return left + (v - lo) / max(hi - lo, 1e-9) * (width - left - right)

    def y(v: float) -> float:
        return top + (1.0 - min(max(v, 0.0), 1.05) / 1.05) * (height - top - bottom)

    e = html.escape
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{e(title)}">',
           f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="bold">{e(title)}</text>']
    for tick in range(lo, hi + 1):  # x ticks at powers of ten seconds
        label = _human_seconds(10.0 ** tick)
        out.append(f'<line x1="{x(tick):.1f}" y1="{top}" x2="{x(tick):.1f}" y2="{height - bottom}" stroke="#ddd"/>'
                   f'<text x="{x(tick):.1f}" y="{height - bottom + 18}" text-anchor="middle" font-size="11">{label}</text>')
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{width - right}" y2="{y(tick):.1f}" stroke="#ddd"/>'
                   f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 15}" text-anchor="middle" font-size="12">training time (log scale) →</text>'
               f'<text x="18" y="{(top + height - bottom) / 2}" text-anchor="middle" font-size="12" '
               f'transform="rotate(-90 18 {(top + height - bottom) / 2})">relative score (1 = best) →</text>')
    for (method, r), xv in zip(data.iterrows(), xs):
        colour = "#e65100" if str(r["runs_on"]).upper().startswith("GPU") else "#1976d2"
        cx, cy = x(float(xv)), y(float(r["relative"]))
        if r.get("pareto"):
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="none" stroke="#2e7d32" stroke-width="2"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{colour}"><title>{e(method)}: '
                   f'{r["relative"]:.2f}, {r["train_s"]:.1f} s</title></circle>'
                   f'<text x="{cx + 8:.1f}" y="{cy - 6:.1f}" font-size="10">{e(method)}</text>')
    out.append(f'<text x="{width - right}" y="{top + 12}" text-anchor="end" font-size="11">'
               '<tspan fill="#1976d2">● CPU</tspan>  <tspan fill="#e65100">● GPU</tspan>  <tspan fill="#2e7d32">◯ Pareto front</tspan></text>')
    out.append("</svg>")
    return "\n".join(out)


def render_cost(cost: pd.DataFrame, title: str) -> str:
    if cost.empty:
        return "_No results yet for this source._\n"
    has_jobs = cost["job_min"].notna().any()
    header = ["Method", "Relative score", "Train time", "Scoring s / 1k users", "Peak RAM MB", "Peak GPU MB", "Runs on"]
    header += ["Job time incl. tuning"] if has_jobs else []
    header += ["Pareto"]
    rows = []
    for method, r in cost.sort_values("relative", ascending=False).iterrows():
        row = [_link(method), fmt(r["relative"], 2), fmt(r["train_s"], 1) + " s", fmt(r["score_s"], 2), fmt(r["rss_mb"], 0),
               fmt(r["gpu_mb"], 0) if r["gpu_mb"] and r["gpu_mb"] == r["gpu_mb"] else "–", r["runs_on"]]
        row += [fmt(r["job_min"], 0) + " min" if r["job_min"] == r["job_min"] else "–"] if has_jobs else []
        row += ["**yes**" if r.get("pareto") else ""]
        rows.append(row)
    chart = svg_scatter(cost, title)
    text = (f'<div class="overall-chart">\n{chart}\n</div>\n\n' if chart else "") + _table(header, rows)
    text += ("\nTrain time: the median over datasets of the final run's training (tuning not included). A method is on the "
             "**Pareto front** when no other method is both more accurate and faster: those are the sensible choices at "
             "each budget. Times are only comparable within one source (one machine).\n")
    return text


# ----- beyond accuracy, status, scorecard ------------------------------------------------------------------------------

def beyond_table(frame: pd.DataFrame, methods: list[str]) -> str:
    if frame.empty or not methods:
        return "_No results yet for this source._\n"
    means = {}
    for metric, _, _ in BEYOND:
        col = f"metrics.{metric}"
        means[metric] = frame.groupby("tags.method")[col].mean() if col in frame else pd.Series(dtype=float)
    best = {}
    for metric, _, higher in BEYOND:
        series = means[metric].reindex(methods).dropna()
        if not series.empty:
            best[metric] = series.max() if higher else series.min()
    rows = []
    for m in methods:
        row = [_link(m)]
        for metric, _, _ in BEYOND:
            value = means[metric].get(m, np.nan)
            cell = fmt(value, 3) if value == value else "–"
            if value == value and metric in best and abs(value - best[metric]) < 1e-12:
                cell = f"**{cell}**"
            row.append(cell)
        rows.append(row)
    header = ["Method"] + [f"{title} {'↑' if higher else '↓'}" for _, title, higher in BEYOND]
    return (_table(header, rows) + "\nAverages over the datasets each method ran on; ↑ higher is better, ↓ lower is "
            "better; the best value in each column is bold. – means not measured (for example, no `explain()` yet).\n")


def status_matrix(status: pd.DataFrame, methods: list[str], datasets: list[str], summaries: dict[tuple[str, str], dict]) -> str:
    """finished / failed / unsupported / over budget / not attempted, per method and dataset."""
    if not methods:
        return "_No runs yet for this source._\n"
    latest = {}
    if not status.empty:
        for _, r in status.iterrows():
            latest[(r["method"], r["dataset"])] = (r["status"], str(r.get("reason") or ""))
    rows = []
    for m in methods:
        row = [_link(m)]
        for d in datasets:
            summary = summaries.get((m, d))
            st, why = (summary.get("status"), summary.get("reason") or "") if summary else latest.get((m, d), ("—", ""))
            label = {"finished": "✓", "timeout": "over budget", "over_budget": "over budget"}.get(st, st)
            why = html.escape(why[:60]).replace("|", "/")  # a pipe would split the table cell
            row.append(label + (f" ({why})" if st not in ("finished", "—") and why else ""))
        rows.append(row)
    return _table(["Method"] + datasets, rows) + "\n✓ finished; — not attempted.\n"


def scorecard(catalog: dict[str, Any], registry: Any, queue_cfg: dict[str, Any]) -> str:
    resources = {e["name"]: e.get("resource", "cpu").upper() for e in (queue_cfg.get("methods") or [])}
    rows = []
    for name in sorted(registry.methods, key=lambda n: (catalog["methods"][n]["rung"], n)):
        spec, entry = registry.methods[name].spec, catalog["methods"][name]
        rubric = entry["rubric"]
        if spec.managed:
            gate = "managed service"
        elif name in resources:
            gate = f"yes ({resources[name]})"
        else:
            gate = "held back"
        idea = str(entry.get("summary", "")).split(". ")[0].rstrip(".") + "."
        yes = lambda flag: "yes" if flag else ""  # noqa: E731
        rows.append([f"[{entry['title']}](../dictionary/algorithms/{name.replace('_', '-')}.md)", str(entry["rung"]), entry["family"], idea,
                     gate, yes(spec.sequence_aware), yes(spec.scores_cold_items), yes(spec.requires_side_features),
                     *(str(rubric[k][0]) for k in ("implementation", "tuning", "data_hunger", "controllability", "explainability")),
                     entry["fidelity"]])
    header = ["Method", "Rung", "Family", "Idea", "In the bake-off", "Order-aware", "New items", "Needs content",
              "Build effort ↓", "Tuning effort ↓", "Data hunger ↓", "Control ↑", "Explanations ↑", "Fidelity"]
    return (_table(header, rows) + "\nRubric scores 1–5 from `dictionary/catalog.yaml` (anchors: "
            "[qualitative rubric](../dictionary/metrics/qualitative-rubric.md)); ↓ lower is better, ↑ higher is better.\n")


# ----- building -------------------------------------------------------------------------------------------------------

def _summaries(tier: str, stage: str, methods: list[str], datasets: list[str]) -> dict[tuple[str, str], dict]:
    from recbench.tuning.job import read_summary

    found = {}
    for m in methods:
        for d in datasets:
            summary = read_summary(tier, d, m, stage)
            if summary:
                found[(m, d)] = summary
    return found


def build(docs_dir: Path | None = None, out_dir: Path | None = None, root: Path | None = None) -> dict[str, str]:
    """Write every fragment; returns {fragment name: markdown}."""
    from recbench.config import load_benchmark_yaml
    from recbench.dictionary.build import load_catalog
    from recbench.registry import ensure_loaded
    from recbench.results import latest_status, load_runs

    root = (root or Path.cwd()).resolve()
    catalog, registry = load_catalog(root), ensure_loaded()
    quick_path = root / "configs" / "benchmarks" / "quick.yaml"
    queue_cfg = (load_benchmark_yaml(quick_path, root).get("queue") or {}) if quick_path.exists() else {}
    resources = {e["name"]: e.get("resource", "cpu").upper() for e in (queue_cfg.get("methods") or [])}
    fragments: dict[str, str] = {"scorecard": scorecard(catalog, registry, queue_cfg)}
    available = []
    for key, label, tier, tuning in SOURCES:
        try:
            frame = load_runs(tier, tuning=tuning)
            status = latest_status(tier, tuning=tuning)
        except Exception:  # noqa: BLE001 - docs still build without MLflow results
            frame, status = pd.DataFrame(), pd.DataFrame()
        table, info = results_matrix(frame)
        methods = list(table.index)
        datasets = info["datasets"] or list(DATASET_ORDER)
        job_minutes: dict[str, float] = {}
        summaries: dict[tuple[str, str], dict] = {}
        if tier == "quick" or (tier == "full" and tuning == "tuned"):
            stage = "tune" if tier == "quick" else "confirm"
            names = sorted(set(methods) | set(resources))
            summaries = _summaries(tier, stage, names, datasets)
            for m in names:
                spans = [(s["ended"] - s["started"]) / 60 for (mm, _), s in summaries.items() if mm == m and s.get("ended") and s.get("started")]
                if spans:
                    job_minutes[m] = float(np.median(spans))
        # Tuned sources ran through the queue, so its resource column says where each method ran; for the older
        # untuned runs, the GPU memory they used tells it.
        ran_on = resources if tuning == "tuned" else {}
        cost = cost_table(frame, table, ran_on, job_minutes) if not table.empty else pd.DataFrame()
        fragments[f"matrix-{key}"] = render_matrix(table, info)
        fragments[f"cost-{key}"] = render_cost(cost, f"{label}: accuracy against training time")
        fragments[f"beyond-{key}"] = beyond_table(frame, methods)
        others = (set(status["method"]) if not status.empty else set()) | {m for (m, _) in summaries}
        status_methods = methods + sorted(others - set(methods))  # ranked methods first, then those that never finished
        fragments[f"status-{key}"] = status_matrix(status, status_methods, datasets, summaries)
        if not table.empty:
            available.append((label, table, info))
    fragments["headline"] = headline(available)
    if docs_dir is not None:
        target = docs_dir / "generated" / "overall"
        target.mkdir(parents=True, exist_ok=True)
        for name, text in fragments.items():
            (target / f"{name}.md").write_text("<!-- generated by recbench.report.overall; do not edit -->\n\n" + text)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = ["# Overall comparison", "", fragments["headline"]]
        for key, label, _, _ in SOURCES:
            parts += [f"## {label}", "", fragments[f"matrix-{key}"], "### Accuracy vs cost", "", fragments[f"cost-{key}"],
                      "### Beyond accuracy", "", fragments[f"beyond-{key}"], "### Status", "", fragments[f"status-{key}"]]
        parts += ["## Qualitative scorecard", "", fragments["scorecard"]]
        (out_dir / "overall.md").write_text("\n".join(parts) + "\n")
    return fragments


def headline(available: list[tuple[str, pd.DataFrame, dict[str, Any]]]) -> str:
    if not available:
        return "_No results yet: run a benchmark (see Start here) and rebuild._\n"
    label, table, info = available[0]  # the most informative source that has results (SOURCES order)
    lines = [f"**Source shown first: {label}.** {len(table)} ranked methods on {len(info['datasets'])} datasets."]
    for d in info["datasets"]:
        col = f"{d}:rank"
        if col in table:
            winners = table[table[col] == 1].index.tolist()
            tied = [m for m in table.index if _is_true(table.loc[m].get(f"{d}:tied")) and table.loc[m].get(col) != 1]
            lines.append(f"- **{d}:** {', '.join(winners)}" + (f" (≈ {', '.join(tied)})" if tied else ""))
    ranked = table[table["mean_rank"].notna()]
    if not ranked.empty:
        top = ranked.index[0]
        lines.append(f"\nBest mean rank: **{top}** ({ranked.loc[top, 'mean_rank']:.1f}) over the {len(info['common'])} methods "
                     f"that ran everywhere" + (f"; critical difference {info['cd']:.1f}." if info["cd"] else "."))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare every method across every dataset.")
    parser.add_argument("--docs", action="store_true", help="write docs/generated/overall/*.md")
    parser.add_argument("--out", type=Path, default=None, help="also write <out>/overall.md")
    args = parser.parse_args()
    fragments = build(docs_dir=Path("docs") if args.docs else None, out_dir=args.out)
    print(fragments["headline"])


if __name__ == "__main__":
    main()
