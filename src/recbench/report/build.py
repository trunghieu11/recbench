"""Write leaderboards as Markdown (+ HTML) from MLflow results.

    python -m recbench.report.build --tier smoke --out reports/smoke-latest
    python -m recbench.report.build --tier full --out reports/full-latest --docs

--docs also writes docs/generated/leaderboards/<tier>/<dataset>.md, which the
documentation site embeds. Numbers in the docs therefore always come from here.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import pandas as pd

from recbench.protocol import PROTOCOL_NOTE
from recbench.results import fmt, latest_status, leaderboard, load_runs


def _m(row: pd.Series, name: str, digits: int = 3) -> str:
    return fmt(row.get(f"metrics.{name}"), digits)


def _ci(row: pd.Series, name: str) -> str:
    value, low, high = row.get(f"metrics.{name}"), row.get(f"metrics.{name}_ci_low"), row.get(f"metrics.{name}_ci_high")
    if value is None or pd.isna(value):
        return "–"
    if low is None or pd.isna(low):
        return fmt(value, 4)
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines + [""]


def _split_facts(data_dir: Path, dataset: str, tier: str) -> list[str]:
    meta_path = data_dir / "splits" / dataset / tier / "meta.json"
    if not meta_path.exists():
        return []
    m = json.loads(meta_path.read_text())
    facts = [
        ["Users / items", f"{m['n_users']:,} / {m['n_items']:,}"],
        ["Events before / after the test cutoff", f"{m['n_pretest']:,} / {m['n_test']:,}"],
        ["Test window starts (UTC)", m["test_start"]],
        ["Warm eval users (have history and a new test item)", f"{m['n_eval_warm']:,}"],
        ["Cold test users (no history; popularity is their only option)", f"{m['n_cold_test_users']:,}"],
        ["Share of test interactions that repeat a past item", f"{m['repeat_share']:.1%}"],
        ["Repeat policies", ", ".join(m["repeat_policies"]) + f" (primary: {m['primary_policy']})"],
    ]
    return ["**Split at a glance**", ""] + _table(["", ""], facts)


def dataset_section(frame: pd.DataFrame, dataset: str, tier: str, data_dir: Path, status: pd.DataFrame) -> list[str]:
    out = [f"## {dataset}", ""] + _split_facts(data_dir, dataset, tier)
    board = leaderboard(frame, dataset, "ndcg_at_10")
    if not board.empty:
        out += ["### Top-N for warm users (full-catalog ranking)", "",
                "Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.", ""]
        rows = []
        for _, r in board.iterrows():
            tied = "≈" if r["tied_with_best"] is True or r["tied_with_best"] == 1.0 else ""
            rows.append([str(int(r["rank"])), f"{r['tags.method']} {tied}".strip(), _ci(r, "ndcg_at_10"), _m(r, "recall_at_10"),
                         _m(r, "hitrate_at_10"), _m(r, "coverage_at_10"), _m(r, "train_seconds", 1),
                         _m(r, "score_seconds_per_1k_users", 2), _m(r, "peak_rss_mb", 0), _m(r, "personal_explanation_rate", 2)])
        out += _table(["#", "Method", "NDCG@10 [95% CI]", "Recall@10", "HitRate@10", "Coverage@10", "Train s",
                       "Score s / 1k users", "Peak RSS MB", "Personal expl."], rows)
    nxt = leaderboard(frame, dataset, "next_ndcg_at_10")
    if not nxt.empty:
        out += ["### Next-item prediction", "", "Can the method put the user's very next item in its top 10?", ""]
        rows = [[str(int(r["rank"])), r["tags.method"], _ci(r, "next_ndcg_at_10"), _m(r, "next_hitrate_at_10")] for _, r in nxt.iterrows()]
        out += _table(["#", "Method", "Next NDCG@10 [95% CI]", "Next HitRate@10"], rows)
    repeat_col = "metrics.allow_repeats/ndcg_at_10"
    has_repeats = repeat_col in frame and frame.loc[frame["tags.dataset"] == dataset, repeat_col].notna().any()
    if has_repeats and not board.empty:
        out += ["### Same users, repeats allowed", "", "Re-listening/re-buying counts as a hit here.", ""]
        rows = [[r["tags.method"], _m(r, "ndcg_at_10"), _m(r, "allow_repeats/ndcg_at_10")] for _, r in board.iterrows()]
        out += _table(["Method", "NDCG@10 (new items only)", "NDCG@10 (repeats allowed)"], rows)
    if not board.empty:
        out += ["### Beyond accuracy", ""]
        rows = [[r["tags.method"], _m(r, "coverage_at_10"), _m(r, "gini_at_10"), _m(r, "popularity_percentile_at_10"),
                 _m(r, "long_tail_share_at_10"), _m(r, "novelty_at_10", 2), _m(r, "ild_at_10"), _m(r, "serendipity_at_10"),
                 _m(r, "calibration_kl_at_10"), _m(r, "user_group_ndcg_gap_at_10"), _m(r, "item_cold_recall_at_10")]
                for _, r in board.iterrows()]
        out += _table(["Method", "Coverage", "Gini ↓", "Popularity pct ↓", "Long-tail share", "Novelty", "ILD",
                       "Serendipity", "Calibration KL ↓", "Group gap ↓", "Cold-item recall"], rows)
        sampled = leaderboard(frame, dataset, "sampled_ndcg_at_10")
        if not sampled.empty:
            sampled_rank = dict(zip(sampled["tags.method"], sampled["rank"]))
            rows = []
            for _, r in board.iterrows():
                s_rank = sampled_rank.get(r["tags.method"])
                moved = "" if s_rank is None or s_rank == r["rank"] else ("↑" if s_rank < r["rank"] else "↓")
                rows.append([r["tags.method"], str(int(r["rank"])), "–" if s_rank is None else f"{int(s_rank)} {moved}".strip(),
                             _m(r, "sampled_ndcg_at_10")])
            out += ["### Full ranking vs. sampled (1 + 100 negatives)", "",
                    "If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.", ""]
            out += _table(["Method", "Rank (full)", "Rank (sampled)", "Sampled NDCG@10"], rows)
    unranked = frame[(frame["tags.dataset"] == dataset) & (frame.get("tags.ranked", "true") == "false")]
    if not unranked.empty:
        out += ["### Experimental (not ranked)", ""]
        rows = [[r["tags.method"], _ci(r, "ndcg_at_10"), _m(r, "next_ndcg_at_10")] for _, r in unranked.iterrows()]
        out += _table(["Method", "NDCG@10", "Next NDCG@10"], rows)
    if not status.empty:
        problems = status[(status["dataset"] == dataset) & (status["status"] != "finished")]
        if not problems.empty:
            out += ["### Did not run", ""]
            out += _table(["Method", "Status", "Reason"], [[p["method"], p["status"], str(p["reason"])[:160]] for _, p in problems.iterrows()])
    return out


def build(tier: str, out_dir: Path, *, docs_dir: Path | None = None, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or Path(os.environ.get("DATA_DIR", "data"))
    frame = load_runs(tier)
    status = latest_status(tier)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(set(frame["tags.dataset"])) if not frame.empty else []
    header = [f"# Results — {tier} tier", "", f"> {PROTOCOL_NOTE}", ""]
    if tier == "smoke":
        header += ["> Smoke splits are small user samples (about 50K events, at most 2,000 eval users), so confidence "
                   "intervals are wide. Use them to check the pipeline, and the full tier to choose a method.", ""]
    body: list[str] = []
    for dataset in datasets:
        section = dataset_section(frame, dataset, tier, data_dir, status)
        body += section
        if docs_dir is not None:
            target = docs_dir / "generated" / "leaderboards" / tier / f"{dataset}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(section[1:]) + "\n")
    if not datasets:
        body = ["No finished protocol-v2 runs for this tier yet."]
    text = "\n".join(header + body) + "\n"
    (out_dir / "report.md").write_text(text)
    (out_dir / "report.html").write_text(_html(text, f"Results — {tier} tier"))
    return out_dir


def _html(markdown_text: str, title: str) -> str:
    try:
        import markdown

        body = markdown.markdown(markdown_text, extensions=["tables"])
    except ImportError:
        body = f"<pre>{html.escape(markdown_text)}</pre>"
    style = ("body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 16px}"
             "table{border-collapse:collapse;display:block;overflow-x:auto}th,td{border-bottom:1px solid #ddd;padding:.3rem .5rem;text-align:left}")
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{style}</style></head><body>{body}</body></html>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write benchmark leaderboards.")
    parser.add_argument("--tier", default="smoke")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--docs", action="store_true", help="also write docs/generated/leaderboards/<tier>/*.md")
    args = parser.parse_args()
    print(build(args.tier, args.out, docs_dir=Path("docs") if args.docs else None))


if __name__ == "__main__":
    main()
