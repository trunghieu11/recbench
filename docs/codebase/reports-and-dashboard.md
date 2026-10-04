# Reports and dashboard

## Reading results back: `results.py`

`src/recbench/results.py::load_runs` reads the MLflow store and returns one row per run, keeping only:

- runs with `protocol_version` = 2 (older, leaky runs are never mixed in);
- with `tuning="defaults"` or `tuning="tuned"`, only runs of that kind (runs from before the tag count as
  defaults). Without it, both kinds are mixed, so the reports and docs always pass one;
- one run per (dataset, tier, method): the **latest finished** one. When several finished runs share its
  `config_group` (the same settings with different seeds), their metrics are **averaged**, and the row gains
  `metrics.n_seeds` and `metrics.ndcg_at_10_seed_sd` (the spread across seeds).

`src/recbench/results.py::leaderboard` sorts one dataset's runs by a metric (lower-is-better metrics sorted the
other way), drops unranked (experimental) methods, adds ranks, and flags methods whose confidence interval overlaps the best one's (`tied_with_best`).
`TASK_BOARDS` defines the leaderboards: top-N (`ndcg_at_10`), next-item (`next_ndcg_at_10`), and CTR-style
(`sampled_auc`, diagnostic).

## The report

```bash
python -m recbench.report.build --tier smoke --tuning defaults --out reports/smoke-latest --docs
```

`src/recbench/report/build.py::build` writes `report.md` and `report.html` (Markdown converted to HTML). Each
dataset section (`dataset_section`) contains:

1. split facts from `meta.json`;
2. the top-N leaderboard with CIs and ≈ ties, recall, hit rate, coverage, training time, scoring time, memory,
   and the personal-explanation rate;
3. the next-item leaderboard;
4. the repeats-allowed comparison (only for datasets with that policy);
5. the beyond-accuracy table;
6. full ranking vs sampled ranks, with arrows;
7. experimental (unranked) methods;
8. methods that did not run, and why.

With `--docs`, each dataset section is also written to `docs/generated/leaderboards/<tier>/<dataset>.md`
(`<tier>-tuned/` with `--tuning tuned`), which the [leaderboards](../results/leaderboards.md) and
[quick-tier](../results/quick-tier.md) pages embed. The report's header says whether its runs were tuned, and the
datasets appear in run order (`protocol.py::DATASET_ORDER`).

## The overall comparison

```bash
python -m recbench.report.overall --docs --out reports/overall
```

`src/recbench/report/overall.py` compares every method across every dataset for each source of results (quick tier
tuned, full confirmed, full untuned, smoke): a results matrix with wins, ties, mean rank, relative score and the
critical difference; accuracy against cost with the Pareto front and an SVG chart; beyond-accuracy averages; a
status matrix; and a qualitative scorecard of every registered method. With `--docs` it writes
`docs/generated/overall/*.md` for the [overall comparison](../results/overall-comparison.md) page.
`scripts/fetch_results.sh` and the queue's per-dataset reports run it.

## The dashboard

`GET /dashboard` on the API (`src/recbench/serving/app.py::dashboard`) renders the same leaderboards as an HTML
page (template: `src/recbench/dashboard/templates/leaderboard.html`), plus a "Not finished" table. It needs MLflow,
so it works from the development environment; in the serve-only Docker image it answers 404 with an explanation. Use
`?tier_name=full` to switch tiers and `?tuning=defaults` or `?tuning=tuned` to keep one kind of run (without it,
both kinds are mixed).

## Regenerating the docs facts

```bash
python -m recbench.dictionary.build
```

Run it after adding methods or datasets, after editing `dictionary/catalog.yaml`, and after new runs (to refresh
the per-method result tables). Then `mkdocs build --strict` or `mkdocs serve`.
