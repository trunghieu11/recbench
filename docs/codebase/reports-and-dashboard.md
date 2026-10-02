# Reports and dashboard

## Reading results back: `results.py`

`src/recbench/results.py::load_runs` reads the MLflow store and returns one row per run, keeping only:

- runs with `protocol_version` = 2 (older, leaky runs are never mixed in);
- one run per (dataset, tier, method): the **latest finished** one.

`src/recbench/results.py::leaderboard` sorts one dataset's runs by a metric (lower-is-better metrics sorted the
other way), drops unranked (experimental) methods, adds ranks, and flags methods whose confidence interval overlaps the best one's (`tied_with_best`).
`TASK_BOARDS` defines the leaderboards: top-N (`ndcg_at_10`), next-item (`next_ndcg_at_10`), and CTR-style
(`sampled_auc`, diagnostic).

## The report

```bash
python -m recbench.report.build --tier smoke --out reports/smoke-latest --docs
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

With `--docs`, each dataset section is also written to `docs/generated/leaderboards/<tier>/<dataset>.md`, which
the [leaderboards page](../results/leaderboards.md) embeds.

## The dashboard

`GET /dashboard` on the API (`src/recbench/serving/app.py::dashboard`) renders the same leaderboards as an HTML
page (template: `src/recbench/dashboard/templates/leaderboard.html`), plus a "Not finished" table. It needs MLflow,
so it works from the development environment, not from the serve-only Docker image. Use `?tier_name=full` to
switch tiers.

## Regenerating the docs facts

```bash
python -m recbench.dictionary.build
```

Run it after adding methods or datasets, after editing `dictionary/catalog.yaml`, and after new runs (to refresh
the per-method result tables). Then `mkdocs build --strict` or `mkdocs serve`.
