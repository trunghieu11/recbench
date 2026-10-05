# RecBench

A recommender-systems benchmark that doubles as a **dictionary**: methods from a random baseline to Meta's HSTU
and a managed service, 5 public datasets, an honest evaluation protocol, a fair tuned bake-off, a serving API with
monitoring, and documentation that explains every algorithm from scratch.

> **Status (2026-10-05):** protocol v2 (a review of v0.1 found test-label leakage; fixed and guarded by tests, see
> [the review log](docs/review/2026-10-02-review.md)). The first **quick-tier bake-off** has run: 23 low-budget
> methods tuned with the same budget on all five datasets, and each dataset's top 3 confirmed on the full data.
> A short follow-up session comes next: a SASRec re-run, new confirmations for the LightGBM re-ranker and for
> Last.fm, and the serving bundles.

## Results so far

Best on the full data, by test NDCG@10. A tie (≈) means overlapping 95% confidence intervals.

| Dataset | Winner |
|---|---|
| MovieLens-25M | EASE 0.225 ≈ LightGBM re-ranker ≈ PureSVD |
| RetailRocket | RP3beta 0.028 ≈ iALS ≈ LightGBM re-ranker |
| Steam | EASE 0.058 ≈ SANSA ≈ LightGBM re-ranker |
| H&M | LightGBM re-ranker 0.028, clearly ahead of EASE (0.021) |
| Last.fm | EASE 0.106 ≈ PureSVD ≈ SLIM |

Across the five datasets, EASE has the best mean rank, and no neural model placed higher than 8th on any dataset
under the same tuning budget. The findings and their caveats are in `docs/results/quick-tier.md`.

## What is inside

| | |
|---|---|
| **Methods** | a ladder, simplest first; see the table below |
| **Datasets** | MovieLens 25M (movies), RetailRocket and H&M (e-commerce), Last.fm 1K (music), Steam (games) |
| **Protocol** | one global time cutoff (UTC); models see only pre-test events; full-catalog ranking for warm users; sampled 1+100 kept as a secondary check; bootstrap 95% confidence intervals |
| **Metrics** | accuracy (NDCG, Recall, HitRate, MAP, MRR, next-item), beyond-accuracy (coverage, Gini, novelty, diversity, serendipity, calibration, group gap), cold-start slices, efficiency, explainability, time to market |
| **Tuning** | the quick-tier bake-off: about 1M events per dataset, 10 settings per method (Optuna), validation folds, a 3-hour cap per job, each dataset's top 3 confirmed on full data |
| **Serving** | precomputed recommendation bundles behind a FastAPI app (no torch in the container), Cloud Run deployment with budget alerts, an uptime check, quality checks (`/stats`) and teardown |
| **Docs** | MkDocs site: tutorials, how-to guides, the dictionary (concepts, algorithms, metrics, datasets), codebase walkthrough, cloud guide |

<!-- generated:methods (python -m recbench.dictionary.build writes this block; do not edit) -->

**34 methods** on a ladder from simplest to most complex; the 23 in **bold** are in the quick-tier bake-off, the others are held back (heavier) or a managed service.

| Rung | Methods |
|---|---|
| 0 Baselines | **Random**, **MostPopular** |
| 1 Neighbourhood and linear models | **ItemKNN**, **SLIM (ElasticNet)**, **RP3beta**, **EASE^R**, **SANSA** |
| 2 Matrix factorisation | **iALS**, **BPR-MF**, **PureSVD**, **MultVAE**, **RecVAE**, **SimpleX**, **DirectAU** |
| 3 Graph methods | LightGCN, **GF-CF**, **UltraGCN**, XSimGCL, **Turbo-CF** |
| 4 Sequential models | **GRU4Rec**, **SASRec**, **V-SKNN**, BERT4Rec, S3-Rec, TIGER-lite, HSTU |
| 5 CTR-style rankers and re-rankers | **LightGBM re-ranker**, DIN, DCN-V2, **DCN-V2 re-ranker** |
| 6 Content-based methods | Text hash tower, Multimodal tower, **Text-embedding kNN** |
| 7 Managed services | Recombee |

<!-- /generated:methods -->

## Quick start (macOS or Linux, CPU)

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"                 # laptop setup: CPU models, Recombee SDK, docs, pytest
bash scripts/fetch_third_party.sh          # pinned SELFRec, Meta generative-recommenders and official GRU4Rec code
./scripts/run_smoke_cpu.sh --datasets movielens-25m --methods most_popular,ease,sasrec
open reports/smoke-*/report.html           # leaderboards with confidence intervals
mkdocs serve                               # the documentation at http://127.0.0.1:8000
```

H&M and RetailRocket download through Kaggle (put `kaggle.json` in `~/.kaggle/`). The quick-tier bake-off runs on a
rented GPU box (step by step in `docs/start/quick-tier-box.md`: `setup_box.sh`, `upload_splits.sh`,
`run_quick_box.sh`, `fetch_results.sh`). Run the tests with `pytest -q` (or `pytest -q -m "not slow"`).

## Reading the numbers

Smoke runs use small user samples and default settings: they check the pipeline, not which method is best. The
quick-tier bake-off chooses, with the same tuning budget for every method. Compare methods within one dataset, treat
methods whose confidence intervals overlap as tied, and use the overall comparison (`docs/results/overall-comparison.md`)
to see which methods are good everywhere.

## Improve the methods yourself

The labs (`docs/labs/index.md`) take the 11 light methods one week at a time: reproduce the baseline, understand every
setting, try data tricks, and implement one idea from a paper, with hints and folded solutions. Your experiments live
in their own workspace (`runs/lab/`), and `python -m recbench.compare` judges each one with a paired test. The
developer handbook (`docs/handbook/index.md`) covers the daily workflow: set up, run, experiment, compare, test,
debug, Git and docs.

## Layout

```
src/recbench/       package: data pipeline, methods, evaluation, metrics, runner, reports, serving
configs/            benchmark and hardware profiles; tuning/ holds the bake-off's search spaces
labs/               the improvement labs: one notebook and experiments.yaml per method, the test template
dictionary/         catalog.yaml: facts and rubric scores shown in the docs
docs/               documentation sources (docs/generated/ is built from code and results)
scripts/, deploy/   run scripts, Docker/Cloud Run deployment, budget alert, teardown
tests/              leakage, alignment, metric, parity, serving, and docs-sync tests
```

## License

Code: MIT (see [LICENSE](LICENSE)). The datasets have their own, mostly non-commercial, licences; recbench
downloads them from their sources and never redistributes them. See the dataset pages in the docs.
