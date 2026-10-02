# RecBench

A recommender-systems benchmark that doubles as a **dictionary**: 18 methods, from a random baseline to
Meta's HSTU and a managed service, 5 public datasets, an honest evaluation protocol, a serving API, and
documentation that explains every algorithm from scratch.

> **Status (v0.2, 2026-10-02):** protocol v2. A review of v0.1 found test-label leakage and other bugs
> that made earlier numbers invalid; they are fixed and guarded by tests. See
> [the review log](docs/review/2026-10-02-review.md).

## What is inside

| | |
|---|---|
| **Methods** (a ladder, simplest first) | Random, MostPopular · ItemKNN, EASE · BPR-MF, iALS · LightGCN, XSimGCL · SASRec, BERT4Rec, S3-Rec, HSTU, TIGER-lite* · DIN, DCN-V2 · text and multimodal towers · Recombee (managed) |
| **Datasets** | MovieLens 25M (movies), RetailRocket and H&M (e-commerce), Last.fm 1K (music), Steam (games) |
| **Protocol** | one global time cutoff (UTC); models see only pre-test events; full-catalog ranking for warm users; sampled 1+100 kept as a secondary check; bootstrap 95% confidence intervals |
| **Metrics** | accuracy (NDCG, Recall, HitRate, MAP, MRR, next-item), beyond-accuracy (coverage, Gini, novelty, diversity, serendipity, calibration, group gap), cold-start slices, efficiency, explainability, time to market |
| **Serving** | precomputed recommendation bundles behind a FastAPI app (no torch in the container), Cloud Run deployment with budget alerts and teardown |
| **Docs** | MkDocs site: tutorials, how-to guides, the dictionary (concepts, algorithms, metrics, datasets), codebase walkthrough, cloud guide |

\* experimental, not ranked.

## Quick start (macOS or Linux, CPU)

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"                 # laptop setup: CPU models, Recombee SDK, docs, pytest
bash scripts/fetch_third_party.sh          # pinned SELFRec and Meta generative-recommenders code
./scripts/run_smoke_cpu.sh --datasets movielens-25m --methods most_popular,ease,sasrec
open reports/smoke-*/report.html           # leaderboards with confidence intervals
mkdocs serve                               # the documentation at http://127.0.0.1:8000
```

H&M and RetailRocket download through Kaggle (put `kaggle.json` in `~/.kaggle/`). The full tier runs on an
NVIDIA GPU machine with `./scripts/run_full_gpu.sh`. Run the tests with `pytest -q` (or `pytest -q -m "not slow"`).

## Reading the numbers

Smoke runs use small user samples and untuned, step-budgeted training: they check the pipeline and give
first indications, not paper-level results. Compare methods within one dataset and one task, and treat
methods whose confidence intervals overlap as tied.

## Layout

```
src/recbench/       package: data pipeline, methods, evaluation, metrics, runner, reports, serving
configs/            benchmark and hardware profiles
dictionary/         catalog.yaml: facts and rubric scores shown in the docs
docs/               documentation sources (docs/generated/ is built from code and results)
scripts/, deploy/   run scripts, Docker/Cloud Run deployment, budget alert, teardown
tests/              leakage, alignment, metric, parity, serving, and docs-sync tests
```

## License

Code: MIT (see [LICENSE](LICENSE)). The datasets have their own, mostly non-commercial, licences; recbench
downloads them from their sources and never redistributes them. See the dataset pages in the docs.
