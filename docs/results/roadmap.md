# Roadmap

What recbench deliberately does not do yet, why, and how each item could be built. Each item is sized to be a
separate pull request. Good first projects are marked ★.

## Done since this page was first written

- **Hyperparameter tuning with Optuna:** the [quick-tier bake-off](quick-tier.md) tunes every low-budget method
  with the same budget on a validation fold, then tests the best setting once.
- **Two-stage pipeline:** the [LightGBM](../dictionary/algorithms/lgbm-rerank.md) and
  [DCN-V2](../dictionary/algorithms/dcnv2-rerank.md) re-rankers re-order candidates from EASE, ItemKNN and
  trending items, and report candidate recall.
- **Several seeds** for the full-data confirmation of each dataset's top 3 (stochastic methods only). Paired
  significance tests are still open (below).

## Evaluation

| Item | Why | Sketch |
|---|---|---|
| ★ Paired tests and seeds in the bake-off's reports | the lab now has a paired test (`python -m recbench.compare`) and tests random methods with 3 seeds; the bake-off's leaderboards still use interval overlap and one seed | reuse `src/recbench/compare.py::paired_difference` in the leaderboards; `tuning.final_seeds` in `configs/benchmarks/quick.yaml` |
| ★ Scenario-profile leaderboards | the [overall comparison](overall-comparison.md) now has an accuracy-vs-cost Pareto front, but one metric still hides other trade-offs | add coverage to the Pareto front; profiles weight metrics per scenario ("discovery", "low cost") |
| Online-style evaluation | offline accuracy is not user response ([offline vs online](../dictionary/concepts/offline-vs-online.md)) | off-policy estimators on logged data, or a simple simulator; A/B routing in the API |
| Rolling temporal windows | one test window per dataset (the last 10% of events; H&M: 7 days) | repeat the split at several cutoffs and average |

## Methods

| Item | Why | Sketch |
|---|---|---|
| A real LLM recommender | the text-hash tower is a placeholder, and [text-embedding kNN](../dictionary/algorithms/text-knn.md) only uses a pretrained encoder ([LLM and generative recsys](../dictionary/concepts/llm-and-generative-recsys.md)) | the same encoder's vectors in a two-tower model trained on interactions; or an LLM re-ranker over EASE's top 50 |
| Real image (and audio) embeddings | 15 colour features are a toy | precompute CLIP- or SigLIP-style embeddings for H&M images into the split cache; feed them to the multimodal tower |
| Faithful TIGER | TIGER-lite is unranked | RQ-VAE on content embeddings, a seq2seq model generating semantic IDs, beam search, collision handling |
| Live Amazon Personalize and Vertex AI runs | only Recombee is benchmarked live | adapters following `src/recbench/methods/recombee.py`, with a request budget and teardown; check costs first |

## Cost, serving, and operations

| Item | Why | Sketch |
|---|---|---|
| ★ Cost = runtime × price table | dollars are easier to compare than seconds (your Q14) | a `configs/prices.yaml` (machine type → hourly price); compute `train_cost_usd` in the report |
| Real-time serving | bundles are only as fresh as the last export ([serving and bundles](../codebase/serving-and-bundles.md)) | a second image with PyTorch that loads model weights and scores the current history; an ANN index for embedding models |
| ★ GitHub Actions CI | tests only run when someone remembers | run `pytest -q -m "not slow"` and `mkdocs build --strict` on every pull request |
| MLflow 3 | `mlflow<3` is pinned | upgrade, check `load_runs` and the file store, and update `import_runs` |
| Zensical | Material for MkDocs is in maintenance mode | build the same `mkdocs.yml` with Zensical once it supports every extension used here |

## Decisions this project is built on

recbench was planned through two rounds of questions. These answers explain many design choices; change them
deliberately.

### Round 1: scope and protocol

| # | Topic | Decision |
|---|---|---|
| Q1 | Purpose | learning and reference, product decisions, and an open-source portfolio project |
| Q2 | Starting knowledge | none assumed: explain from scratch |
| Q3 | Timeline | a broad v1 in two to three months |
| Q4 | Domains | e-commerce, music, movies and video |
| Q5 | Tasks | top-N, next-item and session, similar items, CTR |
| Q6 | Scale | small and medium datasets (up to about 30 million events) |
| Q7 | Side information | text, plus images and audio where available |
| Q8 | Licences | tag each dataset's licence; allow all for research |
| Q9 | Splits | a global temporal cutoff, plus leave-last-out for next-item |
| Q10 | Candidates | full-catalog ranking |
| Q11 | Leaderboards | scenario profiles and Pareto fronts (accuracy-vs-cost Pareto front done; profiles planned) |
| Q12 | Statistics | seeds, confidence intervals, significance tests (intervals done; 3 seeds in the full-data confirmations; tests planned) |
| Q13 | Qualitative criteria | a 1–5 rubric with reasons, plus automated proxies |
| Q14 | Cost | runtime × a price table (proxies done; price table planned) |
| Q15 | Beyond accuracy | coverage and popularity, diversity and novelty, cold start, fairness and calibration |
| Q16 | Online evaluation | offline now, designed so online evaluation can plug in |
| Q17 | Hardware | GPUs on demand (superseded twice: your own GPU machine in round 2, then short rented boxes for the bake-off) |
| Q18 | Cloud | Google Cloud |
| Q19 | Budget | under $50 a month |
| Q20 | Serving | every model behind one API, load-tested |

### Round 2: after the first review (2026-10-02)

| # | Topic | Decision |
|---|---|---|
| 1 | Order of work | fix the critical validity bugs first, then write the docs |
| 2 | Baselines | add the full ladder: Random, MostPopular, ItemKNN, EASE, BPR-MF, iALS, LightGCN, SASRec |
| 3 | Protocol | full ranking is primary; sampled 1 + 100 is a secondary check |
| 4 | Managed services | Recombee end to end; AWS Personalize and Vertex AI documented only |
| 5 | Docs format | an in-repo MkDocs site |
| 6 | Language | English |
| 7 | Docs style | tutorials plus reference |
| 8 | Docs scope | runbook, recsys dictionary with a detailed page per algorithm, codebase walkthrough, cloud and MLOps |
| 9 | Explanations | intuition first, then the key math symbol by symbol, with tiny worked examples |
| 10 | Full tier | runs on your own GPU machine; no cloud GPU automation (later: the quick-tier bake-off runs on short vast.ai rentals, see [step 5](../start/quick-tier-box.md)) |

The [2026-10-02 review](../review/2026-10-02-review.md) records the bugs that round 2 fixed.
