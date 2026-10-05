# Known limitations

Every benchmark simplifies. This page lists recbench's simplifications so you can judge how far its results
travel. Items that are planned to change are also on the [roadmap](roadmap.md).

## Evaluation

| Limitation | Effect | Mitigation |
|---|---|---|
| **Offline only** | measures how well past behaviour is predicted, not how users react to new recommendations | treat results as hypotheses; validate with an A/B test ([offline vs online](../dictionary/concepts/offline-vs-online.md)) |
| **Few seeds** | the untuned tiers and the quick-tier jobs run one seed, so their training randomness is not measured; confidence intervals cover user sampling only. It can be large: the LightGBM re-ranker's three seeds on full Steam scored 0.0509, 0.0561 and 0.0581 | full-data confirmations of each dataset's top 3 use 3 seeds for methods with random training; treat small differences as ties |
| **One time cap per job, not per setting** | a setting that cannot stop early may use up the job's 3 hours. Only the methods trained by recbench's own epoch loop stop at their fair share of the time; SANSA, SLIM, iALS and GRU4Rec (the authors' code) cannot. In the first bake-off, SANSA finished 2 of its 10 settings on H&M and RetailRocket, and GRU4Rec 1 of 10 on RetailRocket and Steam; their results there are a lower bound | the status table marks these jobs "✓ (n of 10 settings)"; a per-setting limit is on the [roadmap](roadmap.md) |
| **Tuning only in the bake-off** | the smoke and full tiers run default settings, so tuned methods could rank differently; held-back methods are not tuned yet | the [quick-tier bake-off](quick-tier.md) tunes the 23 low-budget methods with the same budget; see [fair baselines](../dictionary/concepts/fair-baselines-and-tuning.md) |
| **Quick tier is a sample** | about 1M events per dataset; validation folds are small on some datasets (about 850 warm users on MovieLens, 220 on Last.fm), so close settings are effectively tied; quick-tier training times were measured with 3 or 4 jobs sharing a GPU | each dataset's top 3 are confirmed on full data |
| **One global temporal cutoff** | results describe one test window per dataset (the last 10% of events; H&M: the last 7 days); seasonal effects are not averaged out | rolling-window evaluation would fix this; not planned |
| **At most 10,000 eval users** (2,000 in the smoke config) | slightly wider intervals on big datasets | raise `max_eval_users` if you have time |
| **CTR, rating, and session tasks are diagnostic only** | there are no real impressions (shown but not clicked items) in these datasets | `sampled_auc` and `sampled_logloss` are reported, never ranked |
| **Cold users are not ranked** | the leaderboards describe warm users only | the split box reports how many cold users there are; they get the popularity fallback |
| **Smoke tier is tiny** | many ties; under-trained deep models | use the quick-tier bake-off and its full-data confirmations for decisions |

## Methods

| Limitation | Details |
|---|---|
| **Simplified methods** | HSTU (small dense version of Meta's model), TIGER-lite (no content-based semantic IDs, no generation; unranked), text-hash tower (hashed bag of words, not an LLM), multimodal tower (15 colour features per image, not a vision model). Each algorithm page has a "Fidelity" box. |
| **Library implementations** | BERT4Rec, S3-Rec, and DIN run through RecBole; iALS and BPR-MF through `implicit`; LightGCN and XSimGCL follow SELFRec; GRU4Rec is the authors' official code; SANSA uses the `sansa` package; the re-rankers use LightGBM and FuxiCTR; text kNN uses sentence-transformers. Their behaviour includes those libraries' choices (for example, RecBole's BERT4Rec ignores the oldest item of every history when recommending). |
| **Item caps** | EASE and Turbo-CF keep dense item × item matrices, so they use at most 20,000 (laptop) or 30,000 (GPU) of the most popular items, fewer if GPU memory is short; the rest cannot be recommended by them. SANSA has no cap. |
| **Re-rankers need a big machine** | the LightGBM re-ranker needed 27 to 43 GB of RAM on the quick tier and up to 62 GB on full RetailRocket; the DCN-V2 re-ranker up to 41 GB. They do not fit on a 24 GB laptop for the larger datasets |
| **Some methods cannot explain yet** | PureSVD, SANSA, GF-CF, Turbo-CF, V-SKNN, MultVAE, RecVAE and GRU4Rec have no `explain()`, so "Personal expl." shows "–" for them |
| **XSimGCL under-trained at smoke defaults** | its contrastive loss needs more steps than the smoke preset gives (the algorithm page explains the settings that close the gap) |
| **DIN not in the smoke config** | full-catalog scoring with a pointwise network is too slow on a laptop CPU; run it with `--methods din` or on the GPU |
| **S3-Rec needs categories** | unsupported where items have fewer than two category values (Last.fm) |
| **Multimodal tower needs images** | only H&M has images |

## Data

| Limitation | Details |
|---|---|
| **Implicit feedback only** | ratings and purchases are all treated as "the user interacted"; dislikes are lost |
| **Ties in timestamps** | H&M and Steam record dates, not times; events on the same day are ordered by file order (deterministic but arbitrary) |
| **Dataset licences** | several datasets forbid commercial use; see the [datasets index](../dictionary/datasets/index.md) |
| **Domains** | e-commerce, fashion, movies, music, and games; no news, video platforms, or social feeds |

## Serving and cost

| Limitation | Details |
|---|---|
| **Precomputed lists** | recommendations reflect events before the test cutoff, not the user's latest clicks ([serving and bundles](../codebase/serving-and-bundles.md)) |
| **Bundles cover the 20,000 most recently active users** | others get the popularity fallback |
| **Cost is reported as proxies** | seconds and megabytes, plus a qualitative cost band; no dollar figures yet ([cost](../dictionary/metrics/cost.md)) |
| **Peak GPU memory is CUDA only** | Apple GPU (MPS) memory is not measured |
| **Managed services** | only Recombee is benchmarked, on a small slice; Amazon Personalize and Vertex AI Search for commerce are documented only |

## Tooling

| Limitation | Details |
|---|---|
| **MLflow pinned below 3** | the local file store and query code target MLflow 2.x |
| **No continuous integration** | tests run locally (`pytest -q`); a CI workflow is planned |
| **Docs theme** | Material for MkDocs is in maintenance mode; the configuration uses only features that Zensical, its successor, also supports |
