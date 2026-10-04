# Decision guide

This page turns benchmark results into a choice of method for a real product. It does not give you a single
winner, because there is none: the right method depends on your data, your task, and your constraints. It gives
you a shortlist and tells you what to check for a dataset like yours.

!!! warning "Use tuned results"
    Base decisions on the [quick-tier bake-off](quick-tier.md), where every method gets the same tuning budget,
    and on its full-data confirmations. The [overall comparison](overall-comparison.md) shows which methods are
    good everywhere. The untuned [full-tier leaderboards](leaderboards.md) understate methods with many settings,
    and smoke results are for learning the workflow.

## Step 1: find the dataset most like yours

| Your product | Closest dataset | Why |
|---|---|---|
| online shop, browsing sessions, many anonymous visitors | [RetailRocket](../dictionary/datasets/retailrocket.md) | sparse, short histories, many cold users |
| fashion or catalogue retail with product images | [H&M](../dictionary/datasets/hm.md) | purchases, seasonal items, images and metadata |
| media library (films, series, books) with long histories | [MovieLens-25M](../dictionary/datasets/movielens-25m.md) | dense, long histories, stable catalog |
| music or podcasts, repeated consumption | [Last.fm-1K](../dictionary/datasets/lastfm.md) | heavy repeats, strong sequential habits |
| games or apps with metadata | [Steam](../dictionary/datasets/steam.md) | popularity-heavy, tags and genres |

## Step 2: shortlist by situation

| Situation | Start with | Then try | Why |
|---|---|---|---|
| you need something this week | [MostPopular](../dictionary/algorithms/most-popular.md) + [EASE](../dictionary/algorithms/ease.md), [RP3beta](../dictionary/algorithms/rp3beta.md) or [ItemKNN](../dictionary/algorithms/itemknn.md) | [iALS](../dictionary/algorithms/ials.md), [SLIM](../dictionary/algorithms/slim.md) | no GPU, few settings, strong baselines, easy to explain |
| order matters (next song, next click) | [SASRec](../dictionary/algorithms/sasrec.md), [GRU4Rec](../dictionary/algorithms/gru4rec.md) | [V-SKNN](../dictionary/algorithms/vsknn.md) (no training); later [BERT4Rec](../dictionary/algorithms/bert4rec.md) or [HSTU](../dictionary/algorithms/hstu.md) | sequence models use the order of events; compare them on the next-item board |
| catalog over ~50,000 items, millions of users | [SANSA](../dictionary/algorithms/sansa.md) (EASE without the item cap), [iALS](../dictionary/algorithms/ials.md) | [GF-CF](../dictionary/algorithms/gfcf.md), [MultVAE](../dictionary/algorithms/multvae.md) | EASE's memory grows with items squared; SANSA and factor models scale much better |
| many new items (fast-moving catalog) | [text-embedding kNN](../dictionary/algorithms/text-knn.md) | the LightGBM re-ranker with its text feature | content features can score items with no interactions (check cold-item recall) |
| reranking a short candidate list with rich features | [LightGBM re-ranker](../dictionary/algorithms/lgbm-rerank.md) | [DCN-V2 re-ranker](../dictionary/algorithms/dcnv2-rerank.md) | two stages: cheap models propose ~200 candidates, a ranker orders them with many signals; check candidate recall first |
| explanations are required ("because you watched X") | [ItemKNN](../dictionary/algorithms/itemknn.md), [EASE](../dictionary/algorithms/ease.md), [RP3beta](../dictionary/algorithms/rp3beta.md) or [SLIM](../dictionary/algorithms/slim.md) | embedding models with cited neighbours | item-to-item weights are directly readable |
| no ML team, need a hosted service | [Recombee](../dictionary/algorithms/recombee.md) | other [managed services](../dictionary/algorithms/managed-services.md) | you trade control and cost for time to market |

## Step 3: check the results for your dataset

For each shortlisted method, look up on the dataset from step 1, on the [quick-tier page](quick-tier.md) and the
[overall comparison](overall-comparison.md):

1. **Accuracy:** is it tied with the best (≈) on the board for your task (top-N or next-item)?
2. **Beyond accuracy:** coverage and popularity percentile, if discovery matters; calibration, if users have
   distinct tastes; group gap, if light users matter to you.
3. **Efficiency:** training time and scoring time per 1,000 users, scaled to your catalog and user counts.
4. **Rubric:** each algorithm page's facts box scores implementation effort, tuning effort, data hunger,
   controllability, and explainability from 1 to 5, with reasons. See the
   [qualitative rubric](../dictionary/metrics/qualitative-rubric.md).

Prefer the **simplest method in the tied group**. A simpler method is cheaper to run, easier to debug, and easier
to explain, and it gives up nothing measurable.

## Step 4: build or buy

| Question | Favours building (in-house) | Favours a managed service |
|---|---|---|
| Do you have someone to own the model? | yes | no |
| Do you need custom business rules, explanations, or audits? | yes | partly (services offer filters and boosters) |
| Is your data allowed to leave your infrastructure? | no | yes |
| How soon must it be live? | weeks are fine | days |
| Expected traffic cost | high traffic: in-house is cheaper per request | low or uncertain traffic: pay as you go |

The Recombee benchmark compares a service with in-house methods on the same slice; see
[run Recombee](../how-to/run-recombee.md) and the [time to market](../dictionary/metrics/time-to-market.md) metric.

## Step 5: validate online

An offline winner is a hypothesis. Before switching a product over, run an A/B test with a business metric
(clicks, purchases, listening time), and keep the old method as the control. See
[offline vs online](../dictionary/concepts/offline-vs-online.md).

## Worked example

> A small online bookshop: 20,000 titles, 50,000 customers, mostly short visits, one data scientist, no GPU.

1. Closest datasets: RetailRocket (short visits) and MovieLens (book-like long-term tastes). Check both.
2. Shortlist: MostPopular as the fallback for new visitors, EASE (20,000 items fits on a laptop), ItemKNN for
   "customers who bought this also bought", and iALS as a scalable alternative.
3. On both datasets' quick-tier boards, see which of these are tied with the best, and compare coverage and
   training time on the overall comparison's accuracy-vs-cost chart.
4. Build in-house: the methods are simple, run on a CPU, and explain themselves.
5. A/B test EASE against the current "bestsellers" list.
