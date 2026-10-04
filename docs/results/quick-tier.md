# Quick-tier bake-off

> Which low-budget method is best on each dataset, when every method gets the same tuning budget? 23 methods are
> tuned on about one million events per dataset, and each dataset's top 3 are confirmed on the full data.
> The quick tier is also the **permanent entry gate**: a new method joins the comparisons by passing it.

How to run it: [the quick-tier bake-off on a rented GPU box](../start/quick-tier-box.md).

## Status

| # | Dataset | Quick tier (23 jobs) | Confirmed on full data | Verdict |
|---|---|---|---|---|
| 1 | MovieLens-25M | not run yet | — | — |
| 2 | RetailRocket | not run yet | — | — |
| 3 | Steam | not run yet | — | — |
| 4 | H&M | not run yet | — | — |
| 5 | Last.fm | not run yet | — | — |

The overall ranking (each method's mean rank across the five datasets) is added when all five are done.

## Why a quick tier?

The first full-tier run used one set of default settings for every method ([leaderboards](leaderboards.md)).
EASE, and sometimes MostPopular, won everywhere, but that run cannot say whether the neural models are worse or
only untuned. Tuning every method properly on full data would cost days of GPU time. The quick tier makes a fair
comparison affordable: smaller data, the same budget for everyone, and a full-data check for the winners only.

## The rules

1. **Same data.** Each dataset's `quick` split keeps about one million events, sampled by user, with the same
   test window as the full data. Its `quick-val` fold has the same users with every test event deleted; the
   validation window plays the role of the test window.
2. **Same budget.** Every method tries 10 settings (Random tries 1), drawn by Optuna's seeded TPE sampler (the
   first 5 at random) from the search spaces in `configs/tuning/quick.yaml`. The spaces follow published
   tuning recipes; the file names the source for each method.
3. **Time-aware settings for everyone.** Every method may train on only the last 30, 90 or 365 days, or on all
   history (`train_window_days`; each user keeps their last 10 events). Popularity and neighbourhood methods also
   tune a recency decay.
4. **Validation, then one test.** Settings are scored on the same 3,000 validation users. The neural models stop
   early on the validation fold, within each method's epoch limit from its paper (200 for SASRec and MultVAE, for
   example) and a patience of 5 to 10 epochs. GRU4Rec, iALS and BPR-MF tune their number of epochs or iterations
   instead. The best setting then runs **once** on the test split, with the best epoch count. The test split
   never influences a choice.
5. **3 hours per job**, tuning included. Each setting may train for its fair share of the time left (the time left
   divided by the settings still to try plus the final run); training then stops and keeps its best epoch. A job
   stops searching when one more setting and the final run would not fit. If no setting finishes in time, the job
   is `over_budget`.
6. **The best pick** is the highest test NDCG@10 on the dataset. "≈" marks methods whose 95% confidence interval
   overlaps the best one's: they are tied, not beaten. Training time is shown but does not affect the ranking.
7. **Confirmation on full data.** A dataset's top 3 re-check their size-sensitive setting (for example EASE's λ at
   ×0.5, ×1 and ×2 after scaling by the number of users) on `full-val`, then run the final test on `full`, three
   times with different seeds when training is random.
8. **The gate.** A method added later, heavy ones included, runs a quick-tier job on every dataset first. It is
   confirmed on full data only if it reaches a dataset's top 3.

## The 23 methods

| Runs on | Methods |
|---|---|
| CPU (11) | [Random](../dictionary/algorithms/random.md), [MostPopular](../dictionary/algorithms/most-popular.md), [ItemKNN](../dictionary/algorithms/itemknn.md), [iALS](../dictionary/algorithms/ials.md), [BPR-MF](../dictionary/algorithms/bpr-mf.md), [RP3beta](../dictionary/algorithms/rp3beta.md), [PureSVD](../dictionary/algorithms/puresvd.md), [SLIM](../dictionary/algorithms/slim.md), [V-SKNN](../dictionary/algorithms/vsknn.md), [SANSA](../dictionary/algorithms/sansa.md), [LightGBM re-ranker](../dictionary/algorithms/lgbm-rerank.md) |
| GPU (12) | [EASE](../dictionary/algorithms/ease.md), [SASRec](../dictionary/algorithms/sasrec.md), [DCN-V2 re-ranker](../dictionary/algorithms/dcnv2-rerank.md), [GF-CF](../dictionary/algorithms/gfcf.md), [Turbo-CF](../dictionary/algorithms/turbocf.md), [SimpleX](../dictionary/algorithms/simplex.md), [DirectAU](../dictionary/algorithms/directau.md), [UltraGCN](../dictionary/algorithms/ultragcn.md), [MultVAE](../dictionary/algorithms/multvae.md), [RecVAE](../dictionary/algorithms/recvae.md), [GRU4Rec](../dictionary/algorithms/gru4rec.md), [Text-embedding kNN](../dictionary/algorithms/text-knn.md) |

GF-CF is listed with the GPU methods in the plan but runs on the CPU: it only needs sparse products and a
truncated SVD. EASE runs on a CPU worker when the machine has no GPU.

Held back for later, because they are heavy or never finished: LightGCN, XSimGCL, BERT4Rec, S3-Rec, HSTU, DIN,
full-catalog DCN-V2, the text and multimodal towers, and TIGER-lite. They come back through this gate.

## Results by dataset

The datasets appear in run order: the most studied first, then contrasting regimes, the flagship shop, and the
statistically weakest last.

### 1. MovieLens-25M

--8<-- "generated/leaderboards/quick-tuned/movielens-25m.md"

**Confirmed on full data**

--8<-- "generated/leaderboards/full-tuned/movielens-25m.md"

### 2. RetailRocket

--8<-- "generated/leaderboards/quick-tuned/retailrocket.md"

**Confirmed on full data**

--8<-- "generated/leaderboards/full-tuned/retailrocket.md"

### 3. Steam

--8<-- "generated/leaderboards/quick-tuned/steam.md"

**Confirmed on full data**

--8<-- "generated/leaderboards/full-tuned/steam.md"

### 4. H&M

--8<-- "generated/leaderboards/quick-tuned/hm.md"

**Confirmed on full data**

--8<-- "generated/leaderboards/full-tuned/hm.md"

### 5. Last.fm

--8<-- "generated/leaderboards/quick-tuned/lastfm.md"

**Confirmed on full data**

--8<-- "generated/leaderboards/full-tuned/lastfm.md"

## Read these results with care

- **Item caps.** EASE and Turbo-CF keep a dense item × item matrix, so they use only the 30,000 most popular
  items, because three GPU jobs share one GPU. This affects RetailRocket (about 160,000 items in the quick tier),
  H&M (about 70,000) and Last.fm (about 47,000). SANSA approximates EASE without a cap: compare the two.
- **Small validation folds.** The tuning fold has about 850 warm users on MovieLens, 2,500 on H&M and 220 on
  Last.fm, so settings that score within a few percent of each other there are effectively tied.
- **Re-rankers depend on other jobs.** The LightGBM and DCN-V2 re-rankers take their candidates from the tuned
  EASE and ItemKNN of the same dataset, so they run after those two. Their `candidate_recall` is the best recall
  any re-ranking could reach.
- **Quick is not full.** A method can rank differently on one million events than on 31 million. The
  confirmation step checks the top 3, and the analysis after all five datasets compares the quick and full
  rankings.
- **One dataset is one regime.** Read each verdict next to the dataset's split box (test window, repeats, cold
  users), as explained in [how to read results](how-to-read-results.md).
