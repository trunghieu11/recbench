# Quick-tier bake-off

> Which low-budget method is best on each dataset, when every method gets the same tuning budget? 23 methods are
> tuned on about one million events per dataset, and each dataset's top 3 are confirmed on the full data.
> The quick tier is also the **permanent entry gate**: a new method joins the comparisons by passing it.

How to run it: [the quick-tier bake-off on a rented GPU box](../start/quick-tier-box.md).

## Status

The first bake-off ran on 2026-10-04 on one rented box (8 CPU workers and 1 GPU), from the code at commit `8d541ad`.
It took 15.1 hours from start to finish and 84.8 hours of job time. All 115 jobs finished. Each dataset's top 3 were
then confirmed on the full data.

| # | Dataset | Quick tier (23 jobs) | Confirmed on full data | Verdict on full data (test NDCG@10) |
|---|---|---|---|---|
| 1 | MovieLens-25M | done | EASE, LightGBM re-ranker, PureSVD | a three-way tie: EASE 0.2249 ≈ re-ranker 0.2223 ≈ PureSVD 0.2151 |
| 2 | RetailRocket | done (SASRec: re-run pending) | iALS, LightGBM re-ranker, RP3beta | a three-way tie: RP3beta 0.0275 ≈ iALS 0.0272 ≈ re-ranker 0.0270 |
| 3 | Steam | done | LightGBM re-ranker, EASE, SANSA | a three-way tie: EASE 0.0582 ≈ SANSA 0.0568 ≈ re-ranker 0.0550 |
| 4 | H&M | done | LightGBM re-ranker, EASE, ItemKNN | **the re-ranker, clearly**: 0.0283, against EASE's 0.0213 |
| 5 | Last.fm | done | PureSVD, EASE, SLIM (chosen by test score, see below) | a three-way tie: EASE 0.1061 ≈ PureSVD 0.0994 ≈ SLIM 0.0988 |

Four things from this run are still open:
- SASRec's collapsed final run on RetailRocket;
- the re-ranker's full-data confirmations, whose EASE candidates were badly scaled on two datasets;
- Last.fm's confirmations;
- the serving bundles.

The [notes below](#open-items-from-the-first-run) explain them, and a
[follow-up session](../start/box-5-follow-up.md) on a rented box closes all four.

## What it found

1. **Simple methods win.** On every dataset, the best method, or one tied with it, is either a linear or
   neighbourhood model (EASE, RP3beta, PureSVD, iALS) or the LightGBM re-ranker, which is built on two of them.
   Mean ranks over the five datasets, out of 23:
    - EASE 2.2;
    - the re-ranker 3.6;
    - SANSA 5.8;
    - ItemKNN 6.2;
    - RP3beta 6.6.

    The best neural models are in the middle of the table:
    - the DCN-V2 re-ranker (11.6), which re-orders EASE's and ItemKNN's candidates;
    - MultVAE (12.2).

    No neural model placed higher than 8th on any dataset. SASRec on RetailRocket may change that once it is
    re-run.
2. **Only the re-ranker clearly beats EASE, and only on H&M.** On the full H&M data it scores 0.0283
   [0.0260, 0.0305], 33% above EASE's 0.0213 [0.0192, 0.0234]. H&M is a fashion shop with new collections every
   week, which is where the re-ranker's trend and item-age features help. On the other datasets it ties. It costs
   about 30 times EASE's training time and up to 62 GB of memory.
3. **Tuning matters a lot.** Compare the [untuned v0.2 run](leaderboards.md) with the tuned results on the same
   full data (the same splits). The v0.2 run did not have the time-aware settings yet. With tuning:
    - EASE improved by 12% (Last.fm) to 118% (H&M);
    - ItemKNN on H&M improved by 102%;
    - iALS on RetailRocket improved by 51%;
    - the best result per dataset rose by 10% to 190%.
4. **Recent data matters.** Of the 90 jobs that could train on only the last 30, 90 or 365 days, 67 chose to. Of
   the 45 that could give old interactions less weight, 40 chose to. MostPopular did best when it counted only the
   last 7 to 14 days on four datasets (112 days on MovieLens). Each job tried only 10 settings, so read these as
   tendencies, not proofs.
5. **The quick tier mostly predicted the full data.** Two verdicts changed:
    - on Steam, the re-ranker's clear quick-tier win became a tie on full data, with EASE ahead;
    - on H&M, a three-way quick-tier tie became a clear win for the re-ranker.

    Within the tied groups the order moved around, as ties allow.
6. **Simple is also cheap.** The [Pareto front](overall-comparison.md#2-accuracy-against-cost) is the set of
   methods that no other method beats on both accuracy and training time. Here it is V-SKNN, Turbo-CF, PureSVD,
   EASE and the re-ranker (and Random, trivially). Median training times on the quick tier:
    - EASE: 9 seconds (on the GPU);
    - the re-ranker: 4.5 minutes;
    - SANSA: 12 minutes.
7. **Five datasets cannot rank 23 methods precisely.** The critical difference is 15.5 ranks, so EASE's mean rank
   is significantly better only than those of GRU4Rec, text kNN and Random. The per-dataset confidence intervals
   are the stronger evidence.
8. **The cheaper sampled protocol would have misled.** On the full H&M data, the
   [sampled protocol](../dictionary/concepts/evaluation-protocols.md) ranks the re-ranker 3rd. That protocol ranks
   each test item against only 100 random items. Ranking the whole catalog, as recbench does, puts the re-ranker
   1st.

## Why a quick tier?

The first full-tier run used one set of default settings for every method ([leaderboards](leaderboards.md)).
EASE, and sometimes MostPopular, won everywhere, but that run cannot say whether the neural models are worse or
only untuned. Tuning every method properly on full data would cost days of GPU time. The quick tier makes a fair
comparison affordable: smaller data, the same budget for everyone, and a full-data check for the winners only.

## The rules

1. **Same data.** Each dataset's `quick` split keeps about one million events, sampled by user, with the same
   test window as the full data. Its `quick-val` fold has the same users with every test event deleted; the
   validation window plays the role of the test window.
2. **Same budget.** Every method tries 10 settings (Random tries 1), taken from the search spaces in
   `configs/tuning/quick.yaml`. The spaces follow published tuning recipes; the file names the source for each
   method. The settings are chosen by Optuna's **TPE sampler**: the first 5 at random, then each new setting near
   the ones that scored well so far. A fixed seed makes the sequence repeatable.
3. **Time-aware settings.**
   - Every method except Random, MostPopular, text kNN and the two re-rankers may train on only the last 30, 90 or
     365 days, or on all history (`train_window_days`; each user keeps their last 10 events). MostPopular has its
     own window instead.
   - MostPopular, ItemKNN, EASE, iALS, RP3beta, PureSVD, SLIM, SANSA, GF-CF and Turbo-CF also tune a recency
     **decay**: an interaction counts half as much after a "half-life" of 30, 90 or 365 days (MostPopular: 3 to
     30 days).
4. **Validation, then one test.** Settings are scored on the same 3,000 validation users. The neural models stop
   early on the validation fold ("early stopping": training ends once the validation score has not improved for a
   number of epochs, the **patience**). Each uses its paper's epoch limit: 200 for SASRec and MultVAE (patience
   10), 100 for SimpleX, DirectAU and UltraGCN (patience 5), 50 for RecVAE (patience 5), and 30 for the DCN-V2
   re-ranker (patience 3). GRU4Rec, iALS and BPR-MF tune their number of epochs or iterations instead. The best
   setting then runs **once** on the test split, with the best epoch count. The test split never influences a
   choice: even the methods to confirm (rule 7) are picked by validation score. The first run's code still
   picked them by test score; see the [open items](#open-items-from-the-first-run).
5. **3 hours per job**, tuning included. Each setting may train for its fair share of the time left (the time left
   divided by the settings still to try plus the final run); training then stops and keeps its best epoch. Only
   the methods trained in epochs by recbench can stop this way. The others run each setting to the end
   ([searches cut short](#read-these-results-with-care)). A job stops searching when one more setting and the
   final run would not fit. If no setting finishes in time, the job is `over_budget`.
6. **The best pick** is the highest test NDCG@10 on the dataset. "≈" marks methods whose 95% confidence interval
   overlaps the best one's: they are tied, not beaten. Training time is shown but does not affect the ranking.
7. **Confirmation on full data.** The dataset's top 3 by **validation** score are re-checked on the full data:
   - EASE, iALS and SANSA re-check the setting that depends on data size (EASE's and SANSA's λ, iALS's
     regularisation) at ×0.5, ×1 and ×2 on `full-val`. The value is first scaled by the number of users, because
     regularisation must grow with the amount of data it balances. The other methods re-check their quick-tier best.
   - The final test runs on `full`, three times with different seeds when training is random.
   - The first final run also writes the **serving bundle** that [step 7](../start/deploy-cloud-run.md) can deploy.
     The first bake-off ran before this existed; `scripts/export_bundles_box.sh` writes its bundles afterwards.
   - A re-ranker's confirmation waits for the confirmations of its candidate generators (EASE, ItemKNN) on the
     same dataset, and keeps their settings fixed for the whole job.
8. **The gate.** A method added later, heavy ones included, runs a quick-tier job on every dataset first. It is
   confirmed on full data only if it reaches a dataset's top 3.

## The 23 methods

| Runs on | Methods |
|---|---|
| CPU (12) | [Random](../dictionary/algorithms/random.md), [MostPopular](../dictionary/algorithms/most-popular.md), [ItemKNN](../dictionary/algorithms/itemknn.md), [RP3beta](../dictionary/algorithms/rp3beta.md), [PureSVD](../dictionary/algorithms/puresvd.md), [GF-CF](../dictionary/algorithms/gfcf.md), [iALS](../dictionary/algorithms/ials.md), [BPR-MF](../dictionary/algorithms/bpr-mf.md), [SLIM](../dictionary/algorithms/slim.md), [V-SKNN](../dictionary/algorithms/vsknn.md), [SANSA](../dictionary/algorithms/sansa.md), [LightGBM re-ranker](../dictionary/algorithms/lgbm-rerank.md) |
| GPU (11) | [EASE](../dictionary/algorithms/ease.md), [Turbo-CF](../dictionary/algorithms/turbocf.md), [SimpleX](../dictionary/algorithms/simplex.md), [DirectAU](../dictionary/algorithms/directau.md), [UltraGCN](../dictionary/algorithms/ultragcn.md), [MultVAE](../dictionary/algorithms/multvae.md), [RecVAE](../dictionary/algorithms/recvae.md), [GRU4Rec](../dictionary/algorithms/gru4rec.md), [SASRec](../dictionary/algorithms/sasrec.md), [Text-embedding kNN](../dictionary/algorithms/text-knn.md), [DCN-V2 re-ranker](../dictionary/algorithms/dcnv2-rerank.md) |

This is the order of `queue.methods` in `configs/benchmarks/quick.yaml`. GF-CF needs only sparse products and a
truncated SVD, so it runs on the CPU. EASE runs on a CPU worker when the machine has no GPU.

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
  items, because three or four GPU jobs share one GPU. This affects RetailRocket (about 160,000 items in the quick tier),
  H&M (about 70,000) and Last.fm (about 47,000). SANSA approximates EASE without a cap: compare the two.
- **Small validation folds.** The tuning fold has about 850 warm users on MovieLens, 2,500 on H&M and 220 on
  Last.fm, so settings that score within a few percent of each other there are effectively tied.
- **Re-rankers depend on other jobs.** The LightGBM and DCN-V2 re-rankers take their candidates from the tuned
  EASE and ItemKNN of the same dataset, so they run after those two. Their `candidate_recall` is the best recall
  any re-ranking could reach.
- **Quick is not full.** A method can rank differently on one million events than on 31 million. The
  confirmation step checks the top 3, and [finding 5](#what-it-found) compares the quick and full verdicts.
- **One dataset is one regime.** Read each verdict next to the dataset's split box (test window, repeats, cold
  users), as explained in [how to read results](how-to-read-results.md).
- **Searches cut short.** A job stops searching when one more setting and its final run would not fit in 3
  hours. The status table on the [overall comparison](overall-comparison.md#5-what-ran-and-what-did-not) marks the
  six jobs where that happened:
    - **SANSA** finished only 2 of its 10 settings on H&M and RetailRocket, and 7 on MovieLens. A setting
      cannot stop early, so one slow setting can use up the job's time. On H&M and RetailRocket the third setting
      (all history, 249 weights per item) ran for 140 and 85 minutes before the cap stopped it.
        - The tuner's first two settings are the same on every dataset: a 30-day window and a small λ (about 5).
          Wherever more settings finished, those two were SANSA's weakest; on MovieLens they scored 0.098 on
          validation, against 0.236 for the best.
        - On H&M and RetailRocket, SANSA was therefore judged with one of its weakest settings. It is understated
          there.
    - **GRU4Rec**'s first setting took about an hour on RetailRocket and Steam, so it tried only that one.
    - **SLIM** tried 8 of its 10 settings on RetailRocket.
- **Settings that blew up while tuning.** A few settings made training collapse: the loss jumped and the
  model's scores fell, some to 0. This happened with MultVAE, UltraGCN and the DCN-V2 re-ranker on RetailRocket,
  Steam and H&M, and with SASRec on RetailRocket. Early stopping kept the best epoch of each, and the tuner moved
  on to other settings. Those settings simply scored badly; the results stay valid.
- **Seeds.** Methods whose training is random were confirmed with three seeds. The re-ranker varied the most.
  On full Steam its three runs scored 0.0581, 0.0561 and 0.0509. The quick-tier results use one seed, so treat
  small gaps between random methods as noise.
- **Memory.** The LightGBM re-ranker needed 27 to 43 GB of RAM on the quick tier, and up to 62 GB on full
  RetailRocket. The DCN-V2 re-ranker needed up to 41 GB. Rent a box with at least 128 GB of RAM to run them in
  parallel with other jobs.

### Open items from the first run

The run used the code at commit `8d541ad`. Four of its results need another look, and a
[follow-up session](../start/box-5-follow-up.md) settles all four:

- **SASRec's final run on RetailRocket collapsed.** It scored 0.0200 on validation, but only 0.0014 on test. The
  final run trains for a fixed number of epochs, here 25, with nothing watching the validation score. At epoch 21
  the learning rate was too high: the loss jumped from 5.8 to 9.9, and the model never recovered.
    - **The fix.** The training loop now keeps the best weights when that happens, that is when an epoch's loss is
      not a number or is more than 50% worse than the best epoch's.
    - **Re-run.** SASRec's implementation version is now 4, so all five of its jobs run again.
    - **Until then**, read SASRec's 22nd place on RetailRocket as this bug, not as a result. For comparison, the
      untuned v0.2 run scored 0.0221 with SASRec on the full RetailRocket data.
    - **Only this run.** A check of every other final run's loss curve found no other collapse.
- **Last.fm's confirmations were chosen by test score.** The code at `8d541ad` picked each dataset's top 3 by
  their test score. Today's code picks them by validation score, so the test data never influences a choice.
    - **On four datasets** this changes nothing: the top 3 are the same either way.
    - **On Last.fm**, validation would have chosen SANSA, the LightGBM re-ranker and EASE. The run confirmed
      PureSVD, EASE and SLIM instead.
    - **What stands.** EASE's confirmation stands. PureSVD's and SLIM's are kept as extra information, but they
      were chosen by the very score they are judged on, which flatters them.
    - **To do.** The follow-up session confirms SANSA and the re-ranker.
- **No serving bundles.** The code at `8d541ad` did not write bundles from confirmations.
  `scripts/export_bundles_box.sh` writes them for all 15 confirmed methods, with the settings each confirmation
  chose.
- **The re-ranker's EASE candidates.** A re-ranker reads EASE's and ItemKNN's settings when it trains: their
  confirmed settings if those confirmations have finished, else their quick-tier ones. Two problems followed.
    - **Timing.** The confirmations ran at the same time, so the outcome depended on which finished first. On
      MovieLens and RetailRocket the re-ranker's test runs used the quick-tier settings of both generators. On
      Steam they used EASE's confirmed settings, and on H&M both generators' confirmed settings.
    - **Scale.** A quick-tier λ is tuned for the quick tier's users. On full data a confirmation multiplies it by
      the ratio of users, then checks ×0.5, ×1 and ×2. Where the re-ranker fell back to EASE's quick-tier λ, it
      used it unscaled. On full MovieLens that was λ = 340, where EASE's own confirmation chose 7,775. On full
      RetailRocket it was 262, where scaling gives about 770. Its EASE was under-regularised, and the re-ranker
      still tied for first on both.
    - **The fix.** Today's queue makes a re-ranker's confirmation wait for its generators' confirmations. It
      scales quick-tier settings to the full data, and pins them for the whole job.
    - **To do.** The follow-up session confirms the re-ranker again on all five datasets.
