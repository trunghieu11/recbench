# 3. Reading the results

**Goal:** understand what the report, the MLflow UI, and the dashboard tell you, and avoid the common misreadings.
**Time:** 30 minutes.

## Three places to look

| Tool | Command | Best for |
|---|---|---|
| Report | open `reports/smoke-<time>/report.html` | the complete picture, per dataset |
| MLflow UI | `mlflow ui --backend-store-uri file://$PWD/runs/mlflow --port 5001`, then <http://127.0.0.1:5001> | one run's settings, all metrics, artifacts |
| Dashboard | `uvicorn recbench.serving.app:app --port 8080`, then <http://127.0.0.1:8080/dashboard> | quick leaderboards in the browser |

## The report, section by section

For each dataset:

1. **Split at a glance:** users, items, events, the test start date, how many warm evaluation users, cold users,
   and the share of repeats. Small numbers (a few hundred users) mean wide confidence intervals.
2. **Top-N leaderboard:** sorted by NDCG@10, with a 95% confidence interval for each method. **≈** marks methods
   whose interval overlaps the best one's: treat them as tied. The other columns show recall, hit rate,
   coverage, training time, scoring time, memory, and the share of personal explanations.
3. **Next-item prediction:** can the method rank the very next item in its top 10?
4. **Same users, repeats allowed:** only for Last.fm.
5. **Beyond accuracy:** coverage, Gini, popularity, novelty, diversity, serendipity, calibration, group gap,
   cold-item recall.
6. **Full ranking vs sampled:** each method's rank under both protocols. Arrows show who would move if you
   trusted sampled metrics.
7. **Experimental:** unranked methods such as TIGER-lite.
8. **Did not run:** unsupported, failed, or timed-out runs, each with a reason.

## A worked reading (illustrative numbers)

| # | Method | NDCG@10 [95% CI] | Coverage@10 |
|---|---|---|---|
| 1 | ease ≈ | 0.080 [0.059, 0.101] | 0.029 |
| 2 | most_popular ≈ | 0.079 [0.059, 0.100] | 0.003 |
| 3 | itemknn ≈ | 0.066 [0.046, 0.087] | 0.014 |
| 4 | sasrec | 0.037 [0.026, 0.049] | 0.010 |

How to read it:

- EASE, MostPopular, and ItemKNN are **tied**: their intervals overlap EASE's. Do not claim EASE "wins".
- MostPopular is tied on accuracy but covers ten times less of the catalog than EASE: every user gets nearly
  the same list. If discovery matters, EASE is the better choice.
- SASRec's interval ends (0.049) below EASE's start (0.059), so it is genuinely worse *here*. But this is a
  400-step untuned laptop run. Check its training budget before concluding anything (see
  [fair baselines](../dictionary/concepts/fair-baselines-and-tuning.md)).

## In MLflow

Click a run to see:

- **Parameters:** every setting (dimension, steps, seed, ...), plus `fit.*` values the method reported (for example
  EASE's `fit.item_cap_coverage`).
- **Metrics:** everything the report shows and more (`ndcg_at_20`, `recall_at_50`, ...).
- **Tags:** dataset, method, tier, `protocol_version` (should be 2), `config_hash`, status, reason.
- **Artifacts:** `per_user_metrics.npz` (one value per user, for your own analyses), `explanations.json` (sample
  explanations), `metric_errors.json` (if any metric failed), `error.txt` (for failed runs).

Try it: download `per_user_metrics.npz` for a run and recompute the mean NDCG:

```python
import numpy as np
values = np.load("per_user_metrics.npz")["ndcg_at_10"]
print(values.mean(), (values > 0).mean())   # mean NDCG, and the share of users with any hit
```

## Common misreadings

- **Comparing across datasets.** An NDCG of 0.006 on RetailRocket can be excellent; 0.08 on MovieLens can be
  ordinary. Compare within one dataset.
- **Ignoring ties.** With hundreds of users, small differences are noise.
- **Trusting sampled metrics.** They are much higher and can reorder methods.
- **Treating smoke results as final.** Smoke runs are small and untuned; use the full tier to choose a method.

**Next:** [your first new method](your-first-method.md).
