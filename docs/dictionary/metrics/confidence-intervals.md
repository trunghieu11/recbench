# Confidence intervals

## The question

Method A has NDCG@10 = 0.079 and method B has 0.066. Is A really better, or would another sample of users
have reversed them? A **confidence interval** (CI) gives the range of values the metric could plausibly take.

## Intuition: the bootstrap

We evaluated on one sample of users. To see how much the average would wobble with a different sample,
**resample**: draw the same number of users *with replacement* from the ones we have (some appear twice, some not
at all), recompute the average, and repeat 1,000 times. The middle 95% of those 1,000 averages is the 95%
bootstrap confidence interval.

## A worked example

Per-user NDCG@10 for 8 users: 0, 0, 0.5, 1.0, 0, 0.63, 0, 0.39. The mean is **0.315**. With 1,000 bootstrap
resamples (seed 0), the 2.5th and 97.5th percentiles of the resampled means are **0.098 and 0.565**. With only 8
users the interval is wide: any method whose mean falls inside it is hard to tell apart from this one.

## In recbench

- Computed for `ndcg_at_10`, `recall_at_10`, `hitrate_at_10`, and `next_ndcg_at_10` (`CI_METRICS` in
  `src/recbench/evaluation.py`), with 1,000 resamples: `src/recbench/evaluation.py::bootstrap_ci`.
- Stored as `<metric>_ci_low` and `<metric>_ci_high` in MLflow.
- Leaderboards mark with **≈** every method whose interval overlaps the best method's interval: "statistically
  tied with the best" (`src/recbench/results.py::leaderboard`).
- Per-user values are saved with each run (`per_user_metrics.npz`), so you can run your own tests later.

## What the intervals do *not* cover

- **Training randomness.** A different seed gives a different model. The bootstrap only resamples users.
  Small runs can move noticeably between seeds. Multi-seed runs are on the [roadmap](../../results/roadmap.md).
- **Different datasets:** an interval is valid for one split only.

## When it misleads

- **Overlap is a heuristic, not a formal test.** Two intervals can overlap while a paired test (same users,
  both methods) still finds a significant difference. Paired tests over the saved per-user values are the
  stronger tool.
- **Many comparisons:** with 15 methods, some "significant" differences appear by chance.

## Check your understanding

??? question "Why resample with replacement?"
    Sampling with replacement mimics drawing a new, equally large sample from the same population. Without
    replacement you would get the same users back every time.

??? question "Why are smoke-tier intervals so wide?"
    Smoke splits evaluate a few hundred to two thousand users, and per-user NDCG is often 0 or a single hit.
    Few, highly variable values give wide intervals.

## Further reading

- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall.
- Sakai (2014), [Statistical Reform in Information Retrieval?](https://dl.acm.org/doi/10.1145/2641383.2641385) (SIGIR Forum).
