# Primer: comparing methods fairly

!!! abstract "In plain words"
    A number like "NDCG@10 = 0.2005" is an average over the users we happened to test. With other users it would be
    a bit different. Before saying "B beats A" you need to know how much that average can wobble, and you need to
    make sure you did not pick B *because* of its lucky wobble. This page explains the four habits the lab is built
    on: look at per-user results, compare in pairs, choose on validation and test once, and give everyone the same
    budget.

## 1. A score is an average over users

The evaluator ranks the whole catalog for each test user and scores their top 10. NDCG@10 is 1 when all their test
items are at the top, 0 when none is in the top 10. The method's score is the **mean over users**. recbench keeps the
per-user values (`per_user_metrics.npz` for every run) because everything else is computed from them.

Per-user values are very uneven: on MovieLens most users score 0 or close to it, a few score high. That unevenness is
why a mean over 1,862 users still wobbles by about ±0.006.

## 2. How much does the mean wobble? The bootstrap

We cannot test new users, but we can **resample** the ones we have. Draw 1,862 users *with replacement* from the
1,862 (some twice, some not at all), take the mean, and repeat 2,000 times. The middle 95% of those means is the 95%
**bootstrap interval**. Eight users by hand: [confidence intervals](../dictionary/metrics/confidence-intervals.md).

```python
values = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.63, 0.0, 0.39])          # NDCG@10 of 8 users
rng = np.random.default_rng(0)
means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
np.quantile(means, [0.025, 0.975])                                     # the 95% interval
```

## 3. Compare in pairs

Two methods are scored on the **same** users. Their per-user differences cancel the users' own spread (easy users stay
easy under any method), so the interval of the *differences* is much narrower than the intervals of the two means.
[Compare results](../handbook/compare-results.md#why-paired-matters) shows a six-user example.

The interval's width tells you what a dataset can detect. From the real comparison of MostPopular and RP3beta on
MovieLens (1,862 users, interval ±0.0058), with users that vary like MovieLens' do:

| Test users | Smallest difference the interval can show (about) |
|---|---|
| 723 (Last.fm) | ±0.009 |
| 1,862 (MovieLens) | ±0.006 |
| 10,000 (RetailRocket, Steam, H&M) | ±0.003 |

A real gain of 0.004 will usually show up as "better" on the three large datasets and as "no clear difference" on
Last.fm. That is expected, not a failure.

## 4. Choose on validation, test once

Suppose 20 settings are all truly equally good (NDCG@10 = 0.150) and each measurement wobbles by ±0.005. In one
simulation (seed 0):

| Choose the "winner" by | Its test score |
|---|---|
| its **test** score, the best of 20 | 0.157 (on average 0.159): looks like a gain of 0.009, which is pure luck |
| its **validation** score, then test it once | 0.146: an honest estimate, unbiased on average |

Picking the maximum of many noisy numbers always finds a lucky one. This is the **winner's curse**. The lab therefore
never uses a test split to choose anything: tuning chooses on the validation fold, the chosen setting is tested
once, and the scoreboard chooses among your experiments by their validation scores.

## 5. Same budget, same seeds

- **Budget.** An experiment that may try 50 settings will often beat a baseline that tried 10, even with no better
  idea. Every lab experiment gets the baseline's 10 unless you change `trials` on purpose, and `compare` prints both.
- **Seeds.** iALS, BPR-MF and the LightGBM re-ranker start from random numbers, so two runs differ a little. The lab
  tests them with 3 seeds and averages, so a lucky seed cannot decide a comparison.

## 6. Many experiments, a few false alarms

A 95% interval excludes zero by chance about once in 20 comparisons where nothing changed. Over a season of
experiments, some "better" verdicts will be such false alarms. Two defences are built in: promotion needs no "worse"
on any of five datasets, and the new default needs "better" on all five. A false alarm rarely repeats across
datasets.

## Exercises

**1.** Method A scores 0.150 [0.140, 0.160] and B scores 0.155 [0.145, 0.165] on the same users. The intervals overlap.
Can B still be significantly better?

??? success "Solution"
    Yes. The two intervals describe each mean on its own, including the users' shared spread. The paired interval of
    B − A removes that spread and can be entirely above 0, for example [+0.002, +0.008], if B is better for most
    users. Only the paired test answers the question.

**2.** You ran 12 experiments for EASE. On MovieLens, experiment 7 has the highest *test* score but only the fourth
highest *validation* score. Which one do you report as your best, and why?

??? success "Solution"
    The one with the highest validation score. Picking by test score is the winner's curse: experiment 7 may simply
    have had a lucky test sample. Reporting it would overstate the gain.

**3.** On Last.fm the verdict for your change is "no clear difference", and on the other four it is "better". What
does the promotion rule say, and is the Last.fm result a problem?

??? success "Solution"
    Better on four, worse on none: promote it as a search-space option. Last.fm's 723 test users can only detect
    differences of about 0.009, so "no clear difference" there is expected for a smaller gain.
