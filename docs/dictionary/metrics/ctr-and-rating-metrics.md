# CTR and rating metrics

These metrics come from *classification* and *regression*. recbench computes some of them as
**diagnostics only**: its datasets have no real impressions (what users saw but did not click), so true CTR
evaluation is not possible.

## AUC and per-user AUC (GAUC)

**Question:** if you pick one positive and one negative at random, how often does the model score the positive higher?

$$
\text{AUC} = \frac{\#\{(p, n): s_p > s_n\} + \tfrac12\#\{(p, n): s_p = s_n\}}{\#\text{positives}\cdot\#\text{negatives}}
$$

| Symbol | Meaning |
|---|---|
| $s_p, s_n$ | model scores of a positive and a negative item |
| ties | count half |

**Per-user AUC (GAUC):** compute AUC separately for each user, then average. That avoids comparing scores
across users, whose score scales differ. recbench's `sampled_auc` is a per-user AUC of the next item against
its 100 sampled negatives.

Worked example: the positive scores 0.8; four negatives score 0.9, 0.5, 0.3, 0.8. One negative is above, one
ties, two are below: AUC = 1 − (1 + 0.5)/4 = **0.625**.

Range 0–1; 0.5 = random; higher is better. Code: `src/recbench/evaluation.py::_sampled_stats`.

## Log loss

**Question:** how well calibrated are predicted probabilities?

$$
\text{LogLoss} = -\frac{1}{n}\sum_{k}\big[y_k\ln p_k + (1-y_k)\ln(1-p_k)\big]
$$

A positive predicted at 0.7685 costs 0.263; a negative predicted at 0.5744 costs 0.854. Lower is better.

recbench reports `sampled_logloss` **only for methods that output probabilities** (DCN-V2, DIN). Even then it
reflects the 1:100 candidate sampling, not real click rates, so treat it as a sanity check.

## RMSE and MAE (rating prediction)

$$
\text{RMSE} = \sqrt{\frac1n\sum_k (\hat r_k - r_k)^2},\qquad \text{MAE} = \frac1n\sum_k|\hat r_k - r_k|
$$

They measure how close predicted star ratings are to real ones. This was the focus of the Netflix Prize era.
Today it is considered a **legacy task**: predicting the exact rating of items a user *already chose* says little
about which items to *show*. No recbench method is trained to predict ratings, so these metrics are not
computed; they would return as soon as a rating model is added.

## When these metrics mislead

- **AUC on sampled negatives** is inflated, like all sampled metrics (see [full ranking vs sampled](sampled-vs-full.md)).
- **Pooled AUC** across users mixes score scales: a model can look good by scoring active users higher overall.
- **Low RMSE** can coexist with poor top-N lists.

## Check your understanding

??? question "What AUC does a model get if the positive ties with all negatives?"
    0.5, since every pair counts half.

??? question "Why is pooled AUC less meaningful than per-user AUC for recommenders?"
    Recommenders rank items *within* a user. Pooling compares a score for one user with a score for another,
    which never happens when serving.

## Further reading

- Zhou et al. (2018), [Deep Interest Network](https://arxiv.org/abs/1706.06978) (defines GAUC as used in advertising).
- [DCN-V2](../algorithms/dcnv2.md) and [DIN](../algorithms/din.md).
