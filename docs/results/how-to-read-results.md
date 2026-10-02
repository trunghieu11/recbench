# How to read results

The [reading results tutorial](../start/reading-results.md) walks through one report. This page goes further: a
checklist to apply to any leaderboard, how to compare two methods properly, and what each column can and
cannot tell you.

## The five-question checklist

Ask these in order before you draw any conclusion.

**1. Is the split big enough?**
Look at "Warm eval users" in the split box. With a few hundred users, confidence intervals are wide and most
methods tie. With thousands, real differences show. The [confidence intervals](../dictionary/metrics/confidence-intervals.md)
page explains why the width shrinks roughly with the square root of the number of users.

**2. Which methods are tied with the best?**
The ≈ mark means a method's 95% interval overlaps the best method's interval. Treat every ≈ method as a
candidate. Ranking positions inside the tied group are noise.

**3. Did each method get a fair budget?**
Compare `max_steps`, `dim`, and training time in MLflow. A neural model trained for 400 steps on a laptop is not
the same model as one trained for 30,000 on a GPU. See [fair baselines and tuning](../dictionary/concepts/fair-baselines-and-tuning.md).

**4. What does the method give up?**
Read the beyond-accuracy table: a popularity-heavy method can be tied on NDCG while covering a tiny part of the
catalog. Read the efficiency columns: a tie at ten times the cost is not a tie for a product decision.

**5. Does the conclusion hold on the other datasets?**
A method that wins only on one dataset may be exploiting that dataset's quirks (repeats, popularity, session
length). Compare *rankings* across datasets, never raw values.

## Comparing two methods properly

The ≈ mark is a quick, conservative screen: two separate intervals can overlap even when one method beats the
other for most users. A **paired** comparison is sharper because every method is evaluated on the same seeded
users, in the same order (when the split, `seed`, and `max_eval_users` are the same).

Download `per_user_metrics.npz` for both runs from MLflow, then:

```python
import numpy as np

a = np.load("ease/per_user_metrics.npz")["ndcg_at_10"]
b = np.load("sasrec/per_user_metrics.npz")["ndcg_at_10"]
ok = ~np.isnan(a) & ~np.isnan(b)
diff = a[ok] - b[ok]                       # one difference per user

rng = np.random.default_rng(0)
means = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
low, high = np.quantile(means, [0.025, 0.975])
print(f"mean difference {diff.mean():.4f}, 95% CI [{low:.4f}, {high:.4f}]")
```

If the interval of the **difference** excludes 0, the difference is unlikely to be noise on this split. It still
says nothing about other datasets, other seeds, or other training budgets.

!!! note "Keys with a slash"
    Secondary-policy metrics are stored with `__` instead of `/`, for example `allow_repeats__ndcg_at_10`.

## Column by column

### Accuracy

| Column | Read it as | Watch out for |
|---|---|---|
| NDCG@10 [CI] | how high the relevant items appear in the top 10, averaged over users | values are only comparable within one dataset and tier |
| Recall@10 | the share of a user's test items found in the top 10 (divided by min(10, number of test items)) | users with many test items |
| HitRate@10 | the share of users with at least one hit | saturates for easy datasets |
| Next NDCG@10 | can the method predict the very next item | sequence models should shine here; if they do not, check their training budget |

### Beyond accuracy

| Column | Good direction | Typical trade-off |
|---|---|---|
| Coverage | higher | popularity methods cover almost nothing |
| Gini ↓, Popularity pct ↓ | lower | accurate methods often concentrate on popular items |
| Long-tail share, Novelty | higher (in moderation) | random scores high here and is useless |
| ILD (diversity) | higher (in moderation) | very diverse lists can be irrelevant |
| Calibration KL ↓ | lower | how well the list's category mix matches the user's history |
| Group gap ↓ | lower | the largest NDCG@10 difference between light, medium, and heavy users (activity terciles) |
| Cold-item recall | higher | ID-only models score 0 by design; only content-aware models can be non-zero |

Random is a useful anchor: it shows what a method with no information scores on each beyond-accuracy metric.

### Efficiency

| Column | Measured as | Caveats |
|---|---|---|
| Train s | wall-clock seconds of `fit` | includes data preparation inside `fit`; depends on the machine and preset |
| Score s / 1k users | seconds to score the full catalog for 1,000 users | pointwise models (DIN, DCN-V2) score every (user, item) pair and are much slower |
| Peak RSS MB | the child process's peak memory | includes Python and libraries (a few hundred MB even for Random) |
| Personal expl. | share of explanations that cite the user's own history | "–" when the method gives none; popularity methods give non-personal ones |

## Protocol effects to keep in mind

- **Full ranking vs sampled.** The sampled table shows where the cheaper 1 + 100 protocol would reorder
  methods. Trust the full-ranking leaderboard. See [sampled vs full](../dictionary/metrics/sampled-vs-full.md).
- **Repeats.** On Last.fm many test events repeat past items. The primary leaderboard excludes items the user
  has already consumed; the "repeats allowed" table shows the other view. A method that ranks well only when
  repeats are allowed is mostly re-recommending history. See [evaluation protocols](../dictionary/concepts/evaluation-protocols.md).
- **Cold users** are not in the leaderboards. They get the popularity fallback; the split box shows how many
  there are. On datasets with many cold users, the fallback matters as much as the model. See
  [cold start](../dictionary/concepts/cold-start.md).
- **Item cap.** EASE uses only the most recently popular items up to a cap (20,000 on the laptop, 30,000 on the
  GPU machine). MLflow logs the share of pre-test interactions those items cover as `fit.item_cap_coverage`.

## "Did not run" is a result too

| Status | Usually means |
|---|---|
| unsupported | the method needs something the dataset lacks (images, categories) or is switched off (managed services) |
| failed | a bug or a resource problem; read `error.txt` in MLflow |
| timeout | the method exceeded `timeout_minutes`; consider a smaller preset or a GPU |

A method that cannot run on your data is not a candidate, however well it does elsewhere.
