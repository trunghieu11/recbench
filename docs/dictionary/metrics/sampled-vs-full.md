# Full ranking vs sampled metrics

## The question

Should the true item compete against the **whole catalog** (full ranking) or against a **small random
sample** of items (sampled metrics, often 1 positive + 100 negatives)?

## Intuition

Sampled evaluation is cheap: score 101 items instead of 200,000. It was standard in many papers. But a random
sample of 100 items contains almost none of the popular, strong competitors that the true item really
competes with. The task becomes much easier, and, worse, *differently* easier for different methods. Krichene
& Rendle (2020) proved that sampled metrics can **reverse** which method looks better.

## How recbench computes the sampled version

For each warm evaluation user: the next item (positive) plus 100 random *warm* items the user never
interacted with (`candidates.parquet`). The rank of the positive among these 101 items gives
`sampled_hitrate_at_10`, `sampled_ndcg_at_10`, and `sampled_auc`. Ties count half.

## A simulation: a reversed ranking

A catalog of 10,000 items and 5,000 users. Two hypothetical methods:

- **X** puts the true item at rank 1 for 20% of users, and around rank 5,000 for the others.
- **Y** puts the true item at rank 500 for every user.

| | Full HitRate@10 | Sampled HitRate@10 (1 + 100) |
|---|---|---|
| X | **0.21** | 0.21 |
| Y | 0.00 | **0.97** |

Full ranking says X is clearly better: Y never puts the true item in a real top 10. Sampled evaluation says Y
is far better: at rank 500 of 10,000, only about 5 of 100 random items beat the true item, so it lands in the
sampled top 10 almost every time. (Computed with the binomial model of how many sampled negatives fall above the
true item; see "Check your understanding".)

## What recbench shows you

Every leaderboard has a table "Full ranking vs. sampled (1 + 100 negatives)" with both ranks for each method.
Arrows show methods that would move if you trusted sampled metrics. On the smoke tier this happens on most
datasets. See the [leaderboards](../../results/leaderboards.md).

## Range, direction, and when it misleads

- Range 0–1, higher is better, like the full versions.
- **Always optimistic:** sampled numbers are much higher than full-ranking ones; never mix them in one table.
- **Unstable rankings:** use sampled metrics only to compare with older papers, never to choose a method.

## In recbench

- Candidates: `src/recbench/pipeline/materialize.py::materialize`.
- Ranks, AUC, log loss: `src/recbench/evaluation.py::_sampled_stats`.
- Metrics: `sampled_hitrate_at_10`, `sampled_ndcg_at_10` (kind `sampled`); `sampled_auc`, `sampled_logloss` (kind `diagnostic`).

## Check your understanding

??? question "Why does method Y look so good under sampling?"
    Its true item is beaten by 499 of 9,999 other items, about 5%. Of 100 random negatives, about 5 beat it on
    average, so its sampled rank is about 6, inside the top 10.

??? question "Why does recbench sample negatives only from warm items?"
    ID models cannot score cold items, so cold negatives would be automatically ranked last for them, giving
    them an unfair advantage over content models.

## Further reading

- Krichene & Rendle (2020), [On Sampled Metrics for Item Recommendation](https://arxiv.org/abs/2005.06338) (KDD 2020).
- [Evaluation protocols](../concepts/evaluation-protocols.md).
