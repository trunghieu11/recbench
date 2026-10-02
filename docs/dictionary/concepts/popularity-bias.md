# Popularity bias

## Why it matters

Most interactions go to a small set of popular items: the **head**. Models trained on that data learn to
recommend the head even more, so niche items (the **long tail**) are rarely shown, rarely clicked, and
therefore stay unpopular: a feedback loop. Accuracy metrics often *reward* this, because popular items are
what users clicked in the test data too.

## Intuition

Five items with 50, 30, 10, 5, and 5 interactions: the top item alone has 50% of all interactions. A model
that always recommends it is "right" often, but it never helps users discover anything, and the other four
items' creators never get an audience.

## A small example: measuring concentration

Suppose a recommender shows items 40, 30, 20, 10, and 0 times across all users' top-10 lists. The
**Gini coefficient** of this exposure is 0.40 (0 = perfectly equal exposure, 1 = one item gets everything).
Showing each item 20 times gives Gini 0. Showing only one item gives a Gini close to 1.

## How recbench exposes it

| Metric | What it tells you | Page |
|---|---|---|
| `coverage_at_10` | share of the catalog that appears in at least one list | [coverage and popularity](../metrics/coverage-and-popularity.md) |
| `gini_at_10` | inequality of exposure across items | same |
| `popularity_percentile_at_10` | how popular the recommended items are on average | same |
| `long_tail_share_at_10` | share of recommendations outside the top 20% most popular items | same |
| `novelty_at_10` | how surprising the items are on average ($-\log_2$ of popularity) | [novelty, diversity, serendipity](../metrics/novelty-diversity-serendipity.md) |

A useful habit: read NDCG@10 *next to* coverage and Gini. In recbench's results, MostPopular is often near
the top on NDCG while covering well under 1% of the catalog.

## In recbench

- [MostPopular](../algorithms/most-popular.md) is the pure-popularity reference.
- [XSimGCL](../algorithms/xsimgcl.md) is designed to reduce popularity concentration (more uniform embeddings).
- Popularity is always computed from **pre-test** interactions only (`TrainView.item_pop`).
- Metric code: `src/recbench/metrics/catalog.py::gini` and the beyond-accuracy metrics in the same file.

## Pitfalls

- **Celebrating accuracy alone.** A popularity-heavy model can top the leaderboard and still be a poor product.
- **Over-correcting.** Recommending obscure items nobody wants lowers satisfaction. Diversity must be traded
  off, not maximised.
- **Evaluation bias.** Test data comes from a system that already favoured popular items, so offline metrics
  are biased towards popularity too (see [offline vs online](offline-vs-online.md)).

## Check your understanding

??? question "Why can a popularity-biased model look good offline?"
    The test interactions were themselves shaped by popularity (people click what they are shown, and popular
    items are shown most), so predicting popular items matches the test data.

??? question "What Gini would 'everyone gets the same single item' produce?"
    Close to 1: all exposure concentrated on one item.

## Further reading

- Abdollahpouri, Burke and Mobasher (2019), [Managing Popularity Bias in Recommender Systems with
  Personalized Re-ranking](https://arxiv.org/abs/1901.07555).
- [Calibration and fairness](../metrics/calibration-and-fairness.md).
