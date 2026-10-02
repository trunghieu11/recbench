# Metrics overview

A recommender can be judged in many ways. recbench measures seven kinds of things, so a leaderboard never
reduces to a single accuracy number.

| Kind | Question | Pages |
|---|---|---|
| **accuracy** | does the list contain what the user actually interacted with? | [ranking accuracy](ranking-accuracy.md) |
| **beyond** | is the list varied, fresh, fair, and not just popular items? | [coverage and popularity](coverage-and-popularity.md), [novelty, diversity, serendipity](novelty-diversity-serendipity.md), [calibration and fairness](calibration-and-fairness.md) |
| **slice** | does it work for new items and new users? | [cold-start slices](cold-start-slices.md) |
| **sampled** | the cheaper 1 + 100 protocol, for comparison only | [full ranking vs sampled](sampled-vs-full.md) |
| **diagnostic** | click-style metrics on sampled candidates | [CTR and rating metrics](ctr-and-rating-metrics.md) |
| **efficiency** | how long does training take, how fast is scoring and serving, how much memory? | [efficiency](efficiency.md), [cost](cost.md) |
| **explainability** | can it say why it recommended something? | [qualitative rubric](qualitative-rubric.md), [explainability concept](../concepts/explainability.md) |

How sure can we be about a difference? See [confidence intervals](confidence-intervals.md). How long does it take
to go from raw data to a working API? See [time to market](time-to-market.md).

## Choosing metrics for a goal

| Business goal | Primary metric | Watch next to it |
|---|---|---|
| "Show people what they will buy next" | NDCG@10, HitRate@10 | coverage, popularity percentile |
| "Fill a page of 50 products" (retrieval) | Recall@50 | coverage |
| "Help users discover new things" | NDCG@10 on new items (the default `exclude_seen`) | novelty, serendipity, long-tail share |
| "Autoplay the next song" | next-item NDCG@10 | repeats-allowed variants |
| "Support new products from day one" | cold-item recall | NDCG@10 on all items |
| "Be fair to small sellers or artists" | long-tail share, Gini | NDCG@10 (do not sacrifice too much) |
| "Run cheaply" | train time, serving latency, memory | accuracy |

## Every metric recbench computes

Generated from the code (`src/recbench/metrics/catalog.py`):

--8<-- "generated/metrics.md"

A metric is computed for a method only when its task list overlaps the method's tasks, so CTR-style metrics
never appear for pure top-N methods.
