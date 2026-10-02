# Explicit vs implicit feedback

## Why it matters

What your data *means* decides which models and metrics make sense. Treating clicks as if they were ratings,
or the absence of a click as a "dislike", leads to wrong conclusions.

## Intuition

- **Explicit feedback** is a user telling you what they think: 4.5 stars, a thumbs-down, a written review score.
  It is clear but rare: most people never rate.
- **Implicit feedback** is what users *do*: views, plays, add-to-carts, purchases, watch time. It is abundant
  but ambiguous. A play may be accidental, and no play may just mean "never saw it".

With implicit data you mostly see **positives**. The "negatives" must be inferred. That is why
[negative sampling](negative-sampling.md) exists, and why [iALS](../algorithms/ials.md) uses *confidence*
rather than certainty.

## Examples from recbench's datasets

| Dataset | Raw signal | Type | How recbench uses it |
|---|---|---|---|
| MovieLens 25M | 0.5–5 star ratings | explicit | as implicit "watched" events (any rating = an interaction) |
| RetailRocket | view, add-to-cart, transaction | implicit | every event is an interaction; the type is kept as a value (1, 2, 3) |
| H&M | purchases | implicit | every purchase is an interaction |
| Last.fm 1K | plays of an artist | implicit | every play is an interaction; repeats matter |
| Steam | written reviews | implicit | a review counts as an interaction with the game |

## A small example: confidence

A user played artist A 20 times, artist B once, and never played C. Under the iALS view:

| Artist | Plays | Preference $p$ | Confidence $c = 1 + 10 \cdot \text{plays}$ |
|---|---|---|---|
| A | 20 | 1 | 201 |
| B | 1 | 1 | 11 |
| C | 0 | 0 | 1 |

All three are evidence, with very different weights: the model is pushed hard to score A high, gently to
score B high, and only weakly to score C low.

## In recbench

- The clean data contract stores `feedback_type` ("explicit" or "implicit") and a numeric `value`
  (rating, event weight, price, or playtime): `src/recbench/schema.py`.
- All ranking methods treat every pre-test event as a positive interaction. Ratings are not predicted:
  rating prediction (RMSE) is a legacy task, explained in [CTR and rating metrics](../metrics/ctr-and-rating-metrics.md).
- `TrainView.interaction_counts` keeps repeat counts (used by iALS); `TrainView.seen` is the 0/1 version.

## Pitfalls

- **Using low ratings as positives.** A 1-star rating is a strong *negative*. recbench keeps it simple and
  treats any rating as "watched", which matches how the top-N task is usually defined, but you lose that signal.
- **Treating "not clicked" as "disliked"** in training *and* evaluation. Exposure was never observed.
- **Mixing event types blindly.** One purchase may mean more than ten views.

## Check your understanding

??? question "Why does MovieLens, an explicit dataset, work as implicit data?"
    Rating a movie implies the user watched it. Predicting *which* movies a user will watch next is the
    top-N task, regardless of the score given.

??? question "Name one source of false positives in implicit data."
    Accidental clicks, autoplay, gifts bought for someone else, or bots.

## Further reading

- Hu, Koren and Volinsky (2008), [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf).
- [Negative sampling](negative-sampling.md) and [the interaction matrix](interaction-matrix.md).
