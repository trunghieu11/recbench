# Recommender systems 101

## Why it matters

Recommenders decide much of what people see online: videos on a home page, products under "you may also
like", songs in an autoplay queue, articles in a news feed. Most catalogs are far too large to browse.
A recommender picks a short list that each person is likely to want.

## The core problem

Given what we know about a user (mainly what they did before) and about items, produce a **ranked list
of items** that the user is likely to interact with next.

Three things make this hard:

1. **Sparsity.** Each user touches a tiny fraction of the catalog. In MovieLens 25M, users rated
   0.26% of the movies; in RetailRocket, 0.0008% of the products. See [the interaction matrix](interaction-matrix.md).
2. **Implicit signals.** Most data are clicks, plays, or purchases, not ratings. A missing interaction may mean
   "not interested" or simply "never saw it". See [feedback types](feedback-types.md).
3. **Change.** Tastes, catalogs, and trends move over time, so models must be judged on the *future*, not on
   a random sample of the past. See [data leakage and time splits](data-leakage-and-splits.md).

## A tiny example

| | Toy Story | Up | Heat | Alien |
|---|---|---|---|---|
| Ana | ✓ | ✓ | | |
| Ben | ✓ | ✓ | ✓ | |
| Chi | | | ✓ | ✓ |

What should Ana see next? Ben shares Ana's tastes (both liked Toy Story and Up) and also liked Heat, so Heat
is a reasonable guess. That reasoning, "people similar to you liked X", is **collaborative filtering**,
the idea behind most methods on this site.

## The main families

| Family | Idea | recbench examples |
|---|---|---|
| Non-personalised | the same list for everyone | [MostPopular](../algorithms/most-popular.md) |
| Neighbourhood | items or users that co-occur | [ItemKNN](../algorithms/itemknn.md) |
| Linear / matrix factorisation | users and items as vectors or weights | [EASE](../algorithms/ease.md), [BPR-MF](../algorithms/bpr-mf.md), [iALS](../algorithms/ials.md) |
| Graph neural networks | propagate over the user-item graph | [LightGCN](../algorithms/lightgcn.md), [XSimGCL](../algorithms/xsimgcl.md) |
| Sequential | the order of actions | [SASRec](../algorithms/sasrec.md), [BERT4Rec](../algorithms/bert4rec.md), [HSTU](../algorithms/hstu.md) |
| Feature-based rankers | click prediction from many features | [DIN](../algorithms/din.md), [DCN-V2](../algorithms/dcnv2.md) |
| Content-based | items understood from text or images | [Text hash tower](../algorithms/text-hash-tower.md) |
| Managed services | rent a recommender | [Recombee](../algorithms/recombee.md) |

## Tasks you will meet

- **Top-N recommendation:** a ranked list per user ("recommended for you").
- **Next-item prediction:** the very next item in a sequence ("up next").
- **Similar items:** items like a given item ("more like this").
- **Click-through-rate (CTR) prediction:** the probability that a user clicks a specific item.

recbench evaluates the first two directly; [evaluation protocols](evaluation-protocols.md) explains how.

## In recbench

- 17 methods on a ladder from simple to complex ([the method ladder](../algorithms/index.md)).
- 5 public datasets across movies, e-commerce, music, and games ([datasets](../datasets/index.md)).
- One honest protocol: models learn only from the past and are judged on the future.

## Pitfalls

- **Judging a recommender on the past it was trained on.** Always evaluate on later events.
- **Optimising one number.** Accuracy, diversity, freshness, cost, and explainability all matter; see the
  [metrics overview](../metrics/index.md).

## Check your understanding

??? question "Why is 'Ana did not watch Alien' weak evidence that she dislikes it?"
    She may never have seen it offered. Implicit data rarely contains real negatives.

??? question "What makes recommendation different from classification?"
    The output is a *ranked list* chosen from a huge catalog, and almost all items are unlabeled. Only the
    few the user interacted with are known positives.

## Further reading

- Aggarwal (2016), *Recommender Systems: The Textbook*, Springer.
- [Glossary](glossary.md) for every term on this site.
