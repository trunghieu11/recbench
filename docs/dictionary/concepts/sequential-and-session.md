# Sequential and session-based recommendation

## Why it matters

What you did *last* often predicts what you do *next* better than everything you ever did. Sequential models
read a history in order; static models (EASE, matrix factorisation) treat it as an unordered set.

## Intuition

Someone who just bought a camera is likely to want a memory card next, not another camera. A music
listener in the middle of a jazz session probably wants more jazz now, even if most of their history is pop.
**Order and recency** carry signal that a set of items loses.

**Session-based** recommendation is the extreme case: only the current visit's clicks are known (anonymous
users). recbench splits histories into sessions with a 30-minute gap rule, but evaluates users rather than
anonymous sessions.

## A small example: the copy task

Every user walks through items in order: 5 → 6 → 7 → 8. What comes next?

- A sequential model learns "the next item is the last one + 1" and predicts 9.
- A set-based model sees {5, 6, 7, 8} and recommends items that co-occur with *any* of them, perhaps 4 or 6,
  but it has no idea which comes *next*.

recbench uses exactly this synthetic task as a test (`tests/test_methods.py`). SASRec, HSTU, TIGER-lite and
GRU4Rec must predict the next item with at least 90% accuracy, BERT4Rec with at least 60%. ItemKNN
scores close to 0 on it.

## How histories are fed to models: padding and alignment

Histories have different lengths, but a batch must be a rectangle. Shorter histories are **padded** with
item 0. Where the padding goes matters:

```
right-aligned (recbench's HistoryBatch):   [0, 0, 0, 5, 6, 7, 8]   newest item in the LAST column
left-aligned (RecBole's layout):           [5, 6, 7, 8, 0, 0, 0]   newest item at position length-1
```

A model must read the user's state at the position of the **newest** item. With right-aligned data that is the
last column; with left-aligned data it is position `length - 1`. Mixing these up makes the model read padding.
That was a real bug in recbench v0.1 (see the [review log](../../review/2026-10-02-review.md)).

## How sequence models are trained (shifted targets)

From a window [5, 6, 7, 8] the model reads [5, 6, 7] and must predict [6, 7, 8]. Each position predicts the
item that follows it, like next-word prediction in a language model. Windows are cut only from **pre-test**
history, so no test item can leak into training.

## In recbench

- `src/recbench/data.py::HistoryBatch`: right-aligned `items`, `lengths`, event `times`, and
  `left_aligned()` for RecBole.
- `src/recbench/methods/seq_trainer.py::sequence_windows`: random training windows from pre-test histories.
- Sessions: `src/recbench/pipeline/materialize.py::materialize` (30-minute gap, unless the dataset provides
  session ids).
- Next-item metrics (`next_hitrate_at_10`, `next_ndcg_at_10`) are computed for **every** method, so static and
  sequential models can be compared on the same "what comes next" task.
- The order-aware methods (the "Order-aware" column of the [capability matrix](../algorithms/index.md)):
  [GRU4Rec](../algorithms/gru4rec.md) (a recurrent network, the authors' code), [SASRec](../algorithms/sasrec.md),
  [BERT4Rec](../algorithms/bert4rec.md), [S3-Rec](../algorithms/s3rec.md), [HSTU](../algorithms/hstu.md),
  [TIGER-lite](../algorithms/tiger-lite.md), and [V-SKNN](../algorithms/vsknn.md), which needs no training at all: it
  finds past sessions similar to the user's latest one, weighting recent items more.

## Pitfalls

- **Bad timestamps.** MovieLens ratings are often entered in bulk, so their order says little about viewing order.
- **Too short a window** (`seq_len`) loses context; too long costs memory quadratically in attention models.
- **Repeat consumption.** In music, the "next item" is often one already played; see repeat policies in
  [evaluation protocols](evaluation-protocols.md).

## Check your understanding

??? question "A right-aligned history of length 3 in a window of 6: which column holds the newest item?"
    The last column, index 5. Indices 0–2 are padding.

??? question "Why does recbench compute next-item metrics for static methods too?"
    To show directly whether modelling order pays off. If a static method matches the sequential ones on
    next-item metrics, the order carries little signal in that dataset.

## Further reading

- [SASRec](../algorithms/sasrec.md), [BERT4Rec](../algorithms/bert4rec.md), [HSTU](../algorithms/hstu.md).
- Hidasi et al. (2016), [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939) (GRU4Rec).
