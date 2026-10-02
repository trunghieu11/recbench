# Collaborative, content-based, hybrid

## Why it matters

Where a model gets its knowledge from decides what it can and cannot do. Collaborative models need
behaviour data; content models need item descriptions; hybrids combine both.

## Intuition

- **Collaborative filtering (CF):** "people who liked what you liked also liked this". It learns only from
  the interaction matrix. It knows nothing about what an item *is*, so it finds surprising connections, but
  cannot handle items nobody has touched.
- **Content-based:** "this is similar to what you liked, judging by its description". It learns from item
  attributes (text, categories, images). It handles new items, but tends to recommend more of the same.
- **Hybrid:** both signals in one model, or one model's output feeding another.

## A small example

You liked *Toy Story*. A brand-new animated film is released today.

| Approach | Can it recommend the new film? | Why |
|---|---|---|
| Collaborative | no | nobody has watched it yet, so it has no co-occurrence and no trained ID embedding |
| Content-based | yes | its description ("animated, family, comedy") matches what you liked |
| Hybrid | yes | content gets it started; behaviour takes over as data arrives |

## Which recbench methods are which

| Kind | Methods |
|---|---|
| Collaborative | [ItemKNN](../algorithms/itemknn.md), [EASE](../algorithms/ease.md), [BPR-MF](../algorithms/bpr-mf.md), [iALS](../algorithms/ials.md), [LightGCN](../algorithms/lightgcn.md), [XSimGCL](../algorithms/xsimgcl.md), [SASRec](../algorithms/sasrec.md), [BERT4Rec](../algorithms/bert4rec.md), [HSTU](../algorithms/hstu.md) |
| Content-based | [Text hash tower](../algorithms/text-hash-tower.md) (items from text only) |
| Hybrid | [Multimodal tower](../algorithms/multimodal-tower.md) (content plus an ID residual), [S3-Rec](../algorithms/s3rec.md) (IDs plus attributes), [DIN](../algorithms/din.md) and [DCN-V2](../algorithms/dcnv2.md) (IDs plus category features) |
| Non-personalised | [MostPopular](../algorithms/most-popular.md), [Random](../algorithms/random.md) |

## In recbench

- A method declares `scores_cold_items=True` when it can score items with no pre-test interactions (content
  models). For other methods, the evaluator removes such items from the rankings
  (`src/recbench/evaluation.py::Evaluator.rank`).
- `requires_side_features=True` makes the runner skip the method on datasets without text or categories.
- The slice metric `item_cold_recall_at_10` shows how well a method finds *new* items, which is where content
  models should shine ([cold-start slices](../metrics/cold-start-slices.md)).

## Pitfalls

- **Expecting content models to beat CF on mature catalogs.** When behaviour data is plentiful, CF usually wins.
- **Weak content.** Meaningless or leaky item text (see the Steam fix in the
  [review log](../../review/2026-10-02-review.md)) hurts content models.

## Check your understanding

??? question "Why can't EASE recommend a brand-new item?"
    Its weight matrix connects items through co-occurrence. A new item has none, so its column of weights is
    zero, and recbench caps EASE's catalog to items with pre-test history anyway.

??? question "What is the main risk of a purely content-based recommender?"
    A "filter bubble" of similar items, with low novelty and serendipity.

## Further reading

- [Cold start](cold-start.md) and [novelty, diversity, serendipity](../metrics/novelty-diversity-serendipity.md).
