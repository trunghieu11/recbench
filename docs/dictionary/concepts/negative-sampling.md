# Negative sampling

## Why it matters

With implicit feedback you observe what users did, never what they disliked. To train a model to tell
"good for this user" from "bad for this user", you need examples of both. **Negative sampling** invents
the second kind by picking items the user did not interact with.

## Intuition

To teach a child what a dog is, you show dogs *and* some non-dogs. In recommendation, the non-dogs are
random items the user never touched. Most of them are genuinely uninteresting to that user, so they are
good enough as negatives, even though a few would actually have been liked (they are "false negatives").

## A small example

User u interacted with items {3, 7}. The catalog has items 1–10.

1. Draw a random item: 7. The user already has it (an **accidental hit**), so draw again: 4.
2. Training pair: (u, positive 3, negative 4). [BPR](../algorithms/bpr-mf.md) then pushes score(3) above score(4).

## Where negatives appear

| Place | What is sampled | recbench code |
|---|---|---|
| Pairwise training (BPR, LightGCN, XSimGCL) | 1 negative per positive, re-drawn if seen | `src/recbench/methods/_torch.py::sample_negatives` |
| Pointwise training (DCN-V2, DIN) | 4 negatives per positive, labelled 0 | `src/recbench/methods/_torch.py::edge_batches`; RecBole's sampler for DIN |
| Softmax training (SASRec, HSTU, TIGER-lite, towers) | the full catalog, or 1,024 shared random negatives when it is too large | `src/recbench/methods/seq_trainer.py::next_item_loss` |
| Evaluation, secondary protocol | 100 random unseen *warm* items per user | `src/recbench/pipeline/materialize.py::materialize` (candidates.parquet) |

## Sampled softmax in one paragraph

Full softmax cross-entropy compares the true next item against **every** item, which needs a score for each
of possibly 200,000 items at every position. Sampled softmax compares it against a random subset (recbench:
1,024 items shared by the whole batch) and masks any sample that equals the true item. It is much cheaper and
approximates the full loss well when the sample is large. recbench switches automatically when
batch × positions × items exceeds a budget of 200 million scores.

## Training negatives vs evaluation negatives

They play different roles:

- In **training**, negatives are a tool to learn; many tricks exist (popularity-weighted, "hard" negatives).
- In **evaluation**, ranking the true item against only 100 random negatives (*sampled metrics*) is cheap but
  can reorder methods. recbench therefore uses full-catalog ranking as its main protocol and reports sampled
  metrics only as a secondary check. See [full ranking vs sampled](../metrics/sampled-vs-full.md).

## Pitfalls

- **Not re-drawing accidental hits:** the model is told an item the user likes is a negative.
- **Uniform negatives are easy:** most random items are obscure, so the model may only learn "popular beats unpopular".
- **Evaluating with sampled negatives and reporting it as full ranking.** recbench v0.1 did only sampled evaluation.

## Check your understanding

??? question "Why does recbench draw evaluation negatives only from 'warm' items?"
    Pure ID models cannot score cold items (no pre-test data), so cold negatives would sit at the bottom
    for them and make the task artificially easier for ID models than for content models.

??? question "When is sampled softmax a poor approximation?"
    When the sample is tiny compared with the catalog and contains mostly easy (unpopular) items: the model
    rarely sees the hard competitors it must beat.

## Further reading

- Krichene & Rendle (2020), [On Sampled Metrics for Item Recommendation](https://arxiv.org/abs/2005.06338).
- [Loss functions](loss-functions.md).
