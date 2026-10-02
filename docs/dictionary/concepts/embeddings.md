# Embeddings

## Why it matters

Most modern recommenders represent every user and every item as a short list of numbers, an **embedding**.
Scores, similar-item lists, and explanations all come from comparing these vectors.

## Intuition

Place every movie on a map where similar movies sit close together: animated films in one corner, crime
dramas in another. A movie's embedding is its coordinates on that map. Real embeddings have 32–256
dimensions instead of 2, and the model chooses the axes itself while training: they rarely correspond to
named genres.

A user also gets coordinates, positioned so that the items they like are "in the direction" they point.
The score of an item for a user is the **dot product** of the two vectors.

## A small example

Item vectors: $\mathbf{a} = (0.9, 0.1)$, $\mathbf{b} = (0.8, 0.3)$, $\mathbf{c} = (-0.2, 1.0)$.

| Pair | Dot product $\mathbf{x}\cdot\mathbf{y}$ | Cosine similarity |
|---|---|---|
| a, b | 0.9·0.8 + 0.1·0.3 = 0.75 | 0.969 |
| a, c | 0.9·(−0.2) + 0.1·1.0 = −0.08 | −0.087 |

a and b point in almost the same direction (cosine near 1): similar items. a and c are nearly
perpendicular: unrelated.

- **Dot product** $\mathbf{x}^\top\mathbf{y} = \sum_k x_k y_k$ grows with both direction and length.
  Popular items often end up with longer vectors.
- **Cosine similarity** $\frac{\mathbf{x}^\top\mathbf{y}}{\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert}$ compares
  direction only. It always lies in [−1, 1].

## How models produce embeddings

| Method | Where the vectors come from |
|---|---|
| [BPR-MF](../algorithms/bpr-mf.md), [iALS](../algorithms/ials.md) | learned directly, one vector per user and item |
| [LightGCN](../algorithms/lightgcn.md) | learned, then smoothed over the interaction graph |
| [SASRec](../algorithms/sasrec.md), [HSTU](../algorithms/hstu.md) | items learned; the user vector is computed from the sequence of history items |
| [Text hash tower](../algorithms/text-hash-tower.md) | computed from item text, so new items get one too |

## In recbench

- PyTorch embedding tables: `nn.Embedding(n_items + 1, dim, padding_idx=0)`. Row 0 is padding and stays at zero.
- Scores: `src/recbench/methods/_torch.py::EmbeddingRecommender.score_users` multiplies user vectors by the
  item matrix.
- Explanations for embedding models cite the history items closest (by cosine) to the recommendation:
  `src/recbench/methods/_explain.py::embedding_explanations`.
- The size is the `dim` setting of the preset (32 on the laptop, 64 or 128 on the GPU).

## Pitfalls

- **Reading meaning into single dimensions.** Axes are learned and rotate freely; only distances and
  directions are meaningful.
- **Comparing embeddings from two different trainings.** Each run has its own coordinate system.
- **IDs without data.** A new item's ID embedding is never trained; it is random, which is why ID models
  cannot score new items (see [cold start](cold-start.md)).

## Check your understanding

??? question "Two items have cosine similarity 0.97. What does that say?"
    Their vectors point in nearly the same direction, so the model treats them as very similar.

??? question "Why might a popular item have a long vector?"
    A longer vector raises its dot product with many users, which helps the model fit the many interactions
    the item has.

## Further reading

- Koren, Bell and Volinsky (2009), [Matrix Factorization Techniques for Recommender
  Systems](https://ieeexplore.ieee.org/document/5197422), IEEE Computer.
- [Loss functions](loss-functions.md): how embeddings are trained.
