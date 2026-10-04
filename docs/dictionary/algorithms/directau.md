# DirectAU

> Matrix factorisation without negative sampling: pull each user towards their items ("alignment") and spread
> all users and all items evenly over the sphere ("uniformity").

!!! abstract "In plain words"
    DirectAU learns user and item vectors with two plain goals. A user's vector should point the same way as the
    vectors of the items they used (**alignment**), and all vectors together should spread evenly over a sphere
    instead of bunching up (**uniformity**). Spreading out is part of the goal, so it needs no random "negative" items
    at all: that is the whole trick.

--8<-- "generated/methods/directau.md"

!!! tip "When to use it"
    - As a modern MF baseline with only one important loss setting (γ).
    - To understand *why* contrastive losses work: DirectAU optimises the two properties they implicitly
      encourage.

!!! warning "When not to"
    - When order matters, or for cold items.
    - With tiny batches: uniformity is measured within a batch, so small batches give noisy estimates.

## 1. Intuition

Good embeddings need two properties:

- **Alignment:** a user and the items they interacted with should be close.
- **Uniformity:** embeddings should be spread out. If every vector collapsed to one point, alignment would be
  perfect, but nothing could be told apart.

Losses with negative samples (BPR, softmax) reach these two properties indirectly. DirectAU writes them down
and optimises them directly, with a weight γ that balances them. No negatives need to be sampled.

## 2. A tiny worked example

**Alignment.** A user at [1, 0] and an item they interacted with at [0.8, 0.6] (both unit vectors). The
squared distance is (1 − 0.8)² + (0 − 0.6)² = 0.04 + 0.36 = **0.4**. Training shrinks it.

**Uniformity** is log of the average of $e^{-2\lVert x - y \rVert^2}$ over pairs:

- Three identical vectors: every distance is 0, so every term is $e^0 = 1$ and the uniformity is
  log 1 = **0**, the worst value.
- Three orthogonal unit vectors: every squared distance is 2, so every term is $e^{-4}$ and the uniformity is
  **−4**. Lower is better.

The loss is alignment + γ × (uniformity of users + uniformity of items) / 2.

## 3. How it works

1. Each step samples (user, item) pairs from the training events. No negatives.
2. Normalise the user and item embeddings to unit length.
3. Loss = alignment(u, i) + γ · (uniformity(users) + uniformity(items)) / 2, over the batch.
4. Train in epochs with Adam and weight decay; stop early on the validation fold.
5. Score with the dot product of the (unnormalised) embeddings, as the authors do.

```mermaid
flowchart LR
    P[(user, item) pairs] --> N[normalise embeddings]
    N --> A[alignment: pull pairs together]
    N --> U[uniformity: spread users and items]
    A --> L[loss = A + gamma * U]
    U --> L
```

## 4. The math, symbol by symbol

$$
\mathcal{L} = \underbrace{\mathbb{E}_{(u,i)} \lVert \tilde e_u - \tilde e_i \rVert^2}_{\text{alignment}}
+ \frac{\gamma}{2}\Big(\underbrace{\log \mathbb{E}_{u,u'} e^{-2\lVert \tilde e_u - \tilde e_{u'} \rVert^2}}_{\text{uniformity of users}}
+ \log \mathbb{E}_{i,i'} e^{-2\lVert \tilde e_i - \tilde e_{i'} \rVert^2}\Big)
$$

| Symbol | Meaning |
|---|---|
| $\tilde e_u, \tilde e_i$ | unit-normalised user and item embeddings |
| $(u, i)$ | an observed interaction from the batch |
| $u, u'$ and $i, i'$ | pairs of users (items) within the batch |
| $\gamma$ | balance between the two terms (`directau_gamma`) |

## 5. Training and inference

- **Training:** per step, pairwise distances within the batch, $O(\text{batch}^2 \cdot d)$.
- **Inference:** a dot product against all item vectors.
- **Hardware:** GPU recommended; it works on a CPU for small data.

## 6. Hyperparameters

| Name in recbench config | What it does | Default | Searched over |
|---|---|---|---|
| `directau_gamma` | weight of uniformity | 1.0 | 0.01–10 (the authors' grid) |
| `directau_l2` | weight decay | 1e-6 | 0, 1e-8, 1e-6, 1e-4 (the authors' grid) |
| `batch_size` | pairs per step; uniformity is measured within the batch | 256 | 256–2048 |
| `dim`, `lr` | the usual | quick preset | `dim` 64–128, `lr` 1e-4–1e-2 |
| `max_epochs`, `patience` | early stopping on the validation fold | 30, 3 | fixed: 100, 5 |

## 7. In recbench

- Code: `src/recbench/methods/mf_losses.py::DirectAU`, with the two terms in
  `src/recbench/methods/mf_losses.py::alignment` and `src/recbench/methods/mf_losses.py::uniformity`.
- `tests/test_new_methods.py` checks both terms on simple vectors, and checks that the model beats Random on
  toy data.

!!! info "Fidelity: faithful"
    It follows the authors' code (MIT) with the MF encoder: normalised embeddings in the loss, raw dot product
    for scoring, and Xavier-normal initialisation. Their LightGCN encoder variant is not included.

## 8. Results in this benchmark

--8<-- "generated/methods/directau-results.md"

## 9. Strengths and weaknesses

- **Strengths:** no negative sampling to get wrong; one main setting; a clear explanation of what good
  embeddings are.
- **Weaknesses:** uniformity costs batch² per step; the batch size changes what "uniform" means; ignores
  order.

## 10. Common pitfalls

- **γ too large:** everything spreads out, users drift away from their items, and accuracy drops.
- **γ too small:** embeddings collapse towards each other.
- **A batch that is too small.** Uniformity is measured within each batch, so a tiny batch gives a noisy picture of
  how spread out the vectors are. The search tries batches of 256 to 2,048.

## 11. Check your understanding

??? question "Why does DirectAU not need negative samples?"
    The uniformity term already pushes all embeddings apart, which is the job negatives do in other losses.

??? question "What would happen with alignment alone?"
    Every embedding would collapse to the same point: perfect alignment, zero ability to rank.

??? question "Why does DirectAU need no negative sampling?"
    Its uniformity term already pushes every vector away from every other vector in the batch. In other losses, that
    is the job of the sampled negatives.

## 12. Further reading

- Wang, Yu, Ma, Zhang, Liu, Ma (2022). *Towards Representation Alignment and Uniformity in Collaborative
  Filtering.* KDD. [arXiv](https://arxiv.org/abs/2206.12811)
- Wang & Isola (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on
  the Hypersphere.* ICML.
- [SimpleX](simplex.md), [BPR-MF](bpr-mf.md), [loss functions](../concepts/loss-functions.md).
