# SimpleX

> Matrix factorisation done well: cosine similarity, many negatives with a margin (the "cosine contrastive
> loss"), and a user vector that also looks at the user's history.

!!! abstract "In plain words"
    SimpleX is matrix factorisation with three sensible habits. It compares directions, not lengths, so popular items
    cannot win just by being trained more. It shows each user many wrong items, but only learns from the ones that still
    look too similar to the right ones. And it describes a user partly by the items they used, so light users borrow
    strength from their history.

--8<-- "generated/methods/simplex.md"

!!! tip "When to use it"
    - As a strong, cheap learned-embedding baseline. Its authors showed that a simple model with the right loss
      matches or beats many graph and neural models.
    - When you want embeddings for nearest-neighbour retrieval later (cosine-normalised vectors).

!!! warning "When not to"
    - When order matters (next item): it averages the history and ignores the order.
    - When you have no GPU and a large catalog: many negatives per step cost compute.

## 1. Intuition

Three choices make SimpleX strong:

1. **Cosine similarity:** only the direction of vectors matters, so popular items cannot win just by
   having long vectors.
2. **A cosine contrastive loss (CCL):** push the user's real items towards similarity 1. Push many random
   items down, but only those still above a **margin** (say 0.8). Negatives that are already far away cost
   nothing, so learning focuses on the confusing ones.
3. **A history-aware user:** the user vector mixes the user's own embedding with the average of the items in
   their history. Users with little training data borrow strength from their items.

## 2. A tiny worked example

**The loss.** Cosine scores for one user: the positive item 0.7; three negatives 0.9, 0.5, and 0.85. With
margin 0.8 and negative weight 2:

- positive term: 1 − 0.7 = **0.3**
- negative terms: max(0, 0.9 − 0.8) = 0.1; max(0, 0.5 − 0.8) = 0; max(0, 0.85 − 0.8) = 0.05. Their mean is
  0.05, times the weight 2 gives **0.1**
- loss = 0.3 + 0.1 = **0.4**. The easy negative (0.5) contributes nothing.

**The user vector.** With γ = 0.5, an own embedding [1, 0], and a history average of [0, 1] (after the
linear layer), the user vector is 0.5·[1, 0] + 0.5·[0, 1] = [0.5, 0.5], normalised to [0.71, 0.71].

## 3. How it works

1. Each training step samples (user, positive item) pairs from the events, plus `simplex_negatives` random
   items per pair.
2. User vector: $\gamma e_u + (1-\gamma) W_v \cdot \text{mean}(e_{\text{history}})$, normalised, then dropout.
3. Cosine scores against the normalised item vectors; CCL loss; Adam.
4. Train in epochs; on a validation fold, stop early when the validation NDCG stops improving.

```mermaid
flowchart LR
    U[user id] --> M[gamma * e_u + (1 - gamma) * W mean(history)]
    H[history items] --> M
    M --> C[cosine with item vectors]
    C --> L[CCL: 1 - pos + w * mean relu(neg - margin)]
```

## 4. The math, symbol by symbol

$$
\mathcal{L} = \big(1 - \hat y_{u,i}\big) + \frac{w}{|\mathcal{N}|} \sum_{j \in \mathcal{N}} \max\big(0,\ \hat y_{u,j} - m\big),
\qquad \hat y_{u,i} = \cos(v_u, e_i)
$$

| Symbol | Meaning | Range |
|---|---|---|
| $v_u$ | user vector: $\gamma e_u + (1-\gamma) W_v \bar e_{H_u}$ | — |
| $e_i$ | item embedding | `dim` |
| $\mathcal{N}$ | sampled negative items (`simplex_negatives`) | 50–1000 |
| $m$ | margin (`simplex_margin`) | 0.2–0.9 |
| $w$ | negative weight (`simplex_neg_weight`) | 1–500 |
| $\gamma$ | own-embedding versus history mix (`simplex_gamma`) | 0–1 |

## 5. Training and inference

- **Training:** one embedding lookup per positive and per negative, $O(\text{batch} \cdot |\mathcal{N}| \cdot d)$ per step.
- **Inference:** a dot product against all normalised item vectors.
- **Hardware:** GPU recommended; it runs on a CPU for small data.

## 6. Hyperparameters

| Name in recbench config | What it does | Default | Searched over |
|---|---|---|---|
| `simplex_negatives` | negatives per positive | 100 | 50–1000 |
| `simplex_margin` | margin for negatives | 0.8 | 0.2–0.9 |
| `simplex_neg_weight` | weight of the negative term | 150 | 1–500 |
| `simplex_gamma` | own embedding versus history | 0.5 | 0–1 |
| `simplex_history` | history items averaged | 50 | fixed (not searched) |
| `dim`, `lr` | the usual | quick preset | `dim` 64, 128; `lr` 1e-4–1e-2 |
| `max_epochs`, `patience` | early stopping on the validation fold | 30, 3 | fixed: 100, 5 |

## 7. In recbench

- Code: `src/recbench/methods/mf_losses.py::SimpleX`.
- Trained with the shared epoch loop (`src/recbench/methods/_torch.py::train_epochs`), with early stopping on
  the validation fold and learning curves in MLflow.
- `tests/test_new_methods.py` checks that it clearly beats Random on toy data.

!!! info "Fidelity: faithful"
    It follows the authors' SimpleX (RecZoo, Apache-2.0): mean aggregation, cosine similarity, the CCL loss,
    and a 1e-4 normal initialisation. The optional user and item biases are left out.

## 8. Results in this benchmark

--8<-- "generated/methods/simplex-results.md"

## 9. Strengths and weaknesses

- **Strengths:** strong for its cost; easy to understand; normalised embeddings are ready for ANN search.
- **Weaknesses:** several interacting settings (margin, negatives, weight); ignores order; random
  negatives can accidentally be items the user would like.

## 10. Common pitfalls

- **A margin that is too low:** almost every negative is "too close", the loss is dominated by noise, and
  training becomes slow.
- **Too few negatives:** the main advantage of CCL disappears.
- **Too few epochs with many negatives.** Each step is costly, and early stopping may end training before the
  model has settled. Check `fit.epochs_run` against `max_epochs` and the learning curve in MLflow.

## 11. Check your understanding

??? question "Why does the margin make training more efficient?"
    Negatives already below the margin contribute zero loss and zero gradient, so each step focuses on the
    negatives the model still confuses with the positive.

??? question "Why use cosine instead of a plain dot product?"
    A dot product can grow just by making vectors longer, which favours frequently trained (popular) items.
    Cosine compares directions only.

??? question "What does the negative weight `simplex_neg_weight` balance?"
    How much the "push wrong items down" part of the loss counts against the "pull the right item up" part. With many
    negatives and a small weight, the positive dominates; with a large weight, the model mostly learns to say no.

## 12. Further reading

- Mao, Zhu, Wang, Dai, Dong, Xiao, He (2021). *SimpleX: A Simple and Strong Baseline for Collaborative
  Filtering.* CIKM. [arXiv](https://arxiv.org/abs/2109.12613)
- [Loss functions](../concepts/loss-functions.md), [negative sampling](../concepts/negative-sampling.md),
  [DirectAU](directau.md).
