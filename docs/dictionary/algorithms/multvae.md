# MultVAE

> Reads a user's whole history, squeezes it into a small random code, and decodes that code into a probability
> for every item. Items with high probability that the user has not seen yet are the recommendations.

--8<-- "generated/methods/multvae.md"

!!! tip "When to use it"
    - As the classic neural autoencoder baseline for top-N recommendation; strong on dense data such as
      MovieLens.
    - When new users must be scored without retraining: it needs only their history vector.

!!! warning "When not to"
    - When order matters (it treats the history as a set).
    - On very large catalogs without a GPU: the first and last layers have one weight per item.

## 1. Intuition

An **autoencoder** learns to rebuild its input after passing it through a narrow bottleneck. Asked to rebuild
"this user watched these 30 films" from a 200-number summary, it must learn the broad tastes behind the
films. When it rebuilds the vector, it also gives high scores to films that *fit* those tastes but are
missing from the history. Those are the recommendations.

Two extra ideas make it work well:

- **Variational:** the code is a small random cloud, not a single point. This regularises the model: similar
  histories must decode similarly.
- **Multinomial likelihood:** the decoder outputs one softmax over all items, so items compete for probability
  mass, much like ranking.

## 2. A tiny worked example

A user has items 1 and 2: $x = [1, 1, 0, 0]$. Suppose the decoder outputs logits $[2.0, 1.5, 1.0, -1.0]$.
The softmax turns them into probabilities:

| Item | exp(logit) | Probability |
|---|---|---|
| 1 | 7.389 | 0.494 |
| 2 | 4.482 | 0.300 |
| 3 | 2.718 | 0.182 |
| 4 | 0.368 | 0.025 |

The multinomial log-likelihood counts only the items the user has: log 0.494 + log 0.300 = −0.705 − 1.205 =
**−1.910**. Training pushes it up. Items 1 and 2 are already seen, so the recommendation is **item 3**
(0.182), well ahead of item 4.

## 3. How it works

1. Input: the user's interaction vector, L2-normalised, with dropout (half the items are hidden during
   training).
2. Encoder: items → 600 (tanh) → mean and log-variance of a 200-number code.
3. Sample a code (training), or take its mean (scoring).
4. Decoder: 200 → 600 (tanh) → one logit per item.
5. Loss: −multinomial log-likelihood + β × KL(code ‖ N(0, I)), with β growing from 0 to `vae_beta_cap`.

```mermaid
flowchart LR
    X[history vector, dropout] --> E[encoder 600]
    E --> Z[code: mean and variance, 200]
    Z --> D[decoder 600]
    D --> P[softmax over all items]
```

## 4. The math, symbol by symbol

$$
\mathcal{L}(x_u) = -\sum_i x_{ui} \log \pi_i(z_u) \;+\; \beta\, \mathrm{KL}\big(q(z_u \mid x_u)\,\|\,\mathcal N(0, I)\big)
$$

| Symbol | Meaning |
|---|---|
| $x_u$ | user $u$'s interaction vector (binary, or time-decayed) |
| $q(z_u \mid x_u)$ | the encoder's Gaussian over the code $z_u$ (200 numbers) |
| $\pi(z_u)$ | the decoder's softmax over all items |
| $\beta$ | KL weight, annealed from 0 to `vae_beta_cap` over `vae_anneal_epochs` |

## 5. Training and inference

- **Training:** batches of users; each step costs about (items × 600) multiplications per user, twice.
- **Inference:** one encoder and decoder pass per user (fold-in, no per-user parameters).
- **Hardware:** GPU recommended for large catalogs.

## 6. Hyperparameters

| Name in recbench config | What it does | Default | Searched over |
|---|---|---|---|
| `vae_hidden`, `vae_latent` | layer sizes | 600, 200 | 200–1000, 64–256 |
| `vae_dropout` | input dropout | 0.5 | 0.2–0.7 |
| `vae_beta_cap` | maximum KL weight | 0.2 | 0.05–1.0 |
| `vae_anneal_epochs` | epochs for β to reach the cap | 10 | 5–20 |
| `lr` | Adam learning rate | 0.001 | 1e-4–1e-2 |

## 7. In recbench

- Code: `src/recbench/methods/vae.py::MultVAE`.
- Early stopping on the validation fold through the shared loop, with learning curves in MLflow.
- `tests/test_new_methods.py` checks that it clearly beats Random on toy data.

!!! info "Fidelity: faithful"
    Same architecture, multinomial likelihood, input dropout, and KL annealing as Liang et al. (2018). The
    annealing is scheduled over epochs rather than a fixed number of steps, so it suits datasets of any size.

## 8. Results in this benchmark

--8<-- "generated/methods/multvae-results.md"

## 9. Strengths and weaknesses

- **Strengths:** strong on dense data; fold-in scoring; a principled probabilistic model.
- **Weaknesses:** parameters grow with the catalog; ignores order; less effective for users with very few
  interactions.

## 10. Common pitfalls

- **No KL annealing:** with the full β from the start, the code can collapse (it carries no information).
- **Forgetting input dropout:** the model learns to copy the input and recommends what the user already has.

## 11. Check your understanding

??? question "Why does the softmax output suit ranking?"
    All items share one unit of probability. Raising one item's probability lowers the others', exactly like
    competing for the top of a list.

??? question "Why can MultVAE score users it never saw in training?"
    Its parameters describe items and the mapping from histories to codes, not individual users. A new user's
    history goes through the same encoder and decoder.

## 12. Further reading

- Liang, Krishnan, Hoffman, Jebara (2018). *Variational Autoencoders for Collaborative Filtering.* WWW.
  [arXiv](https://arxiv.org/abs/1802.05814)
- [RecVAE](recvae.md), [EASE](ease.md) (a linear autoencoder), [embeddings](../concepts/embeddings.md).
