# Loss functions

## Why it matters

A loss function is the number a model tries to make small during training. It encodes *what "good" means*:
predicting clicks one by one, ordering pairs correctly, or picking the right item out of the whole catalog.
Different losses suit different data and tasks.

## Overview

| Loss | Question it trains | Used by in recbench |
|---|---|---|
| [Binary cross-entropy](#binary-cross-entropy) | "will this user interact with this item, yes or no?" | DCN-V2, DIN |
| [BPR](#bpr) | "is this interacted item above that random one?" | BPR-MF, LightGCN, XSimGCL |
| [Softmax cross-entropy](#softmax-cross-entropy) | "which of all items comes next?" | SASRec, HSTU, BERT4Rec, S3-Rec (fine-tuning), TIGER-lite, towers |
| [Sampled softmax](#sampled-softmax) | the same, against a random subset of items | the sequence models, when the catalog is large |
| [InfoNCE](#infonce) | "can you find your own twin among all others?" | XSimGCL's contrastive term |
| Weighted least squares | "fit 1s and 0s, trusting observed ones more" | iALS ([its page](../algorithms/ials.md)) |

## Binary cross-entropy

$$
\ell = -\big[y\ln\sigma(z) + (1-y)\ln(1-\sigma(z))\big]
$$

| Symbol | Meaning |
|---|---|
| $z$ | the model's raw score (logit) for one (user, item) pair |
| $\sigma(z) = 1/(1+e^{-z})$ | predicted probability of an interaction |
| $y$ | 1 for an observed pair, 0 for a sampled negative |

Example: a positive with $z = 1.2$ has $\sigma = 0.7685$ and loss 0.263. A negative with $z = 0.3$ has loss 0.854.
A positive scored $z = -2.0$ ($\sigma = 0.119$) costs 2.127: confident mistakes are punished hardest.

## BPR

$$
\ell = -\ln\sigma(\hat{x}_{ui} - \hat{x}_{uj})
$$

| Symbol | Meaning |
|---|---|
| $\hat{x}_{ui}$ | score of an item $i$ the user interacted with |
| $\hat{x}_{uj}$ | score of a random item $j$ they did not |

Only the *difference* matters, so it trains ranking directly. Example: a difference of −0.15 (wrong order)
costs 0.771; a difference of +5 costs 0.007. Worked through on the [BPR-MF page](../algorithms/bpr-mf.md).

## Softmax cross-entropy

$$
\ell = -\ln\frac{\exp(z_{\text{true}})}{\sum_{j=1}^{N}\exp(z_j)}
$$

| Symbol | Meaning |
|---|---|
| $z_j$ | score of item $j$ (for example $\mathbf{h}^\top\mathbf{e}_j$) |
| $N$ | catalog size |

Example: four items with scores (2.0, 1.0, 0.1, −1.0) give probabilities (0.638, 0.235, 0.095, 0.032). If
the true next item is the first, the loss is $-\ln 0.638 = 0.449$; if it is the third, the loss is 2.349. One
loss call teaches the model about *all* items at once, which is why this beats BPR for sequence models.

## Sampled softmax

Same formula, but the sum runs over the true item plus a random subset $S$:

$$
\ell = -\ln\frac{\exp(z_{\text{true}})}{\exp(z_{\text{true}}) + \sum_{j\in S}\exp(z_j)}
$$

With the example above and $S$ = {item 2, item 4}: $-\ln\frac{e^{2}}{e^{2}+e^{1}+e^{-1}} = 0.349$. That is
lower than the full loss (0.449), because the subset left out competitor item 3. Sampled losses are
optimistic; larger samples get closer to the truth. recbench uses 1,024 shared negatives and masks accidental hits.

## InfoNCE

$$
\ell_i = -\ln\frac{\exp(\cos(\mathbf{z}_i', \mathbf{z}_i'')/\tau)}{\sum_{j}\exp(\cos(\mathbf{z}_i', \mathbf{z}_j'')/\tau)}
$$

| Symbol | Meaning |
|---|---|
| $\mathbf{z}_i', \mathbf{z}_i''$ | two views (for example, two noisy copies) of node $i$ |
| $\tau$ | temperature: small values make the comparison sharper |

It is softmax cross-entropy where the "classes" are the other nodes in the batch. Minimising it pulls each
node towards its twin and pushes it away from everyone else, spreading embeddings out. A worked example is on
the [XSimGCL page](../algorithms/xsimgcl.md).

## In recbench

- BCE: `src/recbench/methods/dcnv2.py::DCNV2` (and RecBole's DIN); the DCN-V2 re-ranker on candidate rows;
  UltraGCN, with weights that mimic graph convolution ([UltraGCN](../algorithms/ultragcn.md)).
- BPR: `implicit` for BPR-MF; SELFRec's `bpr_loss` for the graph models. GRU4Rec's official code adds **BPR-max**,
  a BPR variant that compares each positive with a softmax-weighted set of negatives ([GRU4Rec](../algorithms/gru4rec.md)).
- Softmax and sampled softmax: `src/recbench/methods/seq_trainer.py::next_item_loss`, applied at every non-padding
  position (SASRec can switch between full softmax, sampled softmax and the original BCE with `sasrec_loss`).
- **Cosine contrastive loss (CCL)**: [SimpleX](../algorithms/simplex.md) pushes negatives below a cosine margin.
- **Alignment and uniformity**: [DirectAU](../algorithms/directau.md) needs no negatives at all; it pulls positive
  pairs together and spreads all vectors over the sphere.
- **Multinomial likelihood with a KL term**: the VAEs ([MultVAE](../algorithms/multvae.md),
  [RecVAE](../algorithms/recvae.md)) reconstruct a user's whole history from a compressed code.
- **LambdaRank**: the [LightGBM re-ranker](../algorithms/lgbm-rerank.md) weights each pair of candidates by how much
  swapping them would change NDCG, so mistakes at the top cost most.

## Pitfalls

- **BCE with sampled negatives gives biased probabilities.** The predicted click rates reflect the sampling ratio.
- **Sampled softmax with too few negatives** under-trains against popular competitors.
- **Comparing loss values across different losses:** a lower BCE says nothing about a BPR model.

## Check your understanding

??? question "Why is softmax cross-entropy usually better than BPR for next-item prediction?"
    It compares the true item with every other item in one step, while BPR compares it with one random item
    at a time.

??? question "What happens to InfoNCE if τ is very small?"
    The softmax becomes almost a hard maximum: the loss focuses on the single most similar wrong node, and
    gradients get large and unstable.

## Further reading

- Rendle et al. (2009), [BPR](https://arxiv.org/abs/1205.2618).
- Klenitskiy & Vasilev (2023), [Turning Dross Into Gold Loss](https://arxiv.org/abs/2309.07602).
- van den Oord, Li and Vinyals (2018), [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (InfoNCE).
