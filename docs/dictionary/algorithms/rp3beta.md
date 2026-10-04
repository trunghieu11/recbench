# RP3beta

> Recommends the items you reach most often by "walking" from items you liked, to people who liked them, to
> what else those people liked, with a penalty that stops every walk from ending at the bestsellers.

!!! abstract "In plain words"
    Suppose you liked a book. RP3beta asks: who else read that book, and what else did those readers read? Items
    that many such short "walks" reach get high scores. Walks tend to end at bestsellers, because everyone reads those,
    so each score is divided by the item's popularity raised to a power β. β is a dial: 0 leaves popularity alone, and
    larger values turn it down.

--8<-- "generated/methods/rp3beta.md"

!!! tip "When to use it"
    - As a strong, cheap item-to-item baseline next to ItemKNN and EASE. Published re-evaluations found it
      among the best simple methods.
    - When popular items crowd out everything else: `beta` turns that down directly.
    - When you need explanations: every score is a sum over the user's own history items.

!!! warning "When not to"
    - For brand-new items or users: a walk needs interactions to walk along.
    - When the order of events matters (next song, next click): it ignores order.

## 1. Intuition

Picture the interaction data as a map. Items and users are towns, and every interaction is a road between a
user and an item. To recommend for you, start at an item you liked and take two random steps: first to a
person who liked that item, then to another item that person liked. Items you land on often are good
recommendations. This is **P3** ("probability after three steps", counting the step from you to your item).

Two settings shape the walk:

- **alpha** sharpens or flattens each step's probabilities. Above 1, strong links get stronger; below 1, the
  walk spreads out.
- **beta** divides by the destination's popularity. Bestsellers have roads from everywhere, so walks end
  there almost by default. Penalising them (beta > 0) gives **RP3beta**, "re-ranked P3".

## 2. A tiny worked example

Three users and four items:

| User | Items |
|---|---|
| A | 1, 2 |
| B | 1, 3 |
| C | 1, 2, 4 |

Item popularity: item 1 has 3 users, item 2 has 2, items 3 and 4 have 1 each.

A new user D has interacted only with item 2. Walk from item 2 (alpha = 1):

1. Item 2 has two users, A and C, so each is reached with probability 1/2.
2. From A (items 1, 2), each item has probability 1/2. From C (items 1, 2, 4), each has probability 1/3.

| Destination | Probability (P3alpha) | RP3beta, beta = 0.5 (divide by √pop) | RP3beta, beta = 1 (divide by pop) |
|---|---|---|---|
| item 1 | 1/2·1/2 + 1/2·1/3 = 0.417 | 0.417 / √3 = 0.241 | 0.417 / 3 = 0.139 |
| item 4 | 1/2·1/3 = 0.167 | 0.167 / 1 = 0.167 | 0.167 / 1 = 0.167 |
| item 3 | 0 (nobody has both 2 and 3) | 0 | 0 |

With beta = 0, D gets the bestseller (item 1) first. With beta = 1, the niche item 4 that co-occurs with item 2
wins. The tuner picks the beta that predicts the validation window best.

## 3. How it works

1. Build the user × item matrix $X$ (1 for an interaction, or a time-decayed weight).
2. Turn rows into probabilities: user → item ($P_{ui}$) and item → user ($P_{iu}$).
3. Raise every probability to the power alpha.
4. Multiply: $W = P_{iu}^{\alpha} P_{ui}^{\alpha}$ gives item → item walk probabilities.
5. Divide column $j$ by $\text{pop}_j^{\beta}$, drop self-loops, and keep each row's top-k neighbours.
6. Score: a user's history vector times $W$.

```mermaid
flowchart LR
    I[item you liked] -->|P(u given i)^alpha| U[people who liked it]
    U -->|P(j given u)^alpha| J[their other items]
    J -->|divide by popularity^beta| S[score]
```

## 4. The math, symbol by symbol

$$
W_{ij} = \frac{1}{\text{pop}_j^{\,\beta}} \sum_{u} P(u \mid i)^{\alpha}\, P(j \mid u)^{\alpha},
\qquad \text{score}(u, j) = \sum_{i \in H_u} x_{ui}\, W_{ij}
$$

| Symbol | Meaning | Shape / range |
|---|---|---|
| $P(u \mid i)$ | probability of stepping from item $i$ to user $u$: $1 / \text{pop}_i$ for each of $i$'s users | [0, 1] |
| $P(j \mid u)$ | probability of stepping from user $u$ to item $j$: $x_{uj}$ divided by $u$'s total | [0, 1] |
| $\alpha$ | sharpening exponent (`rp3_alpha`) | about 0.5–2 |
| $\text{pop}_j$ | number of users who interacted with item $j$ | ≥ 1 |
| $\beta$ | popularity penalty (`rp3_beta`); 0 = P3alpha | about 0–1 |
| $W$ | item × item matrix, top-k entries per row | items × items, sparse |
| $H_u$, $x_{ui}$ | user $u$'s history and its weights (1, or a time decay) | — |

## 5. Training and inference

- **Training:** two sparse matrix products, computed in blocks of items, keeping the top-k per row. On a CPU
  this costs about as much as ItemKNN: seconds on the quick tier.
- **Inference:** a sparse vector times a sparse matrix per user.
- **Hardware:** CPU only.

## 6. Hyperparameters

| Name in recbench config | What it does | Default | Searched over |
|---|---|---|---|
| `rp3_alpha` | sharpens (> 1) or flattens (< 1) step probabilities | 1.0 | 0.3–2 |
| `rp3_beta` | popularity penalty; 0 = P3alpha | 0.5 | 0–1 |
| `rp3_neighbors` | neighbours kept per item | 200 | 20–1000 |
| `decay_half_life_days` | recent interactions weigh more | none | none, 30, 90, 365 |
| `train_window_days` | train on the last N days only | none | none, 30, 90, 365 |

## 7. In recbench

- Code: `src/recbench/methods/neighbourhood.py::RP3beta`, with the walk in
  `src/recbench/methods/neighbourhood.py::rp3beta_similarity`.
- `tests/test_new_methods.py` checks it against a dense, line-by-line implementation of the formula above.
- Explanations cite the history items with the largest $W_{ij}$, like ItemKNN.

!!! info "Fidelity: faithful"
    It follows Paudel et al. (2016) and the reference implementation used by Ferrari Dacrema et al. (2019),
    with that implementation's default of not re-normalising rows after the top-k cut.

## 8. Results in this benchmark

--8<-- "generated/methods/rp3beta-results.md"

## 9. Strengths and weaknesses

- **Strengths:** cheap, no training loop, one interpretable knob for popularity, exact explanations.
- **Weaknesses:** no notion of order or time (except through decay); cannot score items without
  interactions; the item × item matrix grows with the catalog (top-k keeps it sparse).

## 10. Common pitfalls

- **Tuning only alpha.** Beta usually matters more, because it decides how much the bestsellers dominate.
- **Comparing with an untuned ItemKNN.** Both deserve the same tuning budget.
- **Keeping every neighbour.** Without a top-k per item (`rp3_neighbors`), the item × item matrix is dense: slow,
  memory-hungry, and full of tiny weights that add noise.

## 11. Check your understanding

??? question "What does beta = 0 give you, and what does a very large beta do?"
    Beta = 0 is plain P3alpha: walks naturally end at popular items. A very large beta divides popular items
    away almost completely, so lists fill with rare items, which is usually too much.

??? question "Why is RP3beta called a 'graph' method when it has no neural network?"
    It works directly on the user-item graph: its scores are random-walk probabilities on that graph.
    LightGCN also uses the graph, but learns embeddings by gradient descent.

??? question "A very large β fills the lists with rare items. Where does the overall comparison show the problem?"
    Accuracy (NDCG@10) drops, while the beyond-accuracy table shows coverage and long-tail share going up and the
    popularity percentile going down. Tuning on the validation fold picks the β that balances the two for each
    dataset.

## 12. Further reading

- Paudel, Christoffel, Newell, Bernstein (2016). *Updatable, Accurate, Diverse, and Scalable Recommendations
  for Interactive Applications.* ACM TiiS.
- Ferrari Dacrema, Cremonesi, Jannach (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent
  Neural Recommendation Approaches.* RecSys. [arXiv](https://arxiv.org/abs/1907.06902)
- Compare with [ItemKNN](itemknn.md) and [EASE](ease.md).
