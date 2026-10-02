# Ranking accuracy

These metrics answer one question in different ways: **did the recommended list contain the items the user
actually interacted with after the cutoff, and how high?** All of them are computed per user and then
averaged over warm evaluation users, with bootstrap confidence intervals for the headline ones.

Notation used on this page:

| Symbol | Meaning |
|---|---|
| $L_u$ | user $u$'s ranked list (top 50 in recbench) |
| $L_u^{(K)}$ | its first $K$ items |
| $R_u$ | user $u$'s relevant items: test-window items, not seen before the cutoff by default |
| $\text{rel}(r)$ | 1 if the item at rank $r$ is relevant, else 0 |

The worked examples below come from `docs/assets/metric_examples.yaml`. The test suite (`tests/test_metrics.py`)
recomputes every number on this page, so they cannot silently drift from the code.

| Example | Ranked list | Relevant | K |
|---|---|---|---|
| A | 11, 12, **13**, 14, 15 | {13} | 5 |
| B | **21**, 22, 23, **24**, 25 | {21, 24, 99} | 5 |
| C | 31, 32, 33 | {40} | 3 |
| D | **1, 2, 3, 4** | {1, …, 8} | 4 |

## HitRate

**Question:** did the user get *at least one* relevant item in the top K?

$$
\text{HitRate@}K(u) = \mathbb{1}\left[\,|L_u^{(K)} \cap R_u| > 0\,\right]
$$

Examples: A = 1, B = 1, C = 0, D = 1. Easy to explain to non-experts ("x% of users found something");
ignores how many hits there were and where. Code: `src/recbench/metrics/catalog.py::hitrate`.

## Precision

**Question:** what share of the K recommended slots were relevant?

$$
\text{Precision@}K(u) = \frac{|L_u^{(K)} \cap R_u|}{K}
$$

Examples: A = 1/5 = 0.2, B = 2/5 = 0.4, C = 0, D = 4/4 = 1.0. Its maximum is limited when users have fewer
relevant items than K. Code: `src/recbench/metrics/catalog.py::precision`.

## Recall

**Question:** what share of the user's relevant items did the list find?

$$
\text{Recall@}K(u) = \frac{|L_u^{(K)} \cap R_u|}{\min(K, |R_u|)}
$$

recbench divides by $\min(K, |R_u|)$ instead of $|R_u|$, so a perfect list always scores 1 even for users with
more relevant items than slots (example D: 4 found, 8 relevant, K = 4 → 4/4 = 1.0). Examples: A = 1/1 = 1.0,
B = 2/3 = 0.667 (item 99 was missed), C = 0. Code: `src/recbench/metrics/catalog.py::recall`.

## NDCG

**Question:** how close is the list's order to the best possible order? Hits near the top count more.

$$
\text{DCG@}K = \sum_{r=1}^{K}\frac{\text{rel}(r)}{\log_2(r+1)},
\qquad
\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}
$$

| Symbol | Meaning |
|---|---|
| $1/\log_2(r+1)$ | the discount: rank 1 → 1.0, rank 2 → 0.631, rank 3 → 0.5, rank 4 → 0.431, rank 5 → 0.387 |
| IDCG@K | the DCG of a perfect list: all $\min(K, \lvert R_u\rvert)$ relevant items at the top |

Worked example B: hits at ranks 1 and 4, so DCG = 1 + 0.4307 = 1.4307. The ideal list puts 3 relevant items at
ranks 1–3: IDCG = 1 + 0.6309 + 0.5 = 2.1309. NDCG@5 = 1.4307 / 2.1309 = **0.671**.
Example A: a single hit at rank 3, so DCG = 0.5 and IDCG = 1, giving NDCG = 0.5.

NDCG@10 is recbench's headline metric. Code: `src/recbench/metrics/catalog.py::ndcg`.

## MAP

**Question:** averaged over the relevant items found, how precise was the list at each of them?

$$
\text{AP@}K(u) = \frac{1}{\min(K, |R_u|)}\sum_{r=1}^{K}\text{Precision@}r(u)\cdot\text{rel}(r)
$$

Example B: precision at rank 1 = 1/1, at rank 4 = 2/4, so AP = (1 + 0.5) / min(5, 3) = **0.5**. Example A:
(1/3) / 1 = 0.333. MAP@10 is the mean of AP@10 over users. Code: `src/recbench/metrics/catalog.py::average_precision`.

## MRR

**Question:** how far down is the *first* relevant item?

$$
\text{RR}(u) = \frac{1}{\text{rank of the first relevant item}}\quad(0 \text{ if none in the top 50})
$$

Examples: A = 1/3 = 0.333, B = 1/1 = 1.0, C = 0. recbench reports `mrr_at_50`. Code:
`src/recbench/metrics/catalog.py::reciprocal_rank`.

## Next-item HitRate and NDCG

The sequential versions use **one** relevant item, the user's very first relevant test item:
`next_hitrate_at_10` (is it in the top 10?) and `next_ndcg_at_10` ($1/\log_2(\text{rank}+1)$ if it is). They are
computed for every method, so you can see directly whether sequence models help.

## Range, direction, and when they mislead

- All range from 0 to 1, and higher is better.
- **Absolute values depend on the dataset** (catalog size, sparsity). Compare methods within one dataset only.
- **Small samples:** with a few hundred users, differences of 0.01 are often noise. Read the confidence intervals.
- **Popularity:** high accuracy can come from recommending popular items; read
  [coverage and popularity](coverage-and-popularity.md) next to it.
- **Offline ≠ online:** see [offline vs online](../concepts/offline-vs-online.md).

## Check your understanding

??? question "A list has hits at ranks 2 and 3, and the user has 2 relevant items. What is NDCG@5?"
    DCG = 0.6309 + 0.5 = 1.1309; IDCG = 1 + 0.6309 = 1.6309; NDCG = 0.693.

??? question "Why does recbench divide recall by min(K, |R|)?"
    So the best achievable recall is always 1. Otherwise users with many relevant items could never score well
    with a short list.

??? question "Which metric would you show a product manager, and which would you optimise?"
    HitRate@10 is easy to explain; NDCG@10 is more sensitive to ordering, so it is a better optimisation target.

## Further reading

- Järvelin & Kekäläinen (2002), [Cumulated Gain-based Evaluation of IR Techniques](https://dl.acm.org/doi/10.1145/582415.582418) (the origin of NDCG).
- [Evaluation protocols](../concepts/evaluation-protocols.md) and [confidence intervals](confidence-intervals.md).
