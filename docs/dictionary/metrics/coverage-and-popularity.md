# Coverage and popularity

Accuracy says whether lists contain the right items; these metrics say **which part of the catalog** a method
uses. A method can be accurate while showing only a handful of bestsellers to everyone. All four are computed
on the top-10 lists of the full-ranking evaluation.

## Coverage

**Question:** what share of the catalog appears in at least one user's top 10?

$$
\text{Coverage@10} = \frac{\big|\bigcup_u L_u^{(10)}\big|}{N}
$$

| Symbol | Meaning |
|---|---|
| $L_u^{(10)}$ | user $u$'s top-10 list |
| $N$ | number of items in the split |

Example: 3 users, 4-item lists drawn from 100 items: {1, 2, 3, 4}, {1, 2, 5, 6}, {1, 2, 3, 7}. The union is
{1, …, 7}, so coverage = 7/100 = 0.07. MostPopular gives everyone the same list, so its coverage is close to
10/N, tiny. Code: `src/recbench/metrics/catalog.py::_coverage`.

## Gini of exposure

**Question:** how unequally are recommendation slots spread across items?

$$
G = \frac{2\sum_{i=1}^{n} i\,x_{(i)}}{n\sum_i x_i} - \frac{n+1}{n}
$$

| Symbol | Meaning |
|---|---|
| $x_{(i)}$ | the $i$-th smallest number of times an item was recommended |
| $n$ | number of items |

Examples (also checked by the tests): exposure (5, 5, 5, 5) gives $G = 0$ (perfectly equal); (0, 0, 0, 12)
gives $G = 0.75$ (one item gets everything; the maximum for 4 items is 0.75). Lower means more evenly spread.
Code: `src/recbench/metrics/catalog.py::gini`.

## Popularity percentile

**Question:** how popular, on average, are the recommended items?

Each recommended item gets its popularity percentile among all items (by pre-test interactions): 1.0 for the
most popular, near 0 for items nobody touched. Example: ten items with counts 100, 50, 30, 20, 10, 5, 3, 2, 1, 0;
recommending the items with counts 100, 50, and 1 gives percentiles 1.0, 0.9, and 0.2, a mean of **0.70**.
Lower means less popularity-driven. Code: `src/recbench/metrics/catalog.py::_pop_percentile`.

## Long-tail share

**Question:** what share of recommendations comes from outside the **head** (the 20% most popular items)?

In the ten-item example the head is the top 2 items (counts 100 and 50). Recommending items with counts 100, 50,
and 1 gives a long-tail share of 1/3 = **0.33**. Higher means more long-tail exposure. Code:
`src/recbench/metrics/catalog.py::_long_tail`.

## Range, direction, and when they mislead

| Metric | Range | Better |
|---|---|---|
| coverage_at_10 | 0–1 | higher (usually) |
| gini_at_10 | 0–1 | lower |
| popularity_percentile_at_10 | 0–1 | lower (more personal) |
| long_tail_share_at_10 | 0–1 | higher |

- **Random scores "perfectly"** on all four, so never read them without accuracy.
- **Coverage depends on the number of users:** more users, more distinct items.
- **The catalog includes cold items**, which ID models never recommend, so their coverage has a ceiling.

## Check your understanding

??? question "Why is MostPopular's coverage tiny but not exactly 10/N?"
    The evaluator removes items each user has already seen, so users get slightly different lists.

??? question "A method has high coverage but low NDCG. What might be going on?"
    It spreads recommendations widely but not accurately, for example close to random. Check its NDCG against Random's.

## Further reading

- [Popularity bias](../concepts/popularity-bias.md) and [novelty, diversity, serendipity](novelty-diversity-serendipity.md).
