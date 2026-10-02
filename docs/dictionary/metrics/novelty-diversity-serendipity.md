# Novelty, diversity, serendipity

Three ways to ask whether a list is *interesting*, not just accurate. All are computed on the top-10 lists.

## Novelty

**Question:** how surprising are the recommended items, judged by how rarely people interact with them?

$$
\text{novelty}(i) = -\log_2 p(i), \qquad p(i) = \frac{c_i + 1}{\sum_j c_j + N}
$$

| Symbol | Meaning |
|---|---|
| $c_i$ | pre-test interactions of item $i$ |
| $N$ | number of items (the +1 and +N are Laplace smoothing, so unseen items do not get infinite novelty) |

Example: 1,000 interactions over 100 items. An item with 50 interactions has novelty 4.43 bits; with 10, 6.64
bits; with 1, 9.10 bits; with 0, 10.10 bits. The metric averages over all recommended items. Higher means less
obvious. Code: `src/recbench/metrics/catalog.py::_novelty`.

## Intra-list diversity (ILD)

**Question:** how different are the items *within* one list, judged by their categories?

$$
\text{ILD}(L) = \frac{2}{|L|(|L|-1)}\sum_{i<j}\left(1 - \frac{|C_i \cap C_j|}{|C_i \cup C_j|}\right)
$$

| Symbol | Meaning |
|---|---|
| $C_i$ | the set of category tokens of item $i$ (for example {Action, Comedy}) |
| $1 - \frac{\lvert C_i \cap C_j\rvert}{\lvert C_i \cup C_j\rvert}$ | Jaccard distance: 0 = same categories, 1 = nothing in common |

Example: items with {Action, Comedy}, {Action}, and {Drama}. Pair distances: 1 − 1/2 = 0.5, 1 − 0/3 = 1, and
1 − 0/2 = 1, so ILD = **0.833**. Not computed for datasets without categories (Last.fm). Code:
`src/recbench/metrics/catalog.py::_ild`.

## Serendipity

**Question:** how many recommendations were both *relevant* and *unexpected*?

recbench counts, per user, the top-10 items that are:

1. relevant (in the user's test items), **and**
2. not among the 5% most popular items, **and**
3. (when the dataset has categories) of a category the user had never interacted with before,

and divides by 10. Example: a top-10 list has 2 relevant items: a top-5% bestseller, and a niche documentary
from a genre new to the user. Only the documentary counts, so serendipity = 1/10 = **0.1**. Code:
`src/recbench/metrics/catalog.py::_serendipity`.

## Range, direction, and when they mislead

- Higher is "more interesting" for all three, but none is good in isolation: Random maximises novelty and
  diversity while being useless. Always read them next to accuracy.
- ILD depends on category quality: broad categories (H&M's product types) behave differently from detailed ones.
- Serendipity is rare by design and needs many users to measure reliably.

## Check your understanding

??? question "Why add +1 and +N in the novelty formula?"
    An item with zero interactions would otherwise have probability 0 and infinite novelty.

??? question "A list contains ten action movies. What is its ILD (categories = {Action} for each)?"
    0: every pair has Jaccard distance 0.

## Further reading

- Castells, Hurley and Vargas (2015), "Novelty and Diversity in Recommender Systems", in *Recommender Systems
  Handbook*, Springer.
- [Coverage and popularity](coverage-and-popularity.md).
