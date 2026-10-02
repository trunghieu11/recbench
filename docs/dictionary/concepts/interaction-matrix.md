# The interaction matrix

## Why it matters

Almost every recommender starts from one object: a table with users as rows and items as columns,
marking who interacted with what. Its size and emptiness decide which algorithms are feasible, and how
data must be stored.

## Intuition

Picture a spreadsheet with 162,541 rows (MovieLens users) and 59,047 columns (movies): 9.6 billion cells. Only
25 million are filled. If you printed it, it would look blank. That emptiness is called **sparsity**.

## Real sparsity (full datasets, as cleaned by recbench)

| Dataset | Users | Items | Interactions | Filled cells |
|---|---|---|---|---|
| MovieLens 25M | 162,541 | 59,047 | 25,000,095 | 0.26% |
| RetailRocket | 1,407,580 | 235,061 | 2,756,101 | 0.0008% |
| H&M | 1,362,281 | 104,547 | 31,788,324 | 0.022% |
| Last.fm 1K | 992 | 176,892 | 19,150,868 | 10.9% |
| Steam | 2,567,538 | 15,474 | 7,793,069 | 0.020% |

Last.fm is the exception: few users, each with years of listening. Most datasets are 99.9%+ empty.

## A small example: storing only the filled cells (CSR)

Four users, six items, seven interactions (sparsity 1 − 7/24 = 70.8%):

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| user 0 | 1 | | | 1 | | |
| user 1 | | 1 | | | | |
| user 2 | 1 | 1 | | | | 1 |
| user 3 | | | | | 1 | |

**Compressed Sparse Row (CSR)** format keeps three arrays:

- `data` = [1, 1, 1, 1, 1, 1, 1]: the stored values.
- `indices` = [0, 3, 1, 0, 1, 5, 4]: the column (item) of each value.
- `indptr` = [0, 2, 3, 6, 7]: where each row starts. User $u$'s items are `indices[indptr[u]:indptr[u+1]]`.
  For example, user 2 → positions 3..6 → items 0, 1, 5.

Memory is proportional to the 7 stored values, not the 24 cells. For MovieLens that means 25 million
instead of 9.6 billion.

## In recbench

- **IDs:** users and items get integer indices starting at 1. Index 0 is reserved for *padding* (an empty
  slot in a sequence), so every matrix has an empty row 0 and column 0.
- **Pre-test histories** are stored in exactly the CSR layout above, as three NumPy files per split:
  `pretest_offsets.npy` (like `indptr`), `pretest_items.npy` (like `indices`, in time order), and
  `pretest_ts.npy` (timestamps). See `src/recbench/pipeline/materialize.py::materialize`.
- `TrainView.seen` is the binary CSR matrix and `TrainView.interaction_counts` the count version
  (`src/recbench/data.py::TrainView`).
- `TrainView.user_items(u)` returns a user's history in time order, like `indices[indptr[u]:indptr[u+1]]`.

## Pitfalls

- **Densifying by accident.** `matrix.todense()` on a large sparse matrix can exhaust memory. EASE needs a
  dense *item × item* matrix, which is why it caps the catalog.
- **Forgetting padding index 0.** Off-by-one mistakes between "row index" and "user index" are a classic bug.

## Check your understanding

??? question "Which items does user 0 have, according to the CSR arrays?"
    `indptr[0]=0`, `indptr[1]=2` → `indices[0:2]` = items 0 and 3.

??? question "Why does a sparse format matter more for RetailRocket than for Last.fm?"
    RetailRocket is 99.9992% empty, so a dense matrix would waste almost all its memory on zeros. Last.fm is
    "only" 89% empty and has few users.

## Further reading

- SciPy sparse matrices: <https://docs.scipy.org/doc/scipy/reference/sparse.html>.
- [Embeddings](embeddings.md): how models compress this matrix into vectors.
