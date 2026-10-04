# Primer: sparse matrices

!!! abstract "In plain words"
    Every lab method starts from one table: users as rows, items as columns, and a 1 where a user interacted with an
    item. Almost every cell is 0: a user has seen a few dozen of tens of thousands of items. A **sparse matrix**
    stores only the non-zero cells, which makes the table thousands of times smaller and the arithmetic much faster.
    `scipy.sparse` provides it. This page covers what you need for the labs, with numpy-style examples.

## Why not a numpy array?

| Quick tier | Users × items | Cells | As a dense float32 array | Non-zero cells |
|---|---|---|---|---|
| MovieLens | 3,398 × 32,049 | 109 million | 436 MB | about 1 million |
| RetailRocket | 477,235 × 158,520 | 76 billion | **303 GB** | about 1 million |

A dense array of RetailRocket's matrix would not fit in any laptop. Stored sparsely, it takes a few megabytes.

## A tiny example

Four users, five items, eight interactions:

```python
import numpy as np
import scipy.sparse as sp

users = [0, 0, 1, 1, 1, 2, 3, 3]
items = [0, 2, 0, 1, 2, 1, 2, 3]
X = sp.csr_matrix((np.ones(8), (users, items)), shape=(4, 5))
X.toarray()
```

```text
[[1. 0. 1. 0. 0.]      user 0 interacted with items 0 and 2
 [1. 1. 1. 0. 0.]      user 1 with items 0, 1 and 2
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 1. 0.]]     item 4: nobody (a column of zeros)
```

`toarray()` makes a dense copy. Use it only on small pieces, never on a whole quick-tier matrix.

## How CSR stores it

**CSR** (compressed sparse row) keeps three arrays:

```text
data    = [1 1 1 1 1 1 1 1]   the non-zero values, row by row
indices = [0 2 0 1 2 1 2 3]   their column (item) numbers
indptr  = [0 2 5 6 8]         where each row starts in data/indices
```

Row `u` is `data[indptr[u]:indptr[u+1]]` at columns `indices[indptr[u]:indptr[u+1]]`. User 1: positions 2 to 5,
items `[0, 1, 2]`. Two consequences:

- **Reading a row is cheap**: `X[1]`, or a set of rows: `X[[0, 3]]`. Columns are slow in CSR. Convert with
  `X.tocsc()` (compressed sparse *column*) when you need many columns.
- **The number of items per user** is `np.diff(X.indptr)` → `[2, 3, 1, 2]`, without touching the values.

## The operations the labs use

| You want | numpy (dense) | scipy.sparse |
|---|---|---|
| users per item (item popularity) | `X.sum(axis=0)` | `np.asarray(X.sum(axis=0)).ravel()` → `[2, 2, 3, 1, 0]` (sums come back as a 1 × n matrix; `ravel()` flattens it) |
| scale each row (user) by a number | `X * w[:, None]` | `sp.diags(w) @ X` |
| scale each column (item) | `X * w[None, :]` | `X @ sp.diags(w)`, or `X.multiply(w[None, :])` |
| element-wise product | `A * B` | `A.multiply(B)`; in scipy, `*` used to mean matrix product, so use `@` for that |
| matrix product | `A @ B` | `A @ B` (sparse @ sparse stays sparse; sparse @ dense gives dense) |

## X transposed times X: co-occurrence counts

The single most useful product in recommendation:

```python
(X.T @ X).toarray()
```

```text
[[2. 1. 2. 0. 0.]
 [1. 2. 1. 0. 0.]
 [2. 1. 3. 1. 0.]
 [0. 0. 1. 1. 0.]
 [0. 0. 0. 0. 0.]]
```

Cell (i, j) counts the users who interacted with **both** item i and item j. The diagonal (i, i) is item i's
popularity. Items 0 and 2 share two users (users 0 and 1); items 0 and 3 share none. ItemKNN turns these counts into
similarities (lab 1), and EASE inverts this matrix (lab 3). It is also why catalogs are expensive: the item × item
matrix has (items)² cells, so 32,049 items make 1 billion of them.

## Scoring a user: a row times a matrix

If `S` is an item × item similarity matrix, a user's scores are their row times `S`:

```python
scores = X[[1]] @ S        # user 1: the sum of S's rows for items 0, 1 and 2
```

"The sum of the similarity rows of everything you interacted with" is exactly how ItemKNN, RP3beta, EASE, SLIM and
SANSA score. Only the matrix `S` differs.

## Top-k without sorting everything

```python
row = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
k = 2
top = np.argpartition(-row, k - 1)[:k]        # the k largest, in no particular order: [1, 3]
top = top[np.argsort(-row[top])]              # now sorted: [1, 3]
```

`argpartition` finds the k largest without sorting the whole row: important with 158,520 items per row.

## In recbench

- `TrainView.seen` is the user × item matrix of a split (CSR), with a 1 for every interaction before the cutoff.
  Row 0 and column 0 are padding (index 0 means "nobody" / "no item"), so the real users and items start at 1.
- `TrainView.weighted_matrix(half_life_days)` is the same matrix with recent interactions weighted more (the
  `decay_half_life_days` setting). `binary=False` gives counts instead of 0/1.
- `TrainView.item_pop` is the number of interactions per item (`X.sum(axis=0)`, as an array).

## Exercises

**1.** Without running code, what is `np.diff(X.indptr)` for a user with no interactions? What does the user's row
look like?

??? success "Solution"
    0: two consecutive `indptr` entries are equal, so the row holds no values. Its dense form is all zeros.

**2.** Build the 4 × 5 matrix above from a pandas DataFrame with columns `user` and `item`.

??? tip "Hint"
    `sp.csr_matrix((values, (rows, cols)), shape=...)`, with the DataFrame's columns as `rows` and `cols`.

??? success "Solution"
    ```python
    import pandas as pd
    events = pd.DataFrame({"user": [0, 0, 1, 1, 1, 2, 3, 3], "item": [0, 2, 0, 1, 2, 1, 2, 3]})
    X = sp.csr_matrix((np.ones(len(events)), (events["user"], events["item"])), shape=(4, 5))
    ```
    If a (user, item) pair appears twice, CSR **adds** the two values (a count of 2). Use `X.data[:] = 1` to get
    0/1 back.

**3.** With the co-occurrence matrix above, which item would you recommend to user 2 (who has only item 1), and
why?

??? success "Solution"
    User 2's row of `X.T @ X` is row 1: `[1, 2, 1, 0, 0]`. Without item 1 itself, items 0 and 2 tie at 1: each was
    seen together with item 1 by one user. This raw count is the core of ItemKNN; lab 1 shows how a similarity
    measure breaks such ties.

**4.** Why would `X.toarray()` on RetailRocket's quick-tier matrix crash the notebook, while `X.T @ X` might not?

??? success "Solution"
    `toarray()` allocates all 76 billion cells (303 GB). `X.T @ X` stays sparse: it only stores item pairs that share
    at least one user. With about 2 interactions per user on RetailRocket, that is a few million pairs.
