# Primer: linear algebra for recommenders

!!! abstract "In plain words"
    Most lab methods are a few lines of linear algebra. You need five ideas: the **dot product** (a score is a
    weighted sum), the **matrix product** (scoring every user at once), the **inverse** (undoing a matrix, used to
    solve for the best weights in one step), **ridge regression** (the best weights, kept small by a penalty), and the
    **SVD** (the main "directions" in the data). Part 1 covers the first four and leads to EASE (lab 3). Part 2 covers
    the SVD and leads to PureSVD and GF-CF (labs 6 and 7). Every example is small enough to check by hand.

## Part 1: scores, products and the inverse

### The dot product: a score is a weighted sum

A user's history is a vector with a 1 for each item they interacted with: `x = [1, 0, 1]` means items A and C. An
item-to-item model has, for a candidate item j, one weight per history item: how much having seen A, B or C points to
j. Say `w = [0.5, 0.2, 0.3]`. The score is the dot product:

$$\text{score} = x \cdot w = 1 \times 0.5 + 0 \times 0.2 + 1 \times 0.3 = 0.8$$

Only the items in the history count, and each adds its weight. That is all an item-to-item recommender does.

### The matrix product: every user, every item at once

Stack the users' history vectors as rows of **X** (users × items) and the weights for every candidate item as columns
of **B** (items × items). Then **X B** holds every user's score for every item: row u, column j is the dot product
of user u's history with column j of B. In numpy: `X @ B`. ItemKNN, RP3beta, EASE, SLIM and SANSA all score exactly
like this; they differ only in how they build B.

### The identity and the inverse

The **identity matrix** I has 1s on its diagonal and 0s elsewhere: `I @ v = v`. The **inverse** of a square matrix A
is the matrix $A^{-1}$ with $A A^{-1} = I$. It undoes A. For example:

$$A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}, \qquad A^{-1} = \begin{pmatrix} 0.6 & -0.2 \\ -0.2 & 0.4 \end{pmatrix}$$

Check the first entry: $2 \times 0.6 + 1 \times (-0.2) = 1$. In numpy: `np.linalg.inv(A)`. The inverse matters because
it solves equations: if $A w = b$, then $w = A^{-1} b$. Not every matrix has an inverse; adding a positive number to
the diagonal ($A + \lambda I$) makes sure it has one.

### Ridge regression: the best weights, kept small

Regression finds weights w so that $X w$ is as close as possible to a target y. The least-squares solution is
$w = (X^\top X)^{-1} X^\top y$. **Ridge** regression adds a penalty λ on large weights:

$$w = (X^\top X + \lambda I)^{-1} X^\top y$$

A larger λ gives smaller, more cautious weights that fit the training data less closely and new data better.
$X^\top X$ is the co-occurrence matrix from the [sparse matrices primer](primer-sparse-matrices.md).

### Putting it together: EASE in three items

EASE predicts each item's column of X from the *other* items' columns, with ridge regression. The solution for all
items at once is one inverse (Steck 2019):

$$P = (X^\top X + \lambda I)^{-1}, \qquad B_{ij} = -\frac{P_{ij}}{P_{jj}}, \qquad B_{jj} = 0$$

The zero diagonal forbids the trivial answer "item j predicts itself". Four users, three items (A, B, C):

```python
X = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=float)
G = X.T @ X                       # [[3, 2, 1], [2, 3, 1], [1, 1, 2]]
P = np.linalg.inv(G + 1.0 * np.eye(3))
B = -P / np.diag(P)[None, :]
np.fill_diagonal(B, 0)
```

```text
B (lambda = 1)          B (lambda = 10)
[[0.    0.455 0.167]    [[0.    0.148 0.067]
 [0.455 0.    0.167]     [0.148 0.    0.067]
 [0.182 0.182 0.   ]]    [0.071 0.071 0.   ]]
```

User 3 has seen A and C. Their scores `X[3] @ B` are `[0.182, 0.636, 0.167]` for λ = 1: the unseen item B is
recommended. With λ = 10 the scores are `[0.071, 0.219, 0.067]`: the same order, smaller weights. A and B are
linked most strongly (two users saw both). In a real catalog the order *does* change with λ: a large λ pushes towards
recommending popular items. Lab 3 sweeps it.

## Part 2: SVD and filters

### Directions in the data

Four users and four movies, two of them action (A, B) and two romance (C, D):

```text
        A  B  C  D
user 0  1  1  0  0
user 1  1  1  1  0
user 2  0  0  1  1
user 3  0  1  1  1
```

The **singular value decomposition** (SVD) writes any matrix as $R = U \Sigma V^\top$:

- the columns of **V** are *directions* over the items, ordered from most to least important;
- **Σ** holds the importance of each direction (the *singular values*): here 2.618, 1.618, 0.618, 0.382;
- **U** says how much each user follows each direction.

```text
movie   direction 1   direction 2
A          -0.372        0.602
B          -0.602        0.372
C          -0.602       -0.372
D          -0.372       -0.602
```

Direction 1 has the same sign everywhere: it is roughly "how popular" (B and C, seen by three users, weigh more).
Direction 2 is "action (+) versus romance (−)". Nobody told the SVD about genres: it found the taste axis from the
data. In a real catalog, the first few hundred directions capture most of what users have in common. These
directions are the **latent factors** of matrix factorisation.

### PureSVD: project onto the main directions

Keep the top k directions ($V_k$, items × k). A user's scores are their history, projected onto those directions and
back: $x V_k V_k^\top$. For k = 2, user 0 (who saw A and B) gets 0.224 for C and −0.224 for D. They are pointed
towards C, the romance movie that action fans (user 1) also saw. Fewer directions mean smoother, more general
recommendations; more directions mean more detail and more noise. That trade-off is `svd_factors` in lab 6.

`np.linalg.svd` computes all directions. For big sparse matrices recbench uses
`sklearn.utils.extmath.randomized_svd`, which finds only the top k directions, fast and approximately.

### Graph filters: smoothing

Think of users and items as a graph: an edge for each interaction. Multiplying a user's history by a normalised
co-occurrence matrix $P = \tilde{R}^\top \tilde{R}$ spreads their history one step along the graph: from the items
they saw to the items those users also saw. That is a *smoothing* filter. Keeping only the top singular directions is
another, stronger smoothing (an "ideal low-pass filter": keep the slow, broad patterns, drop the noisy ones). GF-CF
(lab 7) adds the two. The normalisation $\tilde{R} = D_U^{-1/2} R D_I^{-1/2}$ divides by the square roots of the
users' and items' degrees (numbers of interactions), so very active users and very popular items do not dominate.

## Exercises

**1.** With B (λ = 1) above, compute user 0's scores by hand. Which item would EASE recommend to user 0?

??? tip "Hint"
    User 0 has seen A and B: add rows A and B of B.

??? success "Solution"
    Row A `[0, 0.455, 0.167]` + row B `[0.455, 0, 0.167]` = `[0.455, 0.455, 0.333]`. A and B are already seen, so
    EASE recommends C (0.333).

**2.** Why does $P = (X^\top X + \lambda I)^{-1}$ always exist when λ > 0?

??? success "Solution"
    $X^\top X$ never has negative "stretch" in any direction (it is positive semi-definite). Adding λ to the diagonal
    stretches every direction by at least λ > 0, so no direction collapses to zero and the matrix can be undone.

**3.** In the SVD example, what would PureSVD with k = 1 recommend to user 0, and why is that less personal?

??? success "Solution"
    With only direction 1 ("popularity"), $x V_1 V_1^\top$ is proportional to each movie's weight on that direction:
    B and C (−0.602) score highest, then A and D. User 0 gets C, as for everyone whose history is similar in total
    popularity. One direction cannot tell action from romance; k = 2 can.
