# Lab 7 · GF-CF: smoothing over the graph, without training

!!! abstract "In plain words"
    Graph neural networks such as LightGCN learn by passing signals along the user-item graph. GF-CF showed that much
    of their power comes from a fixed **smoothing filter**, which needs no training at all. It adds two parts. One
    spreads a user's history one step along the graph, through a normalised co-occurrence matrix. The other keeps the
    graph's broadest patterns, its top singular directions as in lab 6. This week you will measure what each part
    contributes, make scoring faster, and replace the filter's sharp cut-off with a smooth one, following more
    recent graph-filter papers.

## Your starting point

--8<-- "generated/lab/gfcf.md"

## Before you start

- Lab 6 (PureSVD) first; read [linear algebra, part 2](primer-linear-algebra.md#graph-filters-smoothing) and sections
  1 to 5 of the [GF-CF page](../dictionary/algorithms/gfcf.md).
- Open `labs/07-gfcf/gfcf.ipynb` and `labs/07-gfcf/experiments.yaml`; `git switch -c lab/07-gfcf`.
- Plan about 6 hours, plus box runs.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method gfcf --dataset movielens-25m --best
```

### 1.2 Read the code

Open `src/recbench/methods/graph_filters.py`: `normalise` and `GFCF`. The module's docstring has the formula:

$$\text{score}(u) = r_u P + \alpha\, r_u D_I^{-1/2} V_k V_k^\top D_I^{1/2}, \qquad P = \tilde R^\top \tilde R, \qquad \tilde R = D_U^{-1/2} R\, D_I^{-1/2}$$

**a.** In `normalise(x, a)`, which matrices are $D_U$ and $D_I$, and what does dividing by their square roots do?

??? success "Answer"
    $D_U$ holds each user's number of interactions, $D_I$ each item's. With a = 1/2, each interaction is divided by
    $\sqrt{d_u d_i}$, so an interaction of a very active user, or with a very popular item, counts less. Without
    this, the heavy users and the blockbusters would dominate P.

**b.** Which line is the "linear filter" and which is the "ideal low-pass filter"?

??? success "Answer"
    `self.linear = (norm.T @ norm)` is P, used in `x @ self.linear`. The low-pass part is the SVD of the normalised
    matrix: `randomized_svd(norm, ...)` keeps `self.vt`, and scoring computes
    `alpha * ((x D^-1/2 V) V^T) D^1/2`. "Ideal" means every one of the top k directions counts fully and every other
    direction not at all.

**c.** P is never pruned. On MovieLens it has about 70 million non-zero weights, and scoring takes about 4 seconds
per 1,000 users, 30 times slower than ItemKNN. Why so many?

??? success "Answer"
    Two items get a non-zero weight in P as soon as one user shares them. With users who rated hundreds of films,
    almost every pair of popular films is connected. ItemKNN and RP3beta keep only the top-k weights per item; GF-CF
    keeps them all.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `gfcf_alpha` | the weight of the low-pass part (0: the linear filter alone) | 0.0 to 1.0 |
| `gfcf_k` | the number of singular directions in the low-pass part | 64, 128, 256, 512, 1024 |
| `decay_half_life_days`, `train_window_days` | recency | |
| the normalisation exponent `a` | how strongly degrees are divided out | fixed at 0.5 today |

**a.** Sweep `gfcf_alpha` (0.0, 0.3, 0.6, 1.0) and `gfcf_k` (64, 256, 1024) on MovieLens. Which part, the linear
filter or the low-pass, carries more of the result?

**b.** Make the normalisation exponent a setting, `gfcf_norm` (default 0.5), then sweep 0.3, 0.5, 0.7.

??? tip "Hint"
    `normalise(self.seen, float(cfg.get("gfcf_norm", 0.5)))`. Turbo-CF (another recbench method) searches the same
    exponent: look at its `turbocf_alpha`.

??? success "What to look for"
    Below 0.5 the active users and popular items count more; above 0.5 less. A dataset with a few huge users (Last.fm)
    may prefer a different exponent from MovieLens.

## Level 3: data tricks

### 3.1 What does the low-pass part add?

`linear-only` fixes `gfcf_alpha: 0` and searches the rest. Run it on the box and compare: this measures the SVD part's
contribution on each dataset.

### 3.2 Prune the linear filter

Keep only each item's k largest weights in P: much faster scoring, and possibly less noise. Add `gfcf_prune`
(0: no pruning).

??? tip "Hint"
    RP3beta's `rp3beta_similarity` keeps the top k of each row with `np.argpartition`. Write a small `top_k_rows(matrix,
    k)` that does the same for any CSR matrix, then apply it to `self.linear`.

??? success "Solution"
    ```python
    def top_k_rows(matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """Keep each row's k largest entries."""
        rows, cols, vals = [], [], []
        for r in range(matrix.shape[0]):
            lo, hi = matrix.indptr[r], matrix.indptr[r + 1]
            idx, val = matrix.indices[lo:hi], matrix.data[lo:hi]
            if len(val) > k:
                top = np.argpartition(-val, k - 1)[:k]
                idx, val = idx[top], val[top]
            rows.append(np.full(len(idx), r, dtype=np.int64))
            cols.append(idx.astype(np.int64))
            vals.append(val)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=matrix.shape, dtype=matrix.dtype)
    ```

    and in `GFCF.fit`, after `self.linear` is computed:

    ```python
    prune = int(cfg.get("gfcf_prune", 0))
    if prune:  # keep each item's k largest weights: faster scoring, and maybe less noise
        self.linear = top_k_rows(self.linear, prune)
    ```

    A test: with `gfcf_prune=5`, `np.diff(model.linear.indptr).max() <= 5`.

Compare accuracy *and* the `score s/1k` column. A large speed-up at the same accuracy is worth promoting by itself.

## Level 4: one change from the literature

### The idea: a smooth low-pass filter

GF-CF's "ideal" filter keeps the top k directions at full weight and drops the rest: a sharp cut-off. Later work on
graph filters for recommendation, such as BSPM (Choi et al. 2023) and SGFCF (Peng et al. 2024), finds that *smooth*
filters, which weigh directions by how strong they are, work as well or better and depend less on the exact k. The
simplest smooth filter weighs direction k by a power of its singular value:

$$w_k = \left(\frac{\sigma_k}{\sigma_1}\right)^{\gamma}$$

γ = 0 is GF-CF's ideal filter. A larger γ fades out the weaker directions gradually.

### Write the variant

Add `gfcf_sv_power` = γ.

??? tip "Hint"
    `randomized_svd` returns the singular values as its second result; today's code throws them away (`_, _, vt`). The
    weights multiply the k columns of the projection `low` before it goes back through `self.vt`.

??? success "Solution"
    In `GFCF.fit`:

    ```python
    self.sv_weight = None
    if self.alpha > 0:
        k = max(1, min(int(cfg.get("gfcf_k", 256)), min(norm.shape) - 1))
        _, s, vt = randomized_svd(norm, n_components=k, n_iter=5, random_state=int(cfg.get("seed", 42)))
        power = float(cfg.get("gfcf_sv_power", 0.0))
        if power:  # a smooth low-pass: direction k weighted by (sigma_k / sigma_1)^power instead of 1
            self.sv_weight = np.power(s / s[0], power).astype(np.float32)
    ```

    and in `score_users`:

    ```python
    low = np.asarray(x.multiply(self.d_inv[None, :]) @ self.vt.T, dtype=np.float32)
    if self.sv_weight is not None:
        low = low * self.sv_weight[None, :]
    out += self.alpha * ((low @ self.vt) * self.d[None, :])
    ```

### Test it

??? success "Solution"
    ```python
    METHOD = "gfcf"
    VARIANT = {"gfcf_sv_power": 1.0}
    DEFAULT = {"gfcf_sv_power": 0.0, "gfcf_prune": 0, "gfcf_norm": 0.5}


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
        """The direction weights are (sigma_k / sigma_1)^power: 1 for the first direction, then decreasing."""
        view, _ = toy
        weights = checks.fit(METHOD, view, **VARIANT).sv_weight
        assert weights[0] == 1.0 and np.all(np.diff(weights) <= 1e-6)
    ```

### Run and compare

Uncomment `smooth` (it re-tunes `gfcf_alpha`, whose best value changes with the filter's shape), run it on the box,
then compare with `--segments`.

??? success "What to look for"
    Check whether the chosen `gfcf_k` matters less now: run `lab sweep` over `gfcf_k` with and without the smooth
    filter. A filter that depends less on k is easier to tune, which is a benefit even at equal accuracy.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| linear-only | | | | | | |
| pruned | | | | | | |
| smooth | | | | | | |

## Further reading

- Shen et al. (2021), *How powerful is graph convolution for recommendation?*, CIKM 2021: GF-CF.
- Choi et al. (2023), *Blurring-sharpening process models for collaborative filtering*, SIGIR 2023: BSPM.
- Peng et al. (2024), *How powerful is graph filtering for recommendation*, KDD 2024: SGFCF.
- Park et al. (2024), *Turbo-CF: matrix decomposition-free graph filtering for fast recommendation*, SIGIR 2024.
