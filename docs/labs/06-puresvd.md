# Lab 6 · PureSVD: the main directions of taste

!!! abstract "In plain words"
    PureSVD finds the few main "directions" in which users' tastes differ: popular against niche, family films against
    horror, and so on. It describes every item by them, and recommends what lies in the same directions as a user's
    history. It is a single SVD, takes about a second, and is often surprisingly strong. This week you will see what
    those directions are, how many to keep, and how a small rescaling from the EigenRec paper changes which
    directions the SVD finds.

## Your starting point

--8<-- "generated/lab/puresvd.md"

## Before you start

- Read [linear algebra, part 2](primer-linear-algebra.md#part-2-svd-and-filters) and sections 1 to 5 of the
  [PureSVD page](../dictionary/algorithms/puresvd.md).
- Open `labs/06-puresvd/puresvd.ipynb` and `labs/06-puresvd/experiments.yaml`; `git switch -c lab/06-puresvd`.
- Plan about 5 hours, plus box runs.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method puresvd --dataset movielens-25m --best
```

### 1.2 Read the code

Open `src/recbench/methods/linear.py::PureSVD`.

**a.** `randomized_svd(self.seen, n_components=factors, ...)` returns three things; PureSVD keeps only `vt`. Why is
V enough, without U and Σ?

??? success "Answer"
    The score is $x_u V V^\top$: project the user's history onto the kept directions and back onto the items. That
    needs only the item side, V. A user does not need a learned vector; their history *is* their description. So a user
    who was not in the training data can still be scored, and that is why PureSVD counts as "uses history".

**b.** `n_iter=5` is hard-coded. What is it?

??? success "Answer"
    Randomised SVD approximates the top directions with random projections, refined by a few "power iterations".
    More iterations give more exact directions and cost more time. It is not a setting today: Level 2 makes it one.

**c.** `factors = min(svd_factors, min(shape) - 1)`. What happens if you ask for more factors than there are users
or items?

??? success "Answer"
    It is cut to one less than the smaller dimension. Asking for nearly all directions makes $V V^\top$ close to the
    identity: each user's projection is just their own history, and PureSVD learns nothing. You can see this on the
    40-item toy data in the tests, which is why the toy checks use 8 factors.

### 1.3 Look inside

In the notebook, fit the baseline's best setting and compare each item's weight on the first direction,
`model.v[:, 0]`, with its popularity, `data.item_pop`. Then list the films at both ends of the second direction.

??? question "What does the first direction mean? And the second?"
    The first direction usually follows popularity closely (a strong correlation, up to the sign). Later directions
    contrast groups of films. Name them from the titles at their two ends; it is like reading the axes of a map that
    nobody labelled.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `svd_factors` | the number of directions kept | 16, 32, 64, 128, 256, 512, 1024 |
| `decay_half_life_days`, `train_window_days` | recency | |
| the SVD's power iterations | how exact the directions are | fixed at 5 today |

**a.** Predict, then sweep the number of directions:

```bash
python -m recbench.lab sweep --method puresvd --dataset movielens-25m --param svd_factors --values 16,64,256,1024
```

??? success "What to look for"
    Too few directions give generic, popular-leaning lists; too many bring back noise, and in the limit return the
    user's own history (1.2c). The best value depends on how many distinct tastes the data holds, and on its size.

**b.** Make the power iterations a setting, `svd_iters` (default 5), then sweep 1, 2, 5 and 10.

??? tip "Hint"
    Replace `n_iter=5` with `n_iter=int(cfg.get("svd_iters", 5))`.

??? success "What to look for"
    Accuracy usually stops changing after a few iterations, while the time keeps growing. If 2 is as good as 5, you
    have found a free speed-up. Report it, even though it is not an accuracy gain.

## Level 3: data tricks

### 3.1 How much does recency matter?

`no-time-knobs` searches the number of factors alone. Run it on the box and compare with the baseline, as in lab 1.

### 3.2 Normalise users by activity

A user with 2,000 ratings pulls the SVD's directions towards their own taste far more than a user with 20. Add
`svd_user_norm` = γ: before the SVD, divide each user's row by (their number of items)$^\gamma$. γ = 0 is today's
PureSVD.

??? tip "Hint"
    The number of items per user is `np.diff(x.indptr)` (the [sparse matrices primer](primer-sparse-matrices.md)).
    Scaling rows is a diagonal matrix on the left: `sp.diags(w) @ x`. Scale only the matrix you decompose; scoring keeps
    `self.seen`.

??? success "Solution"
    In `PureSVD.fit`, before the SVD:

    ```python
    x = self.seen
    gamma = float(cfg.get("svd_user_norm", 0.0))
    if gamma:  # long histories weigh less: each user's row divided by (number of items)^gamma
        lengths = np.maximum(np.diff(x.indptr), 1).astype(np.float64)
        x = (sp.diags(np.power(lengths, -gamma).astype(np.float32)) @ x).tocsr()
    ```

    and decompose `x` instead of `self.seen`.

## Level 4: one change from the literature

### The idea: EigenRec's scaling (Nikolakopoulos et al. 2019)

EigenRec shows that PureSVD is one member of a family. It builds an item × item "proximity" matrix, scales each item
by a power of its popularity, and takes the main directions of the result. The scaling exponent changes which
directions the method finds. Divide popular items down and the directions describe niche tastes more; push them up
and the directions follow the mainstream. In PureSVD's terms: scale each item's column of X by
$\text{popularity}_i^{\,d}$ before the SVD. d = 0 is PureSVD.

### Write the variant

Add `svd_pop_exponent` = d.

??? tip "Hint"
    Column scaling is a diagonal matrix on the right: `x @ sp.diags(w)`. `data.item_pop` holds the popularity; use
    `np.maximum(..., 1)` so that items without interactions do not become `0 ** -0.3 = inf`.

??? success "Solution"
    After the user normalisation, before the SVD:

    ```python
    d = float(cfg.get("svd_pop_exponent", 0.0))
    if d:  # EigenRec (Nikolakopoulos et al. 2019): each item's column scaled by its popularity^d
        pop = np.maximum(data.item_pop, 1).astype(np.float64)
        x = (x @ sp.diags(np.power(pop, d).astype(np.float32))).tocsr()
    ```

### Test it

??? success "Solution"
    ```python
    METHOD = "puresvd"
    VARIANT = {"svd_pop_exponent": -0.3}
    DEFAULT = {"svd_pop_exponent": 0.0, "svd_user_norm": 0.0, "svd_iters": 5}


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
        """Whatever the scaling, the kept directions are orthonormal: V^T V = I."""
        view, _ = toy
        v = checks.fit(METHOD, view, svd_factors=8, **VARIANT).v
        np.testing.assert_allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-4)
    ```

### Run and compare

Uncomment `eigenrec` (it re-tunes the number of factors, since the scaling changes how many directions matter), run
it on the box, then:

```bash
python -m recbench.compare puresvd puresvd:eigenrec --segments
```

??? success "What to look for"
    Which sign of d did the tuner choose on each dataset? A negative d means niche directions helped, a positive d
    mainstream ones. Check the **taste** segments and the coverage column: they should move in the same direction as
    the sign of d.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| no-time-knobs | | | | | | |
| user-norm | | | | | | |
| eigenrec | | | | | | |

## Further reading

- Cremonesi, Koren & Turrin (2010), *Performance of recommender algorithms on top-N recommendation tasks*, RecSys 2010: PureSVD.
- Nikolakopoulos, Kalantzis, Gallopoulos & Garofalakis (2019), *EigenRec: generalizing PureSVD for effective and efficient top-N recommendations*, Knowledge and Information Systems.
- Halko, Martinsson & Tropp (2011), *Finding structure with randomness*, SIAM Review: the randomised SVD.
