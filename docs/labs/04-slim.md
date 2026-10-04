# Lab 4 · SLIM: many small sparse regressions

!!! abstract "In plain words"
    SLIM learns the same kind of item-to-item weights as EASE, but one item at a time: for each item, a regression
    predicts "did the user have this item?" from the user's other items. A penalty (ElasticNet: L1 plus L2) sets most
    weights to exactly zero, so each item keeps a short list of predictors, and the weights must be positive. This
    week you will see how the penalty shapes the model, test whether its shortcuts hold it back, and lift the
    positivity constraint, which EASE's author showed to be a mistake.

## Your starting point

--8<-- "generated/lab/slim.md"

## Before you start

- Read the [optimisation primer](primer-optimisation.md) (loss and penalties) and sections 1 to 5 of the
  [SLIM page](../dictionary/algorithms/slim.md). Lab 3's EASE is the comparison point.
- Open `labs/04-slim/slim.ipynb` and `labs/04-slim/experiments.yaml`; `git switch -c lab/04-slim`.
- Plan about 5 hours, plus box runs.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method slim --dataset movielens-25m --best
```

### 1.2 Read the code

Open `src/recbench/methods/linear.py`: `SLIM.fit` and `_slim_column`.

**a.** Each item j gets its own regression. What are its inputs (features) and its target?

??? success "Answer"
    The target is item j's column of X: for every user, whether they had j. The features are the columns of j's
    candidate neighbours only (`neighbours[j].indices`, without j itself). `item_cosine_topk` pre-selects them by
    cosine (`slim_neighbors` of them). This pre-selection (fsSLIM) is what makes SLIM fast: a regression over 100
    features instead of the whole catalog.

**b.** Where is "weights must be positive" in the code, and where is "most weights are zero"?

??? success "Answer"
    `ElasticNet(..., positive=True)` forbids negative weights. The L1 part of the penalty
    (`alpha * l1_ratio`) makes many weights exactly zero, and `keep = model.coef_ > 0` stores only the non-zero ones.
    `fit_info["nonzeros"]` reports how many remain.

**c.** scikit-learn's ElasticNet minimises
$\frac{1}{2n}\lVert y - Xw\rVert^2 + \alpha\, \rho\, \lVert w \rVert_1 + \frac{\alpha (1-\rho)}{2} \lVert w \rVert_2^2$,
with n the number of users and ρ = `slim_l1_ratio`. What does the $\frac{1}{2n}$ mean for the best `slim_alpha` on
a dataset with 10 times more users?

??? success "Answer"
    The error term is an *average* over users, so it does not grow with the number of users. The penalty does not
    shrink either, so the same α means the same strength. Compare with EASE, whose λ is added to a *sum* over users
    ($X^\top X$) and must grow with the data.

**d.** The convergence warnings of ElasticNet are silenced (`warnings.simplefilter("ignore")`). What could that hide?

??? success "Answer"
    With `max_iter` 100 (`slim_max_iter`, not searched), the coordinate descent may stop before converging, especially
    with a small α. The weights are then a little off, without any message. Level 2 checks whether that matters.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `slim_alpha` | total penalty strength α | 0.00001 to 0.1 (log scale) |
| `slim_l1_ratio` | the share of L1 in the penalty, ρ: more L1 means sparser | 0.01 to 1.0 (log scale) |
| `slim_neighbors` | candidate neighbours per item (pre-selected by cosine) | 50, 100, 200 |
| `decay_half_life_days`, `train_window_days` | recency, as in lab 1 | |
| `slim_max_iter` | coordinate-descent iterations per regression | not searched (100) |

**a.** Predict how NDCG@10 and the number of non-zero weights change as α grows, then sweep:

```bash
python -m recbench.lab sweep --method slim --dataset movielens-25m --param slim_alpha --values 0.00001,0.0001,0.001,0.01,0.1
```

In the notebook, also print `model.fit_info["nonzeros"]` for each value.

??? success "What to look for"
    The non-zero count falls steeply with α. At large α almost every weight is zero and NDCG@10 collapses (an item
    with no weights is never recommended). At tiny α the model is dense and may overfit. The SLIM page's pitfalls
    warn about "a penalty too strong gives all zeros".

**b.** Sweep `slim_max_iter` (50, 100, 500) at the baseline's best setting. Does convergence matter?

??? success "What to look for"
    If 500 iterations score the same as 100, the silenced warnings are harmless at this setting. If not, you have
    found a cheap improvement: put `slim_max_iter` in an experiment.

## Level 3: data tricks

### 3.1 Do the shortcuts hold it back?

`more-neighbours` searches 200 to 800 candidate neighbours instead of 50 to 200, with every other setting at the
baseline's best. It is slower: more features per regression. Run it on the box and compare:

```bash
python -m recbench.compare slim slim:more-neighbours
```

??? success "What to look for"
    Better accuracy means the cosine pre-selection was dropping useful predictors. Also compare the `train s`
    column: is the gain worth the time?

### 3.2 Leave out very rare items

An item with 2 interactions gets a regression fitted on 2 positive examples: its weights are mostly noise, and it
almost never appears in a test window. Add `slim_min_count`: items with fewer interactions get no weights.

??? tip "Hint"
    The regressions run over `warm`, the items with at least one interaction. Change that selection; the default 1
    must keep it identical.

??? success "Solution"
    In `SLIM.fit`:

    ```python
    warm = np.flatnonzero(data.item_pop >= max(1, int(cfg.get("slim_min_count", 1))))  # rarer items get no weights
    warm = warm[warm > 0]
    ```

    A test:

    ```python
    def test_rare_items_get_no_weights(toy):
        view, _ = toy
        weights = checks.fit("slim", view, slim_min_count=50).weights.tocsc()
        rare = np.flatnonzero((view.item_pop > 0) & (view.item_pop < 50))
        assert all(weights[:, j].nnz == 0 for j in rare)
    ```

## Level 4: one change from the literature

### The idea: allow negative weights (Steck 2019)

SLIM's positivity was meant to keep the model interpretable. Steck found that in EASE, which has no such constraint,
**many learned weights are negative**, and that forcing them to be positive hurts accuracy. A negative weight says
"having i makes j less likely" (lab 3's Level 1.3). That is useful information, and SLIM throws it away.

### Write the variant

Add `slim_positive` (true by default).

??? tip "Hint"
    Two places depend on positivity: the `ElasticNet(..., positive=True)` argument, and `keep = model.coef_ > 0`.
    `_slim_column` runs in parallel threads, so pass the flag to it as an argument.

??? success "Solution"
    ```python
    def _slim_column(x, j, features, alpha, l1_ratio, max_iter, positive=True):
        ...
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=positive, fit_intercept=False, copy_X=False,
                           precompute=True, selection="random", max_iter=max_iter, tol=1e-4, random_state=0)
        ...
        keep = model.coef_ > 0 if positive else model.coef_ != 0
        return features[keep].astype(np.int64), model.coef_[keep].astype(np.float32)
    ```

    and in `SLIM.fit`:

    ```python
    positive = bool(cfg.get("slim_positive", True))  # False: negative weights allowed (Steck 2019)
    results = Parallel(n_jobs=jobs, batch_size=256, prefer="threads")(
        delayed(_slim_column)(x, int(j), neighbours[j].indices, alpha, l1_ratio, int(cfg.get("slim_max_iter", 100)), positive)
        for j in warm
    )
    ```

### Test it

??? success "Solution"
    ```python
    METHOD = "slim"
    VARIANT = {"slim_positive": False}
    DEFAULT = {"slim_positive": True, "slim_min_count": 1}


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
        """Without the constraint some weights are negative; with it, none is."""
        view, _ = toy
        assert checks.fit(METHOD, view, **VARIANT).weights.data.min() < 0
        assert checks.fit(METHOD, view).weights.data.min() >= 0
    ```

### Run and compare

Uncomment `negative-weights` in `experiments.yaml`. It also re-tunes α and ρ, because negative weights change what
the best penalty is. Run it on the box, then:

```bash
python -m recbench.compare slim slim:negative-weights --segments
```

??? success "What to look for"
    Does SLIM move closer to EASE's score on each dataset? Compare the three: `python -m recbench.compare ease
    slim:negative-weights`. If SLIM with negative weights matches EASE, you have reproduced, from the other side, the
    finding that made EASE famous.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| more-neighbours | | | | | | |
| min-count | | | | | | |
| negative-weights | | | | | | |

## Further reading

- Ning & Karypis (2011), *SLIM: sparse linear methods for top-N recommender systems*, ICDM 2011: SLIM, and fsSLIM, its
  neighbour pre-selection.
- Steck (2019), [Embarrassingly shallow autoencoders for sparse data](https://arxiv.org/abs/1905.03375), WWW: section 4 on negative weights.
