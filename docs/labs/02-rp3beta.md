# Lab 2 · RP3beta: a random walk with a popularity penalty

!!! abstract "In plain words"
    RP3beta imagines a walker on the graph of users and items. Start at an item you know, step to a random user who
    has it, then step to a random item of theirs. The chance of landing on item j says how strongly your item points
    to j. A penalty, popularity to the power β, stops the walker from always ending on bestsellers. Like ItemKNN it is
    counting, but with probabilities, and it was one of the strongest simple methods in a famous 2019
    re-evaluation. This week you will also hunt down a real numerical bug.

## Your starting point

--8<-- "generated/lab/rp3beta.md"

## Before you start

- Read sections 1 to 5 of the [RP3beta page](../dictionary/algorithms/rp3beta.md). Lab 1's ItemKNN is the
  comparison point.
- Open `labs/02-rp3beta/rp3beta.ipynb` and `labs/02-rp3beta/experiments.yaml`; `git switch -c lab/02-rp3beta`.
- Plan about 5 hours, plus box runs.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method rp3beta --dataset movielens-25m --best
```

The validation NDCG@10 must equal the "Validation" column above. The notebook does the same with `lab.fit`.

### 1.2 Read the code

Open `src/recbench/methods/neighbourhood.py`: `rp3beta_similarity`, `_row_normalise` and `RP3beta`.

**a.** `p_ui` and `p_iu` are both row-normalised. Which one is "from a user, step to one of their items", and which
is "from an item, step to one of its users"? Why is one built from `x` and the other from a 0/1 copy?

??? success "Answer"
    `p_ui = _row_normalise(x)`: each user's row sums to 1, so it is P(item | user), weighted by `x`, the (decayed)
    interactions. `p_iu` is built from `binary_t`, the transposed matrix with every value set to 1, row-normalised:
    P(user | item), uniform over the item's users. So recency and weights affect the second step only.

**b.** What does `p_ui.data **= alpha` do for α > 1 and for α < 1?

??? success "Answer"
    It raises every step probability to α. α > 1 sharpens: likely steps get relatively likelier, so the walk follows
    the strongest links. α < 1 flattens it. Note that after the power the rows no longer sum to 1: RP3beta is a
    *scoring rule inspired by* a walk, not an exact probability.

**c.** Where is popularity penalised, and which popularity is it?

??? success "Answer"
    `penalty = 1 / pop^beta`, with `pop` the number of users per item (`binary_t.sum(axis=1)`), applied to the
    destination item j: `walk.data * penalty[idx]`. It is all-time popularity over the training data. Keep that in
    mind: on MovieLens, recently popular films matter most.

### 1.3 Look inside

In the notebook, compare `lab.neighbours` for the same film under RP3beta and under ItemKNN (lab 1). Then fit RP3beta
with `rp3_beta=0.0` and `rp3_beta=1.0` and look at the neighbours' `popularity` column.

??? question "What changes in the neighbour lists as β grows?"
    The neighbours get less popular. With β = 0 (P3alpha), blockbusters appear in almost every list; with β = 1 they
    are divided by their full popularity and niche films rise.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `rp3_alpha` | sharpens (α > 1) or flattens (α < 1) the walk's steps | 0.3 to 2.0 |
| `rp3_beta` | popularity penalty on the destination item | 0.0 to 1.0 |
| `rp3_neighbors` | how many weights each item keeps | 20 to 1,000 (log scale) |
| `decay_half_life_days` | recent interactions weigh more in the walk and the profile | none, 30, 90, 365 |
| `train_window_days` | learn only from the last N days | none, 30, 90, 365 |

**a.** Predict the shape of NDCG@10 against β on MovieLens, then sweep it:

```bash
python -m recbench.lab sweep --method rp3beta --dataset movielens-25m --param rp3_beta --values 0.0,0.2,0.4,0.6,0.8,1.0
```

??? success "What to look for"
    Usually a hump: β = 0 recommends too many blockbusters, β = 1 too many niche items, and the best value sits
    between. Where the peak sits depends on how much the dataset rewards popularity. Repeat on Last.fm and compare.

**b.** Sweep `rp3_alpha` (0.5, 1.0, 1.5, 2.0) and `rp3_neighbors` (20, 100, 500). Which of the three settings
moves NDCG@10 most? Write it down: it is where future tuning effort should go.

**c.** Sweep `decay_half_life_days` with the values null, 30, 90, 365. Watch the terminal for a warning while the
30-day run fits.

??? success "What to look for"
    A `RuntimeWarning: overflow encountered in divide` from `_row_normalise`, and a 30-day score that is suspiciously
    low. That is Level 3.2.

## Level 3: data tricks

### 3.1 What does the popularity penalty add?

`no-popularity-penalty` in `experiments.yaml` fixes β = 0 (P3alpha) and searches the rest as the baseline did. Run
it on the box and compare:

```bash
python -m recbench.compare rp3beta rp3beta:no-popularity-penalty
```

??? success "What to look for"
    If the penalty matters, removing it is worse on most datasets. Look at the `coverage A -> B` column too: without the
    penalty, coverage drops, because fewer distinct items reach anyone's top 10.

### 3.2 A numerical trap

With `decay_half_life_days = 30`, an interaction from 12 years ago weighs $2^{-146}$, about $10^{-44}$. That is
still a valid float32, but a *tiny* one. Find out what `_row_normalise` does with a user whose every interaction is
that old.

??? tip "Hint 1"
    Build a 3 × 3 matrix where one user's weights are `np.float32(2.0 ** -140)`, call
    `rp3beta_similarity(x, 1.0, 0.5, 10)`, and print the result.

??? tip "Hint 2"
    `1.0 / 1e-42` is `1e42`. float32 stops at about `3.4e38`. What does numpy return when a float32 division overflows?

??? success "Solution"
    The division gives `inf`, and `inf` times a tiny weight is still `inf`. The old user's two items then get an `inf`
    similarity to each other, so *every* user who has one of them gets the other at the top of their list, whatever
    else they did. The fix is to compute the normalisation in float64, which reaches about `1e308`:

    ```python
    def _row_normalise(matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Each row divided by its sum, in float64: with strong time decay, an old user's weights can be ~1e-42, and
        1 / 1e-42 overflows float32 to inf (which then spreads into the item-item weights)."""
        sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        scale = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums > 0)
        return sp.diags(scale) @ sp.csr_matrix(matrix, dtype=np.float64)
    ```

    and at the top of `rp3beta_similarity`: `x = sp.csr_matrix(x, dtype=np.float64)`. The final weights are still
    stored as float32.

    A test that would have caught it:

    ```python
    def test_very_old_weights_do_not_overflow():
        old = np.float32(2.0 ** -140)
        x = sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [old, 0.0, old], [0.0, 1.0, 1.0]], dtype=np.float32))
        assert np.isfinite(rp3beta_similarity(x, 1.0, 0.5, 10).data).all()
    ```

**Measure the fix.** It changes no setting, so define an experiment with nothing in it but a note:

```yaml
  numerics:
    note: The baseline's search, with the float64 fix
```

Because the lab puts the code into each run's identity, this re-runs the baseline's search *with the fixed code*,
and `python -m recbench.compare rp3beta rp3beta:numerics` measures what the bug cost. Decay 30 may now become a good
setting, which the buggy search could never find.

!!! note "Not only RP3beta"
    ItemKNN has a cousin of this problem: with strong decay and `knn_shrink = 0`, `0 / 0` and `x / 0` appear in its
    cosine (a `divide by zero` warning). A fix that changes results, even a bug fix, is promoted like any improvement:
    a new `impl_version` and a re-run of the bake-off job ([promote](../handbook/promote.md)).

### 3.3 Walk along counts

RP3beta's second step follows the user's 0/1 (or decayed) interactions. On Last.fm a user may play one artist
thousands of times and another once. Add `rp3_counts`: when true, the walk follows interaction counts.

??? tip "Hint"
    `data.weighted_matrix(half_life, binary=False)` gives (decayed) counts. Use it only for the similarity; the
    profile used for scoring stays as it is.

??? success "Solution"
    In `RP3beta.fit`:

    ```python
    self.seen = data.weighted_matrix(cfg.get("decay_half_life_days"))
    # rp3_counts: the walk's step from a user to an item follows interaction counts instead of 0/1.
    walk = data.weighted_matrix(cfg.get("decay_half_life_days"), binary=False) if cfg.get("rp3_counts") else self.seen
    self.sim = rp3beta_similarity(walk, float(cfg.get("rp3_alpha", 1.0)), float(cfg.get("rp3_beta", 0.5)),
                                  int(cfg.get("rp3_neighbors", 200)))
    ```

??? question "Before running it: on which datasets can `rp3_counts` change anything at all?"
    Only where users repeat items. MovieLens has no repeats (each film is rated once), so the result there is
    identical. Last.fm repeats 47% of its user-artist pairs, RetailRocket 16%, H&M 14%, Steam 12%.

Uncomment `counts` in `experiments.yaml` (it re-tunes α with the counts) and run it on the box.

## Level 4: one change from the literature

### The idea: normalise each item's weights

After the top-k cut, items have very different total weight: an item with many strong links "votes" with much more
weight than an item with a few weak ones. The reference implementation in Ferrari Dacrema et al.'s re-evaluation
offers `normalize_similarity`: divide each item's kept weights by their sum, so every history item contributes the
same total. A user's score then averages what their items point to instead of being dominated by their best-connected
item.

### Write the variant

Add `rp3_normalize` (default false).

??? tip "Hint"
    You already have a function that makes every row sum to 1.

??? success "Solution"
    At the end of `RP3beta.fit`:

    ```python
    if cfg.get("rp3_normalize"):  # each item's kept weights sum to 1 (Ferrari Dacrema et al.'s normalize_similarity)
        self.sim = _row_normalise(self.sim).astype(np.float32).tocsr()
    ```

### Test it

Copy the template to `tests/test_rp3beta_variant.py`.

??? success "Solution"
    ```python
    METHOD = "rp3beta"
    VARIANT = {"rp3_normalize": True}
    DEFAULT = {"rp3_normalize": False, "rp3_counts": False}


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
        """After normalisation, every item's kept weights sum to 1."""
        view, _ = toy
        sums = np.asarray(checks.fit(METHOD, view, **VARIANT).sim.sum(axis=1)).ravel()
        np.testing.assert_allclose(sums[sums > 0], 1.0, rtol=1e-5)


    def test_counts_change_the_walk(toy):
        view, _ = toy
        checks.check_differs(METHOD, view, {}, {"rp3_counts": True})
    ```

    Add the overflow test from Level 3.2 to the same file.

### Run and compare

Uncomment `normalised` (it also re-tunes `rp3_neighbors`, which decides how many weights get normalised), run it on
the box, then:

```bash
python -m recbench.compare rp3beta rp3beta:normalised --segments
```

??? success "What to look for"
    Normalisation helps users whose history mixes one hub item (linked to everything) with specific ones: the hub stops
    drowning the rest. Check the **activity** segments: if the gain sits with heavy users (long, mixed histories),
    that is the mechanism at work.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| no-popularity-penalty | | | | | | |
| numerics | | | | | | |
| counts | | | | | | |
| normalised | | | | | | |

Two or three sentences: which setting matters most for RP3beta, and did the bug fix change the tuned result?

## Further reading

- Christoffel, Paudel, Newell & Bernstein (2015), *Blockbusters and wallflowers: accurate, diverse, and scalable recommendations with random graph walks*, RecSys 2015.
- Paudel, Christoffel, Newell & Bernstein (2016), *Updatable, accurate, diverse, and scalable recommendations for interactive applications*, ACM TiiS: RP3beta.
- Ferrari Dacrema, Cremonesi & Jannach (2019), [Are we really making much progress?](https://arxiv.org/abs/1907.06902), RecSys; their code, [RecSys2019_DeepLearning_Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation).
