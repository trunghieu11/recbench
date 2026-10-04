# Lab 9 · BPR-MF: learning to rank, one pair at a time

!!! abstract "In plain words"
    BPR-MF learns user and item vectors like iALS (lab 8), with a different goal. It does not predict 1s and 0s; for
    each user it learns that an item they chose should score **above** an item they did not. It trains by stochastic
    gradient descent, millions of tiny updates, each for one (user, chosen item, other item) triple. Which "other
    item" you draw, the **negative**, matters a lot: a random one is usually too easy. This week you will write your
    own BPR training loop and teach it harder negatives.

## Your starting point

--8<-- "generated/lab/bpr_mf.md"

## Before you start

- Read the [optimisation primer, SGD and BPR](primer-optimisation.md#stochastic-gradient-descent) (it does one BPR
  update by hand), the [negative sampling page](../dictionary/concepts/negative-sampling.md) and sections 1 to 5 of
  the [BPR-MF page](../dictionary/algorithms/bpr-mf.md).
- Open `labs/09-bpr-mf/bpr_mf.ipynb` and `labs/09-bpr-mf/experiments.yaml`; `git switch -c lab/09-bpr-mf`.
- Plan about 7 hours: Level 4 is a real coding exercise.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method bpr_mf --dataset movielens-25m --best
```

??? question "The number may differ slightly from the baseline's best trial. Why, when the seed is the same?"
    `implicit`'s BPR runs SGD on several threads at once, and each thread updates the shared vectors without waiting
    for the others ("Hogwild"). The order of updates, and so the result, depends on timing. That is why the lab tests
    BPR-MF with 3 seeds and averages them. Your own trainer in Level 4 runs on one thread and repeats exactly.

### 1.2 Read the code

Open `src/recbench/methods/implicit_mf.py::BPRMF`.

**a.** Which matrix does BPR-MF train on? Can it use `decay_half_life_days`?

??? success "Answer"
    `data.seen`: 0/1, without time weights. BPR only needs to know which items a user chose, so decay cannot enter, and
    the baseline does not search it. Only `train_window_days` (which events are kept) affects it.

**b.** How are the negatives drawn in `implicit`'s BPR?

??? success "Answer"
    Uniformly from all items. Most random items are obscure and already scored low, so the update factor
    $\sigma(-x_{uij})$ is close to 0 and they teach almost nothing (the primer's exercise 3).

**c.** In `_ImplicitMF._factors`, the comment says BPR "appends a bias column to both matrices". What is a bias here?

??? success "Answer"
    An extra number per item that is added to every user's score of it: a learned popularity. `implicit` stores it as
    one more column of the item vectors, with a constant 1 in every user vector, so that the plain dot product already
    includes it.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `dim` | the vectors' length | 64, 128, 256 |
| `bpr_lr` | the learning rate η | 0.001 to 0.1 (log scale) |
| `bpr_reg` | the L2 penalty on the vectors | 0.00001 to 0.1 (log scale) |
| `bpr_iterations` | passes over the data (epochs) | 50, 100, 200 |
| `train_window_days` | learn only from the last N days | none, 30, 90, 365 |

**a.** Sweep `bpr_lr` (0.001, 0.01, 0.05, 0.1) at the baseline's best setting. Then repeat with `--set
bpr_iterations=50` and with `--set bpr_iterations=200`. How do learning rate and iterations trade off?

??? success "What to look for"
    A small learning rate needs more passes to get anywhere; a large one gets there fast, then becomes unstable. The
    best pair lies on a diagonal: halving η roughly needs twice the iterations.

**b.** Sweep `bpr_reg`. Is BPR-MF more or less sensitive to its penalty than iALS was to `ials_reg` (lab 8)?

## Level 3: data tricks

### 3.1 Is it under-trained?

`more-iterations` searches 200 to 800 passes instead of 50 to 200, with every other setting at the baseline's best.
Run it on the box and compare. Look at both the accuracy and the `train s` column.

### 3.2 Leave out very rare items

An item with 1 interaction is a positive in one triple per epoch and a negative almost never: its vector barely
learns. Add `bpr_min_count`: interactions with rarer items are left out of training.

??? tip "Hint"
    As in lab 4: multiply the 0/1 matrix by a diagonal 0/1 matrix over items, then `eliminate_zeros()`. Pass the
    result to `model.fit`.

??? success "Solution"
    In `BPRMF.fit`:

    ```python
    seen = data.seen.tocsr()
    min_count = int(cfg.get("bpr_min_count", 1))
    if min_count > 1:  # interactions with very rare items are left out of training
        seen = (seen @ sp.diags((data.item_pop >= min_count).astype(np.float32))).tocsr()
        seen.eliminate_zeros()
    ...
    model.fit(seen, show_progress=False)
    ```

## Level 4: one change from the literature

### The idea: harder negatives (Rendle & Freudenthaler 2014)

Two years after BPR, Rendle and Freudenthaler showed that uniform sampling makes BPR learn slowly. Most sampled pairs
are already in the right order, so their gradient is almost zero. Drawing negatives more often from **popular**
items helps: users skip popular items for a reason, so these are harder, more informative negatives. Their paper also
adapts the sampler to the current model; the simplest version, "static oversampling", draws item j with probability

$$P(j) \propto \text{pop}_j^{\,\beta}$$

β = 0 is uniform; β = 1 follows popularity. `implicit` cannot do this, so you will write the training loop yourself
with [numba](https://numba.pydata.org), which compiles Python loops to fast machine code.

### Write the variant

Add `bpr_backend` (`implicit` by default, or `numba`) and `bpr_neg_power` (β). With `numba`, `BPRMF.fit` calls your
own `train_bpr(...)`, which returns the user and item vectors.

??? tip "Hint 1: the data"
    Turn the 0/1 matrix into two arrays, `users` and `items`, one entry per interaction:
    `users = np.repeat(np.arange(n_users), np.diff(seen.indptr))`, `items = seen.indices`. One epoch draws as many
    random positions k as there are interactions: the pair `(users[k], items[k])`.

??? tip "Hint 2: the negative sampler"
    Make a cumulative distribution, `cdf = np.cumsum(pop ** beta)`, divided by its last value. Then for a random number
    r in [0, 1), the negative is the first index whose `cdf` exceeds r, found by binary search. Give item 0 (padding)
    and items without interactions weight 0. If the user has the sampled item, skip the triple: their items are sorted
    in `seen.indices`, so another binary search tells you.

??? tip "Hint 3: the update"
    Exactly the primer's: $x = p_u \cdot (q_i - q_j)$, $g = 1 / (1 + e^{x})$, then
    $p_u \mathrel{+}= \eta (g (q_i - q_j) - \lambda p_u)$, $q_i \mathrel{+}= \eta (g p_u - \lambda q_i)$,
    $q_j \mathrel{+}= \eta (-g p_u - \lambda q_j)$, coordinate by coordinate, reading all three old values before
    writing.

??? tip "Hint 4: numba"
    Write the loop as a plain function at module level using only numpy arrays, numbers and loops. Compile it with
    `from numba import njit; kernel = njit(cache=True)(your_function)`, then call `kernel(...)`. Inside, use
    `np.random.seed(seed)`, `np.random.randint(n)` and `np.random.random()`, which numba supports and which make the
    run repeatable.

??? success "Solution"
    In `BPRMF.fit`, before the `implicit` code:

    ```python
    if cfg.get("bpr_backend", "implicit") == "numba":  # your own trainer: popularity-based negatives
        self.user_factors, self.item_factors = train_bpr(
            seen, dim=int(cfg.get("dim", 64)), lr=float(cfg.get("bpr_lr", 0.01)), reg=float(cfg.get("bpr_reg", 0.01)),
            epochs=int(cfg.get("bpr_iterations", 100)), neg_power=float(cfg.get("bpr_neg_power", 0.0)),
            seed=int(cfg.get("seed", 42)))
        return
    ```

    And at the end of the module:

    ```python
    def train_bpr(seen, *, dim, lr, reg, epochs, neg_power, seed):
        """BPR-MF trained by stochastic gradient descent (Rendle et al. 2009), written with numba.

        Negatives are drawn with probability proportional to popularity^neg_power: 0 is uniform, as in `implicit`;
        1 follows popularity, which gives harder negatives (Rendle & Freudenthaler 2014). Returns (user, item) factors."""
        from numba import njit

        if "kernel" not in _BPR:
            _BPR["kernel"] = njit(cache=True)(_bpr_epochs)  # compiled once (and cached on disk)
        rng = np.random.default_rng(seed)
        seen = seen.tocsr().astype(np.float32)
        seen.sort_indices()
        n_users, n_items = seen.shape
        users = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(seen.indptr))
        items = seen.indices.astype(np.int64)
        pop = np.asarray((seen > 0).sum(axis=0)).ravel().astype(np.float64)
        weights = np.where(pop > 0, np.power(np.maximum(pop, 1.0), neg_power), 0.0)
        weights[0] = 0.0  # item 0 is padding
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]  # ends exactly at 1
        p = (rng.standard_normal((n_users, dim)) * 0.01).astype(np.float32)
        q = (rng.standard_normal((n_items, dim)) * 0.01).astype(np.float32)
        _BPR["kernel"](p, q, users, items, seen.indptr.astype(np.int64), cdf, np.float32(lr), np.float32(reg), int(epochs), int(seed))
        return p, q


    _BPR: dict = {}


    def _bpr_epochs(p, q, users, items, indptr, cdf, lr, reg, epochs, seed):
        """The SGD loop (compiled by numba in train_bpr). One epoch draws as many (user, item, negative) triples as there
        are interactions."""
        np.random.seed(seed)
        n, dim = len(users), p.shape[1]
        for _ in range(epochs):
            for _ in range(n):
                k = np.random.randint(n)
                u, i = users[k], items[k]
                r = np.random.random()  # the negative: the first item whose cumulative weight exceeds r
                lo, hi = 0, len(cdf)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cdf[mid] <= r:
                        lo = mid + 1
                    else:
                        hi = mid
                j = min(lo, len(cdf) - 1)
                seen_by_u = False  # u's items are sorted: binary search for j among them
                a, b = indptr[u], indptr[u + 1]
                while a < b:
                    mid = (a + b) // 2
                    if items[mid] < j:
                        a = mid + 1
                    else:
                        b = mid
                if a < indptr[u + 1] and items[a] == j:
                    seen_by_u = True
                if seen_by_u:
                    continue
                x = 0.0
                for f in range(dim):
                    x += p[u, f] * (q[i, f] - q[j, f])
                g = 1.0 / (1.0 + np.exp(x))  # sigma(-x): large when the pair is badly ordered
                for f in range(dim):
                    pu, qi, qj = p[u, f], q[i, f], q[j, f]
                    p[u, f] += lr * (g * (qi - qj) - reg * pu)
                    q[i, f] += lr * (g * pu - reg * qi)
                    q[j, f] += lr * (-g * pu - reg * qj)
    ```

    `import scipy.sparse as sp` goes at the top of the module. Unlike `implicit`, these vectors have no bias column.

!!! warning "numba does not check array bounds"
    An index one past the end of an array does not raise an error in compiled code: it silently reads or writes
    memory it should not. That is why the sampler clips `j` to the last item and divides `cdf` by its last value.
    When something inside a numba function behaves strangely, run the plain Python function on a tiny input first:
    without `njit` it is slow, but it checks every index.

### Test it

??? success "Solution"
    ```python
    METHOD = "bpr_mf"
    VARIANT = {"bpr_backend": "numba", "bpr_neg_power": 0.5, "bpr_iterations": 30, "bpr_lr": 0.05}


    def test_the_numba_trainer_is_deterministic(toy):
        view, _ = toy
        checks.check_deterministic(METHOD, view, **VARIANT)


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand():
        """One user, one item seen, one not: after training, the seen item scores higher."""
        import scipy.sparse as sp

        from recbench.methods.implicit_mf import train_bpr

        seen = sp.csr_matrix(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=np.float32))  # row and column 0: padding
        p, q = train_bpr(seen, dim=4, lr=0.1, reg=0.0, epochs=200, neg_power=0.0, seed=0)
        assert p[1] @ q[1] > p[1] @ q[2]
    ```

    Remove the template's `test_the_default_is_unchanged` and `test_the_variant_is_deterministic`: `implicit`'s BPR
    differs between two runs anyway. Keep the contract, "changes the model" and "still learns" tests.

### Run and compare

Uncomment `popular-negatives`. It searches β and re-tunes the learning rate and the number of passes, which your own
trainer needs. Run it on the box, then:

```bash
python -m recbench.compare bpr_mf bpr_mf:popular-negatives --segments
```

??? success "What to look for"
    First the chosen β per dataset: if the tuner prefers β near 0, harder negatives did not help there. Then the
    **taste** segments: popular negatives push popular items *down* for users who skip them, so a gain often comes
    from users with niche tastes. Also compare the `train s` column: one compiled thread against `implicit`'s many.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| more-iterations | | | | | | |
| min-count | | | | | | |
| popular-negatives | | | | | | |

## Further reading

- Rendle, Freudenthaler, Gantner & Schmidt-Thieme (2009), *BPR: Bayesian personalized ranking from implicit feedback*, UAI 2009.
- Rendle & Freudenthaler (2014), *Improving pairwise learning for item recommendation from implicit feedback*, WSDM 2014.
- Niu, Recht, Ré & Wright (2011), *Hogwild!: a lock-free approach to parallelizing stochastic gradient descent*, NeurIPS 2011.
