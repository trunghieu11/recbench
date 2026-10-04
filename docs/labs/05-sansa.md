# Lab 5 · SANSA: EASE without the item cap

!!! abstract "In plain words"
    EASE inverts a dense item × item matrix, so it must drop items beyond a cap (20,000 in the lab). SANSA computes
    *almost* the same model with sparse matrices: it factorises the co-occurrence matrix approximately and keeps only
    the largest weights. It can therefore cover the whole catalog. You choose how many weights each item may keep,
    and with that you trade accuracy against time and memory. This week is about that trade-off, and about bringing
    lab 3's EDLAE penalty to SANSA with a neat trick: extra, made-up users.

## Your starting point

--8<-- "generated/lab/sansa.md"

## Before you start

- Lab 3 (EASE) first: SANSA approximates it. Read sections 1 to 5 of the [SANSA page](../dictionary/algorithms/sansa.md).
- SANSA needs the `sansa` extra (`brew install suite-sparse`, then `uv pip install -e ".[sansa]"`). In a notebook,
  `lab.setup()` must run before anything else: SuiteSparse crashes in a process that has loaded PyTorch on macOS.
- Open `labs/05-sansa/sansa.ipynb` and `labs/05-sansa/experiments.yaml`; `git switch -c lab/05-sansa`.
- Plan about 5 hours, plus box runs.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method sansa --dataset movielens-25m --best
```

### 1.2 Read the code

Open `src/recbench/methods/linear.py::SANSA`.

**a.** `sansa_weights_per_item` is turned into the package's `weight_matrix_density`. How, and why does recbench
prefer weights per item?

??? success "Answer"
    `density = per_item / n_items`: the share of all item pairs that may have a weight. A density fixed in advance
    means very different things on catalogs of 13,000 and 158,000 items. Weights per item mean the same everywhere,
    so one search range suits all five datasets.

**b.** How does SANSA score a user, compared with EASE?

??? success "Answer"
    `(x @ self.w1) @ self.w2`: two sparse products instead of one dense one. The package factorises EASE's matrix
    approximately as $W \approx W_1 W_2$ with both factors sparse, so a user's history times $W_1$ times $W_2$ gives
    nearly EASE's scores, for every item, cap-free.

**c.** Why can SANSA not run in the same process as PyTorch on a Mac, and how does recbench avoid it?

??? success "Answer"
    Homebrew's SuiteSparse and PyTorch load two different OpenMP runtimes, and once CHOLMOD runs in parallel the
    process aborts. recbench starts each run in a child process that imports only the method's own module
    (`RECBENCH_METHOD_MODULES`, see `src/recbench/methods/__init__.py`), and `lab.setup()` does the same for notebooks.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `sansa_lambda` | the ridge penalty, as EASE's λ | 1 to 20,000 (log scale) |
| `sansa_weights_per_item` | how many weights each item keeps | 20 to 500 (log scale) |
| `decay_half_life_days`, `train_window_days` | recency | |
| `sansa_factorizer` | `cholmod` (exact sparse Cholesky) or `icf` (incomplete, cheaper) | not searched (cholmod) |

**a.** Accuracy against time. Before recbench used weights per item, a laptop run on MovieLens' validation fold
measured this:

| Weights per item | NDCG@10 | Training time |
|---|---|---|
| about 28 | 0.096 | 12 s |
| about 277 | 0.171 | 104 s |
| about 830 | 0.173 | 496 s |
| EASE (exact, 20,000 items) | 0.190 | 75 s |

Plot NDCG@10 against seconds in your notebook. Where is the "knee", the point after which more weights cost much
more time for little accuracy?

??? success "What to look for"
    The jump from 28 to 277 weights is large (0.096 to 0.171); from 277 to 830 almost nothing is gained for 5 times
    the time. On MovieLens, which fits under EASE's cap, exact EASE is both better and faster. SANSA earns its place
    where the cap cuts, on H&M and RetailRocket.

**b.** Sweep λ at the baseline's best weights per item:

```bash
python -m recbench.lab sweep --method sansa --dataset movielens-25m --param sansa_lambda --values 10,100,500,2000,10000
```

Does the best λ match EASE's on the same dataset (lab 3)? It should be close, since both solve the same problem.

## Level 3: data tricks

### 3.1 The other factoriser

The `sansa` package logs a hint: "You may want to try ICFGramianFactorizer instead of CHOLMODGramianFactorizer
(requires less memory and may be faster)." `icf` in `experiments.yaml` fixes `sansa_factorizer: icf` and searches the
rest as the baseline did. Run it on the box and compare accuracy **and** the `train s` column.

### 3.2 Leave out very rare items

As in lab 4: add `sansa_min_count`. Items with fewer interactions are left out of the Gram matrix, so they get no
weights.

??? tip "Hint"
    Zero their columns before fitting: multiply the matrix by a diagonal 0/1 matrix, `x @ sp.diags(keep)`, then
    `x.eliminate_zeros()`. Scoring keeps using the full `self.seen`.

??? success "Solution"
    In `SANSA.fit`, after `self.seen` is built:

    ```python
    x = self.seen
    min_count = int(cfg.get("sansa_min_count", 1))
    if min_count > 1:  # rare items are left out of the Gram matrix, so they get no weights
        x = (x @ sp.diags((data.item_pop >= min_count).astype(np.float32))).tocsr()
        x.eliminate_zeros()
    ```

    and fit on `x`: `model.fit(x)`.

## Level 4: one change from the literature

### The idea: EDLAE's penalty, as made-up users

Lab 3 gave each item its own penalty, $\lambda + \frac{p}{1-p} G_{ii}$. The `sansa` package only accepts one number,
`l2`. But adding to the diagonal of $X^\top X$ can be done *through the data*. Append one extra row per item, with a
single value $\sqrt{c_i}$ in that item's column. That row adds $c_i$ to item i's diagonal and nothing else:

$$\begin{pmatrix} X \\ D \end{pmatrix}^{\!\top}\! \begin{pmatrix} X \\ D \end{pmatrix} = X^\top X + D^2, \qquad D = \operatorname{diag}\!\left(\sqrt{\tfrac{p}{1-p} G_{ii}}\right)$$

"A penalty is the same as fake data" is a useful idea in itself: ridge regression can always be written this way.

### Write the variant

Add `sansa_dropout` (p, default 0: off).

??? tip "Hint"
    $G_{ii}$ is the sum of the squared values in item i's column: `x.multiply(x).sum(axis=0)`. Stack the pseudo-user
    rows under the matrix with `sp.vstack`. Only fit on the stacked matrix; score with the real one.

??? success "Solution"
    ```python
    p = float(cfg.get("sansa_dropout", 0.0))
    if p > 0:  # EDLAE (Steck 2020): p / (1 - p) * G_ii on the diagonal, added as one pseudo-user row per item
        g_diag = np.asarray(x.multiply(x).sum(axis=0)).ravel()
        x = sp.vstack([x, sp.diags(np.sqrt(p / (1.0 - p) * g_diag).astype(np.float32))]).tocsr()
    ```

    placed before `model.fit(x)`.

### Test it

SANSA's tests must run in a child process (see Before you start), so this test file is not a copy of the template.
It runs the same checks inside a small script:

??? success "Solution"
    `tests/test_sansa_variant.py`:

    ```python
    SCRIPT = """
    import json, sys
    from recbench.data import TrainView
    from recbench.evaluation import EvalSplit
    from recbench.lab import checks

    view, split = TrainView(sys.argv[1]), EvalSplit(sys.argv[1])
    small = {"sansa_weights_per_item": 10}
    variant = {**small, "sansa_dropout": 0.5}
    checks.check_scores(checks.fit("sansa", view, **variant), view)
    checks.check_differs("sansa", view, small, variant)
    checks.check_same_scores("sansa", view, small, {**small, "sansa_dropout": 0.0, "sansa_min_count": 1})
    checks.check_deterministic("sansa", view, **variant)
    score = checks.toy_ndcg("sansa", view, split, **variant)
    learns = score > checks.toy_ndcg("random", view, split) and score > 0.8 * checks.toy_ndcg("sansa", view, split, **small)
    print(json.dumps({"learns": bool(learns)}))
    """


    def test_the_sansa_variants_in_their_own_process(toy_split):
        pytest.importorskip("sansa")
        env = {**os.environ, "RECBENCH_METHOD_MODULES": "linear,baselines"}
        out = subprocess.run([sys.executable, "-c", SCRIPT, str(toy_split)], env=env, capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-3000:]
        assert json.loads(out.stdout.strip().splitlines()[-1])["learns"]
    ```

    with `import json, os, subprocess, sys` and `import pytest` at the top.

### Run and compare

Uncomment `edlae` (it re-tunes λ too) and run it on the box. Compare with SANSA's baseline, and with lab 3's EDLAE
result for EASE:

```bash
python -m recbench.compare sansa sansa:edlae --segments
python -m recbench.compare ease:edlae sansa:edlae --datasets hm,retailrocket
```

??? success "What to look for"
    Two questions. Does EDLAE help SANSA the way it helped (or did not help) EASE? And on the datasets where EASE's cap
    cuts the catalog, does cap-free SANSA with EDLAE overtake EASE?

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| icf | | | | | | |
| min-count | | | | | | |
| edlae | | | | | | |

## Further reading

- Spišák et al. (2023), *Scalable approximate nonsymmetric autoencoder for collaborative filtering*, RecSys 2023: SANSA.
- Steck (2020), *Autoencoders that don't overfit towards the identity*, NeurIPS 2020.
