# SANSA

> EASE without the item cap: the same "predict each item from all the others" model, computed approximately
> with sparse matrices so that catalogs of hundreds of thousands of items fit in memory.

!!! abstract "In plain words"
    EASE learns "how much does owning item A predict item B" for every pair of items, which needs a huge square table.
    SANSA learns the same thing but keeps only the important entries, the way a map keeps the main roads and drops
    the footpaths. The model is almost as accurate, fits catalogs of hundreds of thousands of items, and still says
    exactly which of your items led to each recommendation.

--8<-- "generated/methods/sansa.md"

!!! example "Improve it yourself"
    [Lab 5](../../labs/05-sansa.md) takes you through SANSA in four levels: reproduce its baseline, understand
    every setting, trade accuracy against time, and add EDLAE's penalty as made-up users. The [lab scoreboard](../../labs/scoreboard.md) tracks your results.

!!! tip "When to use it"
    - When EASE works well but your catalog is too large for its dense item × item matrix. In recbench, EASE
      keeps at most 20,000–60,000 items; SANSA keeps them all.
    - When you want a compact model to serve: SANSA's weights are sparse.

!!! warning "When not to"
    - On small catalogs, where exact EASE is simpler and slightly more accurate.
    - If you cannot install SuiteSparse (SANSA needs it for sparse factorisations).

## 1. Intuition

[EASE](ease.md) finds the best item × item weight matrix in one step: invert the item co-occurrence matrix
(plus λ on the diagonal). With 200,000 items that matrix has 40 billion entries, too big for memory, so EASE
keeps only the most popular items.

Most item pairs never co-occur, and the important structure of the inverse is concentrated in relatively few
entries. SANSA (Scalable Approximate NonSymmetric Autoencoder) computes a sparse approximation of the inverse:

1. a **sparse factorisation** of the co-occurrence matrix, $LDL^\top$, keeping only the largest entries. This
   splits the matrix into a triangular matrix $L$, a diagonal $D$ and $L$ again, pieces that are much cheaper to
   invert than the whole matrix (like solving a system of equations one unknown at a time);
2. a sparse **approximate inverse** of $L$.

The result is two sparse matrices whose product behaves like EASE's weights.

## 2. A tiny worked example

On a small catalog the approximation can be made exact by keeping every entry (`sansa_density = 1`). Then
SANSA's scores for unseen items rank exactly like EASE's: `tests/test_new_methods.py` checks this, with a
correlation above 0.99.

The only difference is on items the user already has. EASE sets the diagonal to 0, while SANSA's diagonal is
-1, so seen items differ by exactly the user's own interaction. The evaluator masks those items anyway.

On a large catalog, the number of weights kept controls the trade-off. At a density of 0.001, the weights keep
about 0.1% of the entries a dense matrix would have: a 100,000-item catalog keeps about 10 million weights
instead of 10 billion.

Too few weights cost accuracy. On the MovieLens quick tier (32,000 items), SANSA scored NDCG@10 0.096 with about
32 weights per item, 0.171 with 320, and 0.173 with 960, against 0.190 for EASE. The time grew from 12 to 104 to
496 seconds on a laptop. The quick tier therefore searches the number of weights **per item**
(`sansa_weights_per_item`, 20 to 500), which means the same thing on a small and a large catalog.

## 3. How it works

1. Build the user × item matrix $X$ (binary, or time-decayed weights).
2. Factorise $P(X^\top X + \lambda I)P^\top \approx L D L^\top$ sparsely, with a fill-reducing reordering $P$
   (CHOLMOD, or incomplete Cholesky for very large catalogs).
3. Compute a sparse approximate inverse of $L$.
4. Assemble two sparse factors $W_1, W_2$ with $W_1 W_2 \approx -(X^\top X + \lambda I)^{-1} / \text{diag}$.
5. Score: $x_u W_1 W_2$.

```mermaid
flowchart LR
    X[user x item matrix] --> G[sparse Gram matrix + lambda]
    G --> F[sparse LDL^T factorisation]
    F --> I[sparse approximate inverse of L]
    I --> W[two sparse factors W1, W2]
    W --> S[score = x_u W1 W2]
```

## 4. The math, symbol by symbol

EASE's closed form, which SANSA approximates:

$$
B = I - P \,\mathrm{diagMat}(1 / \mathrm{diag}(P)), \qquad P = (X^\top X + \lambda I)^{-1}
$$

| Symbol | Meaning | Shape |
|---|---|---|
| $X$ | user × item interactions | users × items |
| $\lambda$ | L2 regularisation (`sansa_lambda`) | > 0 |
| $P$ | the inverse SANSA approximates sparsely | items × items |
| $L, D$ | sparse lower-triangular factor and diagonal of the factorisation | items × items |
| density | share of entries kept (`sansa_density`) | about 1e-5–1e-2 |

## 5. Training and inference

- **Training:** a sparse factorisation and a sparse inverse: minutes for very large catalogs, CPU only.
- **Inference:** two sparse matrix products per batch of users.
- **Hardware:** CPU, with SuiteSparse installed (`brew install suite-sparse` or `apt install libsuitesparse-dev`).

## 6. Hyperparameters

| Name in recbench config | What it does | Default | Searched over |
|---|---|---|---|
| `sansa_lambda` | L2 regularisation, as EASE's λ | 500 | 1–20000 (log) |
| `sansa_weights_per_item` | weights kept per item (sets the density as this number ÷ the number of items) | not set | 20–500 (log) |
| `sansa_density` | share of all item pairs kept as weights (used when `sansa_weights_per_item` is not set) | 0.001 | — |
| `sansa_factorizer` | `cholmod` (exact sparse) or `icf` (incomplete, for huge catalogs) | cholmod | — |

## 7. In recbench

- Code: `src/recbench/methods/linear.py::SANSA`, which wraps the `sansa` package (installed with the
  `sansa` extra).
- `tests/test_new_methods.py` checks that it ranks unseen items like exact EASE on a small catalog.

!!! info "Fidelity: faithful"
    It uses the authors' package with its default factorisation and inverter settings.

## 8. Results in this benchmark

--8<-- "generated/methods/sansa-results.md"

## 9. Strengths and weaknesses

- **Strengths:** EASE-quality models on catalogs EASE cannot handle; sparse weights; fast scoring.
- **Weaknesses:** an extra system dependency (SuiteSparse); a second setting (density) to tune;
  explanations are less direct than EASE's, because the weights are a product of two factors.

## 10. Common pitfalls

- **A density that is too low:** too few weights survive, and accuracy drops sharply. On MovieLens' validation
  fold the package's default density (about 28 weights per item) scored 0.096 NDCG@10, against 0.171 with about
  277. That is why recbench searches `sansa_weights_per_item` (20 to 500).
- **One slow setting can use up a tuning job.** A SANSA fit cannot stop early. In the first bake-off, the setting
  with all history and 249 weights per item ran for 85 minutes on RetailRocket and 140 on H&M before the 3-hour
  cap stopped it. SANSA was therefore judged there on its first two settings, which were its weakest on every
  other dataset.
- **Comparing SANSA to a capped EASE without saying so.** The difference may come from the extra items, not
  the method.
- **Running it in a process that has loaded PyTorch, on macOS.** SuiteSparse and PyTorch then bring two OpenMP
  runtimes and the process aborts. recbench's runs import only SANSA's module, so this only bites in a notebook:
  fit SANSA before importing PyTorch there.

## 11. Check your understanding

??? question "Why can't EASE simply use all 235,000 RetailRocket items?"
    Its dense item × item matrix would need 235,000² × 4 bytes ≈ 220 GB, several times that while it is being
    inverted.

??? question "What does SANSA give up compared with exact EASE?"
    A little accuracy, from approximating the inverse with a limited number of non-zero entries.

??? question "Why does the quick tier search `sansa_weights_per_item` instead of `sansa_density`?"
    Density is a share of all item pairs, so the same value keeps very different amounts per item on a 32,000-item
    and a 158,000-item catalog: 320 versus 1,580 weights per item at 0.01, with very different costs. A number of
    weights per item means the same thing on every catalog.

## 12. Further reading

- Spišák, Bartyzal, Hoskovec, Peska, Tůma (2023). *Scalable Approximate NonSymmetric Autoencoder for
  Collaborative Filtering.* RecSys. [Code](https://github.com/glami/sansa)
- [EASE](ease.md), [SLIM](slim.md).
