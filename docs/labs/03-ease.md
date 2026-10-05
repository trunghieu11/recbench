# Lab 3 · EASE: a whole model in one formula

!!! abstract "In plain words"
    EASE learns, for every pair of items, how much having one predicts the other, so that each item's column is
    rebuilt as well as possible from the other items. There is no training loop: the answer is one matrix inverse.
    It is a very strong simple method. In the first bake-off it had the best mean rank of all 23 methods: it was
    first, or tied with the first, on four of the five datasets. This week you will:
    - open its weight matrix;
    - see what λ really does;
    - deal with its one big weakness, the item cap;
    - implement **EDLAE**, a 2020 refinement by EASE's own author.

## Your starting point

--8<-- "generated/lab/ease.md"

## Before you start

- Read the [linear algebra primer, part 1](primer-linear-algebra.md) (it builds EASE on 3 items by hand) and sections
  1 to 5 of the [EASE page](../dictionary/algorithms/ease.md).
- Open `labs/03-ease/ease.ipynb` and `labs/03-ease/experiments.yaml`; `git switch -c lab/03-ease`.
- Plan about 6 hours, plus box runs. A MovieLens fit takes about a minute: the 20,000 × 20,000 inverse.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method ease --dataset movielens-25m --best
```

The validation NDCG@10 must equal the "Validation" column above.

### 1.2 Read the code

Open `src/recbench/methods/baselines.py` and find `EASE.fit`.

**a.** Which items does EASE keep, and why does it keep only some?

??? success "Answer"
    The `ease_max_items` most recently popular items (`dense_item_cap`; the lab keeps 20,000). EASE needs three dense
    item × item float32 matrices: 20,000 items make 400 million cells of 4 bytes, 1.6 GB each. Items outside the cap
    are never recommended (their score is −∞). `fit_info["item_cap_coverage"]` reports the share of interactions the
    kept items cover.

**b.** Match the code to the formula $P = (X^\top X + \lambda I)^{-1}$, $B_{ij} = -P_{ij}/P_{jj}$, $B_{jj} = 0$:
which line does what?

??? success "Answer"
    `gram = x.T @ x` is $X^\top X$; `gram[np.diag_indices_from(gram)] += self.lam` adds λI; `np.linalg.inv` gives
    P; `inverse /= -diag[None, :]` divides each column j by $-P_{jj}$; `inverse[np.diag_indices_from(inverse)] = 0`
    sets the diagonal to 0. The result is stored as `self.weights`.

**c.** Scoring is `x @ self.weights`. Can a score be negative? What would that mean?

??? success "Answer"
    Yes: EASE's weights can be negative. A negative $B_{ij}$ means "having i makes j *less* likely", typically for
    substitutes (two versions of the same thing) or for items from very different audiences. SLIM (lab 4) forbids
    negative weights; that difference is lab 4's Level 4.

### 1.3 Look inside

```python
model = lab.fit("ease", data, **best)
row = model.weights[model.position[toy_story]]   # Toy Story's weights to the kept items
```

The notebook gives the exact cells. Find the five largest and the five most negative weights of a film you know.

??? question "Do the most negative weights make sense to you?"
    Often they are films from a very different audience: for example a children's film against a violent thriller.
    Sometimes they are near-duplicates, such as another edition of the same film, which "replace" each other. EASE
    learned both kinds of relation from co-occurrence alone.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `ease_lambda` | the ridge penalty λ: larger means smaller, more cautious weights | 1 to 20,000 (log scale) |
| `decay_half_life_days` | recent interactions weigh more | none, 30, 90, 365 |
| `train_window_days` | learn only from the last N days | none, 30, 90, 365 |
| `ease_max_items` | the item cap | not searched (20,000 in the lab) |

**a.** Predict, then sweep λ across four orders of magnitude:

```bash
python -m recbench.lab sweep --method ease --dataset movielens-25m --param ease_lambda --values 1,10,100,1000,10000
```

??? success "What to look for"
    A hump. With a tiny λ, EASE fits the training co-occurrences too closely (overfitting): it memorises rare pairs.
    With a huge λ, the penalty dominates. The weights become proportional to plain co-occurrence counts, and those
    favour popular items: EASE drifts towards recommending what is popular. Also note the scale: the best λ grows with
    the number of users, because $X^\top X$ grows with them. That is why the bake-off re-checks λ when a winner moves
    to the full data (`confirm` in `configs/tuning/quick.yaml`).

**b.** How much of MovieLens does the cap keep? Fit once and read `model.fit_info["item_cap_coverage"]`. Then think
about H&M (70,000 items) and RetailRocket (158,000).

??? success "What to look for"
    On MovieLens the 20,000 kept items cover about 99.9% of the interactions: the dropped films are very rare. On the
    big shops a much larger share is cut, so the items EASE may never recommend matter there. Level 3.2 changes which
    items it keeps.

## Level 3: data tricks

### 3.1 Is λ all you need?

`no-time-knobs` searches λ alone, without decay or a training window. Run it on the box and compare:

```bash
python -m recbench.compare ease ease:no-time-knobs
```

??? success "What to look for"
    Compare with ItemKNN's result in lab 1, where removing recency made it clearly worse on MovieLens. Is EASE as
    sensitive? A method that captures the data's structure well may need recency less, or more.

### 3.2 Which items does the cap keep?

Today the cap keeps the *recently* popular items, with all-time popularity breaking ties. Add `ease_keep`
(`recent` by default, or `alltime`) to keep the all-time most popular items instead.

??? tip "Hint"
    `np.lexsort(keys)` sorts by its **last** key first. Today's line is
    `np.lexsort((-data.item_pop[warm], -data.item_recent_pop[warm]))`: recent popularity first.

??? success "Solution"
    In `EASE.fit`:

    ```python
    if len(warm) > cap:
        if cfg.get("ease_keep", "recent") == "alltime":  # all-time popularity first, recent popularity breaks ties
            order = np.lexsort((-data.item_recent_pop[warm], -data.item_pop[warm]))
        else:  # recent popularity first (np.lexsort sorts by its LAST key first)
            order = np.lexsort((-data.item_pop[warm], -data.item_recent_pop[warm]))
        warm = np.sort(warm[order[:cap]])
    ```

    A check that it keeps the right items:

    ```python
    def test_alltime_keeps_the_most_popular_items(toy):
        view, _ = toy
        model = checks.fit("ease", view, ease_max_items=10, ease_keep="alltime")
        warm = np.flatnonzero(view.item_pop > 0)
        warm = warm[warm > 0]
        top = warm[np.lexsort((-view.item_recent_pop[warm], -view.item_pop[warm]))][:10]
        assert sorted(model.kept.tolist()) == sorted(top.tolist())
    ```

??? question "On which datasets can `ease_keep` change anything?"
    Only where the catalog is larger than the cap: not Steam (13,000 items), barely MovieLens (the cut films are very
    rare), more on Last.fm, H&M and RetailRocket.

## Level 4: one change from the literature

### The idea: EDLAE (Steck 2020)

Steck asked why linear autoencoders like EASE work and found that training them with **dropout** is equivalent to a
different penalty. Dropout means hiding a random share p of each user's items while learning to predict them. The
penalty is no longer the same λ for every item. Item i gets

$$\Lambda_{ii} = \lambda + \frac{p}{1 - p}\, G_{ii}$$

where $G_{ii}$ is item i's own co-occurrence, its popularity. Popular items are penalised more, in proportion to how
often they appear. EASE's closed form stays exactly the same with this diagonal; the paper calls it **EDLAE**
(emphasised denoising linear autoencoder). For p = 0 it is EASE.

### Write the variant

Add `ease_variant` (`ease` or `edlae`) and `ease_dropout` (p).

??? tip "Hint"
    Compute the penalty *before* you add anything to `gram`'s diagonal: `np.diag(gram)` is $G_{ii}$. The penalty can
    be a number (EASE) or an array (EDLAE); adding either to the diagonal works the same way. Do not forget the PyTorch
    branch used on GPUs.

??? success "Solution"
    In `EASE.fit`, after `gram` is computed:

    ```python
    penalty = self.lam  # the same lambda for every item
    if cfg.get("ease_variant", "ease") == "edlae":  # Steck 2020: dropout p acts as a per-item penalty
        p = float(cfg.get("ease_dropout", 0.5))
        penalty = self.lam + p / (1.0 - p) * np.diag(gram).astype(np.float64)
    if self.device is None:
        gram[np.diag_indices_from(gram)] += penalty
        ...
    else:
        ...
        g.diagonal().add_(torch.as_tensor(penalty, dtype=g.dtype, device=g.device))
    ```

### Test it

??? success "Solution"
    ```python
    METHOD = "ease"
    VARIANT = {"ease_variant": "edlae", "ease_dropout": 0.5}
    DEFAULT = {"ease_variant": "ease", "ease_keep": "recent"}


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
        """With dropout p = 0 the EDLAE penalty is lambda: exactly EASE."""
        view, _ = toy
        checks.check_same_scores(METHOD, view, {}, {"ease_variant": "edlae", "ease_dropout": 0.0})
    ```

    For a stronger check, build a 4 × 3 matrix X by hand, compute the EDLAE weights with numpy from the formula, and
    compare them with `model.weights` on a tiny TrainView. The [primer](primer-linear-algebra.md#putting-it-together-ease-in-three-items)
    has the code for the EASE part.

### Run and compare

Uncomment `edlae` in `experiments.yaml`. It searches p and re-tunes λ, because the dropout term adds penalty on top of
λ; every other setting stays at the baseline's best. Run it on the box, then:

```bash
python -m recbench.compare ease ease:edlae --segments
```

??? success "What to look for"
    EDLAE penalises popular items' weights more, so look at the **taste** segments (the popularity of the users' test
    items) and at the `coverage A -> B` column. A gain concentrated on users with niche tastes, with higher coverage,
    is the mechanism at work. If accuracy is unchanged but coverage rises, record that too: it is a real benefit
    for a shop.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| no-time-knobs | | | | | | |
| keep-alltime-popular | | | | | | |
| edlae | | | | | | |

## Further reading

- Steck (2019), [Embarrassingly shallow autoencoders for sparse data](https://arxiv.org/abs/1905.03375), WWW: EASE.
- Steck (2020), *Autoencoders that don't overfit towards the identity*, NeurIPS 2020: dropout, DLAE and EDLAE.
- Steck & Liang (2021), *Negative interactions for improved collaborative filtering: don't go deeper, go higher*, RecSys 2021: higher-order EASE, a possible next step.
