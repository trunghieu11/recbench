# Lab 8 · iALS: user and item vectors, learned in turns

!!! abstract "In plain words"
    iALS gives every user and every item a short vector of numbers, and scores a pair by their dot product. It learns
    the vectors by **alternating**: fix the items and solve every user exactly (a ridge regression each), then fix
    the users and solve every item, and repeat. Every empty cell of the user × item table counts as a weak "probably
    not interested". Every interaction counts as a strong "interested", with a **confidence** that grows with how
    often it happened. This week is about the balance between those two signals, and about two ways to change what
    "confidence" means.

## Your starting point

--8<-- "generated/lab/ials.md"

## Before you start

- Read the [optimisation primer](primer-optimisation.md#alternating-least-squares-ials) and sections 1 to 5 of the
  [iALS page](../dictionary/algorithms/ials.md).
- Open `labs/08-ials/ials.ipynb` and `labs/08-ials/experiments.yaml`; `git switch -c lab/08-ials`.
- Plan about 5 hours, plus box runs. iALS starts from random vectors, so the lab tests it with 3 seeds and averages
  them.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method ials --dataset movielens-25m --best
```

??? question "The number should match the baseline's best trial. But iALS starts from random vectors: why does it match?"
    The tuning trial and your run both use seed 42, and for a fixed seed `implicit`'s ALS gives the same vectors. Change
    `--set seed=43` and the number moves a little. The size of that wobble is why the lab tests random methods with
    three seeds.

### 1.2 Read the code

Open `src/recbench/methods/implicit_mf.py::IALS`. It is short: the work happens in the `implicit` library.

**a.** What goes into `model.fit`, and how does it become a confidence?

??? success "Answer"
    `data.weighted_matrix(decay, binary=False)`: (decayed) interaction **counts**, not 0/1. `implicit` turns each
    value r into a confidence for the "interested" target that grows with $\alpha r$, while every empty cell gets
    confidence 1 for the "not interested" target. The paper writes it $c = 1 + \alpha r$. `ials_alpha` sets how much
    more an interaction counts than an empty cell.

**b.** What does `ials_reg` penalise, and why does the iALS literature tie it to `dim`?

??? success "Answer"
    The size of the user and item vectors (L2). More dimensions give the model more freedom to memorise. Rendle et al.
    (2022) showed that iALS with large vectors becomes very strong *if* the regularisation grows with them. Many papers
    had compared against small, poorly regularised iALS and called it weak.

**c.** How is a user scored, and can iALS score a user who was not in the training data?

??? success "Answer"
    `self.user_factors[users] @ self.item_factors.T`: a dot product with every item. A new user has no learned vector,
    so no. That is the difference from the history-based methods of labs 1 to 7.

### 1.3 Look inside

In the notebook, fit the baseline's setting, then use `lab.neighbours(model, data, "Toy Story (1995)")`. For iALS,
neighbours are the items whose vectors point in the most similar direction (cosine).

??? question "Compare with EASE's neighbours of the same film (lab 3). Which list looks more 'thematic', and which more 'co-watched'?"
    There is no single right answer: write down what you see. Vector methods often group films by broad taste
    (genre, era), while EASE and ItemKNN follow what was actually watched together.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `dim` | the vectors' length | 64, 128, 256, 512 |
| `ials_reg` | the L2 penalty on the vectors | 0.001 to 100 (log scale) |
| `ials_alpha` | how much an interaction outweighs an empty cell | 0.3 to 100 (log scale) |
| `ials_iterations` | rounds of alternating | 10, 20 |
| `decay_half_life_days`, `train_window_days` | recency | |

**a.** Predict, then check, how `dim` and `ials_reg` interact. Sweep `ials_reg` at `dim=64` and again at `dim=512`:

```bash
python -m recbench.lab sweep --method ials --dataset movielens-25m --param ials_reg --values 0.01,0.1,1,10,100 --set dim=64
python -m recbench.lab sweep --method ials --dataset movielens-25m --param ials_reg --values 0.01,0.1,1,10,100 --set dim=512
```

??? success "What to look for"
    With 512 dimensions, a weak penalty overfits and the best `ials_reg` is larger than with 64. That is Rendle et al.'s
    point in miniature. Plot both curves in one chart in the notebook.

**b.** Sweep `ials_alpha` (0.3, 3, 30, 100). On which dataset would you expect a larger best α: MovieLens, where every
interaction is a single rating, or Last.fm, where counts run into the thousands?

## Level 3: data tricks

### 3.1 How much does recency matter?

`no-time-knobs` searches the five model settings alone. Run it on the box and compare.

### 3.2 Log-scaled confidence (Hu, Koren & Volinsky 2008)

The original iALS paper offers two confidences: linear, $c = 1 + \alpha r$, and logarithmic,
$c = 1 + \alpha \log(1 + r/\varepsilon)$. With the linear one, a Last.fm user's artist played 5,000 times counts 5,000
times more than one played once. The log version grows much more slowly. Add `ials_confidence` (`linear` or `log`)
and `ials_eps` (ε).

??? tip "Hint"
    `implicit` builds the confidence from each stored value r and α itself. So pass $\log(1 + r/\varepsilon)$ as the
    values: `counts.data = np.log1p(counts.data / eps)`. Copy the matrix first, because `weighted_matrix` may return a
    cached one.

??? success "Solution"
    In `IALS.fit`:

    ```python
    counts = data.weighted_matrix(cfg.get("decay_half_life_days"), binary=False).tocsr().astype(np.float32)
    if cfg.get("ials_confidence", "linear") == "log":  # Hu, Koren & Volinsky 2008: c = 1 + alpha * log(1 + r / eps)
        counts = counts.copy()
        counts.data = np.log1p(counts.data / float(cfg.get("ials_eps", 1.0))).astype(np.float32)
    model.fit(counts, show_progress=False)
    ```

??? question "On MovieLens without decay, every r is 1. What does the log confidence do there?"
    $\log(1 + 1/\varepsilon)$ is the same number for every interaction: it only rescales α. So no new information, and
    the tuner's α absorbs it. The log version can only matter where counts vary: with decay, and on Last.fm, H&M,
    Steam and RetailRocket.

## Level 4: one change from the literature

### The idea: popularity-aware confidence (after Saito et al. 2020)

An interaction with a hugely popular item says little about a user's personal taste: everybody has it. Saito et al.
treat each interaction as observed with a **propensity** that grows with the item's popularity, and weigh it by the
inverse of that propensity (inverse propensity scoring, IPS). The "Rel-MF" loss in their paper is more involved. Its
core is easy to borrow: multiply item j's confidence by

$$w_j = \left(\frac{\text{pop}_j}{\overline{\text{pop}}}\right)^{-\gamma}, \quad \text{clipped to } [0.1, 10]$$

so interactions with rarer items count more. The clipping keeps a handful of very rare items from getting huge
weights.

### Write the variant

Add `ials_ips_power` = γ (0: off).

??? tip "Hint"
    Scaling the columns of the counts matrix scales each item's confidences: `counts @ sp.diags(w)`. Do it after the
    log transform, so both can be combined.

??? success "Solution"
    ```python
    power = float(cfg.get("ials_ips_power", 0.0))
    if power:  # inverse propensity (after Saito et al. 2020): interactions with less popular items count more
        pop = np.maximum(data.item_pop, 1).astype(np.float64)
        weight = np.clip(np.power(pop / pop.mean(), -power), 0.1, 10.0)
        counts = (counts @ sp.diags(weight.astype(np.float32))).tocsr()
    ```

    with `import scipy.sparse as sp` at the top of the module.

### Test it

??? success "Solution"
    ```python
    METHOD = "ials"
    VARIANT = {"ials_confidence": "log", "ials_eps": 1.0}
    DEFAULT = {"ials_confidence": "linear", "ials_ips_power": 0.0}


    def test_inverse_propensity_changes_the_model(toy):
        view, _ = toy
        checks.check_differs(METHOD, view, {}, {"ials_ips_power": 0.5})
    ```

    `implicit`'s ALS is deterministic for a fixed seed, so the template's determinism test can stay.

### Run and compare

Uncomment `ips` (it re-tunes α, since the weights change the confidences' scale), run it on the box, then:

```bash
python -m recbench.compare ials ials:ips --segments
```

??? success "What to look for"
    IPS trades some accuracy on popular items for more on niche ones. Look at the **items** table, recall for the head
    against the long tail, and at coverage. If overall NDCG@10 is flat but long-tail recall and coverage rise, the change
    did exactly what it promises.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| no-time-knobs | | | | | | |
| log-confidence | | | | | | |
| ips | | | | | | |

## Further reading

- Hu, Koren & Volinsky (2008), *Collaborative filtering for implicit feedback datasets*, ICDM 2008: iALS and its two confidences.
- Rendle, Krichene, Zhang & Koren (2022), *Revisiting the performance of iALS on item recommendation benchmarks*, RecSys 2022.
- Saito, Yaginuma, Nishino, Sakata & Nakata (2020), *Unbiased recommender learning from missing-not-at-random implicit feedback*, WSDM 2020: Rel-MF.
