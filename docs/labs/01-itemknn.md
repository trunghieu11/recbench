# Lab 1 · ItemKNN: people who liked this also liked

!!! abstract "In plain words"
    ItemKNN calls two items similar when the same users interacted with both. It recommends the items most similar
    to what you already have. Training is counting co-occurrences, so it takes about a second on MovieLens. This
    week you reproduce its baseline, learn what each of its six settings does, find out how much recency matters
    (a lot), and implement a classic refinement: the **asymmetric cosine**. Along the way you will find a real
    weakness in how the training window treats each user's history.

## Your starting point

--8<-- "generated/lab/itemknn.md"

## Before you start

- Read sections 1 to 5 of the [ItemKNN page](../dictionary/algorithms/itemknn.md) and the
  [sparse matrices primer](primer-sparse-matrices.md).
- Open the notebook `labs/01-itemknn/itemknn.ipynb` ([on GitHub](https://github.com/trunghieu11/recbench/blob/main/labs/01-itemknn/itemknn.ipynb))
  and run its setup cell, then look at `labs/01-itemknn/experiments.yaml`.
- Create the branch: `git switch -c lab/01-itemknn`.
- Plan about 5 hours. Experiments on all five datasets run on the rented box
  ([run experiments](../handbook/run-experiments.md#run-it-on-a-rented-box)).

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method itemknn --dataset movielens-25m --best
```

!!! success "You should see"
    ```text
    itemknn on movielens-25m, validation fold (854 users), decay_half_life_days=365, knn_neighbors=15, knn_shrink=100, knn_weighting=tfidf, train_window_days=365
      NDCG@10   0.1982  [0.1830, 0.2129]
      Recall@10 0.1875   coverage@10 0.023   train 1.1 s   scoring 0.02 s per 1,000 users
      (the baseline's best trial scored 0.1982 on the same users)
    ```

The notebook's Level 1 cells do the same with `lab.fit` and `lab.evaluate`, and keep the model in memory.

??? question "Why is it exactly the same number, not just close?"
    ItemKNN has no randomness: the same data and settings always give the same similarity matrix. The validation users
    are a fixed, seeded sample, so the score is identical. For a random method (iALS, BPR-MF) you would see small
    differences between seeds.

### 1.2 Read the code

Open `src/recbench/methods/baselines.py` and find `ItemKNN`, `weight_users` and `item_cosine_topk`. Answer from the
code:

**a.** What matrix goes into `item_cosine_topk`, and what does each of its rows and columns stand for?

??? success "Answer"
    `weight_users(self.seen, ...)`: the user × item matrix, with time-decayed weights when `decay_half_life_days` is set
    and re-weighted by user activity when `knn_weighting` is `tfidf` or `bm25`. Rows are users, columns items.
    `item_cosine_topk` turns it into an item × item similarity matrix.

**b.** In `val = co.data[lo:hi] / (norms[item] * norms[idx] + shrink)`, what is `co.data`, and what does `shrink`
do to a pair of items seen together by 1 user compared with a pair seen together by 100 users?

??? success "Answer"
    `co` is a block of $X^\top X$: `co.data` holds the co-occurrence of the item with each other item (how many users
    share them, weighted). Without `shrink`, a pair of rare items seen together by a single user can reach cosine 1.0.
    Adding `shrink` to the denominator pulls such small-evidence similarities down a lot, and barely changes a
    similarity backed by 100 users.

**c.** Why does the code drop `idx != item`?

??? success "Answer"
    An item is always most similar to itself. Keeping it would put the diagonal into every score and fill the top-k
    neighbour lists with the item itself.

**d.** How is a user scored, and which of their interactions count?

??? tip "Hint"
    Look at `score_users`, then at where `self.seen` comes from in `fit`, then at `run_single` in
    `src/recbench/runner.py`: which data does `fit` receive?

??? success "Answer"
    `self.seen[users] @ self.sim`: the sum of the similarity rows of the items in the user's history. But `self.seen` is
    built from the data `fit` receives, and the runner first restricts it with `train_window_days`. Each user keeps
    only the events inside the window plus their last 10 (`train_window_keep_last`). So the window does not only decide
    what the similarities are learned from; it also **trims each user's profile** at scoring time. Keep this in mind
    for Level 3.

### 1.3 Look inside

In the notebook, find the neighbours of a film you know:

```python
lab.neighbours(model, data, "Toy Story (1995)")
```

Then try a few films of different popularity, for example a cult film with few ratings.

??? question "A rare film's neighbours look odd: obscure films with very high scores. Why?"
    With few users, a handful of shared ratings make a high cosine. `knn_shrink` exists to damp exactly this. Fit the
    model again with `knn_shrink=0` and with `knn_shrink=500` and compare the neighbours of the rare film.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `knn_neighbors` | how many most-similar items each item keeps | 10 to 1,000 (log scale) |
| `knn_shrink` | lowers similarities that rest on few users | 0, 10, 50, 100, 500 |
| `knn_weighting` | `tfidf`/`bm25` make very active users count less when measuring similarity | none, tfidf, bm25 |
| `decay_half_life_days` | an interaction h days old counts half as much as today's, in the similarity and in the profile | none, 30, 90, 365 |
| `train_window_days` | learn only from the last N days (plus each user's last 10 events) | none, 30, 90, 365 |
| `train_window_keep_last` | how many of each user's latest events are always kept | not searched (10) |

For each setting below: **write down your prediction first**, then run the sweep (each value takes about a second),
then open the explanation.

**a.** How does NDCG@10 change with `knn_neighbors` from 5 to 1,000?

```bash
python -m recbench.lab sweep --method itemknn --dataset movielens-25m --param knn_neighbors --values 5,15,50,200,1000
```

??? success "What we saw, and why"
    0.1959, 0.1982, 0.1913, 0.1819, 0.1891: nearly flat, all within each other's intervals. Each user's score adds up
    the similarities of all their items, so even 5 neighbours per item gives a long candidate list for a user with
    dozens of films. More neighbours add weak, noisy similarities. The setting matters little here; it matters more
    for users with very short histories.

**b.** How does a 30-day training window compare with no window?

```bash
python -m recbench.lab sweep --method itemknn --dataset movielens-25m --param train_window_days --values null,30,90,365
```

??? success "What we saw, and why"
    No window 0.1987, 30 days **0.0980**, 90 days 0.1592, 365 days 0.1982. A 30-day window halves the score. Few
    similarities can be learned from one month, and, as Level 1.2d showed, each user's profile shrinks to their last 10
    events plus that month.

**c.** Sweep `knn_shrink` (0, 10, 100, 1000) and `knn_weighting` (none, tfidf, bm25) yourself. Which matters more on
MovieLens? On Last.fm, where users play the same artists over and over?

??? tip "Hint"
    `--param knn_weighting --values none,tfidf,bm25`. For Last.fm add `--dataset lastfm`.

??? success "What to look for"
    Shrink matters when many items are rare. Weighting matters when a few users are extremely active: on Last.fm some
    users have tens of thousands of plays. In your notes, connect what you see to the formulas on the
    [ItemKNN page](../dictionary/algorithms/itemknn.md#4-the-math-symbol-by-symbol).

## Level 3: data tricks

### 3.1 How much does recency matter?

`experiments.yaml` already contains `no-time-knobs`: the baseline's search without `decay_half_life_days` and
`train_window_days`. Run it on all five datasets on the box and compare:

```bash
./scripts/run_lab_box.sh itemknn:no-time-knobs        # on the box
./scripts/fetch_lab_results.sh vast-gpu               # on the laptop
python -m recbench.compare itemknn itemknn:no-time-knobs
```

??? success "What we saw on MovieLens"
    Test NDCG@10 0.1587 → 0.1453, **worse** (−0.0134 [−0.0206, −0.0060]). Without recency, ItemKNN recommends what was
    co-watched over 25 years of ratings, not what people watch now. Expect the same on the shops (H&M, RetailRocket),
    where fashion and stock change quickly.

### 3.2 Keep more of each user's history

Level 1.2d found that the training window also trims each user's profile to their last 10 events. What if the window
only decided what similarities are learned from, while users kept more of their history? `train_window_keep_last`
controls exactly that, and needs no code. Uncomment `keep-more-history` in `experiments.yaml`:

```yaml
  keep-more-history:
    note: The window trims the profile used for scoring; keep each user's last 50 or 200 events
    from_baseline: true
    params:
      train_window_keep_last: {type: choice, values: [10, 50, 200]}
```

`from_baseline: true` holds every other setting at the baseline's best on each dataset, so the 10 trials only explore
the new setting. Before running it on the box, check the idea on the validation fold:

```bash
python -m recbench.lab sweep --method itemknn --dataset movielens-25m --param train_window_keep_last --values 10,50,200,1000
```

??? success "What we saw"
    Validation NDCG@10 0.1982 (10), 0.2061 (50), **0.2108 (200)**, 0.1987 (1,000, which keeps everything: the same
    as no window). On the test split the focused experiment scored **0.1745 against 0.1587, better** (+0.0158
    [+0.0109, +0.0207]). On Last.fm nothing changed: its baseline uses no window, so there is nothing to keep.

    The lesson goes beyond ItemKNN: *learning* from recent data and *describing* a user with their whole recent
    history are two different choices, and recbench's `train_window_days` couples them.

## Level 4: one change from the literature

### The idea: asymmetric cosine (Aiolli 2013)

Cosine treats the pair (i, j) symmetrically:

$$\text{sim}(i, j) = \frac{|U_i \cap U_j|}{|U_i|^{1/2}\, |U_j|^{1/2}}$$

where $U_i$ is the set of users of item i. Aiolli's **asymmetric cosine** moves the exponents:

$$\text{sim}_\alpha(i, j) = \frac{|U_i \cap U_j|}{|U_i|^{\alpha}\, |U_j|^{1-\alpha}}$$

With α = 0.5 it is the cosine. With α = 1 it is $|U_i \cap U_j| / |U_i| = P(j \mid i)$: "of the users who have i, the
share who also have j", which favours popular candidates j. With α = 0 it favours rare ones. Aiolli also raises
each similarity to a **locality** power q, so a few strong similarities count more than many weak ones. This won the
Million Song Dataset challenge in 2012.

### Write the variant

Add two settings to ItemKNN: `knn_similarity` (`cosine` by default, or `asymmetric`) with `knn_alpha`, and
`knn_locality` (q, default 1). The defaults must give exactly today's scores.

??? tip "Hint"
    `item_cosine_topk` computes `norms[item] * norms[idx]`, and `norms` are square roots of the weighted user counts,
    so `norms[item] ** (2 * alpha) * norms[idx] ** (2 * (1 - alpha))` is the asymmetric denominator. Add an `alpha`
    argument to `item_cosine_topk` with default 0.5 (SLIM calls it too). The locality power applies to the finished
    similarity matrix: `self.sim.data **= q`.

??? success "Solution"
    In `ItemKNN.fit`:

    ```python
    self.seen = data.weighted_matrix(cfg.get("decay_half_life_days"))
    # Asymmetric cosine (Aiolli 2013): alpha 0.5 is the ordinary cosine; towards 1 it becomes P(j | i).
    alpha = float(cfg.get("knn_alpha", 0.5)) if cfg.get("knn_similarity", "cosine") == "asymmetric" else 0.5
    weighted = weight_users(self.seen, str(cfg.get("knn_weighting") or "none"))
    self.sim = item_cosine_topk(weighted, self.k, self.shrink, alpha=alpha)
    locality = float(cfg.get("knn_locality", 1))
    if locality != 1:  # q > 1 makes strong similarities count more than many weak ones
        self.sim.data **= locality
    ```

    In `item_cosine_topk`, add `alpha: float = 0.5` to the arguments, and:

    ```python
    if alpha == 0.5:
        val = co.data[lo:hi] / (norms[item] * norms[idx] + shrink)
    else:
        val = co.data[lo:hi] / (norms[item] ** (2 * alpha) * norms[idx] ** (2 * (1 - alpha)) + shrink)
    ```

    Keeping the original line for α = 0.5 guarantees bit-identical default scores, so `tests/test_lab_defaults.py`
    passes.

### Test it

Copy the template and fill in the TODOs:

```bash
cp labs/templates/test_variant_template.py tests/test_itemknn_variant.py
```

??? success "Solution"
    ```python
    METHOD = "itemknn"
    VARIANT = {"knn_similarity": "asymmetric", "knn_alpha": 0.8, "knn_locality": 2}
    DEFAULT = {"knn_similarity": "cosine", "knn_locality": 1}


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
        """Asymmetric cosine with alpha = 0.5 is the ordinary cosine, so it must give exactly the baseline's scores."""
        view, _ = toy
        checks.check_same_scores(METHOD, view, {}, {"knn_similarity": "asymmetric", "knn_alpha": 0.5})
    ```

Then `pytest -q tests/test_itemknn_variant.py tests/test_lab_defaults.py`.

### Run and compare

Uncomment `asymmetric` in `experiments.yaml`. It searches α and q while every other setting stays at the baseline's
best (`from_baseline: true`). Run it on the box, fetch, and compare with segments:

```bash
python -m recbench.compare itemknn itemknn:asymmetric --segments
```

??? success "What we saw"
    | | MovieLens | Last.fm |
    |---|---|---|
    | chosen α, q | 0.87, 1 | 0.40, 1 |
    | test NDCG@10 | 0.1587 → **0.1811** | 0.0569 → 0.0598 |
    | verdict | **better** (+0.0224 [+0.0167, +0.0281]) | no clear difference (+0.0029 [−0.0009, +0.0067]) |

    On MovieLens, larger α was steadily better on the validation fold (α 0.1: 0.107, 0.4: 0.189, 0.6: 0.202, 0.87:
    0.211). That pushes towards popular candidates, which fits a dataset where MostPopular is strong. On Last.fm the
    best α was *below* 0.5. The right α depends on the data, which is why it belongs in the search space rather than
    as a fixed default.

    **A lesson about experiments:** a first version searched α and q *together with* all the baseline's settings
    (10 trials over 7 settings). It found nothing (+0.0006 on MovieLens). Focusing the 10 trials on the new settings
    found the gain. When you test one idea, give it the trials.

Read the segments: does the gain come from heavy users or light ones? From popular test items or the long tail?

### Decide

Apply the [promotion rule](../handbook/promote.md) to your five-dataset result. If it is better on some datasets and
worse on none, add `knn_similarity`, `knn_alpha` and `knn_locality` to ItemKNN's search space as an option.

## Record your results

Fill in one row per experiment (test NDCG@10 difference to the baseline, with the verdict), then write two or three
sentences: what worked, where, and why you think so.

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| no-time-knobs | | | | | | |
| keep-more-history | | | | | | |
| asymmetric | | | | | | |

## Further reading

- Sarwar, Karypis, Konstan & Riedl (2001), [Item-based collaborative filtering recommendation algorithms](https://dl.acm.org/doi/10.1145/371920.372071), WWW.
- Aiolli (2013), *Efficient top-N recommendation for very large scale binary rated datasets*, RecSys 2013.
- Ferrari Dacrema, Cremonesi & Jannach (2019), [Are we really making much progress?](https://arxiv.org/abs/1907.06902), RecSys: tuned ItemKNN beats many neural methods.
