# Lab 11 · The LightGBM re-ranker: combining methods

!!! abstract "In plain words"
    Production recommenders usually work in two stages. Cheap methods propose a few hundred candidates per user, and
    then a learned ranker orders them using many clues at once. recbench's re-ranker takes candidates from EASE,
    ItemKNN and "popular this week", describes each (user, candidate) pair with 18 features, and orders them with
    LightGBM trained for NDCG. This capstone brings everything together: you will see what the ranker relies on,
    what limits it, and you will add one of your improved methods from the earlier labs as a third source of
    candidates.

## Your starting point

--8<-- "generated/lab/lgbm_rerank.md"

## Before you start

- Read the [trees and learning to rank primer](primer-trees-and-ranking.md), [retrieval and
  ranking](../dictionary/concepts/retrieval-and-ranking.md) and sections 1 to 5 of the
  [LightGBM re-ranker page](../dictionary/algorithms/lgbm-rerank.md).
- Labs 1 and 3 (ItemKNN, EASE) first: the re-ranker's candidates come from their lab baselines.
- Open `labs/11-lgbm-rerank/lgbm_rerank.ipynb` and `labs/11-lgbm-rerank/experiments.yaml`;
  `git switch -c lab/11-lgbm-rerank`.
- Plan about 7 hours, plus box runs. The re-ranker is random (subsampling), so the lab tests it with 3 seeds.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method lgbm_rerank --dataset movielens-25m --best
```

It takes a couple of minutes: the generators (EASE among them) are fitted twice, once for training and once for
scoring.

### 1.2 Read the code

Open `src/recbench/methods/rerank.py`. The module docstring explains the two stages.

**a.** How many candidates does each generator propose, and how are the lists merged?

??? success "Answer"
    With `rerank_candidates` = 200: EASE proposes its top 150 (k × 3/4), ItemKNN its top 100 (k/2), popularity its top
    50 unseen items (k/4). The union is ordered by each item's best rank in any list and cut to 200. Each candidate
    remembers its rank in every list and how many lists proposed it (`n_sources`).

**b.** The ranker needs labels: which candidates did the user really choose? Where do they come from, without
touching the test split?

??? success "Answer"
    From the window between the validation cutoff and the test cutoff (`training_table`). The generators are fitted on
    the events *before* the validation cutoff, the features are computed as of that cutoff, and a label is 1 when the
    user interacted with the candidate in the window. For scoring, everything is fitted again on all pre-test data.
    This is the leakage-free design from the primer.

**c.** What is `train_recall_at_candidates`, and why does it matter?

??? success "Answer"
    The share of users in the label window whose chosen items were among their candidates at all; users without any are
    dropped from training. On the test split the same idea is `candidate_recall`. If the right item is not among the
    candidates, no ranker can find it: it is the ceiling.

### 1.3 Look inside

In the notebook, fit the baseline's setting and show the trees' feature importances next to the feature names:

```python
from recbench.methods.rerank import FEATURES

model = lab.fit("lgbm_rerank", data, **best)
sorted(zip(model.model.feature_importances_, FEATURES), reverse=True)
```

??? question "Which features does the ranker use most? Is anything surprising?"
    Typically the generators' own scores and ranks come first, then recent popularity and trend. A user feature such
    as `user_days_idle` near the top means the ranker learned that returning users want different things. Write down
    your top five: Level 4 adds two features, and you will see where they land.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `rerank_candidates` | candidates per user | 100, 200 |
| `lgbm_leaves` | leaves per tree | 15 to 127 (log scale) |
| `lgbm_lr` | learning rate | 0.01 to 0.2 (log scale) |
| `lgbm_min_child` | minimum examples per leaf | 10 to 100 (log scale) |
| `rerank_text` | adds the text similarity feature (a sentence encoder) | false, true |
| `lgbm_trees`, `rerank_pop_days`, `rerank_train_users` | trees (with early stopping), the popularity window, users for training | not searched: 500, 7, 20,000 |

**a.** Compare `candidate_recall` for 100 and 200 candidates: sweep `rerank_candidates` and print the metric from
the runs (`lab.once(...)["metrics"]["candidate_recall"]`). How much of the gap between the re-ranker and a perfect
score is the ceiling's fault?

**b.** Sweep `lgbm_leaves` (15, 31, 63, 127). The primer says more leaves overfit more easily: do you see it on Last.fm,
the smallest dataset?

!!! note "Text features on a Mac"
    With `rerank_text`, item texts are encoded by a sentence-transformer that needs PyTorch. On macOS PyTorch cannot
    share a process with LightGBM, so recbench encodes any missing vectors in a separate process first
    (`src/recbench/methods/rerank.py::text_vectors`). The first run on a dataset therefore takes longer.

## Level 3: data tricks

### 3.1 More training users

`more-train-users` searches 20,000 or 50,000 training users, with every other setting at the baseline's best. Run it
on the box and compare accuracy and training time.

### 3.2 Fewer negatives per user

Each training user brings about 200 candidates, almost all of them not chosen. Keeping every negative makes training
slow, and the many easy negatives can drown the few useful ones. Add `rerank_neg_per_user` = n (0: keep all): keep
every chosen candidate and at most n others, drawn at random.

??? tip "Hint"
    In `training_table`, each user's rows are selected by `valid = cands[b] > 0`. Turn some negatives off in that mask.
    Use the `rng` that already exists there, so that the default (n = 0) draws no random numbers and stays identical.

??? success "Solution"
    ```python
    neg_cap = int(cfg.get("rerank_neg_per_user", 0))  # 0: keep every candidate the user did not choose
    for start in range(0, len(users), 512):
        ...
        for b in np.flatnonzero(keep_users):
            valid = cands[b] > 0
            if neg_cap:  # keep every chosen candidate and at most neg_cap of the others
                negatives = np.flatnonzero(valid & (labels[b] == 0))
                if len(negatives) > neg_cap:
                    valid = valid.copy()
                    valid[rng.choice(negatives, len(negatives) - neg_cap, replace=False)] = False
    ```

## Level 4: one change from the literature

### The idea: a better candidate pool

Every two-stage system is limited by its candidates. Production systems blend several generators, and winning
competition solutions, such as those of the H&M Kaggle competition, add many sources of candidates before re-ranking. Over the past ten weeks you
have improved several methods. Add one as a **third generator**: its top candidates join the pool, and its score and
rank become two new features. RP3beta (lab 2) is a natural choice: cheap, strong, and different from EASE and ItemKNN.

### Write the variant

Add `rerank_third`: the name of a lab method, for example `rp3beta`. When it is set:

1. `Generators` fits that method too (with its lab-tuned settings, read like EASE's and ItemKNN's) and proposes its
   top k/2 candidates;
2. every candidate gets `third_score` and `third_rank`;
3. the feature builder adds the two columns.

??? tip "Hint"
    The feature list is a fixed tuple, `FEATURES`. Give `FeatureBuilder` an `extra` argument with the additional names,
    and stack `FEATURES + extra`. In `Generators.candidates`, a candidate's ranks live in a list indexed by source,
    `("ease", "knn", "pop")`: add `"third"` when the third generator is on, and size the default list to match.
    `tuned_params(data, name)` already finds any method's tuned settings.

??? success "Solution"
    The full change is about 40 lines. The key parts, in `Generators`:

    ```python
    def __init__(self, cfg, ease_cfg, knn_cfg, third=None, third_cfg=None):
        self.cfg, self.ease_cfg, self.knn_cfg = cfg, ease_cfg, knn_cfg
        self.third, self.third_cfg = third, third_cfg or {}
        self.n_candidates = int(cfg.get("rerank_candidates", 200))

    def fit(self, view):
        ...
        if self.third:  # lab 11: one more generator, with its tuned settings
            from recbench.registry import ensure_loaded

            self.third_model = ensure_loaded().create_method(self.third)
            self.third_model.fit(view, {**self.cfg, **self.third_cfg})
        return self
    ```

    in `Generators.candidates`:

    ```python
    generators = [("ease", self.ease, k * 3 // 4), ("knn", self.knn, k // 2)]
    if self.third:
        generators.append(("third", self.third_model, k // 2))
    sources = ("ease", "knn", "pop", "third") if self.third else ("ease", "knn", "pop")
    for name, model, take in generators:
        ...  # unchanged: score, mask the seen items, keep the top `take`
    names = ["ease_score", "ease_rank", "knn_score", "knn_rank", "pop_rank", "best_rank", "n_sources"]
    names += ["third_score", "third_rank"] if self.third else []
    feats = {f: np.zeros((len(users), k), dtype=np.float32) for f in names}
    ...
        proposals = [("ease", ...), ("knn", ...), ("pop", ...)]
        if self.third:
            proposals.append(("third", [i for i in lists["third"][b] if np.isfinite(scores["third"][b, i])]))
        for source, items in proposals:
            for rank, item in enumerate(items, start=1):
                entry = ranks.setdefault(int(item), [1e4] * len(sources))
                entry[sources.index(source)] = rank
        ...
        for c, item in enumerate(chosen):
            r = ranks[item]
            feats["ease_rank"][b, c], feats["knn_rank"][b, c], feats["pop_rank"][b, c] = r[:3]
            if self.third:
                feats["third_rank"][b, c] = r[3]
            ...
        if self.third:
            feats["third_score"][b, : len(chosen)] = scores["third"][b, chosen]
    for name in ("ease_score", "knn_score", "third_score")[: 3 if self.third else 2]:
        ...  # unchanged: replace -inf by 0
    ```

    in `FeatureBuilder`:

    ```python
    def __init__(self, view, text_vectors=None, extra=()):
        self.view = view
        self.names = FEATURES + tuple(extra)  # extra: the third generator's score and rank (lab 11)
        ...
    # and at the end of build():
    return np.stack([np.asarray(columns[name], dtype=np.float32) for name in self.names], axis=1)
    ```

    and in `TwoStage._setup`, `training_table` and `prepare_scoring`:

    ```python
    self.third = cfg.get("rerank_third")  # lab 11: a third generator, for example rp3beta
    self.third_cfg = tuned_params(data, self.third) if self.third else {}
    self.extra = ("third_score", "third_rank") if self.third else ()
    ...
    gens = Generators(cfg, self.ease_cfg, self.knn_cfg, self.third, self.third_cfg).fit(past)
    builder = FeatureBuilder(past, self.text, self.extra)
    ...
    self.generators = Generators(self.cfg, self.ease_cfg, self.knn_cfg, self.third, self.third_cfg).fit(self.data)
    self.features = FeatureBuilder(self.data, self.text, self.extra)
    ```

### Test it

??? success "Solution"
    ```python
    METHOD = "lgbm_rerank"
    VARIANT = {"rerank_third": "rp3beta"}
    DEFAULT = {"rerank_neg_per_user": 0}


    def test_the_third_generator_adds_two_features(toy):
        view, _ = toy
        model = checks.fit(METHOD, view, **VARIANT)
        assert model.features.names[-2:] == ("third_score", "third_rank")
    ```

    LightGBM with one thread and a fixed seed repeats exactly, so the template's tests can stay; the toy settings use
    one thread.

### Run and compare

Uncomment `third-generator`. If you promoted an improved RP3beta in lab 2, the lab's RP3beta baseline should be
re-run with it first, so the re-ranker gets the improved candidates. Then run it on the box and compare:

```bash
python -m recbench.compare lgbm_rerank lgbm_rerank:third-generator --segments
```

??? success "What to look for"
    Two numbers move together: `candidate_recall`, the ceiling, and NDCG@10. A higher ceiling with the same NDCG@10
    means the ranker cannot yet use the new candidates. Look at the feature importances again: where did `third_score`
    and `third_rank` land?

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| more-train-users | | | | | | |
| downsample | | | | | | |
| third-generator | | | | | | |

Then look back over the eleven weeks: which kinds of changes helped most often, settings, data or algorithms, and on
which datasets? Write it on the [scoreboard page](scoreboard.md) or in your pull request: it is the most useful thing
you can leave for the next person.

## Further reading

- Burges (2010), *From RankNet to LambdaRank to LambdaMART: an overview*, Microsoft Research technical report.
- Ke et al. (2017), *LightGBM: a highly efficient gradient boosting decision tree*, NeurIPS 2017.
- Covington, Adams & Sargin (2016), *Deep neural networks for YouTube recommendations*, RecSys 2016: the two-stage design at scale.
