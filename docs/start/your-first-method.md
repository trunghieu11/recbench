# 4. Your first new method

**Goal:** implement a recommender from scratch, plug it into recbench, benchmark it, and document it.
**Time:** about an hour.
**You will build:** `category_popular`, "popular items, preferring the categories this user already likes".

## Step 1: understand the contract

Every method subclasses `Recommender` (`src/recbench/protocol.py::Recommender`) and provides:

| Piece | Purpose |
|---|---|
| `spec = MethodSpec(...)` | declares the name, tasks, and capabilities (see [the Recommender API](../codebase/recommender-api.md)) |
| `fit(data, cfg)` | learns from `data`, a `TrainView` with pre-test events only |
| `score_users(users, hist)` | returns a float32 matrix of shape [number of users, n_items + 1]; column 0 is padding |
| `explain(users, items, hist)` | optional: reasons for recommendations |

The evaluator does everything else: removing seen items, top-K, metrics, confidence intervals.

## Step 2: write the method

Create a new file, **src/recbench/methods/category_popular.py**:

```python
"""A popularity baseline that prefers items from the categories a user already likes."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recbench.data import HistoryBatch, TrainView
from recbench.protocol import NEG_INF, Explanation, MethodSpec, Recommender, Task
from recbench.registry import register_method


@register_method
class CategoryPopular(Recommender):
    spec = MethodSpec(
        name="category_popular",
        tasks={Task.topn},
        uses_history=True,
        requires_side_features=True,
        upstream="in-repo tutorial method",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.seen = data.seen
        self.boost = float(cfg.get("category_boost", 1.0))
        recent = data.item_recent_pop.astype(np.float64)
        self.popularity = (recent / max(recent.max(), 1.0)).astype(np.float32)  # 0..1
        # Item x category matrix: C[i, c] = 1 if item i has category c.
        item_cats = [[c for c in str(text).split("|") if c] for text in data.item_category]
        names = sorted({c for cats in item_cats for c in cats})
        column = {name: k for k, name in enumerate(names)}
        rows = [i for i, cats in enumerate(item_cats) for _ in cats]
        cols = [column[c] for cats in item_cats for c in cats]
        self.categories = np.array(names, dtype=object)
        self.item_category = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                           shape=(data.n_items + 1, len(names)))

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        liked = (self.seen[users] @ self.item_category) > 0          # users x categories
        match = (liked @ self.item_category.T).toarray() > 0          # users x items
        scores = self.popularity[None, :] + self.boost * match
        scores[:, 0] = NEG_INF
        return scores.astype(np.float32)

    def explain(self, users, items, hist):
        out = []
        for user, row in zip(users, items):
            liked = set(self.categories[((self.seen[user] @ self.item_category) > 0).nonzero()[1]])
            exps = []
            for item in row:
                shared = sorted(liked & set(self.categories[self.item_category[item].nonzero()[1]]))
                if shared:
                    exps.append(Explanation("category", f"Popular in {', '.join(shared)}, which you often choose."))
                else:
                    exps.append(Explanation("popularity", "Popular right now."))
            out.append(exps)
        return out
```

What each part does:

- `bind(data)` stores the catalog size and id maps that every method needs.
- `item_recent_pop` is each item's interaction count in the 28 days before the cutoff (pre-test data only).
- The sparse matrix `item_category` lets us find "items in categories the user likes" with two matrix products
  instead of Python loops.
- Scores = popularity (0–1) + `boost` for items in a liked category. With boost 1.0, any matching item ranks
  above every non-matching one.
- `requires_side_features=True` makes the runner skip datasets without categories (Last.fm).

## Step 3: register it

Add the module to the imports in `src/recbench/methods/__init__.py`:

```python
from recbench.methods import baselines, category_popular, content, dcnv2, graph, hstu, implicit_mf, recbole_models, recombee, sasrec, tiger
```

Check that it is registered:

```bash
python -c "from recbench.registry import ensure_loaded; print('category_popular' in ensure_loaded().methods)"
```

## Step 4: add a catalog entry

The docs and the tests require one. Add this under `methods:` in `dictionary/catalog.yaml`:

```yaml
  category_popular:
    title: CategoryPopular
    family: Baseline
    rung: 0
    year: null
    paper: null
    code: null
    fidelity: faithful
    summary: Popular items, preferring categories the user already likes.
    rubric:
      implementation: [1, "Two sparse matrix products."]
      tuning: [1, "One knob: category_boost."]
      data_hunger: [1, "Needs only recent counts and item categories."]
      controllability: [4, "Transparent; boosts are easy to edit."]
      explainability: [3, "Names the shared category; not item-level."]
```

## Step 5: benchmark it

```bash
python -m recbench.runner --config configs/benchmarks/smoke-cpu.yaml --datasets movielens-25m --methods most_popular,category_popular
python -m recbench.report.build --tier smoke --out reports/smoke-latest --docs
```

When this tutorial was written, `category_popular` clearly beat MostPopular on the synthetic test data (where
every user has one favourite category) but was **tied** with it on MovieLens. Exercise: find out why. Hint: look
at how many genres a typical MovieLens user has already watched. If almost every item gets the boost, the
boost changes nothing. A better version could weight categories by their *share* of the user's history.

## Step 6: test it

Add a test to `tests/test_methods.py`. The existing parametrised test
`test_every_method_fits_and_evaluates` already covers every registered method, so registering is enough to
get a basic test. Run:

```bash
pytest -q tests/test_methods.py -k category_popular
```

## Step 7: document it

1. Run `python -m recbench.dictionary.build` to generate its facts box.
2. Create `docs/dictionary/algorithms/category-popular.md` following the template of the other algorithm pages
   (start with `--8<-- "generated/methods/category_popular.md"`).
3. Add it to the `nav` in `mkdocs.yml`.
4. Run `pytest -q tests/test_docs.py` and `mkdocs build --strict`.

**You now know the full path** from an idea to a benchmarked, documented method. The same steps apply to
complex models; see [add a method](../how-to/add-a-method.md) for the checklist.

**Next:** [the quick-tier bake-off on a rented GPU](quick-tier-box.md).
