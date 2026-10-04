# Add a metric

## 1. Know what a metric receives

Every metric gets a `MetricContext` (`src/recbench/evaluation.py::MetricContext`) for one method, one repeat
policy, and the warm evaluation users:

| Field | Contents |
|---|---|
| `topk` | int array [users, 50]: ranked item indices (0 = empty slot) |
| `hits` | bool array [users, 50]: is the item at each rank relevant? |
| `relevant`, `relevant_sets` | each user's relevant test items (array, set) |
| `n_rel` | number of relevant items per user |
| `next_item`, `next_rank` | the next item and its rank in the list (51 if absent) |
| `item_pop` | pre-test interaction count per item |
| `item_categories`, `has_categories`, `user_profile(u)` | category sets per item; a user's category mix |
| `head_items`, `head_items_5pct` | bool masks of the top 20% / 5% most popular items |
| `cold_items` | bool mask of items without pre-test interactions |
| `user_groups` | "light", "medium", "heavy" per user |
| `sampled_rank`, `sampled_auc`, `sampled_logloss` | secondary-protocol statistics (or None) |
| `extra` | run-level values: `train_seconds`, `score_seconds_per_1k_users`, ... |
| `method_spec` | the method's `MethodSpec` |

## 2. Write it

In `src/recbench/metrics/catalog.py` (or a new module imported by `src/recbench/metrics/__init__.py`):

```python
import numpy as np

from recbench.protocol import Metric, MetricSpec, Task
from recbench.registry import register_metric


@register_metric
class FirstHitDepth(Metric):
    """Average rank of the first relevant item, among users who have one in the top 50."""

    spec = MetricSpec(
        "first_hit_rank_at_50",
        {Task.topn},
        kind="accuracy",
        higher_is_better=False,
        per_user=False,
        description="Mean rank of the first relevant item in the top 50 (users with a hit only).",
    )

    def compute(self, ctx):
        found = ctx.hits.any(axis=1)
        if not found.any():
            return None
        return float((ctx.hits[found].argmax(axis=1) + 1).mean())
```

Rules:

- Return a float (list-level metric), a NumPy array with one value per user (per-user metric; the evaluator
  averages it and keeps the values), or `None` when the metric does not apply.
- `tasks` decides which methods get it: it is computed only if it overlaps the method's `spec.tasks`.
- `kind` groups it in docs and reports: accuracy, beyond, slice, sampled, efficiency, explainability, diagnostic.
- Errors are recorded per metric in `metric_errors.json`, not hidden.

## 3. Add confidence intervals (optional)

For a per-user headline metric, add its name to `CI_METRICS` in `src/recbench/evaluation.py`.

## 4. Add a worked example and a test

For ranking metrics, add a case to `docs/assets/metric_examples.yaml` and extend
`tests/test_metrics.py::test_worked_examples_from_the_docs`, so the docs and the code stay in sync.

## 5. Show it

- `python -m recbench.dictionary.build` adds it to the generated metric list automatically.
- To show it in the report, add a column in `src/recbench/report/build.py::dataset_section`.
- Document it on the matching page under `docs/dictionary/metrics/`.

Existing runs do not get the new metric. Re-run them by bumping `EVAL_VERSION` in `src/recbench/protocol.py`: it is
part of every run's identity (`config_hash`), so all runs then count as new. (The package version is deliberately
not part of it, so bumping that changes nothing.) For a single method, bump its `impl_version` instead, or delete
its runs in MLflow.
