# The Recommender API

Every method is a subclass of `src/recbench/protocol.py::Recommender`. The evaluator, the serving exporter, and
the docs generator talk to methods only through this interface.

## Methods

| Method | Who calls it | Contract |
|---|---|---|
| `bind(data)` | your `fit` | stores `n_users`, `n_items`, `item_ids`, `item_pop` |
| `fit(data, cfg)` | runner | learn from `data` (a `TrainView`); `cfg` is a flat settings dict |
| `score_users(users, hist)` | evaluator, bundle export | float32 array [len(users), n_items + 1]; column 0 is ignored |
| `score_pairs(users, items, hist)` | `full_scores` for pointwise models | float32 [B, C] scores for `items[b, c]` |
| `full_scores(users, hist)` | evaluator | provided: calls `score_users`, or `score_pairs` over item chunks of at most `pair_budget` pairs |
| `topk(users, hist, k, exclude)` | evaluator (for list output) | (items [B, k], scores); the default derives it from `full_scores` |
| `explain(users, items, hist)` | evaluator | one list of `Explanation` per user, aligned with `items` |
| `item_embeddings()` | explanations | optional item vectors [n_items + 1, d] |

`users` are integer user indices (≥ 1); `hist` is a `HistoryBatch` built from the same `TrainView`.

## MethodSpec fields

| Field | Effect |
|---|---|
| `name` | registry key, used in configs, MLflow, and bundle paths |
| `tasks` | which metrics apply (a metric is computed only if its tasks overlap) |
| `output` | `"scores"`, `"pairs"`, or `"list"`, as described above |
| `uses_history` | the method reads the user's past items when scoring |
| `sequence_aware` | it reads them in order (shown as "Order-aware" in the capability matrix) |
| `scores_cold_items` | if False, the evaluator and bundle export remove items with no pre-test events |
| `handles_cold_users` | if True, the method is also evaluated on cold users (`cold_users/*`) |
| `requires_side_features` | the runner skips datasets without text or categories |
| `requires_images` | the runner skips datasets without image files on disk |
| `outputs_probability` | enables `sampled_logloss` |
| `managed` | a remote service: skipped unless `managed_services: true`; no bundle export |
| `ranked` | False = reported as experimental, not on leaderboards |
| `needs_torch`, `upstream`, `fidelity`, `cost_band` | facts shown in the docs (`fidelity` must match the catalog) |
| `feedback` | the feedback kinds it accepts (implicit and explicit by default) |
| `impl_version` | part of every run's identity: bump it when a code change changes the method's results, so old runs stop counting as done |
| `deterministic` | True when fitting has no randomness (closed forms, counting): the full-data confirmation then runs one seed instead of three |

## HistoryBatch

`src/recbench/data.py::HistoryBatch` holds pre-test histories for a batch of users:

| Attribute | Shape | Meaning |
|---|---|---|
| `items` | [B, L] int64 | **right-aligned**: the newest item is in the last column; 0 = padding |
| `lengths` | [B] | number of real items (≤ L) |
| `times` | [B, L] int64 | event times in microseconds, aligned with `items` |
| `left_aligned()` | [B, L] | the same items, first-to-last with padding at the end (RecBole's layout) |
| `last_items()` | [B] | the newest item per user (0 if none) |

## TrainView

`src/recbench/data.py::TrainView` is everything a model may read:

| Member | Contents |
|---|---|
| `n_users`, `n_items`, `dataset`, `tier`, `split_hash`, `meta` | sizes and identity (no test counts) |
| `user_items(u)`, `user_times(u)`, `user_lengths` | one user's pre-test history |
| `history_batch(users, seq_len)` | a `HistoryBatch` |
| `seen`, `interaction_counts` | sparse user × item matrices (binary / counts) |
| `events()` | a DataFrame (user_idx, item_idx, ts_us) of all pre-test events |
| `export_frame()` | pre-test events with original ids (for RecBole and Recombee uploads) |
| `item_ids`, `user_ids`, `item_text`, `item_category`, `item_image_path`, `user_attributes` | metadata |
| `item_pop`, `item_recent_pop` | pre-test counts (all, and the last 28 days) |
| `warm_users()`, `warm_item_mask()` | users and items that have pre-test events |
| `cache_dir` | a per-split cache folder for expensive features |
| `restrict(window_days, keep_last)` | a view with only the last `window_days` of events, plus each user's last `keep_last` (the runner uses it for `train_window_days`) |
| `before(cutoff_us)` | a view of the events before a time (the re-rankers build their training labels with it) |
| `weighted_matrix(half_life_days)`, `event_weights(…)`, `event_age_days()` | interactions weighted by age: an event `half_life_days` old counts half (`decay_half_life_days`) |

It has no attribute pointing at test files; `tests/test_leakage.py` checks that.

## The embedding helper

For models whose score is a dot product, subclass `src/recbench/methods/_torch.py::EmbeddingRecommender` and
implement two methods:

```python
class MyModel(EmbeddingRecommender):
    spec = MethodSpec(name="my_model", tasks={Task.topn}, needs_torch=True)

    def fit(self, data, cfg):
        self.bind(data)
        ...  # train self.user_table and self.item_table

    def user_vectors(self, users, hist):      # torch tensor [B, d]
        return self.user_table[torch.as_tensor(users)]

    def item_matrix(self):                    # torch tensor [n_items + 1, d]
        return self.item_table
```

`score_users`, `item_embeddings`, and `explain` (citing similar history items) come for free.

## Training in epochs

Neural methods train with `src/recbench/methods/_torch.py::train_epochs` (Adam, bf16 autocast on CUDA), or with
`src/recbench/methods/_torch.py::early_stopping_loop` around their own epoch function. Pass `owner=self`:

- on a validation fold, the runner attaches `self.monitor` (a `ValidationMonitor`). After each epoch the loop scores
  the fold's monitor users, keeps the best weights, and stops after `patience` epochs without improvement;
- `epochs` in the config trains exactly that many epochs and never looks at validation data (the final test run);
- `fit_deadline` stops before an epoch that would end after it (the tuning job's time share);
- the loop fills `fit_info` (`epochs_run`, `best_epoch`, `stopped`, logged as `fit.*`) and `fit_curve` (logged as
  `curve/*` metrics, one point per epoch).

## A minimal method

```python
@register_method
class Constant(Recommender):
    spec = MethodSpec(name="constant", tasks={Task.topn})

    def fit(self, data, cfg):
        self.bind(data)

    def score_users(self, users, hist):
        scores = np.zeros((len(users), self.n_items + 1), dtype=np.float32)
        scores[:, 0] = NEG_INF
        return scores
```

See [add a method](../how-to/add-a-method.md) for the full checklist.
