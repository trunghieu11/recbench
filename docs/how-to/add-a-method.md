# Add a method

A checklist for adding any recommender, from a heuristic to a deep model. For a guided first example, follow
the tutorial [your first new method](../start/your-first-method.md).

## 1. Choose how the evaluator gets scores

| `output` | Implement | Use when |
|---|---|---|
| `"scores"` (default) | `score_users(users, hist)` → float32 [B, n_items + 1] | you can score the whole catalog at once (embeddings, linear models) |
| `"pairs"` | `score_pairs(users, items, hist)` → float32 [B, C] | the model scores one (user, item) pair at a time (CTR rankers); set `pair_budget` if each pair is memory-heavy |
| `"list"` | `topk(users, hist, k)` → (items [B, k], scores) | only a ranked list is available (a remote API) |

Column 0 is padding and is always ignored. For embedding models, subclass
`src/recbench/methods/_torch.py::EmbeddingRecommender` and implement only `user_vectors` and `item_matrix`.

## 2. Fill in the MethodSpec honestly

| Field | Set it to True when... |
|---|---|
| `uses_history` | scoring reads the user's past items (as a set or a sequence) |
| `sequence_aware` | it reads that history **in order** (sequential and session models) |
| `scores_cold_items` | it can score items with no pre-test interactions (content-based); otherwise the evaluator removes them |
| `handles_cold_users` | it gives meaningful lists to users with no history (it is then also scored on cold users) |
| `requires_side_features` | it needs item text or categories (skipped on datasets without them) |
| `requires_images` | it needs real image files |
| `outputs_probability` | its scores are logits of a probability (enables `sampled_logloss`) |
| `needs_torch` | it imports PyTorch |
| `ranked` | set to False for experimental methods (reported, not ranked) |
| `deterministic` | fitting involves no randomness (closed forms, counting): one seed is enough when confirming |

Also set `fidelity` ("faithful", "simplified", or "placeholder"; it must match the catalog), `upstream` (where the
implementation comes from), and `impl_version`. **Bump `impl_version`** whenever a code change changes the
method's results: finished runs are recognised by a hash that includes it, so without the bump the old results
would count as done and nothing would re-run.

## 3. Implement `fit(data, cfg)`

- Read data only through `data` (a `TrainView`): `data.seen`, `data.interaction_counts`, `data.events()`,
  `data.user_items(u)`, `data.history_batch(users, L)`, `data.item_text`, `data.item_category`, `data.item_pop`.
  It cannot reach test data, and that is the point.
- Read settings with defaults: `int(cfg.get("dim", 64))`. Prefer the shared preset keys (`dim`, `layers`,
  `heads`, `seq_len`, `batch_size`, `lr`, `seed`, `device`) and prefix method-specific keys (`ease_lambda`).
- **Train neural models in epochs** with `src/recbench/methods/_torch.py::train_epochs` (Adam, bf16 on CUDA) or
  `src/recbench/methods/_torch.py::early_stopping_loop` for a custom loop, and pass `owner=self`. They then honour
  `epochs` (a fixed count), `max_epochs`, `patience` and the tuning job's time limit (`fit_deadline`), stop early on
  a validation fold, and put `best_epoch` in `fit_info`, which the tuning job reuses for the single test run.
  (`max_steps` is the older step-based path, used only by the held-back models.)
- For sequence models, reuse `src/recbench/methods/seq_trainer.py::sequence_windows` and
  `src/recbench/methods/seq_trainer.py::next_item_loss`. For pairwise models, use
  `src/recbench/methods/_torch.py::edge_batches`. For devices, use `src/recbench/methods/_torch.py::resolve_device`.
- Raise `Unsupported("reason")` when the method cannot run on this data; it is recorded, not counted as a failure.
- Optionally set `self.fit_info = {...}`; the values are logged to MLflow as `fit.<key>` parameters.

## 4. Register it

Decorate the class with `@register_method` and add its module's name to the `MODULES` tuple in
`src/recbench/methods/__init__.py`. Each run's child process imports only its method's module, so a method that
does not use PyTorch never loads it.

## 5. Explain (optional, recommended)

Return one list of `Explanation` objects per user. Use `contribution_explanations` for additive models, or
`embedding_explanations` for vector models (`src/recbench/methods/_explain.py`). Cite history items in
`evidence` to count as personal.

## 6. Catalog entry and docs

1. Add the method under `methods:` in `dictionary/catalog.yaml` (title, family, rung, year, paper, code,
   fidelity, summary, and all five rubric scores with reasons).
2. `python -m recbench.dictionary.build`
3. Write `docs/dictionary/algorithms/<name-with-hyphens>.md` following the algorithm page template, and add it
   to the `nav` in `mkdocs.yml`.

## 7. Test

- `tests/test_methods.py::test_every_method_fits_and_evaluates` automatically covers every registered local method.
- Sequence models: add your method to the copy-task test.
- Anything with a reference implementation: add a parity test.
- Run `pytest -q -m "not slow"`, `pytest -q tests/test_docs.py`, and `mkdocs build --strict`.

## 8. Put it through the gate: the quick-tier bake-off

Every new method enters the comparisons the same way: a tuned job on every dataset.

1. **A search space** in `configs/tuning/quick.yaml`, under `methods:`: the settings to try, with their types and
   ranges from the method's paper. Add `fixed: {max_epochs: …, patience: …}` for epoch-trained models, and
   `common: false` if training on only the last N days makes no sense for it. Add a `confirm:` entry if one setting
   depends on data size (like EASE's λ). Without a space, the job runs one untuned trial.
2. **A queue entry** in `configs/benchmarks/quick.yaml`, under `queue.methods`:
   `{name: <name>, resource: cpu or gpu}`, plus `after: [ease, itemknn]` if it needs other methods' results
   (as the re-rankers do). Where it appears in the list decides when it runs inside a dataset's block.
3. **A free dry run** on the laptop:

    ```bash
    python -m recbench.queue run --config configs/benchmarks/quick-smoke.yaml --methods <name> --cpu-workers 3
    python -m recbench.tuning --config configs/benchmarks/quick-smoke.yaml --dataset movielens-25m --method <name>   # or one job
    ```

    Every job should end `finished`, or `unsupported` for a reason you expect.

A method that is not ready for the gate, for example because it is too slow, goes on the `held_back:` list in
`configs/benchmarks/quick.yaml`. `tests/test_docs.py` fails while a registered method is neither queued nor
held back, so nothing is forgotten.

For a quick look without tuning, the smoke tier still works:
`python -m recbench.runner --config configs/benchmarks/smoke-cpu.yaml --datasets movielens-25m --methods <name>`.

## Common errors

| Error | Cause |
|---|---|
| `KeyError: Unknown method` | module missing from the `MODULES` tuple in `methods/__init__.py` |
| a code change does not re-run anything (`skipped_existing`) | bump `impl_version` in the MethodSpec |
| `CatalogError: No catalog entry for` | catalog entry missing |
| `ValueError: ... returned scores of shape` | `score_users` must return [len(users), n_items + 1] |
| results near Random for a sequence model | reading the wrong end of right-aligned histories; use the last column |
