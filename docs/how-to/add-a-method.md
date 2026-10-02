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
| `uses_history` | the model reads the history as a sequence |
| `scores_cold_items` | it can score items with no pre-test interactions (content-based); otherwise the evaluator removes them |
| `handles_cold_users` | it gives meaningful lists to users with no history (it is then also scored on cold users) |
| `requires_side_features` | it needs item text or categories (skipped on datasets without them) |
| `requires_images` | it needs real image files |
| `outputs_probability` | its scores are logits of a probability (enables `sampled_logloss`) |
| `needs_torch` | it imports PyTorch |
| `ranked` | set to False for experimental methods (reported, not ranked) |

Also set `fidelity` ("faithful", "simplified", or "placeholder") and `upstream` (where the implementation comes from).

## 3. Implement `fit(data, cfg)`

- Read data only through `data` (a `TrainView`): `data.seen`, `data.interaction_counts`, `data.events()`,
  `data.user_items(u)`, `data.history_batch(users, L)`, `data.item_text`, `data.item_category`, `data.item_pop`.
  It cannot reach test data, and that is the point.
- Read settings with defaults: `int(cfg.get("dim", 64))`. Prefer the shared preset keys (`dim`, `layers`,
  `heads`, `seq_len`, `batch_size`, `lr`, `max_steps`, `seed`, `device`) and prefix method-specific keys
  (`ease_lambda`).
- For sequence models, reuse `src/recbench/methods/seq_trainer.py::sequence_windows` and
  `src/recbench/methods/seq_trainer.py::next_item_loss`. For pairwise models, use
  `src/recbench/methods/_torch.py::edge_batches`. For devices, use `src/recbench/methods/_torch.py::resolve_device`.
- Raise `Unsupported("reason")` when the method cannot run on this data; it is recorded, not counted as a failure.
- Optionally set `self.fit_info = {...}`; the values are logged to MLflow as `fit.<key>` parameters.

## 4. Register it

Decorate the class with `@register_method` and import its module in `src/recbench/methods/__init__.py`.

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

## 8. Add it to a benchmark config

Add the name under `methods:` in `configs/benchmarks/smoke-cpu.yaml` (and `gpu-full.yaml`), and any
settings under `method_params: {<name>: {...}}`.

## Common errors

| Error | Cause |
|---|---|
| `KeyError: Unknown method` | module not imported in `methods/__init__.py` |
| `CatalogError: No catalog entry for` | catalog entry missing |
| `ValueError: ... returned scores of shape` | `score_users` must return [len(users), n_items + 1] |
| results near Random for a sequence model | reading the wrong end of right-aligned histories; use the last column |
