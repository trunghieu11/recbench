# Testing

## Running the tests

```bash
pytest -q                     # everything (a few minutes)
pytest -q -m "not slow"       # skip the slower training tests
pytest -q tests/test_docs.py  # only the docs-sync checks
RECBENCH_LIVE=1 pytest -q -m live   # live Recombee test (needs credentials)
```

Tests build small synthetic datasets (`src/recbench/pipeline/toy.py`), so they need no downloads. The fixtures
are in `tests/conftest.py`.

## What each test guards against

### The split (`tests/test_split.py`)

| Test | Guards against |
|---|---|
| `test_cutoffs_do_not_depend_on_the_time_zone` | the v0.1 time-zone bug (cutoffs shifted by the machine's zone) |
| `test_events_respect_the_cutoff` | pre-test events after the cutoff |
| `test_rebuilding_gives_identical_files` | non-deterministic splits |
| `test_ties_are_broken_by_file_order` | random order of equal timestamps |
| `test_eval_users_have_history_and_a_new_relevant_item` | wrong eval users or next items |
| `test_candidates_are_positive_plus_unseen_warm_items` | sampled negatives that the user has seen, or that are cold |
| `test_too_few_eval_users_fails_loudly` | silently "finished" datasets with no test users |
| `test_history_batch_is_right_aligned` | padding on the wrong side |

### Leakage (`tests/test_leakage.py`)

| Test | Guards against |
|---|---|
| `test_train_view_has_no_route_to_test_files` | a model reading test data |
| `test_sequence_targets_come_from_pretest_history` | the v0.1 bug of training on evaluation targets |
| `test_pairwise_positives_come_from_pretest_history` | pairwise training on test pairs |
| `test_new_test_items_are_absent_from_training_windows` | test items appearing as training inputs |

### Metrics (`tests/test_metrics.py`)

| Test | Guards against |
|---|---|
| `test_worked_examples_from_the_docs` | the docs' worked examples drifting from the code (reads `docs/assets/metric_examples.yaml`) |
| `test_gini_extremes` | a broken Gini formula |
| `test_bootstrap_interval_brackets_the_mean` | broken confidence intervals |
| `test_oracle_scores_perfectly` | evaluator bugs: a perfect scorer must get 1.0 everywhere |
| `test_evaluator_masks_padding_seen_and_cold_items` | padding, seen, or cold items leaking into lists |

### Methods (`tests/test_methods.py`)

| Test | Guards against |
|---|---|
| `test_every_method_fits_and_evaluates` | any registered local method crashing or producing invalid metrics |
| `test_sequence_models_solve_the_copy_task` (slow) | the v0.1 alignment bug: reading padding instead of the newest item |
| `test_hstu_attention_matches_metas_reference_op` | HSTU's attention diverging from Meta's `pytorch_hstu_mha` |
| `test_bert4rec_inputs_match_recboles_own_training_rows` | train/serve layout mismatch in the RecBole adapter |
| `test_ease_matches_the_closed_form` | errors in EASE's closed form |
| `test_itemknn_matches_implicit_cosine` | errors in ItemKNN's similarity |

### Runner and serving (`tests/test_runner_serving.py`)

| Test | Guards against |
|---|---|
| `test_runner_logs_applicable_metrics_and_resumes` | inapplicable metrics; stale resume (changed settings must re-run) |
| `test_unsupported_methods_are_recorded_not_failed` | skips counted as failures |
| `test_runs_from_another_machine_are_imported_once` | GPU-machine results lost or duplicated when combined with the laptop's |
| `test_child_process_records_peak_memory` (slow) | the child-process path and per-method memory |
| `test_bundle_matches_the_evaluator_and_falls_back_for_unknown_users` | serving different lists than were evaluated |
| `test_api_serves_bundles` | API routing and 404s |
| `test_serving_does_not_import_torch` | a heavy serving image |

### Recombee (`tests/test_recombee.py`)

| Test | Guards against |
|---|---|
| `test_upload_evaluate_and_budget` | wrong uploads (all items, all pre-test events, timestamps in seconds) and budget accounting |
| `test_refuses_a_non_empty_database_without_permission` | overwriting someone's data |
| `test_refuses_runs_that_would_exceed_the_budget` | surprise bills |
| `test_live_recombee` (live) | the real API path |

### Documentation (`tests/test_docs.py`)

| Test | Guards against |
|---|---|
| `test_every_method_and_dataset_has_a_page` | undocumented methods or datasets |
| `test_code_pointers_resolve` | `path::Symbol` references in the docs that no longer exist |
| `test_snippets_point_at_generated_fragments` | snippets that the generator never produces |
| `test_catalog_has_every_method` | missing or malformed rubric scores |

## Writing a new test

Use the `toy` or `toy_split` fixture for a ready split, and `FAST_CFG` from `tests/conftest.py` for tiny model
settings. Mark slow tests with `@pytest.mark.slow`.
