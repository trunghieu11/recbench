# Testing

## Running the tests

```bash
pytest -q                     # everything (a few minutes)
pytest -q -m "not slow"       # skip the slower training tests
pytest -q tests/test_docs.py  # only the docs-sync checks
RECBENCH_LIVE=1 pytest -q -m live   # live Recombee test (needs credentials)
```

Tests build small synthetic datasets (`src/recbench/pipeline/toy.py`), so they need no data downloads. The
fixtures are in `tests/conftest.py`. Text kNN tests use a tiny fake sentence encoder (the autouse fixture
`_no_model_downloads`); only the slow test `test_text_knn_with_the_real_sentence_transformer` downloads the real
model (all-MiniLM-L6-v2, about 90 MB), so `pytest -q -m "not slow"` stays offline.

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
| `test_validation_fold_has_no_real_test_events_and_keeps_the_users` | tuning that could see the real test window |

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
| `test_stats_and_monitor_report_freshness_quality_and_traffic` | a monitoring endpoint or checker that misreports freshness, coverage, fallbacks or traffic |
| `test_api_without_bundles_or_results_says_so` | a service that says "ok" with nothing to serve, or a dashboard that crashes without MLflow |
| `test_serving_does_not_import_torch` | a heavy serving image |

### Newer methods (`tests/test_new_methods.py`)

| Test | Guards against |
|---|---|
| `test_rp3beta_matches_the_dense_definition`, `test_gfcf_matches_the_dense_formula`, `test_turbocf_matches_the_dense_formula`, `test_ultragcn_neighbours_match_the_dense_definition` | fast sparse implementations that drift from their formulas |
| `test_ease_torch_backend_matches_numpy` (needs a GPU) | the GPU backend giving different weights |
| `test_sansa_ranks_like_exact_ease` | SANSA's approximation drifting from EASE (runs in a child process, as queue runs do) |
| `test_puresvd_projects_onto_the_top_singular_vectors`, `test_slim_weights_are_non_negative_without_self_loops`, `test_vsknn_recommends_items_from_similar_sessions` | broken model structure |
| `test_learned_embedding_methods_beat_random_on_toy_data`, `test_alignment_and_uniformity` | models that do not learn, or wrong loss terms |
| `test_text_knn_follows_content_and_scores_new_items` (+ a slow test with the real encoder) | content scoring that ignores text or cold items |
| `test_rerankers_beat_random_and_report_candidate_recall`, `test_reranker_labels_come_only_from_the_window_before_the_test_cutoff` | re-rankers that do not help, or that learn from the future |

### Training loop (`tests/test_training.py`) and time settings (`tests/test_time_knobs.py`)

| Test | Guards against |
|---|---|
| `test_monitor_refuses_a_real_test_split` | early stopping that looks at test data |
| `test_sasrec_stops_early_on_the_fold_and_keeps_its_best_epoch`, `test_fixed_epochs_never_look_at_validation`, `test_runner_logs_learning_curves_on_a_fold` | wrong early stopping, or a final run that peeks at validation |
| `test_next_item_loss_modes`, `test_per_user_arrays_carry_user_ids` | broken loss options; per-user results that cannot be joined back |
| `test_restrict_keeps_the_window_plus_each_users_latest_events`, `test_weighted_matrix_decays_with_age`, `test_before_is_a_past_view_and_views_compose` | wrong training windows and recency decay |

### Tuning and the queue (`tests/test_tuning.py`, `tests/test_queue.py`)

| Test | Guards against |
|---|---|
| `test_every_search_space_samples` | a search space that cannot be sampled |
| `test_job_searches_the_fold_then_tests_once_and_resumes` | tuning on the test split, or re-running finished work |
| `test_retry_starts_a_fresh_attempt_after_failed_trials`, `test_job_whose_trials_are_all_unsupported_is_unsupported`, `test_job_without_time_for_one_trial_is_over_budget`, `test_missing_splits_are_reported` | wrong job statuses |
| `test_jobs_run_dataset_by_dataset_and_backfill_only_when_blocked`, `test_queue_runs_all_jobs_and_resumes` | wrong order, idle workers, lost resumes |
| `test_confirmations_rank_every_method_even_in_a_filtered_session`, `test_interrupted_confirmation_resumes_even_with_stop_after_dataset`, `test_confirmation_rechecks_the_size_setting_on_full_data` | confirming the wrong methods, stranded confirmations, missing bundles |
| `test_status_shows_liveness_cost_confirmations_and_writes_a_page`, `test_jobs_without_splits_run_again_on_resume`, `test_benchmark_files_can_extend_another` | a misleading status, lost jobs, broken config inheritance |

### The overall comparison (`tests/test_overall.py`)

| Test | Guards against |
|---|---|
| `test_matrix_ranks_ties_and_summaries` | wrong ranks, ties or mean ranks across datasets |
| `test_critical_difference_matches_demsar`, `test_pareto_front_keeps_only_undominated_methods`, `test_chart_is_well_formed_svg` | wrong statistics, front or chart |

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
| `test_snippets_point_at_generated_fragments` | snippets that the generator never produces (also inside tabs) |
| `test_catalog_has_every_method` | missing or malformed rubric scores |
| `test_documented_command_flags_exist` | a `python -m recbench.… --flag` in the docs or README that the module no longer accepts (checked against each module's `--help`) |
| `test_documented_scripts_exist` | instructions that name a `scripts/` or `deploy/` file that does not exist |
| `test_method_and_dataset_counts_match_the_code` | stale counts such as "18 methods": a stated count of methods or datasets must be one the code and configs produce |
| `test_method_pages_follow_the_template` | method pages missing a template section, or bake-off methods without an "In plain words" box |
| `test_readme_method_table_is_current` | a README method table that `python -m recbench.dictionary.build` would change |
| `test_code_and_catalog_agree_on_fidelity` | a method called "faithful" in one place and "simplified" in another |
| `test_every_method_is_in_the_bake_off_or_held_back` | a new method forgotten by the bake-off, or queued without a search space |
| `test_docs_build_strictly` | broken links, anchors or snippets (`mkdocs build --strict`) |

## Writing a new test

Use the `toy` or `toy_split` fixture for a ready split, `toy_fold` for its validation fold, and `FAST_CFG` from
`tests/conftest.py` for tiny model settings. Mark slow tests with `@pytest.mark.slow`.
