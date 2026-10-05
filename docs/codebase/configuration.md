# Configuration reference

A run is configured by a **benchmark file** (what to run) that points at a **hardware file** (where it runs),
plus a model-size **preset**. `src/recbench/config.py::resolve_run_config` merges them; each method receives a
flat dictionary from `src/recbench/config.py::method_config`.

## Benchmark file (`configs/benchmarks/*.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `extends` | — | another benchmark file to start from; this file's keys replace its keys, and nested mappings (`tuning:`, `queue:`) merge key by key |
| `tier` | smoke | which split to use: smoke, standard, quick, slice, full (tuning adds the `-val` fold) |
| `hardware` | configs/hardware/local-cpu.yaml | the hardware profile |
| `preset` | (hardware's) | the model-size preset, e.g. `quick` |
| `overrides` | {} | settings that replace the preset's and the hardware's, e.g. `{max_epochs: 30}`. In tuning jobs, a search space's own values win over these; use `tuning.force` to override them |
| `seed` | 42 | training seed (also used for eval-user subsampling) |
| `resume` | true | skip pairs whose `config_hash` already finished |
| `continue_on_error` | true | keep going when a method fails |
| `timeout_minutes` | 240 | wall-clock limit per (dataset, method) |
| `managed_services` | false | allow remote services such as Recombee |
| `export_bundles` | true | write serving bundles after evaluation |
| `datasets` | [] | dataset names |
| `methods` | [] | method names |
| `method_params` | {} | per-method settings, e.g. `{ease: {ease_lambda: 300}}` |
| `max_steps` | (preset) | overrides the preset's training steps |
| `eval` | {} | evaluation settings, e.g. `{max_eval_users: 2000}` |
| `tier_overrides` | {} | per-tier split settings for `prepare`, e.g. `{smoke: {min_eval_users: 10}}` |
| `workspace` | — | keep this file's results apart: `lab` sends runs, summaries and reports to `runs/lab/` and `reports/lab/` (`src/recbench/paths.py`) |
| `track_code` | false | put each method's source into its runs' identity, so runs compute again after a code edit (the lab) |

Quick-tier files (`quick.yaml`, `quick-smoke.yaml`) add four sections, read by `recbench.tuning` and
`recbench.queue`:

| Key | Default | Meaning |
|---|---|---|
| `tuning.spaces` | configs/tuning/quick.yaml | the search spaces |
| `tuning.trials` | 10 | settings tried per job (a method's space may change it, e.g. Random tries 1) |
| `tuning.search_users` | 3,000 | validation users that score each trial (the same seeded sample every time) |
| `tuning.cap_minutes` | 180 | wall-clock limit per job, tuning and final run included |
| `tuning.force` | {} | settings that win over the search spaces, e.g. `{max_epochs: 3}` in the laptop dry run (`quick-smoke.yaml`) |
| `tuning.final_seeds` | — | test methods whose training is random once per seed and average them (the lab: 42, 43, 44) |
| `queue.methods` | [] | job order inside a dataset: `{name, resource: cpu or gpu, after: [methods of the same dataset]}` |
| `queue.jobs_per_gpu` | 3 | GPU jobs that share one GPU |
| `queue.cpu_workers` | cores ÷ 16 (1 to 8) | parallel CPU jobs; `--cpu-workers` overrides it |
| `queue.max_threads_per_job` | 32 | BLAS/OpenMP threads per CPU job |
| `confirm.top`, `confirm.tier`, `confirm.seeds` | 0, full, [] | how many of a dataset's best methods are re-checked, on which tier, with which seeds |
| `write_docs` | true | write docs fragments when a dataset finishes (`quick.yaml` turns it off: the box's checkout stays clean) |

Queue command-line options (`python -m recbench.queue run`): `--datasets`, `--methods`, `--hardware`,
`--deadline-hours`, `--stop-after-dataset`, `--retry-failed`, `--cpu-workers`, `--price-per-hour` (shown as the
session's cost by `status` and on the status page `reports/queue/<tier>.html`), and `--rerun` (with `--methods`: run
those methods' finished jobs again with fresh studies, and confirm them again where they are still top methods,
after a [promotion](../handbook/promote.md)), and `--reconfirm` (with `--methods`: repeat only those methods'
full-data confirmations, keeping their tuning). `python -m recbench.queue status`
takes `--state` (another machine's copied state file) and `--html` (write the page there too).

### Search spaces (`configs/tuning/quick.yaml`)

```yaml
version: 1                      # part of each study's name; change it when the spaces change
common:                         # added to every method unless it says `common: false`
  train_window_days: {type: choice, values: [null, 30, 90, 365]}
methods:
  ease:
    params:
      ease_lambda: {type: log, low: 1.0, high: 20000.0}
    confirm: {param: ease_lambda, factors: [0.5, 1.0, 2.0], scale_with: users}
```

Parameter types: `choice` (a list, `null` allowed), `log` and `float` (real numbers, log or linear scale), `int`
and `logint`. A method can also set `fixed: {...}` (always used) and `trials: N`. `confirm` names the setting that
depends on data size: on full data it is tried at these multiples, after scaling by the ratio of users when
`scale_with: users`. The format is documented in `src/recbench/tuning/spaces.py`.

## Hardware file (`configs/hardware/*.yaml`)

| Key | Meaning |
|---|---|
| `name` | stored on every run as the `hardware` tag |
| `device` | `cpu`, `cuda`, `mps`, or `auto` (CUDA if available); a missing GPU falls back to the CPU |
| `default_preset` | `cpu`, `24gb`, or `48gb` |
| `overrides` | settings that override the preset, e.g. `ease_max_items` |

## Presets (`PRESETS` in `src/recbench/config.py`)

| Preset | dim | layers | heads | seq_len | batch_size | lr | max_steps |
|---|---|---|---|---|---|---|---|
| cpu | 32 | 2 | 2 | 50 | 128 | 1e-3 | 400 |
| 24gb | 64 | 2 | 2 | 50 | 256 | 1e-3 | 10,000 |
| 48gb | 128 | 3 | 4 | 200 | 512 | 1e-3 | 30,000 |
| quick | 64 | 2 | 2 | 50 | 256 | 1e-3 | 10,000, and `max_epochs` 30 |

`./scripts/run_full_gpu.sh` picks `24gb` or `48gb` from the GPU memory; `--preset` on the runner overrides it.

## Method settings (with defaults)

| Method | Keys |
|---|---|
| every method | `train_window_days` (none): train on the last N days only, each user keeping their last `train_window_keep_last` (10) events |
| most_popular | `pop_window_days` 28, `pop_half_life_days` (none) |
| itemknn | `knn_neighbors` 100, `knn_shrink` 10.0, `knn_weighting` none (tfidf, bm25), `decay_half_life_days` (none) |
| ease | `ease_lambda` 500.0, `ease_max_items` 30,000 (20,000 on the laptop profile), `ease_backend` auto (numpy, torch), `decay_half_life_days` |
| rp3beta | `rp3_alpha` 1.0, `rp3_beta` 0.5, `rp3_neighbors` 200 |
| puresvd | `svd_factors` 128 |
| slim | `slim_alpha` 1e-3, `slim_l1_ratio` 0.1, `slim_neighbors` 100, `slim_max_iter` 100 |
| sansa | `sansa_lambda` 500.0, `sansa_weights_per_item` (not set) or `sansa_density` 1e-3, `sansa_factorizer` cholmod (icf) |
| vsknn | `vsknn_k` 100, `vsknn_sample` 1,000, `vsknn_weighting` div, `vsknn_idf` false, `vsknn_last_n` (none: the latest session) |
| gfcf | `gfcf_alpha` 0.3, `gfcf_k` 256 |
| turbocf | `turbocf_alpha` 0.5, `turbocf_power` 1.0, `turbocf_filter` 1, `turbocf_max_items` 30,000 |
| simplex | `simplex_margin` 0.8, `simplex_negatives` 100, `simplex_neg_weight` 150, `simplex_gamma` 0.5, `simplex_history` 50 |
| directau | `directau_gamma` 1.0, `directau_l2` 1e-6 |
| ultragcn | `ultragcn_negatives` 200, `ultragcn_neg_weight` 200, `ultragcn_gamma` 1e-4, `ultragcn_lambda` 1e-3, `ultragcn_neighbors` 10 |
| multvae, recvae | `vae_hidden` 600, `vae_latent` 200, `vae_dropout` 0.5, `vae_beta_cap` 0.2, `vae_anneal_epochs` 10; recvae also `recvae_gamma` 0.005 |
| gru4rec | `gru4rec_loss` bpr-max, `gru4rec_hidden` 224, `gru4rec_batch_size` 80, `gru4rec_lr` 0.05, `gru4rec_epochs` 10, and the official code's other settings |
| text_knn | `text_profile_window` 20, `text_position_decay` 0.9, `text_encoder` all-MiniLM-L6-v2 |
| lgbm_rerank, dcnv2_rerank | `rerank_candidates` 200, `rerank_text` false, `rerank_train_users` 20,000; lgbm also `lgbm_leaves` 31, `lgbm_lr` 0.05, `lgbm_min_child` 20, `lgbm_trees` 500 |
| ials | `ials_reg` 0.01, `ials_alpha` 10.0, `ials_iterations` 15 |
| bpr_mf | `bpr_lr` 0.01, `bpr_reg` 0.01, `bpr_iterations` 100 |
| lightgcn, xsimgcl | `graph_layers` 2, `graph_reg` 1e-4, `graph_batch_size` 2048 |
| xsimgcl | `xsim_eps` 0.2, `xsim_lambda` 0.2, `xsim_tau` 0.15, `xsim_layer_cl` 1 |
| sasrec, hstu, bert4rec, s3rec | `dropout` 0.2; bert4rec also `mask_ratio` 0.2 |
| din | `din_negatives` 4 |
| dcnv2 | `dcn_negatives` 4 |
| text_hash_tower, multimodal_tower | `hash_features` 2048 |
| recombee | `recombee_max_requests` 90,000, `recombee_max_polls` 20, `recombee_poll_seconds` 30, `recombee_reset_wait_seconds` 30, `recombee_scenario` (none) |

All neural methods also read `dim`, `layers`, `heads`, `seq_len`, `batch_size`, `lr`, `seed` and `device` from the
preset. The bake-off's neural methods train in epochs: they read `epochs` (a fixed count), `max_epochs` (30 unless
the search space fixes another limit) and `patience` (3) for early stopping on a validation fold, and `amp` (true:
bf16 on CUDA). Only the older, held-back neural methods read `max_steps`. Evaluation reads `seq_len`, `max_eval_users`, and optionally `eval_batch_users` and
`eval_score_budget` (maximum scores in memory, default 250 million). Bundles read `bundle_k` (100) and
`bundle_users` (20,000).

Changing any of these changes the run's `config_hash`, so the run is redone rather than skipped.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DATA_DIR` | everything | data folder (default `./data`) |
| `RECBENCH_ROOT` | runner, results | repository root for relative paths |
| `MLFLOW_TRACKING_URI` | runner, reports | MLflow store (default `file://<root>/runs/mlflow`) |
| `RECBENCH_BUNDLES`, `RECBENCH_TIER`, `RECBENCH_SERVE_METHODS` | API | where bundles are, which tier to serve, optional allow-list |
| `RECBENCH_REQUEST_LOG` | API | `0` turns off the JSON log line per request |
| `RECBENCH_ID_TOKEN` | load tester | identity token for a private Cloud Run service |
| `RECBENCH_RECOMBEE_DB`, `_TOKEN`, `_REGION`, `_ALLOW_RESET`, `_KEEP` | Recombee | credentials and reset behaviour |
| `RECBENCH_LIVE` | tests | `1` enables live external tests |
| `CUDA_VISIBLE_DEVICES` | PyTorch | the smoke script sets it empty to force the CPU |
| `PYTHONWARNINGS` | Python | the scripts set `ignore` to hide RecBole's pandas warnings |
| `RECBENCH_METHOD_MODULES` | `recbench.methods` | set by the runner for each run's child process: import only these method modules (keeps PyTorch out of CPU-only runs) |
| `RECBENCH_WORKSPACE` | runner, tuning, queue, reports | the active workspace (set from a benchmark file's `workspace`; child processes inherit it). In a workspace, `MLFLOW_TRACKING_URI` is ignored: the store is always `runs/<workspace>/mlflow` |
| `RECBENCH_UPDATE_GOLDEN` | tests | `1` rewrites `tests/golden/lab_defaults.json` (after an intended default change) |

## The bundled configs

| File | Purpose |
|---|---|
| `configs/benchmarks/smoke-cpu.yaml` | laptop: smoke tier, all datasets, the 16 original methods except DIN (add newer ones with `--methods`), ≤ 2,000 eval users |
| `configs/benchmarks/quick.yaml` | rented GPU box: the quick-tier bake-off, dataset by dataset; `queue.methods` lists the bake-off's methods and `held_back` the others ([how](../start/quick-tier-box.md)) |
| `configs/benchmarks/quick-smoke.yaml` | laptop: a free dry run of `quick.yaml` on the smoke splits |
| `configs/benchmarks/lab.yaml` | the improvement lab: the 11 lab methods plus MostPopular on the quick tier, in the lab workspace, 3 test seeds for random methods, top-10 lists saved ([labs](../labs/index.md)) |
| `configs/hardware/lab-cpu.yaml` | the lab's machine profile on any machine: CPUs only, EASE item cap 20,000 |
| `configs/benchmarks/gpu-full.yaml` | GPU machine: full tier, the 17 original methods with default settings, 8-hour timeout per method |
| `configs/benchmarks/archive/gpu-12h.yaml` | the config that produced the untuned v0.2 full results (kept for provenance; see `scripts/archive/`) |
| `configs/hardware/box.yaml` | a rented GPU box: device auto, preset `48gb`, EASE item cap 30,000 |
| `configs/tuning/quick.yaml` | the bake-off's search spaces, one per method |
| `configs/benchmarks/recombee-slice.yaml` | Recombee vs six local methods on the free-plan-sized slice |
