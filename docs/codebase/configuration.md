# Configuration reference

A run is configured by a **benchmark file** (what to run) that points at a **hardware file** (where it runs),
plus a model-size **preset**. `src/recbench/config.py::resolve_run_config` merges them; each method receives a
flat dictionary from `src/recbench/config.py::method_config`.

## Benchmark file (`configs/benchmarks/*.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `tier` | smoke | which split to use: smoke, standard, slice, full |
| `hardware` | configs/hardware/local-cpu.yaml | the hardware profile |
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

`./scripts/run_full_gpu.sh` picks `24gb` or `48gb` from the GPU memory; `--preset` on the runner overrides it.

## Method settings (with defaults)

| Method | Keys |
|---|---|
| most_popular | `pop_window_days` 28 |
| itemknn | `knn_neighbors` 100, `knn_shrink` 10.0 |
| ease | `ease_lambda` 500.0, `ease_max_items` 30,000 (20,000 on the laptop profile) |
| ials | `ials_reg` 0.01, `ials_alpha` 10.0, `ials_iterations` 15 |
| bpr_mf | `bpr_lr` 0.01, `bpr_reg` 0.01, `bpr_iterations` 100 |
| lightgcn, xsimgcl | `graph_layers` 2, `graph_reg` 1e-4, `graph_batch_size` 2048 |
| xsimgcl | `xsim_eps` 0.2, `xsim_lambda` 0.2, `xsim_tau` 0.15, `xsim_layer_cl` 1 |
| sasrec, hstu, bert4rec, s3rec | `dropout` 0.2; bert4rec also `mask_ratio` 0.2 |
| din | `din_negatives` 4 |
| dcnv2 | `dcn_negatives` 4 |
| text_hash_tower, multimodal_tower | `hash_features` 2048 |
| recombee | `recombee_max_requests` 90,000, `recombee_max_polls` 20, `recombee_poll_seconds` 30, `recombee_reset_wait_seconds` 30, `recombee_scenario` (none) |

All neural methods also read `dim`, `layers`, `heads`, `seq_len`, `batch_size`, `lr`, `max_steps`, `seed`, and
`device` from the preset. Evaluation reads `seq_len`, `max_eval_users`, and optionally `eval_batch_users` and
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
| `RECBENCH_ID_TOKEN` | load tester | identity token for a private Cloud Run service |
| `RECBENCH_RECOMBEE_DB`, `_TOKEN`, `_REGION`, `_ALLOW_RESET`, `_KEEP` | Recombee | credentials and reset behaviour |
| `RECBENCH_LIVE` | tests | `1` enables live external tests |
| `CUDA_VISIBLE_DEVICES` | PyTorch | the smoke script sets it empty to force the CPU |
| `PYTHONWARNINGS` | Python | the scripts set `ignore` to hide RecBole's pandas warnings |

## The bundled configs

| File | Purpose |
|---|---|
| `configs/benchmarks/smoke-cpu.yaml` | laptop: smoke tier, all datasets, every method except DIN, ≤ 2,000 eval users |
| `configs/benchmarks/gpu-full.yaml` | GPU machine: full tier, all methods, 8-hour timeout per method |
| `configs/benchmarks/recombee-slice.yaml` | Recombee vs six local methods on the free-plan-sized slice |
