# Training and evaluation flow

## From the command line to MLflow

```mermaid
sequenceDiagram
    participant CLI as python -m recbench.runner
    participant P as run_pair (parent)
    participant C as child process (--single)
    participant M as method
    participant E as Evaluator
    participant ML as MLflow
    CLI->>P: for each dataset, for each method
    P->>ML: already_finished(config_hash)?
    alt finished before
        P-->>CLI: skipped_existing
    else
        P->>C: start child, wall-clock timeout
        C->>M: fit(TrainView, cfg)
        C->>E: run(method)
        E->>M: full_scores / topk / explain
        C->>ML: tags, params, metrics, artifacts
        C->>C: export bundle
        C-->>P: result.json
    end
```

Code: `src/recbench/runner.py::run_matrix` → `src/recbench/runner.py::run_pair` →
`src/recbench/runner.py::run_single`.

## The runner

1. **Resolve the config** (benchmark YAML + hardware YAML + preset): see [configuration](configuration.md).
2. For each dataset with a valid split and each method listed in the config's `methods:`:
    - compute `config_hash` (protocol version, evaluator version, the method's `impl_version`, split hash, and every setting);
    - if a finished run with that hash exists in MLflow, skip it (`skipped_existing`);
    - otherwise start `python -m recbench.runner --single ...` as a **child process** and watch it with a wall-clock
      timeout (`timeout_minutes`). If the child dies or times out, the parent records `failed` or `timeout`, with
      the reason, on the run.
3. **In the child** (`run_single`):
    - **skip checks:** managed services off, no images, or no text/categories → `unsupported`, with the reason;
    - seed NumPy and PyTorch;
    - on a validation fold (`meta.json` has `fold: valid`), attach a `ValidationMonitor` so epoch-trained methods can
      stop early, unless the config fixes `epochs`. The monitor refuses real test splits;
    - restrict the training data to the last `train_window_days` (each user keeps their last 10 events); the
      evaluator still sees the full history;
    - `fit` on the `TrainView` (timed: `train_seconds`), and log the learning curve (`curve/*` metrics, one point
      per epoch);
    - evaluate (below);
    - add `peak_rss_mb` and `peak_gpu_mb`, and managed-service statistics from `finish()`, if the method has one;
    - log everything to MLflow; export a bundle for ranked, non-managed methods.

## Tuning jobs and the queue

The quick-tier bake-off adds two layers on top of `run_pair`:

```mermaid
flowchart TD
    Q[queue.py: jobs ordered dataset by dataset] --> J[tuning/job.py::run_job, one per dataset and method]
    J --> T{time left for one more trial and the final run?}
    T -- yes --> S[Optuna suggests settings from configs/tuning/quick.yaml]
    S --> P[run_pair on the -val fold: train, stop early, score 3,000 users]
    P --> J
    T -- no, or 10 trials done --> F[run_pair once on the test split with the best settings and epoch count]
    F --> SUM[summary in runs/tuning/tier/dataset/method.json]
    SUM --> C{all of the dataset's jobs done?}
    C -- yes --> CF[run_confirm for the top 3 by validation score: full-val checks, then full test with seeds, plus a bundle]
```

- **A job** (`src/recbench/tuning/job.py::run_job`) runs Optuna's seeded TPE sampler over the method's search space
  (`src/recbench/tuning/spaces.py::MethodSpace`), at most `trials` times. Each trial is one `run_pair` on the fold,
  with its own time limit: a fair share of the job's remaining time for training (`fit_deadline`), and everything
  left as the hard timeout. Finished trials live in an Optuna journal, so an interrupted job resumes. With
  `retry=True`, a failed job starts a fresh attempt.
- **The final run** uses the best trial's settings and its best epoch count (`epochs`), so the test split is used
  exactly once per job.
- **A confirmation** (`src/recbench/tuning/job.py::run_confirm`) re-checks the setting that depends on data size on
  `full-val`, then runs the final test on `full` once per seed. Its first final run exports the serving bundle.
- **The queue** (`src/recbench/queue.py::Queue`) orders jobs dataset by dataset, gives CPU jobs to CPU workers with
  a thread cap and GPU jobs to GPU slots (`CUDA_VISIBLE_DEVICES`), starts the next dataset early only when a worker
  would otherwise idle, waits for `after:` dependencies (the re-rankers wait for EASE and ItemKNN), queues the
  confirmations, and writes its state and status page. Everything resumes from the summaries on disk.

## The evaluator, step by step

`src/recbench/evaluation.py::Evaluator.run`:

1. **Users:** warm evaluation users from `eval_users.parquet`, optionally subsampled with the run's seed to
   `max_eval_users` (the smoke config uses 2,000), so every method sees the same users.
2. **Batches:** users are scored in batches sized so that at most ~250 million scores are in memory.
3. **Scores:** `method.full_scores(users, history)`. Pointwise methods are scored over item chunks; list-only
   methods return lists.
4. **Masks** (`Evaluator._masks`): padding item 0; items the user saw before the cutoff (under `exclude_seen`);
   cold items for methods that cannot score them.
5. **Top 50** per user and per repeat policy (`src/recbench/protocol.py::top_k`).
6. **Sampled check:** the method's scores at the 101 candidates → sampled ranks, per-user AUC, log loss.
7. **Explanations:** for 50 users × 3 items → `personal_explanation_rate`, and up to 30 examples saved.
8. **Metrics:** each registered metric whose tasks overlap the method's tasks is computed on a `MetricContext`.
   Secondary repeat policies get prefixed names (`allow_repeats/ndcg_at_10`) and accuracy metrics only.
9. **Confidence intervals:** a bootstrap with 1,000 resamples for the headline metrics.
10. **Cold users:** for methods with `handles_cold_users`, the same procedure on cold users → `cold_users/*`.

## What is logged per run

| MLflow field | Contents |
|---|---|
| tags | dataset, method, tier, preset, hardware, `protocol_version`, `config_hash`, `config_group` (the hash without the seed: runs that differ only in seed share it and are averaged), `split_hash`, managed, ranked, fidelity, tasks, status, reason, bundle, `stage` (benchmark, search, final, confirm), `tuning` (defaults or tuned), `impl_version`, `eval_version`, `recbench_version`, `git_sha`, `git_dirty` |
| params | every numeric or text setting the method received, plus `fit.*` values the method reports |
| metrics | every computed metric, `*_ci_low` and `*_ci_high`, `n_eval_users`, efficiency, and later `served_*` from the load tester |
| artifacts | `per_user_metrics.npz`, `explanations.json`, `metric_errors.json` (if any), `error.txt` (failed runs) |

## Seeds and reproducibility

- Split sampling, eval-user sampling, and candidate negatives use fixed seeds, recorded in `meta.json`.
- Training uses `seed` (default 42). PyTorch on CPU is reproducible run-to-run for most models; GPU kernels can
  differ slightly.
- The full-data confirmations run seeds 42, 43 and 44 for methods whose training is random (`MethodSpec.deterministic`
  is False). The reports average such runs by `config_group` and show the spread (`ndcg_at_10_seed_sd`).
- Bootstrap intervals use their own fixed seed.
