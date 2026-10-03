# 2. Your first smoke run

**Goal:** download a dataset, split it, train and evaluate several methods, and produce a report.
**Time:** about 10 minutes for one dataset and three methods; one to two hours for everything.
**You need:** the [installation](install.md) done and the environment activated.

## Step 1: a small run

```bash
./scripts/run_smoke_cpu.sh --datasets movielens-25m --methods most_popular,ease,sasrec
```

The script prints a lot. Here is what each stage does:

| Stage | Command inside the script | Output |
|---|---|---|
| set-up | creates or uses `.venv`, installs extras, fetches third-party code | — |
| **prepare** | `python -m recbench.pipeline.prepare ...` | `data/raw/`, `data/clean/`, `data/splits/movielens-25m/smoke/` |
| **run** | `python -m recbench.runner ...` | runs in `runs/mlflow/`, bundles in `data/bundles/` |
| **report** | `python -m recbench.report.build --tier smoke ... --docs` | `reports/smoke-<time>/report.md` and `.html`, docs leaderboards |
| **docs facts** | `python -m recbench.dictionary.build` | `docs/generated/` |

The run log is saved under `runs/logs/`.

## Step 2: what happened during "prepare"

1. **Download:** `ml-25m.zip` (about 250 MB) into `data/raw/movielens-25m/`.
2. **Clean:** convert to three standard tables (interactions, items, users) in `data/clean/movielens-25m/`.
3. **Split:** compute the test cutoff on the full data (the last 10% of ratings by time), sample about 50,000
   events' worth of users for the smoke tier, and write the split files. See
   [data pipeline](../codebase/data-pipeline.md) for every file.

Open `data/splits/movielens-25m/smoke/meta.json` to see the cutoff dates and how many warm evaluation users the
split has.

## Step 3: what happened during "run"

For each method, recbench starts a **separate process** that:

1. builds a `TrainView` (pre-test events only) and calls the method's `fit`;
2. ranks the whole catalog for each warm evaluation user, removing seen items;
3. computes metrics with confidence intervals;
4. logs everything to MLflow and exports a serving bundle.

Each method prints one line, for example:

```json
{"status": "finished", "metrics": {"ndcg_at_10": 0.079, "recall_at_10": 0.073}, "dataset": "movielens-25m", "method": "ease", "seconds": 9.3}
```

Possible statuses:

| Status | Meaning |
|---|---|
| `finished` | trained, evaluated, and logged |
| `unsupported` | the method cannot run here, with a reason (for example "no item images on disk") |
| `failed` | an error; the traceback is saved in MLflow as `error.txt` |
| `timeout` | exceeded `timeout_minutes` (wall clock) |
| `skipped_existing` | an identical run (same settings, data, protocol, and method implementation version) already finished |

## Step 4: run everything

```bash
./scripts/run_smoke_cpu.sh
```

This prepares all five datasets (H&M and RetailRocket need the Kaggle token) and runs all default methods.
Expect one to two hours on an M4 Pro, and about 15 GB of disk for raw and clean data. DIN is excluded from the
laptop default because its full-catalog evaluation is very slow on a CPU; add it with `--methods din` if you want.

You can stop and restart at any time: finished pairs are skipped (`skipped_existing`).

## Step 5: look at the report

```bash
open reports/smoke-*/report.html      # macOS; or open the .md in your editor
```

**Next:** [reading the results](reading-results.md).

## Troubleshooting

- *"Kaggle credentials missing"*: see [install, step 5](install.md#step-5-a-kaggle-token-for-hm-and-retailrocket).
- *"Need at least 30 GB free"*: free disk space, or set `DATA_DIR` to a folder on another drive.
- More in [troubleshooting](../how-to/troubleshooting.md).
