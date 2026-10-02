# 5. The full tier on a GPU machine

**Goal:** run the benchmark on the complete datasets with larger models on your Ubuntu NVIDIA machine.
**Time:** setup about an hour; the run takes several hours to a day, depending on the GPU.
**You need:** an NVIDIA GPU with at least 24 GB of memory, 64 GB of RAM (128 GB recommended for the 48 GB
preset), and about 80 GB of free disk.

## What changes compared with the smoke tier

| | Smoke (laptop) | Full (GPU) |
|---|---|---|
| Data | ~50,000 events per dataset (sampled users) | complete datasets (up to 31.8 million events) |
| Eval users | at most 2,000 | at most 10,000 warm users per dataset |
| Model preset | `cpu`: dim 32, 400 steps | `24gb`: dim 64, 10,000 steps; `48gb`: dim 128, 3 layers, length 200, 30,000 steps |
| Methods | DIN excluded | all, including DIN |
| Purpose | check the pipeline | choose a method |

## Step 1: GPU driver and CUDA PyTorch

Check that the driver works:

```bash
nvidia-smi       # should list your GPU and a CUDA version
```

Install recbench as on the laptop (see [install](install.md)), but with the `gpu` extra. PyTorch must be a
**CUDA build**; take the exact command for your CUDA version from <https://pytorch.org/get-started/locally/>.
For example:

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the URL for your CUDA version
uv pip install -e ".[gpu,docs]"
bash scripts/fetch_third_party.sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and your GPU's name.

Then set up the Kaggle token (see [install, step 5](install.md#step-5-a-kaggle-token-for-hm-and-retailrocket)).

## Step 2: run inside tmux

The run takes hours. Use `tmux`, so that it keeps running if your SSH connection drops:

```bash
tmux new -s bench
./scripts/run_full_gpu.sh
# detach with Ctrl-b then d; re-attach later with: tmux attach -t bench
```

The script:

1. checks that it runs on Linux with a visible NVIDIA GPU, and picks the preset from the GPU memory (`24gb` from
   23 GB, `48gb` from 46 GB);
2. checks disk space and Kaggle credentials;
3. prepares the full splits, runs every method in `configs/benchmarks/gpu-full.yaml` (each in its own process,
   with an 8-hour timeout per method), writes the report and the docs leaderboards.

Run a subset with `--datasets hm --methods sasrec,hstu`. Re-running skips finished pairs.

## Step 3: bring the results to your laptop

MLflow stores runs as files, but do **not** copy the GPU machine's folder into the laptop's `runs/mlflow/`.
Each machine created its own `recbench` experiment (with a random id and absolute paths), and MLflow would read
only one of them. Copy it next to yours and import it instead:

```bash
# on the laptop
rsync -a gpu-box:recommendation_benchmark/runs/mlflow/ runs/mlflow-gpu/
python -m recbench.import_runs runs/mlflow-gpu      # re-creates the GPU runs in runs/mlflow
python -m recbench.report.build --tier full --out reports/full-latest --docs
python -m recbench.dictionary.build
```

Importing again later only adds new runs. Runs still in progress on the GPU machine are skipped until they end.

The documentation's leaderboards then include the full tier.

## Expected heavy hitters

- **EASE** on large catalogs is capped at 30,000 items; its memory peaks around 11 GB.
- **RecBole models** (BERT4Rec, S3-Rec, DIN) build in-memory sequence datasets: about 13 GB for MovieLens at
  length 50, and four times that at length 200. recbench checks available memory first and skips with a
  reason if it would not fit.
- **DIN** scores every (user, item) pair: expect around half an hour to an hour per large dataset on a 24 GB GPU.

## Troubleshooting

- `torch.cuda.is_available()` is False: wrong PyTorch build (CPU-only) or a driver mismatch; reinstall from the
  PyTorch site's command.
- Out of GPU memory: use the `24gb` preset (`--preset 24gb` in the runner), or lower `batch_size` in
  `configs/benchmarks/gpu-full.yaml` under `method_params`.
- More in [troubleshooting](../how-to/troubleshooting.md).

**Next:** [deploy to Cloud Run](deploy-cloud-run.md).
