# 1. Install on your Mac

**Goal:** a working Python environment with recbench and its tests passing.
**Time:** about 20 minutes (most of it downloading PyTorch).
**You need:** macOS on Apple Silicon (Intel Macs and Linux work the same way), about 30 GB of free disk for
the datasets later, and an internet connection.

## Step 1: tools

[Homebrew](https://brew.sh) installs developer tools on macOS. Then install `uv`, a fast Python package and
environment manager, and git:

```bash
brew install uv git
uv --version          # e.g. uv 0.9.x
```

**What just happened:** `uv` will create an isolated Python environment for this project, so its packages do not
conflict with anything else on your machine.

## Step 2: get the code

```bash
git clone <your repository URL> recommendation_benchmark
cd recommendation_benchmark
```

(If you already have the folder, just `cd` into it.)

## Step 3: create the environment and install

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

- `.venv` is a *virtual environment*: a private copy of Python for this project. `source .venv/bin/activate`
  switches your terminal to it (your prompt shows `(.venv)`). Run it again in every new terminal.
- `-e` installs recbench in *editable* mode: changes to `src/recbench` take effect without reinstalling.
- `[dev]` selects an **extra**: a named group of optional dependencies from `pyproject.toml`. `dev` is the
  laptop setup: `cpu`, `managed`, and `docs`, plus pytest. The extras are:

| Extra | Adds |
|---|---|
| `serve` | the HTTP API only (what the Docker image uses) |
| `bench` | data pipeline, evaluation, MLflow, simple baselines, scikit-learn, LightGBM (the re-ranker) and Optuna (tuning) |
| `cpu` / `gpu` | `bench` plus PyTorch, RecBole, FuxiCTR, numba, kmeans-pytorch and sentence-transformers for the neural and text methods |
| `sansa` | the `sansa` package for SANSA. It needs SuiteSparse first: `brew install suite-sparse` on a Mac (`setup_box.sh` installs it on a rented box). Without it, SANSA's jobs end `unsupported` |
| `managed` | the Recombee SDK |
| `docs` | MkDocs and Material for this site |
| `dev` | `cpu`, `managed`, and `docs`, plus pytest (use this on the laptop) |
| `lab` | the lab notebooks: a Jupyter kernel for VS Code, matplotlib, and the notebook runner used by the tests ([labs](../labs/index.md)) |

## Step 4: third-party model code

Three models reuse code from other repositories, pinned to exact commits:

```bash
bash scripts/fetch_third_party.sh
```

You should see three lines like `SELFRec 5b0229423c…`, `generative-recommenders ea7b85f1…` and
`GRU4Rec_PyTorch_Official d1fc3110…`. The code lands in `third_party/` (not tracked by git). The official GRU4Rec code
allows research and education only; see [security](../cloud/security.md#third-party-code).

## Step 5: a Kaggle token (for H&M and RetailRocket)

1. Sign in at <https://www.kaggle.com>, open **Settings → API → Create New Token**; a file `kaggle.json` downloads.
2. Move it into place and make it private:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. For H&M, also open the [competition page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
   and accept its rules.

## Step 6: check that everything works

```bash
pytest -q -m "not slow"
```

You should see only dots and a final summary with no failures (a few `s` for skipped live tests are fine). This
takes a minute or two. It builds small synthetic datasets, so no downloads are needed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: uv` | open a new terminal after `brew install`, or run `eval "$(/opt/homebrew/bin/brew shellenv)"` |
| tests skip "third_party ... not fetched" | run step 4 |
| `ModuleNotFoundError: recbench` | activate the environment: `source .venv/bin/activate` |
| very slow install | PyTorch is large; let it finish |

More in [troubleshooting](../how-to/troubleshooting.md).

## Ubuntu or another Linux

The same commands work. Install `uv` with `curl -LsSf https://astral.sh/uv/install.sh | sh`. For an NVIDIA GPU, see
[the full tier on a GPU machine](full-tier-gpu.md).

**Next:** [your first smoke run](first-smoke-run.md).
