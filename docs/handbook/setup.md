# Set up for the labs

!!! abstract "In plain words"
    You need three things: recbench installed with its `lab` extra, the quick-tier data splits on disk, and VS Code
    able to run the lab notebooks. This page checks each one. Step 5 computes the baseline, the numbers to beat, on
    a rented box and brings it back to the laptop.

**Time:** about 15 minutes if recbench is already installed ([Install on your Mac](../start/install.md)).

## Step 1: the lab extra

The labs add a few packages to the usual laptop setup: a Jupyter kernel for VS Code, plotting, and a notebook runner
for the tests.

```bash
cd recommendation_benchmark
source .venv/bin/activate
uv pip install -e ".[dev,lab]"
```

SANSA (lab 5) also needs the `sansa` extra, which needs SuiteSparse. If `brew install suite-sparse` and
`uv pip install -e ".[sansa]"` were done during the install, nothing more is needed.

## Step 2: the data splits

The labs use the **quick tier**, about one million events per dataset. Each dataset has two parts: the validation
fold `quick-val`, where settings are tried, and the test split `quick`, where a chosen setting is tested once.

```bash
ls data/splits/*/quick-val/meta.json data/splits/*/quick/meta.json
```

!!! success "You should see"
    Ten paths: `quick-val` and `quick` for movielens-25m, retailrocket, steam, hm and lastfm.

If some are missing, prepare them (H&M and RetailRocket need a Kaggle token, see
[install, step 5](../start/install.md#step-5-a-kaggle-token-for-hm-and-retailrocket)):

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/quick.yaml --tier quick,quick-val
```

## Step 3: VS Code and the notebooks

1. Install [VS Code](https://code.visualstudio.com), then its **Python** and **Jupyter** extensions (Extensions
   panel, search for each).
2. **File → Open Folder…** and choose the `recommendation_benchmark` folder.
3. Open `labs/01-itemknn/itemknn.ipynb`. At the top right, **Select Kernel → Python Environments → .venv**.
4. Run the first code cell (Shift+Enter). It calls `lab.setup()`.

!!! success "You should see"
    ```text
    recbench repository: /Users/you/recommendation_benchmark; workspace: lab (runs/lab, reports/lab)
    splits ready: movielens-25m, lastfm, hm, steam, retailrocket
    baselines finished (datasets): most_popular 5/5, itemknn 5/5, rp3beta 5/5, ...
    ```

`lab.setup()` finds the repository from the notebook's folder and switches to the lab workspace. It also loads only
the 11 light methods, so PyTorch is never imported. That matters on macOS: SANSA and LightGBM crash when PyTorch's
OpenMP runtime is loaded in the same process. Run it first in every notebook; after a crash, restart the kernel
(**Restart** in the notebook toolbar) and run it again.

### Run the notebooks on the rented box

Cells that fit models are training, which belongs on the box. VS Code can open the box's copy of the repository as if
it were local:

1. Install the **Remote - SSH** extension.
2. Command Palette (Cmd+Shift+P) → **Remote-SSH: Connect to Host…** → `vast-gpu` (the name from `~/.ssh/config`,
   see [renting a box](../start/box-1-before-you-rent.md)).
3. **File → Open Folder…** → the repository on the box (`~/recbench`), then install the Python and Jupyter
   extensions there when VS Code offers it.
4. Open a lab notebook and select the box's `.venv` kernel. Every cell now runs on the box.

Cells that only read stored results (the baseline, `lab.compare`, `lab.segments`) work on the laptop too, after
`./scripts/fetch_lab_results.sh`.

## Step 4: a first check

```bash
./scripts/check.sh --quick
```

!!! success "You should see"
    ```text
    == notebooks are saved without outputs
    ok

    == the lab's tests and your variant tests
    ...........................................                             [100%]

    All checks passed: ready for a pull request.
    ```

## Step 5: the baseline

The baseline is the bake-off's tuning, repeated for each of the 11 methods (and MostPopular, the floor) on each
dataset: each job tries 10 settings on the validation fold and tests the best one once. It is the set of numbers your
experiments must beat. Compute it once on a rented box: all lab methods train on CPUs, so any box with many cores
will do, for example in the same session as the bake-off.

```bash
# on the laptop: the quick splits to the box (0.5 GB; skip if the bake-off's upload already sent them)
QUICK_ONLY=1 ./scripts/upload_splits.sh vast-gpu
# on the box, inside tmux, after ./scripts/setup_box.sh
./scripts/run_lab_box.sh
# on the laptop, when it is done (or while it runs, to see progress)
./scripts/fetch_lab_results.sh vast-gpu
python -m recbench.lab status
```

[Renting a box](../start/box-1-before-you-rent.md) explains `vast-gpu`, SSH and tmux. The lab uses one machine profile
everywhere (`configs/hardware/lab-cpu.yaml`: CPUs only, EASE kept to 20,000 items), so results from the box and from
the laptop can be compared. `fetch_lab_results.sh` imports the box's runs into the laptop's lab workspace and
rebuilds the [scoreboard](../labs/scoreboard.md), which shows every job's test NDCG@10 with its 95% interval. Each lab
page starts with its method's numbers.

!!! note "What is there already"
    A first laptop run stopped part-way: MovieLens and most of Last.fm are done. `lab status` shows them, and the
    box run replaces them.

## Keep the Mac awake for long runs

If you do run something long on the laptop: macOS sleeps when idle, and a sleeping Mac pauses your jobs. Prefix the
command with `caffeinate -i`, which keeps the Mac awake while it runs, and in the background add `nohup`:
`nohup caffeinate -i python -m recbench.lab run … > log.txt 2>&1 &`
([run experiments](run-experiments.md#run-it-in-the-background) explains each part).
