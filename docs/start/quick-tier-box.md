# 5. The quick-tier bake-off on a rented GPU box

**Goal:** find the best low-budget method for each dataset. 23 methods are tuned with the same budget on about
one million events per dataset, then each dataset's top 3 are re-checked on the full data.
**Time:** about 30 minutes to set up, then roughly 1.5 to 2 hours per dataset. That is an estimate; the first
dataset gives the real number.
**Cost:** about $1 to $2 per dataset at $0.5 to $1 per GPU-hour, so $7 to $15 for all five.
**You need:** the laptop set up as in [install](install.md), a [vast.ai](https://vast.ai) account (or any
provider of Linux GPU machines) with some credit, and an SSH key added to that account.

## What runs

| Unit | What it is |
|---|---|
| **Job** | One method on one dataset. It tries up to 10 settings on the validation fold `quick-val`, scoring the same 3,000 users each time, then runs the best setting **once** on the test split `quick`. A job never takes more than **3 hours**, its tuning included. |
| **Block** | All 23 jobs of one dataset, then the **confirmation** of its top 3 by test NDCG@10: the setting that depends on data size (for example EASE's λ) is re-checked on `full-val`, then the final test runs on `full`, three times with different seeds for methods whose training is random. |
| **Order** | MovieLens-25M → RetailRocket → Steam → H&M → Last.fm. A worker that would otherwise wait starts the next dataset early ("backfill"). |

The rules and the reasons behind them are on the [quick-tier page](../results/quick-tier.md).

## Step 0: dry run on the laptop (free)

Run the whole pipeline at toy scale before paying for anything:

```bash
source .venv/bin/activate
python -m recbench.pipeline.prepare --config configs/benchmarks/quick-smoke.yaml --tier smoke,smoke-val
python -m recbench.queue run --config configs/benchmarks/quick-smoke.yaml --cpu-workers 3
python -m recbench.queue status --config configs/benchmarks/quick-smoke.yaml
```

`configs/benchmarks/quick-smoke.yaml` reuses the real queue and search spaces (`extends: quick.yaml`) with 2
settings per job and at most 3 epochs. It takes about 20 minutes on an M4 Pro. Every job should end `finished`,
except the two re-rankers on H&M and Last.fm: their smoke splits have too few recent users to train on ("too few
recent users with a reachable next item"). Ignore the scores: smoke splits have only a few hundred users.

## Step 1: prepare the splits on the laptop

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/quick.yaml --tier quick,quick-val,full,full-val
```

| Tier | What it holds | Used for |
|---|---|---|
| `quick` | about 1M events per dataset, sampled by user, with the real test window | the single test run of each job |
| `quick-val` | the same users with the test window deleted; the validation window plays the test window | the tuning trials |
| `full` | every event | the confirmation's final test |
| `full-val` | the full data's validation fold | the confirmation's size check |

With the raw data already downloaded, this takes about 5 minutes. The full folds need about 22 GB of memory
at their peak, so close other large apps.

## Step 2: rent the box

| | Choose | Why |
|---|---|---|
| GPU | 1 × 48 GB (RTX A6000, A40, L40S) or 2 × 24 GB | three GPU jobs share each GPU; EASE and Turbo-CF hold item × item matrices |
| CPU | 32 cores or more | 11 methods run on the CPU; each CPU worker gets 16 to 32 threads |
| RAM | 128 GB (64 GB at least) | a confirmation job loads up to 31 million events |
| Disk | 100 GB | splits, the Python environment, and the MLflow runs |
| Image | the PyTorch template (Ubuntu 22.04 or 24.04) | the NVIDIA driver comes with it |
| Driver | "CUDA" filter 12.8 or higher | recent PyTorch builds need a recent driver; `setup_box.sh` picks the build that matches, and stops with a clear message if none can use the GPU |

Check the hourly price before you start it. On the instance's **Connect** button, vast.ai shows the SSH
address and port. Add them to `~/.ssh/config` on the laptop, so that `ssh vast-gpu` works:

```text
Host vast-gpu
    HostName <address from vast.ai>
    Port <port from vast.ai>
    User root
```

!!! warning "Stopping is not enough"
    A *stopped* vast.ai instance still bills for its disk. When you are done, **destroy** it (Step 6).

## Step 3: set up the box

```bash
ssh vast-gpu
git clone https://github.com/trunghieu11/recbench.git && cd recbench
./scripts/setup_box.sh
```

`setup_box.sh`:

1. installs SuiteSparse and a C compiler (for SANSA), tmux and rsync;
2. creates `.venv` and installs recbench with CUDA PyTorch, LightGBM, Optuna and sentence-transformers;
3. tries to install SANSA. If its build fails, setup goes on and SANSA's jobs are recorded as unsupported;
4. downloads the text-embedding model once and fetches GRU4Rec and the other third-party code at pinned commits;
5. records the exact package versions in `runs/logs/pip-freeze-*.txt`.

It ends by printing the GPUs it sees. Check that it says `CUDA available: True`. If the driver is too old for
PyTorch, it stops and tells you what to do.

If the instance keeps its files under `/workspace` (vast.ai's default for many templates), clone there instead,
and give that path to the scripts below as a second argument.

## Step 4: copy the splits from the laptop

On the laptop:

```bash
./scripts/upload_splits.sh vast-gpu        # or: ./scripts/upload_splits.sh vast-gpu /workspace/recbench
```

It sends every dataset's quick tiers first (about 0.5 GB), then the full tiers dataset by dataset (about 8 GB).
You can start Step 5 as soon as it prints `uploaded lastfm/quick-val`. A dataset's full splits are only needed
when its block reaches the confirmation, an hour or more later.

??? note "Alternative: prepare the data on the box"
    The box needs your Kaggle token for H&M and RetailRocket:

    ```bash
    ssh vast-gpu 'mkdir -p ~/.kaggle && chmod 700 ~/.kaggle'
    scp ~/.kaggle/kaggle.json vast-gpu:.kaggle/
    ```

    Then, on the box:

    ```bash
    source .venv/bin/activate
    python -m recbench.pipeline.prepare --config configs/benchmarks/quick.yaml --tier quick,quick-val,full,full-val
    ```

    It downloads about 11 GB and takes 20 to 30 minutes. The splits are identical to the laptop's: they have the
    same split hashes.

## Step 5: run, one dataset at a time

On the box:

```bash
cd recbench
tmux new -s quick
./scripts/run_quick_box.sh --stop-after-dataset
```

- `--stop-after-dataset` finishes the first unfinished dataset, with its confirmation, and then stops. That is a
  clean moment to look at the results or end the rental. The same command then runs the next dataset.
- Without it, the queue runs all five datasets.
- `--deadline-hours 4` starts no new job after 4 hours. Jobs already running finish (each within its 3 hours).
- The output is also saved in `runs/logs/quick-queue-*.log`.

Detach from tmux with **Ctrl-b**, then **d**. The run goes on. Re-attach with `tmux attach -t quick`.

To watch it, open a second tmux window (**Ctrl-b**, then **c**):

```bash
source .venv/bin/activate
python -m recbench.queue status --config configs/benchmarks/quick.yaml
nvidia-smi
```

`status` lists every dataset's jobs, best first, with the test NDCG@10 of the finished ones. That is the
provisional leaderboard:

```text
== movielens-25m: 9/23 jobs done
  ease                 finished     test NDCG@10=0.2012  val=0.1830
  rp3beta              finished     test NDCG@10=0.1974  val=0.1801
  sasrec               running      test NDCG@10=   -    val=-
```

(The numbers above are made up to show the layout.)

| Status | Meaning |
|---|---|
| `finished` | tuned and tested; the test NDCG@10 is shown |
| `running` | in progress |
| `interrupted` | started in an earlier session; it continues the next time the queue runs |
| `over_budget` | no setting finished within the 3-hour cap |
| `failed` | an error; the reason is in `runs/tuning/quick/<dataset>/<method>.json` |
| `missing_split` | the splits were not there; it runs again the next time the queue starts |

Interrupting is safe. **Ctrl-c**, or a box that disappears, loses only the trials in progress. Running the command
again resumes: finished jobs are kept, and an interrupted job continues from its finished trials.

## Step 6: bring the results home, then stop paying

On the laptop, at any time, even while the box is still working:

```bash
./scripts/fetch_results.sh vast-gpu        # or: ./scripts/fetch_results.sh vast-gpu /workspace/recbench
```

It copies the MLflow runs, job summaries, queue state, logs and reports from the box. It then imports the runs
into your MLflow store, rebuilds `reports/quick-tuned/` and `reports/full-tuned/` and the docs fragments, and
prints the status. Run it again later to add new results.

When the last block you want has finished, run it once more, then **destroy** the instance in the vast.ai
console.

## What to look at

- `reports/quick-tuned/report.html`: one section per dataset, with confidence intervals, ties ("≈"), training
  time, and coverage. `reports/full-tuned/report.html` has the confirmed top 3.
- The [quick-tier page](../results/quick-tier.md) in the docs (`mkdocs serve`) shows the same leaderboards.
- `runs/tuning/quick/<dataset>/<method>.json`: every setting a job tried, its validation score and time, the best
  setting, and the test result. These feed the sensitivity analysis and the labs.

## If something goes wrong

| Symptom | What to do |
|---|---|
| `CUDA out of memory` in a GPU job | set `queue.jobs_per_gpu: 2` in `configs/benchmarks/quick.yaml`, then rerun with `--retry-failed` |
| A job `failed` | read its reason (`status`, or the job's JSON file); after a fix, rerun with `--retry-failed --methods <name>`. The retry starts a new attempt, so the failed trials do not count. |
| SANSA is unsupported | its optional install failed (see the setup output); the other methods are unaffected |
| `text_knn` cannot download its model | the box needs access to huggingface.co; rerun `./scripts/setup_box.sh` |
| A job seems stuck | `status` shows what runs; a trial can take long, but each has its own time limit inside the job's 3 hours |
| SSH dropped | the run continues inside tmux; `ssh vast-gpu`, then `tmux attach -t quick` |

More in [troubleshooting](../how-to/troubleshooting.md).

**Next:** [the full tier on a GPU machine](full-tier-gpu.md), for the heavy methods once they pass this gate.
