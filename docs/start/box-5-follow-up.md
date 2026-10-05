# 5e. A follow-up session: finish what the first run left open

!!! abstract "In plain words"
    The first bake-off (2026-10-04) finished every job, but left a few things open. This page rents a box once
    more, for 5 to 6 hours, to close them:

    1. SASRec's last training run on RetailRocket blew up. The code now prevents that.
    2. The LightGBM re-ranker was confirmed with candidates from a badly scaled EASE. The code now scales it.
    3. Last.fm's methods were confirmed by test score instead of validation score.
    4. No serving bundles were written.
    5. The labs need their baseline (optional).

    You reuse the scripts from the first session, plus two new ones. One copies the finished job summaries to the
    new box. The other writes the missing bundles.

**Time:** about 5 to 6 hours of box time. **Cost:** a few dollars at the rate you
paid last time.

## What this session does

| Task | Why | Box time |
|---|---|---|
| Confirm the LightGBM re-ranker again on full data, on all five datasets | it takes candidates from EASE. On MovieLens and RetailRocket its confirmation used EASE's quick-tier λ unscaled on the full data: 23 times smaller than EASE's own confirmed λ on MovieLens. The code now scales that λ as a confirmation does, and fixes the generators' settings for the whole job | 3 to 4.5 h (CPU), together with the next row |
| Confirm SANSA on Last.fm (and again on Steam) | the first run's code picked the methods to confirm by test score. On Last.fm that confirmed PureSVD and SLIM; by validation, SANSA, the re-ranker and EASE are the top 3 | (same command as the row above) |
| Re-run SASRec on all five datasets | its final run on RetailRocket collapsed (test NDCG@10 0.0014 after 0.0200 on validation). A new guard in the training loop keeps the best weights when that happens. SASRec's implementation version went from 3 to 4, so all five of its jobs run again | ~1.7 h (GPU) |
| Write the missing serving bundles | the first run's code did not export bundles from confirmations | ~1 h (CPU, alongside) |
| The lab baseline (optional) | the [labs](../labs/index.md) compare your experiments with it | ~5 h (CPU, alongside) |

The first three tasks use the queue. They run as two commands, one after the other, in one tmux window. The
bundles and the lab baseline do not use the queue; they run in their own windows alongside.

## Step 1: before you rent (laptop)

1. **Push your commits.** The box clones the repository from GitHub, so it gets only what you pushed:
   `git push`.
2. **Check the splits.** The confirmations and the bundles need the full splits. SASRec and the lab need the quick
   splits. They are still in `data/splits/` from the first session: `ls data/splits/*/` lists
   `full  full-val  quick  quick-val` for each dataset.

## Step 2: rent and set up

Follow [5b](box-2-rent-and-set-up.md) as before:

- a machine with one GPU of at least 24 GB and at least 128 GB of RAM (the re-ranker alone needed 62 GB);
- `git clone`;
- `./scripts/setup_box.sh`.

## Step 3: copy the data and the job summaries (laptop)

```bash
./scripts/upload_splits.sh vast-gpu
./scripts/upload_results.sh vast-gpu
```

`upload_results.sh` is new. It copies `runs/tuning/quick/` and `runs/tuning/full/`, the summaries of every finished
job. The new box needs them:

- the queue reads them to know which jobs are done. **Without them it would tune all 115 jobs again.**
- the re-rankers read their generators' tuned settings from them.
- the bundle export reads the confirmed settings from them.

!!! success "You should see"
    ```text
    uploaded runs/tuning/quick (115 job summaries)
    uploaded runs/tuning/full (15 job summaries)
    The job summaries are on vast-gpu:recbench/runs/tuning.
    ```

## Step 4: start the work (box, inside tmux)

**Window 1, the queue.** Two queue commands, one after the other:

```bash
cd ~/recbench
./scripts/run_quick_box.sh --methods lgbm_rerank,sansa --reconfirm --cpu-workers 2 && ./scripts/run_quick_box.sh --methods sasrec --rerun
```

- `--reconfirm` moves the full-data confirmations of the re-ranker and SANSA to `archive/` and keeps their tuning.
    - The queue then confirms both methods again wherever they are among a dataset's top 3 by validation. That is
      all five datasets for the re-ranker, and Steam and Last.fm for SANSA.
    - Each confirmation's first final run writes its bundle.
- `--cpu-workers 2` runs two confirmations at a time. A re-ranker confirmation needs up to 62 GB of memory, and
  five at once would not fit in 128 GB. With 256 GB of RAM, `--cpu-workers 4` finishes sooner.
- `--rerun` moves SASRec's five job summaries to `archive/` and starts five fresh jobs.
- If a command is interrupted, start it again **without** `--reconfirm` or `--rerun`, so it resumes instead of
  starting over.

!!! success "You should see"
    ```text
    [queue] 10 jobs over ['movielens-25m', 'retailrocket', 'steam', 'hm', 'lastfm']; ...
    [queue] start confirm:movielens-25m:lgbm_rerank on cpu0
    [queue] start confirm:retailrocket:lgbm_rerank on cpu1
    ...
    [queue] start confirm:lastfm:sansa on cpu5
    ...
    [queue] 5 jobs over ['movielens-25m', 'retailrocket', 'steam', 'hm', 'lastfm']; ...
    [queue] start tune:movielens-25m:sasrec on gpu0
    ```

    SANSA on the full Last.fm data may reach the 3-hour cap and end `over_budget`. That is a result too: SANSA is
    then too slow for that data at that budget.

**Window 2, the bundles** (**Ctrl-b c** opens it). Start it once window 1 has printed its first lines, so the old
confirmations of the re-ranker and SANSA are already archived:

```bash
cd ~/recbench && ./scripts/export_bundles_box.sh
```

It refits each confirmed method that has no bundle yet, once, on its full split, with the settings its
confirmation chose. It writes `data/bundles/<dataset>/full/<method>/`. That is 10 methods: the new confirmations in
window 1 write their own. iALS on RetailRocket takes longest, about 35 minutes. A failure is reported at the end
and does not stop the others. Running the script again skips the bundles already written.

!!! success "You should see"
    ```text
    [export] 10:12:03 movielens-25m/ease ...
    {
      "bundle": "/root/recbench/data/bundles/movielens-25m/full/ease",
      "settings_from": "confirmation on full",
      ...
    }
    [export] 10:12:41 movielens-25m/ease done
    [export] movielens-25m/lgbm_rerank: its confirmation did not finish; skipped
    ```

**Window 3, the lab baseline (optional):** `cd ~/recbench && LAB_WORKERS=3 ./scripts/run_lab_box.sh`, as in the
[handbook](../handbook/setup.md#step-5-the-baseline).
- Start it when window 1 has moved on to SASRec, since the lab's re-ranker jobs need 30 to 45 GB each too.
- With 256 GB of RAM, start it at once and leave out `LAB_WORKERS=3`.

Watch everything as in [5c](box-3-run-and-monitor.md): the status page, `htop`, `nvidia-smi`.

## Step 5: fetch, check, destroy

On the laptop, when windows 1 and 2 are done (and window 3, if you started it):

```bash
./scripts/fetch_results.sh vast-gpu
./scripts/fetch_lab_results.sh vast-gpu          # only if you ran the lab baseline
```

`fetch_results.sh` copies this box's MLflow store into a folder of its own, `runs/mlflow-box-<experiment id>`, and
imports it, as in [5d](box-4-finish.md). Every box's store has its own experiment id, and two stores in one folder
would hide each other's runs.

Check, then destroy the box ([5d, step 3](box-4-finish.md#step-3-destroy-the-box)):

1. `ls data/bundles/*/full/` lists the confirmed methods of each dataset. That is three per dataset, plus SANSA
   and the re-ranker on Last.fm, unless SANSA ran over budget.
2. On the [quick-tier page](../results/quick-tier.md), SASRec's RetailRocket row shows a test score in line with
   its validation score, not 0.0014.
3. On [SASRec's method page](../dictionary/algorithms/sasrec.md#8-results-in-this-benchmark), no quick-tier row is
   marked †. A † would mean the result came from the old implementation.

Then update the "Status" and "Open items" sections of the [quick-tier page](../results/quick-tier.md) with what
changed.

**Next:** [7. Deploy to Cloud Run](deploy-cloud-run.md) can now serve the confirmed winners.
