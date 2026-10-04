# 5d. Finish: bring the results home, then stop paying

!!! abstract "In plain words"
    Results that stay on a rented machine disappear with it. You copy everything to your laptop, check that it
    arrived, and only then destroy the box. Destroying, not stopping, is what ends the bill.

**Time:** about 20 minutes. **Cost:** the last minutes of rent.

## Step 1: fetch the results (on the laptop, at any time)

```bash
./scripts/fetch_results.sh vast-gpu                     # or: ./scripts/fetch_results.sh vast-gpu /workspace/recbench
```

You can run it as often as you like, also while the box is still working. Runs in progress are skipped until
they end, and a second fetch only adds what is new. Fetching after each dataset is a cheap insurance against a box
that disappears.

It copies, then rebuilds:

| What | From the box | To the laptop |
|---|---|---|
| every run (settings, metrics, files) | `runs/mlflow/` | `runs/mlflow-box/`, then **imported** into your `runs/mlflow` |
| job summaries: every trial, the best settings, timings | `runs/tuning/quick/`, `runs/tuning/full/` | `runs/tuning/` |
| the queue's state and the logs | `runs/queue/`, `runs/logs/` | `runs/queue-box/`, `runs/logs-box/` |
| reports written on the box | `reports/` | `reports/box/` |
| serving bundles of the confirmed methods | `data/bundles/<dataset>/full/` | `data/bundles/<dataset>/full/` |

Then it rebuilds `reports/quick-tuned/`, `reports/full-tuned/` and the docs fragments, and prints the status.

!!! success "You should see"
    ```text
    {'imported': 312, 'already_imported': 0, 'still_running': 3} -> file:///Users/you/recbench/runs/mlflow
    ...
    queue: running; state written 1 min ago on C.12345
    == movielens-25m: 23/23 jobs done
    ...
    Reports: reports/quick-tuned/report.html and reports/full-tuned/report.html; docs fragments in docs/generated/.
    ```

    The counts depend on how far the run is. `still_running` runs are imported by a later fetch.

!!! info "Why import instead of copying `runs/mlflow` over yours?"
    Each machine's MLflow store has its own experiment id and absolute paths. Copying one store over another
    would hide one of them. `import_runs` re-creates the box's runs inside your store, once.

## Step 2: check the results

1. Open `reports/quick-tuned/report.html` in your browser: one section per dataset, sorted by NDCG@10, with
   confidence intervals and "≈" for methods tied with the best.
2. Run `mkdocs serve` and open <http://127.0.0.1:8000>. The [quick-tier page](../results/quick-tier.md) now shows
   your results.
3. Check that each finished dataset has its confirmations and bundles:

    ```bash
    ls data/bundles/*/full/
    cat data/bundles/movielens-25m/full/*/manifest.json | grep -E '"method"|"data_cutoff"|"run_id"'
    ```

!!! success "You should see"
    Up to three method folders per finished dataset, for example `data/bundles/movielens-25m/full/ease`. Each
    manifest names its method, the data cutoff ("lists reflect events before …") and the MLflow run that made it.

If a confirmed method has no bundle (its confirmation finished in an earlier session, before bundles were
exported), make one **on the box, before destroying it**, then fetch again:

```bash
python -m recbench.export --dataset movielens-25m --method ease --from-confirm
```

## Step 3: destroy the box

Only after step 2. In the vast.ai console, open **Instances**, use the destroy (trash) button on your instance, and
confirm.

!!! warning "Stop vs destroy"
    **Stop** keeps the disk, and you keep paying for it. **Destroy** deletes the machine and its disk and ends all
    charges. Anything you did not fetch is gone.

!!! success "You should see"
    The instance list is empty, and your credit balance stops going down. Check again 10 minutes later.

## Step 4: compare the bill with your estimate

The billing page lists what each instance cost. Compare it with your estimate from
[5a](box-1-before-you-rent.md#step-5-estimate-the-cost) and write down the real time per dataset; it makes the next
estimate better.

**Next:** read the [quick-tier results](../results/quick-tier.md), then
[6. The full tier on a GPU machine](full-tier-gpu.md) for the heavy methods, or
[7. Deploy to Cloud Run](deploy-cloud-run.md) to serve the winners.
