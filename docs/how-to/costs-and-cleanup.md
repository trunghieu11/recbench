# Costs and cleanup

## What is free

Everything that runs on your own machines: the laptop smoke tier, the full tier on your own GPU machine,
MLflow (a local folder), the docs site (`mkdocs serve`), and the local API.

## What can cost money

| Resource | Created by | Billed for | How to stop it |
|---|---|---|---|
| Cloud Run service | `deploy/cloud_run.sh` | request time and CPU/memory while handling requests (with `min-instances 0`, nothing while idle, beyond a free tier) | `deploy/teardown.sh` |
| Cloud Storage bucket | `deploy/cloud_run.sh` | stored gigabytes per month, operations, data leaving Google Cloud | `deploy/teardown.sh --bucket` |
| Artifact Registry images | Cloud Build during deploy | stored image gigabytes | deleted by `deploy/teardown.sh` |
| Cloud Build | each deploy | build minutes beyond the free allowance | — |
| Recombee | your account | paid plans (the free plan has limits; confirm them on recombee.com) | reset or delete the database; stay on the free plan |

Exact prices change; see [Cloud Run pricing](https://cloud.google.com/run/pricing) and
[Cloud Storage pricing](https://cloud.google.com/storage/pricing).

## Before you deploy anything

```bash
GCP_BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX ./deploy/budget_alert.sh 40
```

You get e-mails at 50%, 90%, and 100% of the monthly budget. **Alerts do not stop spending**; teardown does.

## Cleanup checklist

```bash
GCP_PROJECT=YOUR_PROJECT_ID ./deploy/teardown.sh                                   # service + images
GCP_PROJECT=YOUR_PROJECT_ID RECBENCH_GCS_BUCKET=YOUR_BUCKET ./deploy/teardown.sh --bucket   # + bundles bucket
```

Then:

- [ ] Check <https://console.cloud.google.com/run>: no services left.
- [ ] Check <https://console.cloud.google.com/storage/browser>: no recbench buckets left.
- [ ] Check the billing report a day later.
- [ ] Recombee: reset or delete the benchmark database.
- [ ] Optionally, delete the whole Google Cloud project, the surest way to stop all charges.

## Local disk

| Folder | Typical size | Safe to delete? |
|---|---|---|
| `data/raw/` | ~12 GB for the five datasets | yes, it is re-downloaded on the next prepare |
| `data/clean/` | a few GB | yes, it is re-cleaned (minutes) |
| `data/splits/` | hundreds of MB per tier | yes, it is re-split |
| `data/bundles/` | ~250 MB for the smoke tier; a GB or two more for full | yes, re-exported by the next run |
| `runs/mlflow/` | grows with runs | only if you no longer need the results |
| `runs/mlflow_v1_archive/` | old v0.1 runs | yes, once you no longer need them for the review log |
| `runs/mlflow-gpu/` | a copy of the GPU machine's runs | yes, after `python -m recbench.import_runs` |

See [cost control](../cloud/cost-control.md) for the reasoning behind these defaults.
