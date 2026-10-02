# Cloud Storage and bundles

**Cloud Storage** stores files ("objects") in **buckets**. recbench uses one bucket to hold serving bundles,
which Cloud Run mounts as a read-only folder.

## Layout

```text
gs://YOUR_BUCKET/
└── bundles/
    └── <dataset>/<tier>/<method>/
        ├── manifest.json
        ├── users.parquet
        ├── topk.npy
        ├── scores.npy
        ├── items.parquet
        └── popular.npy
```

It mirrors `data/bundles/` on your machine. The files are described in
[serving and bundles](../codebase/serving-and-bundles.md).

## Upload

`deploy/cloud_run.sh` runs:

```bash
gcloud storage rsync --recursive --delete-unmatched-destination-objects data/bundles/ gs://YOUR_BUCKET/bundles/
```

`rsync` copies only new or changed files. `--delete-unmatched-destination-objects` deletes remote files that no
longer exist locally, so the bucket matches your machine exactly.

!!! warning "rsync deletes"
    Because of `--delete-unmatched-destination-objects`, deleting `data/bundles/` locally and redeploying
    empties the remote folder too. Keep local bundles until you no longer need them online.

To upload without deploying:

```bash
gcloud storage rsync --recursive data/bundles/ gs://YOUR_BUCKET/bundles/
```

## How Cloud Run reads it

The service mounts the bucket with **Cloud Storage FUSE** at `/mnt/gcs` (read-only), and the API reads
`/mnt/gcs/bundles/...` like local files. FUSE fetches file contents over the network on first access, so the
first request for each bundle is slower; the API then keeps the bundle in memory.

`topk.npy` and `scores.npy` are opened with memory mapping (`mmap_mode="r"` in
`src/recbench/serving/bundle.py::Bundle`), so only the rows that requests touch are read.

## Bucket settings used

| Setting | Value | Why |
|---|---|---|
| location | same region as the service | no cross-region transfer, lower latency |
| uniform bucket-level access | on | permissions are managed with IAM only, not per-object ACLs |
| public access | none | only the service account and you can read it |

## Costs

Storage is billed per gigabyte-month, plus per-operation charges and network egress. The smoke-tier bundles
take about 250 MB; a full-tier set (5 datasets × up to 16 methods, each at most about 16 MB of lists plus the item
table) adds a gigabyte or two. Storage costs stay small. See [Cloud Storage pricing](https://cloud.google.com/storage/pricing) and
[cost control](cost-control.md).

## Versions and freshness

Each bundle's `manifest.json` records the split hash, the export time, and a freshness note ("lists reflect
events before ..."). To check what is online:

```bash
gcloud storage cat gs://YOUR_BUCKET/bundles/movielens-25m/smoke/ease/manifest.json
```

## Alternatives (not implemented)

| Option | Trade-off |
|---|---|
| bake bundles into the image | no bucket, but every update means a rebuild and a bigger image |
| a key-value store (e.g. Memorystore, Firestore) | faster lookups for very many users, but an always-on cost and more code |
| model files plus a real-time scorer | fresh recommendations, but needs PyTorch in the image (see the [roadmap](../results/roadmap.md)) |
