# Cloud Run

**Cloud Run** runs a container image as an HTTPS service. It starts instances when requests arrive, stops them
when idle, and bills for the time instances spend handling requests (with request-based billing, the default).
It fits recbench's serving API well: the API is stateless, reads files only, and is used in bursts.

## Key ideas

| Term | Meaning | recbench setting |
|---|---|---|
| service | a named, versioned deployment with a stable URL | `recbench` (`RECBENCH_SERVICE`) |
| revision | an immutable version of the service; each deploy creates one | automatic |
| instance | one running copy of the container | at most 1 |
| min / max instances | how far it scales down and up | 0 and 1 |
| concurrency | requests one instance handles at the same time | 40 |
| cold start | the delay while a new instance starts | a second or two after an idle period |
| volume mount | a bucket or other storage mounted as a folder | bundles bucket at `/mnt/gcs` |

## `deploy/cloud_run.sh`, explained

```bash
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
BUCKET="${RECBENCH_GCS_BUCKET:?Set RECBENCH_GCS_BUCKET (a bucket you own)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${RECBENCH_SERVICE:-recbench}"
TIER="${RECBENCH_TIER:-smoke}"
```

Settings come from environment variables. `${VAR:?message}` stops the script with the message when `VAR` is
missing; `${VAR:-default}` uses a default.

```bash
AUTH="--no-allow-unauthenticated"
[[ "${1:-}" == "--public" ]] && AUTH="--allow-unauthenticated"
```

The service is **private** unless you pass `--public`. Private means Cloud Run rejects calls without an identity
token from a principal holding the Cloud Run Invoker role.

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || gcloud storage buckets create "gs://${BUCKET}" --location "$REGION" --uniform-bucket-level-access
gcloud storage rsync --recursive --delete-unmatched-destination-objects "data/bundles/" "gs://${BUCKET}/bundles/"
```

Enable the APIs (see [GCP basics](gcp-basics.md)), create the bucket in the same region if it does not exist,
and make `gs://BUCKET/bundles/` an exact copy of your local `data/bundles/`. Bundles that you deleted locally
are deleted remotely too. See [Cloud Storage and bundles](storage-and-bundles.md).

```bash
gcloud run deploy "$SERVICE" \
  --source "$ROOT" \
  --region "$REGION" \
  --cpu 1 --memory 1Gi \
  --min-instances 0 --max-instances 1 --concurrency 40 \
  $AUTH \
  --add-volume "name=bundles,type=cloud-storage,bucket=${BUCKET},readonly=true" \
  --add-volume-mount "volume=bundles,mount-path=/mnt/gcs" \
  --set-env-vars "RECBENCH_BUNDLES=/mnt/gcs/bundles,RECBENCH_TIER=${TIER}"
```

| Flag | Why |
|---|---|
| `--source` | upload the repository; Cloud Build builds the `Dockerfile` and stores the image in Artifact Registry |
| `--cpu 1 --memory 1Gi` | enough for memory-mapped bundles; small instances cost less |
| `--min-instances 0` | scale to zero when idle, so idle time is free |
| `--max-instances 1` | a hard cap on cost, even if the URL is flooded with requests |
| `--concurrency 40` | one instance serves many lookups at once (each is a fast array read) |
| `--add-volume ... readonly=true` | mount the bucket with Cloud Storage FUSE, read-only |
| `--set-env-vars` | tell the API where the bundles are and which tier to serve |

Finally the script prints the service URL and, for a private service, how to call it with a token.

## Calling a private service

```bash
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" "$URL/health"
```

To let a teammate call it, grant them the invoker role on this service only:

```bash
gcloud run services add-iam-policy-binding recbench --region us-central1 \
  --member "user:teammate@example.com" --role "roles/run.invoker"
```

## Updating

| What changed | What to do |
|---|---|
| new bundles (a new benchmark run) | run the deploy script again: it uploads only changed files and creates a new revision |
| code (`src/`, `Dockerfile`) | run the deploy script again; a new revision replaces the old one |
| tier | `RECBENCH_TIER=full ./deploy/cloud_run.sh` |

The API caches loaded bundles in memory (`load_bundle`). A running instance keeps serving the old copy of a
bundle it has already loaded; a new revision or an instance restart picks up the new files. Re-running the
deploy script creates a new revision, so the simplest rule is: **after uploading new bundles, redeploy.**

## Logs and monitoring

The Cloud Run console page of the service shows requests, latencies, instance counts, and logs. The API also writes
one JSON line per `/recommend` call (dataset, method, status, latency, fallback), and answers `GET /stats`. From the
terminal:

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="recbench"' --limit 50
```

[Monitoring the API](monitoring.md) covers the whole picture: health, cost and quality signals, the uptime check
and alerts (`deploy/monitoring.sh`), and the checking command `python -m recbench.serving.monitor`.

If the logs show that bundles cannot be read, the service account lacks access to the bucket. Grant it
`roles/storage.objectViewer` on the bucket (see [security](security.md#least-privilege-optional-hardening)).

## Teardown

`deploy/teardown.sh` deletes the service and the `cloud-run-source-deploy` Artifact Registry repository in the
region (where `--source` deploys store images); with `--bucket` it also deletes the bucket. The repository is
shared by every source deploy in that project and region, which is why a [dedicated project](gcp-basics.md#projects)
is recommended.

## Check your understanding

??? question "Someone floods your public URL with requests. What happens to your bill?"
    At most one instance runs, so CPU and memory charges are bounded by one instance running continuously,
    plus per-request charges beyond the free allowance. The service becomes slow, but the bill cannot multiply
    by scaling out. Teardown or switch back to private (redeploy without `--public`).

??? question "Why mount the bucket instead of copying bundles into the image?"
    The image stays small and generic, and updating recommendations only needs an upload, not a rebuild.
