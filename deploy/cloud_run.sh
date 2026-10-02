#!/usr/bin/env bash
# Deploy the serving API to Google Cloud Run, reading bundles from a Cloud Storage bucket.
#
#   GCP_PROJECT=my-project RECBENCH_GCS_BUCKET=my-bucket ./deploy/cloud_run.sh            # private (IAM)
#   GCP_PROJECT=my-project RECBENCH_GCS_BUCKET=my-bucket ./deploy/cloud_run.sh --public   # anyone can call it
#
# Cost control: min-instances 0 (scale to zero when idle), max-instances 1, 1 vCPU / 1 GiB.
# Create a budget alert first (deploy/budget_alert.sh) and remove everything with deploy/teardown.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
BUCKET="${RECBENCH_GCS_BUCKET:?Set RECBENCH_GCS_BUCKET (a bucket you own)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${RECBENCH_SERVICE:-recbench}"
TIER="${RECBENCH_TIER:-smoke}"
AUTH="--no-allow-unauthenticated"
[[ "${1:-}" == "--public" ]] && AUTH="--allow-unauthenticated"

[[ -d "data/bundles" ]] || { echo "No data/bundles. Run the benchmark first (bundles are exported per method)." >&2; exit 1; }

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || gcloud storage buckets create "gs://${BUCKET}" --location "$REGION" --uniform-bucket-level-access
gcloud storage rsync --recursive --delete-unmatched-destination-objects "data/bundles/" "gs://${BUCKET}/bundles/"

gcloud run deploy "$SERVICE" \
  --source "$ROOT" \
  --region "$REGION" \
  --cpu 1 --memory 1Gi \
  --min-instances 0 --max-instances 1 --concurrency 40 \
  $AUTH \
  --add-volume "name=bundles,type=cloud-storage,bucket=${BUCKET},readonly=true" \
  --add-volume-mount "volume=bundles,mount-path=/mnt/gcs" \
  --set-env-vars "RECBENCH_BUNDLES=/mnt/gcs/bundles,RECBENCH_TIER=${TIER}"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
echo "Deployed: $URL"
if [[ "$AUTH" == "--no-allow-unauthenticated" ]]; then
  echo "Private service. Call it with an identity token:"
  echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $URL/health"
else
  echo "  curl $URL/health"
fi
