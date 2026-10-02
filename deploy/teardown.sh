#!/usr/bin/env bash
# Remove everything cloud_run.sh created: the Cloud Run service, its source images, and (with --bucket) the bundles.
#   GCP_PROJECT=my-project ./deploy/teardown.sh            # keep the bucket
#   GCP_PROJECT=my-project RECBENCH_GCS_BUCKET=my-bucket ./deploy/teardown.sh --bucket
set -euo pipefail
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${RECBENCH_SERVICE:-recbench}"
gcloud config set project "$PROJECT" >/dev/null
gcloud run services delete "$SERVICE" --region "$REGION" --quiet || true
gcloud artifacts repositories delete cloud-run-source-deploy --location "$REGION" --quiet || true
if [[ "${1:-}" == "--bucket" ]]; then
  gcloud storage rm --recursive "gs://${RECBENCH_GCS_BUCKET:?Set RECBENCH_GCS_BUCKET}" || true
fi
echo "Removed. Check https://console.cloud.google.com/billing for anything left."
