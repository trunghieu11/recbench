#!/usr/bin/env bash
# Remove everything cloud_run.sh and monitoring.sh created: the Cloud Run service, its source images, the uptime
# check and log-based counters, and (with --bucket) the bundles.
#   GCP_PROJECT=my-project ./deploy/teardown.sh            # keep the bucket
#   GCP_PROJECT=my-project RECBENCH_GCS_BUCKET=my-bucket ./deploy/teardown.sh --bucket
set -euo pipefail
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${RECBENCH_SERVICE:-recbench}"
gcloud config set project "$PROJECT" >/dev/null
gcloud run services delete "$SERVICE" --region "$REGION" --quiet || true
gcloud artifacts repositories delete cloud-run-source-deploy --location "$REGION" --quiet || true
for check in $(gcloud monitoring uptime list-configs --filter="displayName=recbench-health" --format="value(name)" 2>/dev/null); do
  gcloud monitoring uptime delete "$check" --quiet || true   # otherwise it keeps calling the deleted service
done
for metric in recbench_recommend_calls recbench_fallbacks; do
  gcloud logging metrics delete "$metric" --quiet 2>/dev/null || true
done
if [[ "${1:-}" == "--bucket" ]]; then
  gcloud storage rm --recursive "gs://${RECBENCH_GCS_BUCKET:?Set RECBENCH_GCS_BUCKET}" || true
fi
echo "Removed. Delete the alert policies in Monitoring > Alerting if you created them, and check"
echo "https://console.cloud.google.com/billing for anything left."
