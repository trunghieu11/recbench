#!/usr/bin/env bash
# Set up monitoring for the API deployed by deploy/cloud_run.sh:
#   1. an uptime check that calls /health every 15 minutes from several regions, with an identity token (the
#      service is private), so you hear about it when the service is down or has no bundles (503);
#   2. two log-based counters from the API's one-line JSON request log: all /recommend calls, and the calls for
#      unknown users that got the popularity fallback. Their ratio is a quality signal over time.
# Then create the e-mail alert policies in the console: docs/cloud/monitoring.md, step 3.
#
#   GCP_PROJECT=my-project ./deploy/monitoring.sh
set -euo pipefail
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${RECBENCH_SERVICE:-recbench}"
gcloud config set project "$PROJECT" >/dev/null
gcloud services enable monitoring.googleapis.com logging.googleapis.com

# The uptime check signs its calls as the Cloud Monitoring service agent, which may call the private service.
NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
gcloud run services add-iam-policy-binding "$SERVICE" --region "$REGION" --quiet \
  --member="serviceAccount:service-${NUMBER}@gcp-sa-monitoring-notification.iam.gserviceaccount.com" \
  --role=roles/run.invoker >/dev/null
if gcloud monitoring uptime list-configs --filter="displayName=recbench-health" --format="value(name)" | grep -q .; then
  echo "Uptime check recbench-health already exists."
else
  gcloud monitoring uptime create recbench-health \
    --resource-type=cloud-run-revision \
    --resource-labels="project_id=${PROJECT},service_name=${SERVICE},location=${REGION}" \
    --protocol=https --path=/health --service-agent-auth=oidc-token --period=15
fi

BASE="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE}\" AND jsonPayload.message=\"recommend\""
gcloud logging metrics describe recbench_recommend_calls >/dev/null 2>&1 \
  || gcloud logging metrics create recbench_recommend_calls --description="recbench: /recommend calls" --log-filter="$BASE"
gcloud logging metrics describe recbench_fallbacks >/dev/null 2>&1 \
  || gcloud logging metrics create recbench_fallbacks --description="recbench: calls for unknown users (popularity fallback)" \
       --log-filter="$BASE AND jsonPayload.fallback=true"
echo "Done. Next: create the e-mail alert policies in the console (docs/cloud/monitoring.md, step 3)."
