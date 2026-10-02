#!/usr/bin/env bash
# Deploy the CPU image to Cloud Run. Smoke checkpoints for the four small models
# are read from a GCS bucket mounted at /artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${RECBENCH_GCS_BUCKET:?Set RECBENCH_GCS_BUCKET}"
SERVICE="${RECBENCH_SERVICE:-recbench}"

gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

gcloud run deploy "$SERVICE" \
  --source "$ROOT" \
  --region "$REGION" \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 1 \
  --allow-unauthenticated \
  --add-volume "name=artifacts,type=cloud-storage,bucket=${BUCKET}" \
  --add-volume-mount "volume=artifacts,mount-path=/artifacts" \
  --set-env-vars "^|^RECBENCH_SERVE_METHODS=xsimgcl,bert4rec,dcnv2,din|RECBENCH_TIER=smoke|RECBENCH_PRESET=cpu|DATA_DIR=/artifacts"

echo "Cloud Run service ${SERVICE} deployed in ${REGION}."
echo "Put smoke checkpoints at gs://${BUCKET}/artifacts/<dataset>/smoke/cpu/{xsimgcl,bert4rec,dcnv2,din}.pt"
