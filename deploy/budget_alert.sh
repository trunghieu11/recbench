#!/usr/bin/env bash
# Create a monthly budget with e-mail alerts at 50%, 90%, and 100% (default 40 USD).
# Alerts only notify you; they do not stop spending. Use deploy/teardown.sh to stop costs.
#   GCP_BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX ./deploy/budget_alert.sh 40
set -euo pipefail
ACCOUNT="${GCP_BILLING_ACCOUNT:?Set GCP_BILLING_ACCOUNT (gcloud billing accounts list)}"
AMOUNT="${1:-40}"
gcloud billing budgets create \
  --billing-account="$ACCOUNT" \
  --display-name="recbench-monthly-${AMOUNT}usd" \
  --budget-amount="${AMOUNT}USD" \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0
