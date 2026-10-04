#!/usr/bin/env bash
# Start the two monitoring web pages on the rented box, in the background:
#   - the MLflow UI (every run, its settings and metrics)        on 127.0.0.1:5001
#   - the queue's status page and the reports (reports/ folder)   on 127.0.0.1:8081
# Both listen on 127.0.0.1 only, so they are not open to the internet. Reach them from your laptop through an SSH
# tunnel: ./scripts/open_tunnel.sh vast-gpu (see docs/start/box-3-run-and-monitor.md).
#
#   ./scripts/monitor_box.sh          # start (does nothing if they already run)
#   ./scripts/monitor_box.sh stop     # stop both
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -d .venv ]] || { echo "No .venv: run ./scripts/setup_box.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p runs/logs runs/mlflow reports/queue

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "[m]lflow ui --backend-store-uri file://$ROOT/runs/mlflow" || true
  pkill -f "[h]ttp.server 8081 --bind 127.0.0.1" || true
  echo "Stopped the MLflow UI and the status-page server."
  exit 0
fi

if pgrep -f "[m]lflow ui --backend-store-uri file://$ROOT/runs/mlflow" >/dev/null; then
  echo "MLflow UI already running."
else
  nohup mlflow ui --backend-store-uri "file://$ROOT/runs/mlflow" --host 127.0.0.1 --port 5001 > runs/logs/mlflow-ui.log 2>&1 &
  echo "Started the MLflow UI (log: runs/logs/mlflow-ui.log)."
fi
if pgrep -f "[h]ttp.server 8081 --bind 127.0.0.1" >/dev/null; then
  echo "Status-page server already running."
else
  nohup python -m http.server 8081 --bind 127.0.0.1 --directory "$ROOT/reports" > runs/logs/status-server.log 2>&1 &
  echo "Started the status-page server (log: runs/logs/status-server.log)."
fi
echo
echo "On your laptop, run:  ./scripts/open_tunnel.sh vast-gpu"
echo "then open  http://127.0.0.1:8081/queue/quick.html  (status)  and  http://127.0.0.1:5001  (MLflow)."
