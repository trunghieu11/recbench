#!/usr/bin/env bash
# On the laptop: forward the box's two monitoring pages to your browser through SSH (started on the box with
# ./scripts/monitor_box.sh). Keep this window open while you watch; Ctrl-C closes the tunnel, not the run.
#
#   ./scripts/open_tunnel.sh vast-gpu          # "vast-gpu" is the Host name in ~/.ssh/config
#
# Then open:
#   http://127.0.0.1:8081/queue/quick.html   the queue's status page (refreshes itself every 30 seconds)
#   http://127.0.0.1:8081/                   every report the box has written
#   http://127.0.0.1:5001                    the MLflow UI
# If a port is busy on the laptop (something else uses 5001 or 8081), set LOCAL_MLFLOW / LOCAL_STATUS.
set -euo pipefail
HOST="${1:?usage: open_tunnel.sh <ssh-host>}"
LOCAL_MLFLOW="${LOCAL_MLFLOW:-5001}"
LOCAL_STATUS="${LOCAL_STATUS:-8081}"
echo "Status page: http://127.0.0.1:${LOCAL_STATUS}/queue/quick.html"
echo "MLflow UI:   http://127.0.0.1:${LOCAL_MLFLOW}"
echo "Tunnel open; press Ctrl-C to close it (the run on the box is not affected)."
exec ssh -N -o ServerAliveInterval=60 -L "${LOCAL_MLFLOW}:127.0.0.1:5001" -L "${LOCAL_STATUS}:127.0.0.1:8081" "$HOST"
