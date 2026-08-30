#!/bin/bash
# Stop GLM-5.3-Flash NVFP4 on both nodes (head + worker).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

[ -f .env ] || { echo "[glm53] Missing .env"; exit 1; }
set -a; source .env; set +a

# stop the log tailer if running
if [ -f logs/tee.pid ]; then
  kill "$(cat logs/tee.pid)" 2>/dev/null || true
  rm -f logs/tee.pid
fi

echo "[glm53] Stopping on both nodes..."
ssh "$WORKER_SSH_TARGET" "cd $WORKER_DIR && docker compose --env-file .env -f compose.worker.yaml down 2>&1 | tail -1"
echo "  Worker stopped."
docker compose --env-file .env -f compose.head.yaml down 2>&1 | tail -1
echo "  Head stopped."
echo "[glm53] Both nodes stopped."
