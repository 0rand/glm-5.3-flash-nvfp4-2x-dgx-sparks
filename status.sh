#!/bin/bash
# Status: containers, health, served model ID, and boot markers.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

[ -f .env ] || { echo "[glm53] Missing .env"; exit 1; }
set -a; source .env; set +a

echo "=== Containers ==="
docker ps --format '  {{.Names}} {{.Status}}' | grep glm53 || echo "  (head: not running)"
ssh "$WORKER_SSH_TARGET" "docker ps --format '  {{.Names}} {{.Status}}'" 2>/dev/null | grep glm53 || echo "  (worker: not running)"

echo ""
echo "=== Health (head :$PORT) ==="
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
echo "  /health -> $code"
if [ "$code" = "200" ]; then
  echo "  served model:"
  curl -s --max-time 5 "http://localhost:${PORT}/v1/models" | python3 -c "import sys,json; [print('   ', m['id'], 'max_model_len=', m.get('max_model_len')) for m in json.load(sys.stdin)['data']]" 2>/dev/null
fi

echo ""
echo "=== Boot markers (head log) ==="
docker logs glm53-nvfp4 2>&1 | grep -E 'Model loading took|quantprobe|GPU KV cache size|Application startup complete' | tail -6 || echo "  (no markers yet)"
