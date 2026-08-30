#!/bin/bash
# Download/prep everything for GLM-5.3-Flash NVFP4 on both nodes.
#
# Steps (idempotent — safe to re-run):
#   1. Pull the digest-pinned base image on BOTH nodes
#   2. Build $IMAGE on HEAD (Dockerfile + patches/, self-verifying)
#   3. Transfer built image HEAD -> WORKER (docker save | load), verify IDs
#   4. Sync the checkpoint from HEAD's HF cache -> WORKER (rsync)
#
# NOTE: steps 3-4 use the SSH/LAN path for simplicity. For repeated or
#       larger transfers, prefer RoCE (192.168.0.x) per cluster convention.
#
# Usage: ./download.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

[ -f .env ] || { echo "[glm53] Missing .env — copy .env.example to .env and edit"; exit 1; }
set -a; source .env; set +a
: "${WORKER_ROCE_SSH_TARGET:=$WORKER_SSH_TARGET}"

# Base image: digest-pinned in Dockerfile (ARG default). Pull by tag; the
# build uses the pinned digest. Re-derive the digest after upstream updates:
#   docker images --digests | grep glm53-flash-arm64-cu130
BASE_IMAGE="vllm/vllm-openai:glm53-flash-arm64-cu130"

echo "[glm53] === Step 1/4: base image on both nodes ==="
if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "  head: $BASE_IMAGE present"
else
  echo "  head: pulling $BASE_IMAGE ..."
  docker pull "$BASE_IMAGE"
fi
if ssh "$WORKER_SSH_TARGET" "docker image inspect $BASE_IMAGE >/dev/null 2>&1"; then
  echo "  worker: $BASE_IMAGE present"
else
  echo "  worker: pulling $BASE_IMAGE ..."
  ssh "$WORKER_SSH_TARGET" "docker pull $BASE_IMAGE"
fi

echo ""
echo "[glm53] === Step 2/4: build $IMAGE on head ==="
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  $IMAGE already built (skipping — remove with: docker rmi $IMAGE)"
else
  BUILD_DIR=$(mktemp -d /tmp/glm53-build.XXXXXX)
  cp Dockerfile "$BUILD_DIR/"
  cp patches/* "$BUILD_DIR/"
  echo "  building (patch layer only, ~30s)..."
  docker build -t "$IMAGE" "$BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

echo ""
echo "[glm53] === Step 3/4: transfer image head -> worker ==="
HEAD_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
if ssh "$WORKER_SSH_TARGET" "docker image inspect $IMAGE >/dev/null 2>&1"; then
  WORKER_ID=$(ssh "$WORKER_SSH_TARGET" "docker image inspect $IMAGE --format '{{.Id}}'")
  if [ "$HEAD_ID" = "$WORKER_ID" ]; then
    echo "  worker image matches head ($HEAD_ID) — skipping"
  else
    echo "  worker image differs (head=$HEAD_ID worker=$WORKER_ID) — re-transferring"
    docker save "$IMAGE" | ssh "$WORKER_SSH_TARGET" "docker load"
  fi
else
  echo "  transferring (~25 GB over RoCE)..."
  docker save "$IMAGE" | ssh "$WORKER_ROCE_SSH_TARGET" "docker load"
fi
WORKER_ID=$(ssh "$WORKER_SSH_TARGET" "docker image inspect $IMAGE --format '{{.Id}}'")
[ "$HEAD_ID" = "$WORKER_ID" ] || { echo "FATAL: image ID mismatch after transfer"; exit 1; }
echo "  verified: both nodes $HEAD_ID"

echo ""
echo "[glm53] === Step 4/4: sync checkpoint head -> worker ==="
# HF hub layout: $HF_CACHE/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4
REPO_DIR="hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4"
if [ ! -d "$HF_CACHE/$REPO_DIR" ]; then
  echo "FATAL: checkpoint not found at $HF_CACHE/$REPO_DIR"
  echo "  download it first: huggingface-cli download local-inference-lab/GLM-5.3-Flash-NVFP4 --revision $MODEL_REVISION"
  exit 1
fi
HEAD_BLOBS=$(ls "$HF_CACHE/$REPO_DIR/blobs" 2>/dev/null | wc -l)
echo "  head: $HEAD_BLOBS blobs in $REPO_DIR"
if ssh "$WORKER_SSH_TARGET" "[ -d $HF_CACHE/$REPO_DIR ]"; then
  WORKER_BLOBS=$(ssh "$WORKER_SSH_TARGET" "ls $HF_CACHE/$REPO_DIR/blobs 2>/dev/null | wc -l")
  if [ "$WORKER_BLOBS" = "$HEAD_BLOBS" ]; then
    echo "  worker: $WORKER_BLOBS blobs — matches, skipping"
  else
    echo "  worker: $WORKER_BLOBS blobs (head has $HEAD_BLOBS) — syncing..."
    rsync -a --info=progress2 "$HF_CACHE/$REPO_DIR/" "$WORKER_ROCE_SSH_TARGET:$HF_CACHE/$REPO_DIR/"
  fi
else
  echo "  worker: missing — syncing (~186 GB over RoCE)..."
  ssh "$WORKER_ROCE_SSH_TARGET" "mkdir -p $HF_CACHE"
  rsync -a --info=progress2 "$HF_CACHE/$REPO_DIR/" "$WORKER_ROCE_SSH_TARGET:$HF_CACHE/$REPO_DIR/"
fi
WORKER_BLOBS=$(ssh "$WORKER_SSH_TARGET" "ls $HF_CACHE/$REPO_DIR/blobs 2>/dev/null | wc -l")
[ "$WORKER_BLOBS" = "$HEAD_BLOBS" ] || { echo "FATAL: blob count mismatch after sync"; exit 1; }
echo "  verified: both nodes $WORKER_BLOBS blobs"

echo ""
echo "[glm53] === All prep complete ==="
echo "  Next: ./start.sh   (worker-first, ~17 min cold boot)"
