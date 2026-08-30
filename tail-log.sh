#!/bin/bash
# Tail the GLM-5.3 boot log.
#
# Usage:
#   ./tail-log.sh              # tail the LATEST timestamped log (follow)
#   ./tail-log.sh <file>       # tail a specific log file
#   ./tail-log.sh --list       # list available logs
#   ./tail-log.sh -n 100       # show last 100 lines then follow
#
# Ctrl-C stops following.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

[ -d "$LOG_DIR" ] || { echo "[tail-log] no logs dir yet ($LOG_DIR) — run ./start.sh first"; exit 1; }

MODE="latest"
COUNT=""
case "${1:-}" in
  --list|-l)
    MODE="list"
    ;;
  -n)
    COUNT="${2:?usage: $0 -n <lines>}"
    ;;
  "")
    MODE="latest"
    ;;
  *)
    # explicit file (relative to repo or absolute)
    if [ -f "$1" ]; then
      LOGFILE="$1"
    elif [ -f "$LOG_DIR/$1" ]; then
      LOGFILE="$LOG_DIR/$1"
    else
      echo "[tail-log] no such log: $1"; exit 1
    fi
    ;;
esac

if [ "$MODE" = "list" ]; then
  echo "[tail-log] available logs (newest first):"
  ls -1t "$LOG_DIR"/*.head.log 2>/dev/null | while read -r f; do
    echo "  $f  ($(du -h "$f" | cut -f1))"
  done
  exit 0
fi

if [ "$MODE" = "latest" ]; then
  LOGFILE=$(ls -1t "$LOG_DIR"/*.head.log 2>/dev/null | head -1)
  [ -n "$LOGFILE" ] || { echo "[tail-log] no logs found in $LOG_DIR"; exit 1; }
fi

echo "[tail-log] following: $LOGFILE"
if [ -n "$COUNT" ]; then
  tail -n "$COUNT" -f "$LOGFILE"
else
  tail -f "$LOGFILE"
fi
