#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
FOREGROUND=0
ARGS=()
while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if ((FOREGROUND)); then
  exec "$PYTHON_BIN" -u "$PACKAGE_DIR/train.py" "${ARGS[@]}"
fi
mkdir -p "$PACKAGE_DIR/launch_logs"
LOG="$PACKAGE_DIR/launch_logs/train_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" -u "$PACKAGE_DIR/train.py" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf 'pid=%s
log=%s
' "$PID" "$LOG"
