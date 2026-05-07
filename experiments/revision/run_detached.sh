#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-revision_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/root/Rethink/experiments/revision/logs}"
mkdir -p "$LOG_DIR"

cd /root/Rethink/experiments/revision

# Optional explicit API key. Prefer setting ACTOR_API_KEY in the shell rather than editing this file.
if [[ -n "${ACTOR_API_KEY:-}" ]]; then
  export MINIMAX_API_KEY="$ACTOR_API_KEY"
fi

nohup ./run_exp.sh > "$LOG_DIR/${RUN_NAME}.log" 2>&1 &
echo $! > "$LOG_DIR/${RUN_NAME}.pid"
echo "Started ${RUN_NAME}"
echo "PID: $(cat "$LOG_DIR/${RUN_NAME}.pid")"
echo "Log: $LOG_DIR/${RUN_NAME}.log"

