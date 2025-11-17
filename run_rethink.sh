#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
PROMPT=${PROMPT:-"Solve 12 * 13."}
MODEL_PATH=${MODEL_PATH:-"/root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}
CAPTURE_LAYERS=${CAPTURE_LAYERS:-"8,16,24"}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.8}
LOG_FILE=${LOG_FILE:-"outputs/rethink_run.log"}
OUTPUT_JSON=${OUTPUT_JSON:-"outputs/rethink_run.json"}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
DEVICE=${DEVICE:-}

cd "$REPO_ROOT"
CMD=(python rethink_run.py "$PROMPT" \
    --model-path "$MODEL_PATH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --capture-layers "$CAPTURE_LAYERS" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --log-file "$LOG_FILE" \
    --output-json "$OUTPUT_JSON" \
    --torch-dtype "$TORCH_DTYPE")

if [[ -n "$DEVICE" ]]; then
    CMD+=(--device "$DEVICE")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
