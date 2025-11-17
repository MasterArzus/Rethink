#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH=${MODEL_PATH:-/root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct}
OUTPUT_JSONL=${OUTPUT_JSONL:-outputs/gsm8k_llama3_run.jsonl}
LOG_FILE=${LOG_FILE:-outputs/gsm8k_eval.log}
DATASET_NAME=${DATASET_NAME:-gsm8k}
DATASET_CONFIG=${DATASET_CONFIG:-main}
SPLIT=${SPLIT:-test}
MAX_SAMPLES=${MAX_SAMPLES:-50}
BATCH_SIZE=${BATCH_SIZE:-1}
HF_TOKEN=${HF_TOKEN:-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}

if [[ -z "$HF_TOKEN" ]]; then
    echo "HF_TOKEN environment variable must be set (requires GSM8K access)." >&2
    exit 1
fi

python eval.py \
  --model-path "$MODEL_PATH" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --split "$SPLIT" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --output-jsonl "$OUTPUT_JSONL" \
  --log-file "$LOG_FILE" \
  --use-auth-token "$HF_TOKEN" \
  "$@"
