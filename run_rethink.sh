#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)

# Core model / analysis defaults
MODEL_PATH=${MODEL_PATH:-"/root/autodl-fs/LLM-Research/Meta-Llama-3___1-8B-Instruct"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
DEVICE=${DEVICE:-}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# Dataset parameters (focus on gsm8k test[0])
DATASET_NAME=${DATASET_NAME:-gsm8k}
DATASET_CONFIG=${DATASET_CONFIG:-main}
DATASET_SPLIT=${DATASET_SPLIT:-test}
DATASET_INDEX=${DATASET_INDEX:-0}
QUESTION_FIELD=${QUESTION_FIELD:-question}
ANSWER_FIELD=${ANSWER_FIELD:-answer}
PROMPT_FILE=${PROMPT_FILE:-prompts/gsm8k_prompt.txt}

# Analysis output controls
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/analysis/gsm8k_${DATASET_INDEX}"}
VISUALIZE=${VISUALIZE:-1}
TOP_K=${TOP_K:-5}

# Build command ---------------------------------------------------------------
cd "$REPO_ROOT"
CMD=(python rethink_run.py \
    --model-path "$MODEL_PATH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --dataset-split "$DATASET_SPLIT" \
    --dataset-index "$DATASET_INDEX" \
    --question-field "$QUESTION_FIELD" \
    --answer-field "$ANSWER_FIELD" \
    --prompt-file "$PROMPT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --hf-endpoint "$HF_ENDPOINT" \
    --torch-dtype "$TORCH_DTYPE" \
    --top-k "$TOP_K")

if [[ "$VISUALIZE" == "1" ]]; then
    CMD+=(--visualize)
fi

if [[ -n "$DEVICE" ]]; then
    CMD+=(--device "$DEVICE")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
