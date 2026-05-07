#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATASET_PATH="${DATASET_PATH:-/root/Rethink/experiments/revision/data/staged_cases.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/Rethink/experiments/revision/outputs}"
MODEL_ARGS="${MODEL_ARGS:---model qwen2_5_1_5b}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LIMIT_ARGS="${LIMIT_ARGS:-}"
if [[ -n "${CASE_LIMIT:-}" ]]; then
  LIMIT_ARGS="--limit ${CASE_LIMIT}"
fi
RESUME_ARGS="${RESUME_ARGS:---resume}"
SAMPLING_ARGS="${SAMPLING_ARGS:-}"

if [[ -n "${ACTOR_API_KEY:-}" ]]; then
  export MINIMAX_API_KEY="$ACTOR_API_KEY"
fi

cd /root/Rethink/experiments/revision

"$PYTHON_BIN" dataset.py --output "$DATASET_PATH"

for method in reflexion autoLR constraint_decoding chat steer steer_lite; do
  echo "=== Running ${method} ==="
  "$PYTHON_BIN" "${method}.py" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    $RESUME_ARGS \
    $SAMPLING_ARGS \
    $MODEL_ARGS \
    $LIMIT_ARGS
done

"$PYTHON_BIN" summarize.py --input-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/summary.csv"
