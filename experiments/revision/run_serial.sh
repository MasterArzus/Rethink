#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  qwen2_5_1_5b
  deepseek_r1_qwen_1_5b
  llama3_8b
  qwen3_8b
  deepseek_r1
  qwen_14b_instruct
)

RUN_PREFIX="${RUN_PREFIX:-revision}"
LOG_DIR="${LOG_DIR:-/root/Rethink/experiments/revision/logs}"
mkdir -p "$LOG_DIR"

cd /root/Rethink/experiments/revision

# Optional explicit API key. Prefer setting ACTOR_API_KEY in the shell.
if [[ -n "${ACTOR_API_KEY:-}" ]]; then
  export MINIMAX_API_KEY="$ACTOR_API_KEY"
fi

for model in "${MODELS[@]}"; do
  run_name="${RUN_PREFIX}_${model}_$(date +%Y%m%d_%H%M%S)"
  echo "=== Running ${model} (${run_name}) ==="

  MODEL_ARGS="--model ${model}" \
  RUN_NAME="$run_name" \
  ./run_exp.sh 2>&1 | tee "$LOG_DIR/${run_name}.log"

done
