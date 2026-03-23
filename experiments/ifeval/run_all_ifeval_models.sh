#!/usr/bin/env bash
set -u

ROOT_DIR="/root/Rethink"
IFEVAL_DIR="$ROOT_DIR/experiments/ifeval"
OUT_DIR_DEFAULT="$IFEVAL_DIR"

MODELS=(
  "llama3_8b"
  "deepseek_r1"
  "qwen3_8b"
  "deepseek_r1_qwen_1_5b"
  "qwen2_5_1_5b"
  "llama2_13b_chat"
  "qwen_14b_chat"
)

CONTINUE_ON_ERROR=1
MAX_TURNS=5
MAX_NEW_TOKENS=512
OUTPUT_DIR="$OUT_DIR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max-turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --models)
      IFS=',' read -r -a MODELS <<< "$2"
      shift 2
      ;;
    --stop-on-error)
      CONTINUE_ON_ERROR=0
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
RUN_LOG="$OUTPUT_DIR/run_all_ifeval_models_$(date +%Y%m%d_%H%M%S).log"

echo "Running IFEval baselines for models: ${MODELS[*]}" | tee -a "$RUN_LOG"

run_cmd() {
  local cmd="$1"
  echo "[CMD] $cmd" | tee -a "$RUN_LOG"
  eval "$cmd" >> "$RUN_LOG" 2>&1
  local status=$?
  if [[ $status -ne 0 && $CONTINUE_ON_ERROR -eq 0 ]]; then
    echo "Command failed, stop-on-error enabled." | tee -a "$RUN_LOG"
    exit $status
  fi
  return $status
}

for model in "${MODELS[@]}"; do
  echo "===== MODEL: $model =====" | tee -a "$RUN_LOG"

  run_cmd "python $IFEVAL_DIR/run_evaluation.py --method vanilla --max-turns 1 --max-new-tokens $MAX_NEW_TOKENS --model $model --output-dir $OUTPUT_DIR"
  run_cmd "python $IFEVAL_DIR/run_evaluation.py --method regenerate --max-turns $MAX_TURNS --max-new-tokens $MAX_NEW_TOKENS --model $model --output-dir $OUTPUT_DIR"
  run_cmd "python $IFEVAL_DIR/run_constrained_decoding.py --max-new-tokens $MAX_NEW_TOKENS --model $model --output-dir $OUTPUT_DIR"
done

echo "Done. Logs: $RUN_LOG" | tee -a "$RUN_LOG"
