#!/bin/bash
# run_v3_experiments.sh - Run experiments with 2048 tokens, supports resume

set -e

SCRIPT_DIR="/root/Rethink/experiments/ifeval"
OUTPUT_DIR="$SCRIPT_DIR/results_v3"
DATASET="/root/Rethink/dataset/ifeval/taskset_60_hard.json"
MAX_TOKENS=2048
MAX_TURNS=7

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if task already completed
check_done() {
    local model=$1
    local method=$2
    local csv="$OUTPUT_DIR/${model}_${method}_c0c1.csv"

    if [ -f "$csv" ]; then
        local line_count=$(wc -l < "$csv")
        if [ "$line_count" -ge 61 ]; then  # 60 tasks + header
            echo "SKIP: $csv already has 60 lines"
            return 0
        fi
    fi
    return 1
}

# Run automated baseline experiment
run_auto() {
    local model=$1
    local method=$2

    log "=== Running $model $method ==="

    python3 "$SCRIPT_DIR/run_c0c1_experiment.py" \
        --method "$method" \
        --model "$model" \
        --dataset-path "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --max-turns "$MAX_TURNS" \
        --max-new-tokens "$MAX_TOKENS" \
        --resume

    log "=== Done: $model $method ==="
}

# Run LLM Actor simulation
run_llm_actor() {
    local model=$1
    local mode=$2

    log "=== Running $model LLM Actor $mode ==="

    python3 "/root/experiment/llm_actor_simulation/run_simulation.py" \
        --model "$model" \
        --mode "$mode" \
        --dataset-path "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --max-turns "$MAX_TURNS" \
        --max-new-tokens "$MAX_TOKENS"

    log "=== Done: $model LLM Actor $mode ==="
}

########################################
# Phase 1: DeepSeek models
########################################
log "========== PHASE 1: DeepSeek Models =========="

# DeepSeek-1.5B - Automated baselines
for method in regenerate automated_local_repair constrained_decoding; do
    run_auto "deepseek_r1_qwen_1_5b" "$method"
done

# DeepSeek-8B - Automated baselines
for method in regenerate automated_local_repair constrained_decoding; do
    run_auto "deepseek_r1" "$method"
done

# DeepSeek-1.5B - LLM Actor
run_llm_actor "deepseek_r1_qwen_1_5b" "chat"
run_llm_actor "deepseek_r1_qwen_1_5b" "steer"
run_llm_actor "deepseek_r1_qwen_1_5b" "steer_lite_only"

# DeepSeek-8B - LLM Actor
run_llm_actor "deepseek_r1" "chat"
run_llm_actor "deepseek_r1" "steer"
run_llm_actor "deepseek_r1" "steer_lite_only"

########################################
# Phase 2: Qwen-8B
########################################
log "========== PHASE 2: Qwen-8B =========="

for method in regenerate automated_local_repair constrained_decoding; do
    run_auto "qwen3_8b" "$method"
done

run_llm_actor "qwen3_8b" "chat"
run_llm_actor "qwen3_8b" "steer"
run_llm_actor "qwen3_8b" "steer_lite_only"

########################################
# Phase 3: Remaining models
########################################
log "========== PHASE 3: Qwen-1.5B, Llama-8B, Qwen-14B =========="

# Qwen-1.5B
for method in regenerate automated_local_repair constrained_decoding; do
    run_auto "qwen2_5_1_5b" "$method"
done
run_llm_actor "qwen2_5_1_5b" "chat"
run_llm_actor "qwen2_5_1_5b" "steer"
run_llm_actor "qwen2_5_1_5b" "steer_lite_only"

# Llama-8B
for method in regenerate automated_local_repair constrained_decoding; do
    run_auto "llama3_8b" "$method"
done
run_llm_actor "llama3_8b" "chat"
run_llm_actor "llama3_8b" "steer"
run_llm_actor "llama3_8b" "steer_lite_only"

# Qwen-14B
for method in regenerate automated_local_repair constrained_decoding; do
    run_auto "qwen2_5_14b_instruct" "$method"
done
run_llm_actor "qwen2_5_14b_instruct" "chat"
run_llm_actor "qwen2_5_14b_instruct" "steer"
run_llm_actor "qwen2_5_14b_instruct" "steer_lite_only"

log "========== ALL DONE =========="
