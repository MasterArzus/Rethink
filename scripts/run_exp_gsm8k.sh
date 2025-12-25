#!/bin/bash

# Rethink Experiment Script for GSM8K (Math Reasoning)
# Usage: bash scripts/run_exp_gsm8k.sh

export OPENAI_API_KEY="ms-d6f8012b-6730-4632-a67d-e2f4ee43ea71"
export OPENAI_BASE_URL="https://api-inference.modelscope.cn/v1" 
export JUDGE_MODEL="deepseek-ai/DeepSeek-V3.2"

# Default Model Path (Update as needed)
MODEL_PATH="/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3-8B"

OUTPUT_DIR="outputs/experiments_gsm8k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Starting GSM8K Experiments"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================"

# 1. Oracle Baseline
echo "Running Oracle Baseline..."
python -u experiments/math/run_oracle_baseline.py \
    --model-path "$MODEL_PATH" \
    --dataset "gsm8k" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_gsm8k_oracle.json"

# 2. Rethink Simulation
echo "Running Rethink Simulation..."
python -u experiments/math/run_rethink_simulation.py \
    --model-path "$MODEL_PATH" \
    --dataset "gsm8k" \
    --sos-threshold 0.3 \
    --mid-layer 15 \
    --ref-layer 31 \
    --api-key "$OPENAI_API_KEY" \
    --api-base "$OPENAI_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_gsm8k_rethink.json"

# 3. Dialogue Baseline
echo "Running Dialogue Baseline..."
python -u experiments/math/run_dialogue_baseline.py \
    --model-path "$MODEL_PATH" \
    --dataset "gsm8k" \
    --api-key "$OPENAI_API_KEY" \
    --api-base "$OPENAI_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_gsm8k_dialogue.json"

# 4. Sycophancy Test
echo "Running Sycophancy Test..."
python -u experiments/math/run_sycophancy_test.py \
    --model-path "$MODEL_PATH" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_sycophancy.json"

echo "GSM8K Experiments Completed!"
