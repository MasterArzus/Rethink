#!/bin/bash

# Rethink Experiment Script for Math500 (Harder Math Reasoning)
# Usage: bash scripts/run_exp_math500.sh

export OPENAI_API_KEY="ms-d6f8012b-6730-4632-a67d-e2f4ee43ea71"
export OPENAI_BASE_URL="https://api-inference.modelscope.cn/v1" 
export JUDGE_MODEL="deepseek-ai/DeepSeek-V3.2"

# Default Model Path (Update as needed)
MODEL_PATH="/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3-8B"

OUTPUT_DIR="outputs/experiments_math500_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Starting Math500 Experiments"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================"

# Note: We reuse the 'math' experiment scripts but point to 'math500' dataset

# 1. Rethink Simulation
echo "Running Rethink Simulation..."
python -u experiments/math/run_rethink_simulation.py \
    --model-path "$MODEL_PATH" \
    --dataset "math500" \
    --sos-threshold 0.3 \
    --mid-layer 15 \
    --ref-layer 31 \
    --api-key "$OPENAI_API_KEY" \
    --api-base "$OPENAI_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_math500_rethink.json"

# 2. Dialogue Baseline
echo "Running Dialogue Baseline..."
python -u experiments/math/run_dialogue_baseline.py \
    --model-path "$MODEL_PATH" \
    --dataset "math500" \
    --api-key "$OPENAI_API_KEY" \
    --api-base "$OPENAI_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_math500_dialogue.json"

echo "Math500 Experiments Completed!"
