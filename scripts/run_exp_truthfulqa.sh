#!/bin/bash

# Rethink Experiment Script for TruthfulQA (Hallucination/Factuality)
# Usage: bash scripts/run_exp_truthfulqa.sh

export OPENAI_API_KEY="ms-d6f8012b-6730-4632-a67d-e2f4ee43ea71"
export OPENAI_BASE_URL="https://api-inference.modelscope.cn/v1" 
export JUDGE_MODEL="deepseek-ai/DeepSeek-V3.2"

# Default Model Path (Update as needed)
MODEL_PATH="/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3-8B"

OUTPUT_DIR="outputs/experiments_truthfulqa_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Starting TruthfulQA Experiments"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================"

# 1. Standard Baseline
echo "Running Standard Baseline..."
python -u experiments/truthful_qa/run_oracle_baseline.py \
    --model-path "$MODEL_PATH" \
    --dataset "truthful_qa" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_truthfulqa_baseline.json"

# 2. Rethink Simulation
echo "Running Rethink Simulation..."
python -u experiments/truthful_qa/run_rethink_simulation.py \
    --model-path "$MODEL_PATH" \
    --dataset "truthful_qa" \
    --sos-threshold 0.3 \
    --mid-layer 15 \
    --ref-layer 31 \
    --api-key "$OPENAI_API_KEY" \
    --api-base "$OPENAI_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_truthfulqa_rethink.json"

# 3. Dialogue Baseline
echo "Running Dialogue Baseline..."
python -u experiments/truthful_qa/run_dialogue_baseline.py \
    --model-path "$MODEL_PATH" \
    --dataset "truthful_qa" \
    --api-key "$OPENAI_API_KEY" \
    --api-base "$OPENAI_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --output-file "$OUTPUT_DIR/${MODEL_NAME}_truthfulqa_dialogue.json"

echo "TruthfulQA Experiments Completed!"
