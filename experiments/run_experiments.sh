#!/bin/bash

# Rethink Experiment Orchestration Script
# Usage: bash scripts/run_all_experiments.sh

# Configuration
# Set your OpenAI API Key here if you want to use LLM-as-a-Judge for simulation
# Please replace the placeholder with your actual API key.
# If using DeepSeek, set BASE_URL to "https://api.deepseek.com/v1"
export OPENAI_API_KEY="ms-d6f8012b-6730-4632-a67d-e2f4ee43ea71"
export OPENAI_BASE_URL="https://api-inference.modelscope.cn/v1" 
export JUDGE_MODEL="deepseek-ai/DeepSeek-V3.2"

# Set the paths to your models here. 
# You can add more models to this array.
declare -A MODELS
# Default path based on your workspace context. Update if needed.
MODELS["Llama-3-8B"]="/root/autodl-fs/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# Uncomment and update paths for other models
# MODELS["Qwen-2.5-7B"]="/path/to/Qwen2.5-7B-Instruct" 
# MODELS["Mistral-7B"]="/path/to/Mistral-7B-Instruct-v0.3"

# Output Directory
OUTPUT_DIR="outputs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Define Log File
LOG_FILE="$OUTPUT_DIR/execution.log"
echo "Logging all output to: $LOG_FILE"

# Redirect all subsequent output to the log file and console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting Rethink Experiments"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================"

for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    echo "Processing Model: $MODEL_NAME at $MODEL_PATH"
    
    # Check if model path exists (optional, depending on environment)
    # if [ ! -d "$MODEL_PATH" ]; then
    #     echo "Warning: Model path $MODEL_PATH does not exist. Skipping..."
    #     continue
    # fi

    # ---------------------------------------------------------
    # Experiment 1: Theoretical Feasibility (RQ1)
    # ---------------------------------------------------------
    echo "--------------------------------------------------------"
    echo "[Exp 1.1] Running Simulation & Baselines (GSM8K)"
    echo "--------------------------------------------------------"
    
    # 1.1.a Oracle Baseline & Standard Generation
    # This script runs both Standard Generation and Oracle Prompting baselines

    # echo "Running Oracle Baseline..."
    # python -u experiments/math/run_oracle_baseline.py \
    #     --model-path "$MODEL_PATH" \
    #     --dataset "gsm8k" \
    #     --output-file "$OUTPUT_DIR/${MODEL_NAME}_gsm8k_oracle.json"
        
    # 1.1.b Rethink Simulation
    # This script runs the Rethink method with automated intervention

    echo "Running Rethink Simulation..."
    # Add --api-key "$OPENAI_API_KEY" if you have one set
    # Parameters:
    # --sos-threshold: Sensitivity of intervention (0.2-0.5)
    # --mid-layer: The layer representing "internal thought" (e.g., 15 for Llama-3-8B)
    # --ref-layer: The layer representing "final output" (e.g., 31 for Llama-3-8B)
    CMD="python -u experiments/math/run_rethink_simulation.py --model-path \"$MODEL_PATH\" --dataset \"gsm8k\" --sos-threshold 0.3 --mid-layer 15 --ref-layer 31 --output-file \"$OUTPUT_DIR/${MODEL_NAME}_gsm8k_rethink.json\""
    
    if [ ! -z "$OPENAI_API_KEY" ]; then
        CMD="$CMD --api-key \"$OPENAI_API_KEY\""
    fi
    if [ ! -z "$OPENAI_BASE_URL" ]; then
        CMD="$CMD --api-base \"$OPENAI_BASE_URL\""
    fi
    if [ ! -z "$JUDGE_MODEL" ]; then
        CMD="$CMD --judge-model \"$JUDGE_MODEL\""
    fi

    eval $CMD

    # 1.1.c Traditional Dialogue Baseline (Black-box User)
    echo "Running Dialogue Baseline..."
    CMD="python -u experiments/math/run_dialogue_baseline.py --model-path \"$MODEL_PATH\" --dataset \"gsm8k\" --output-file \"$OUTPUT_DIR/${MODEL_NAME}_gsm8k_dialogue.json\""
    if [ ! -z "$OPENAI_API_KEY" ]; then
        CMD="$CMD --api-key \"$OPENAI_API_KEY\""
    fi
    if [ ! -z "$OPENAI_BASE_URL" ]; then
        CMD="$CMD --api-base \"$OPENAI_BASE_URL\""
    fi
    if [ ! -z "$JUDGE_MODEL" ]; then
        CMD="$CMD --judge-model \"$JUDGE_MODEL\""
    fi
    eval $CMD

    # ---------------------------------------------------------
    # Experiment 2: Cooperative Robustness (RQ2)
    # ---------------------------------------------------------
    echo "--------------------------------------------------------"
    echo "[Exp 1.2] Running Sycophancy Test"
    echo "--------------------------------------------------------"
    
    # This script tests if the model flips its answer under user pressure
    python -u experiments/math/run_sycophancy_test.py \
        --model-path "$MODEL_PATH" \
        --output-file "$OUTPUT_DIR/${MODEL_NAME}_sycophancy.json"

done

echo "========================================================"
echo "All Experiments Completed!"
echo "Results saved in $OUTPUT_DIR"
echo "========================================================"
