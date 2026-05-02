#!/bin/bash
# Run all staged experiments for all methods and models

METHODS=("vanilla" "regenerate" "constrained_decoding" "automated_local_repair")
MODELS=("qwen2_5_1_5b" "deepseek_r1_qwen_1_5b" "llama3_8b" "deepseek_r1" "qwen3_8b" "qwen2_5_14b_instruct")

OUTPUT_DIR="/root/Rethink/experiments/ifeval"

echo "Starting all staged experiments..."
echo "=================================="

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo ">>> Running method: $METHOD"
    echo ">>>"

    for MODEL in "${MODELS[@]}"; do
        echo "--- Model: $MODEL ---"
        python run_all_methods_staged.py \
            --method "$METHOD" \
            --model "$MODEL" \
            --max-turns 5 \
            --output-dir "$OUTPUT_DIR" \
            --resume

        if [ $? -ne 0 ]; then
            echo "ERROR: $METHOD / $MODEL failed"
        fi
    done
done

echo ""
echo "=================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null | head -20