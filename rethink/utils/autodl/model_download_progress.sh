#!/usr/bin/env bash
set -u

models=(
  "deepseek_r1_qwen_1_5b|/root/autodl-fs/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "qwen2_5_1_5b|/root/autodl-fs/Qwen/Qwen2.5-1.5B-Instruct"
  "llama2_13b_chat|/root/autodl-fs/LLM-Research/Llama-2-13b-chat-hf"
  "qwen_14b_chat|/root/autodl-fs/Qwen/Qwen-14B-Chat"
)

for item in "${models[@]}"; do
  name="${item%%|*}"
  dir="${item#*|}"
  echo "=== ${name} ==="

  if [ ! -d "$dir" ]; then
    echo "status: MISSING_DIR"
    continue
  fi

  size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
  safes=$(find "$dir" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l)
  bins=$(find "$dir" -maxdepth 1 -type f -name 'pytorch_model*.bin' 2>/dev/null | wc -l)
  idx=$(find "$dir" -maxdepth 1 -type f -name '*.safetensors.index.json' 2>/dev/null | wc -l)

  echo "size: ${size}"
  echo "weights: $((safes + bins)) (safetensors=${safes}, bin=${bins}, index=${idx})"
done

echo "=== active download processes ==="
ps -ef | grep -E 'download_remaining_with_progress.sh|hf download|huggingface-cli download|download_model.py' | grep -v grep || true

echo "=== latest summary ==="
if [ -f /tmp/model_download_logs/remaining_summary.tsv ]; then
  cat /tmp/model_download_logs/remaining_summary.tsv
else
  echo "No summary file yet."
fi
