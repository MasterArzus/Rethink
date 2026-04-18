#!/usr/bin/env bash
set -euo pipefail

# Mirror and timeout settings for unstable links.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

HF_BIN="/root/miniconda3/bin/hf"
PY_BIN="/root/miniconda3/bin/python"
FALLBACK_PY="/root/Rethink/rethink/utils/autodl/download_model.py"

LOG_DIR="/tmp/model_download_logs"
SUMMARY_TSV="$LOG_DIR/remaining_summary.tsv"
mkdir -p "$LOG_DIR"
: > "$SUMMARY_TSV"

# repo|target|min_weights
MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|/root/autodl-fs/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|1"
  "Qwen/Qwen2.5-1.5B-Instruct|/root/autodl-fs/Qwen/Qwen2.5-1.5B-Instruct|1"
  "NousResearch/Llama-2-13b-chat-hf|/root/autodl-fs/LLM-Research/Llama-2-13b-chat-hf|1"
  "Qwen/Qwen-14B-Chat|/root/autodl-fs/Qwen/Qwen-14B-Chat|1"
)

count_weights() {
  local target="$1"
  local safes bins
  safes=$(find "$target" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l)
  bins=$(find "$target" -maxdepth 1 -type f -name 'pytorch_model*.bin' 2>/dev/null | wc -l)
  echo $((safes + bins))
}

size_gb() {
  local target="$1"
  if [ -d "$target" ]; then
    du -sb "$target" 2>/dev/null | awk '{printf "%.2f", $1/1024/1024/1024}'
  else
    echo "0.00"
  fi
}

download_hf() {
  local repo="$1"
  local target="$2"
  mkdir -p "$target"
  # Keep max-workers low for flaky links; HF CLI will show progress bars in console.
  "$HF_BIN" download "$repo" --local-dir "$target" --max-workers 1
}

download_ms() {
  local repo="$1"
  local target="$2"
  mkdir -p "$target"
  "$PY_BIN" "$FALLBACK_PY" "$repo" "$target" modelscope
}

for item in "${MODELS[@]}"; do
  repo="${item%%|*}"
  rest="${item#*|}"
  target="${rest%%|*}"
  minw="${rest##*|}"

  echo "========================================"
  echo "Model: $repo"
  echo "Target: $target"

  current=$(count_weights "$target")
  if [ "$current" -ge "$minw" ]; then
    echo "Status: SKIP_ALREADY_OK (weights=$current, size=$(size_gb "$target")GB)"
    echo -e "$repo\t$target\tSKIP_ALREADY_OK\t$current\t$(size_gb "$target")GB" | tee -a "$SUMMARY_TSV"
    continue
  fi

  echo "Status: HF_DOWNLOAD_START"
  if download_hf "$repo" "$target"; then
    current=$(count_weights "$target")
    if [ "$current" -ge "$minw" ]; then
      echo "Status: HF_OK (weights=$current, size=$(size_gb "$target")GB)"
      echo -e "$repo\t$target\tHF_OK\t$current\t$(size_gb "$target")GB" | tee -a "$SUMMARY_TSV"
      continue
    fi
  fi

  echo "Status: HF_FAILED_OR_INCOMPLETE -> FALLBACK_MODELSCOPE"
  if download_ms "$repo" "$target"; then
    current=$(count_weights "$target")
    if [ "$current" -ge "$minw" ]; then
      echo "Status: MS_OK (weights=$current, size=$(size_gb "$target")GB)"
      echo -e "$repo\t$target\tMS_OK\t$current\t$(size_gb "$target")GB" | tee -a "$SUMMARY_TSV"
    else
      echo "Status: MS_INCOMPLETE (weights=$current, size=$(size_gb "$target")GB)"
      echo -e "$repo\t$target\tMS_INCOMPLETE\t$current\t$(size_gb "$target")GB" | tee -a "$SUMMARY_TSV"
    fi
  else
    current=$(count_weights "$target")
    echo "Status: FAILED (weights=$current, size=$(size_gb "$target")GB)"
    echo -e "$repo\t$target\tFAILED\t$current\t$(size_gb "$target")GB" | tee -a "$SUMMARY_TSV"
  fi

done

echo "========================================"
echo "DONE. Summary file: $SUMMARY_TSV"
cat "$SUMMARY_TSV"
