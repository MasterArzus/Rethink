#!/bin/bash

# =================================================================
# 脚本说明:
# 1. 安装 PyTorch, Transformers, Hugging Face Hub, ModelScope, Accelerate 等核心库。
# 2. 引导用户登录 Hugging Face Hub。
# 3. 提供使用 git-lfs 从 Hugging Face 下载模型。
#
# 使用方法:
# 1. (可选但推荐) 创建并激活一个Python虚拟环境。
# 2. 将此脚本保存为 `init.sh`。
# 3. 在终端中给予脚本执行权限: `chmod +x init.sh`
# 4. 运行脚本: `./init.sh`
# =================================================================

# 如果任何命令执行失败，则立即退出脚本
set -e

# --- 设置国内镜像源以加速下载 ---
# Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com
# Pip 镜像
PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

# --- 1. 环境准备与依赖安装 ---
echo "========================================="
echo "Step 1: Installing Python packages using mirror..."
echo "========================================="

# 确保pip是最新版本
pip install --upgrade pip $PIP_MIRROR

# 安装核心库
# - torch: 深度学习框架
# - transformers: Hugging Face的核心库，用于处理模型
# - huggingface_hub: 用于与Hugging Face Hub交互，包括登录和下载
# - modelscope: 阿里巴巴达摩院的模型社区库
# - accelerate: 帮助在不同硬件上无缝运行PyTorch代码的库
pip install torch transformers huggingface_hub modelscope accelerate $PIP_MIRROR

echo "Python packages installed successfully."
echo ""


# --- 2. 选择下载源并执行下载 ---
echo "========================================="
echo "Step 3: Choose Download Source"
echo "========================================="

# --- 设置模型保存路径 ---
DEFAULT_MODEL_SAVE_PATH="/root/autodl-fs"
MODEL_SAVE_PATH=""

# 询问用户是否使用默认路径
echo "The default path for saving models is: $DEFAULT_MODEL_SAVE_PATH"
read -p "Do you want to use this path? (Y/n): " -r use_default
use_default=${use_default:-Y} # 如果用户直接回车，默认为Y

if [[ "$use_default" =~ ^[Yy]$ ]]; then
    MODEL_SAVE_PATH="$DEFAULT_MODEL_SAVE_PATH"
else
    read -p "Please enter the new path to save models: " -r custom_path
    # 判断用户输入是否为空
    if [ -n "$custom_path" ]; then
        # 使用eval来处理波浪号扩展（~），例如 ~/models
        MODEL_SAVE_PATH=$(eval echo "$custom_path")
    else
        echo "Invalid input. Falling back to default path."
        MODEL_SAVE_PATH="$DEFAULT_MODEL_SAVE_PATH"
    fi
fi

echo "Models will be downloaded to: $MODEL_SAVE_PATH"

# 如果目标目录不存在，则创建它
mkdir -p "$MODEL_SAVE_PATH"


read -p "Choose download source: (1) Hugging Face, (2) ModelScope [Default: 1]: " source_choice
source_choice=${source_choice:-1}

if [ "$source_choice" == "1" ]; then
    DOWNLOAD_SOURCE="huggingface"
    echo "Source: Hugging Face selected."

    echo "========================================="
    echo "Step 2: Checking Hugging Face CLI login status"
    echo "========================================="
    if ! huggingface-cli whoami > /dev/null 2>&1; then
        echo "Not logged in. Please provide your Hugging Face token."
        huggingface-cli login
    fi
else
    DOWNLOAD_SOURCE="modelscope"
    echo "Source: ModelScope selected."
fi

# Seven-model target set: existing 8B + new 1.5B/13B-tier models
declare -A MODEL_TARGETS
MODEL_TARGETS=(
    ["LLM-Research/Meta-Llama-3.1-8B-Instruct"]="$MODEL_SAVE_PATH/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="$MODEL_SAVE_PATH/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    ["Qwen/Qwen3-8B"]="$MODEL_SAVE_PATH/Qwen/Qwen3-8B"
    ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]="$MODEL_SAVE_PATH/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ["Qwen/Qwen2.5-1.5B-Instruct"]="$MODEL_SAVE_PATH/Qwen/Qwen2.5-1.5B-Instruct"
    ["NousResearch/Llama-2-13b-chat-hf"]="$MODEL_SAVE_PATH/LLM-Research/Llama-2-13b-chat-hf"
    ["Qwen/Qwen-14B-Chat"]="$MODEL_SAVE_PATH/Qwen/Qwen-14B-Chat"
)

retry_download() {
    local model_id="$1"
    local target_dir="$2"
    local attempts=0
    local max_attempts=3
    while [ $attempts -lt $max_attempts ]; do
        attempts=$((attempts + 1))
        echo "[$attempts/$max_attempts] Downloading $model_id -> $target_dir"

        mkdir -p "$target_dir"

        if [ "$DOWNLOAD_SOURCE" == "huggingface" ]; then
            huggingface-cli download "$model_id" \
                --local-dir "$target_dir" \
                --resume-download
        else
            python3 download_model.py "$model_id" "$target_dir" "modelscope"
        fi

        if [ $? -eq 0 ]; then
            return 0
        fi
        echo "Download failed for $model_id, retrying..."
        sleep 3
    done
    return 1
}

for model_id in "${!MODEL_TARGETS[@]}"; do
    target_dir="${MODEL_TARGETS[$model_id]}"
    if retry_download "$model_id" "$target_dir"; then
        echo "Downloaded: $model_id"
    else
        echo "Failed after retries: $model_id"
    fi
done



echo "========================================="
echo "All tasks completed!"
echo "Models are located in: $MODEL_SAVE_PATH"
echo "========================================="
