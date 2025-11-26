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


read -p "Choose download source: (1) Hugging Face, (2) ModelScope [Default: 2]: " source_choice
source_choice=${source_choice:-2} # 默认为2 (ModelScope)

if [ "$source_choice" == "1" ]; then
    # --- HUGGING FACE 下载逻辑 ---
    echo "Source: Hugging Face selected."
    
    # ---  检查并登录 Hugging Face ---
    echo "========================================="
    echo "Step 2: Checking and Logging into Hugging Face Hub..."
    echo "========================================="

    # 使用 whoami 检查是否已经登录，将输出重定向到/dev/null以保持界面干净
    if huggingface-cli whoami > /dev/null 2>&1; then
        echo "Already logged in to Hugging Face. Skipping login."
    else
        echo "Not logged in. You need a Hugging Face Access Token."
        echo "Access https://huggingface.co/settings/tokens create a 'write' TOKEN."
        echo "Then paste the token into the prompt below."
        huggingface-cli login
        echo "Hugging Face login successful."
        echo ""
    fi

    # --- 检查 git-lfs 是否安装 ---
    echo "Checking if git-lfs is installed..."
    if ! command -v git-lfs &> /dev/null; then
        echo "------------------------------------------------------------"
        echo "Error: git-lfs is not installed."
        echo "git-lfs is required to download large model files from Hugging Face."
        echo ""
        echo "Please install it using your system's package manager."
        echo "For Debian/Ubuntu, run this command:"
        echo "  sudo apt-get update && sudo apt-get install -y git-lfs"
        echo ""
        echo "After installation, please run this script again."
        echo "------------------------------------------------------------"
        exit 1
    fi
    echo "git-lfs is installed."
    echo "-----------------------------------------"

    # 定义要下载的模型
    # 注意: 截至目前，最新的Qwen系列是Qwen2。这里以Qwen2为例。
    # 您可以替换成您需要的任何Hugging Face上的模型ID。
    declare -A models
    models=(
        # ["Qwen/Qwen2-1.5B-Instruct"]="qwen2-1.5b-instruct"
        # ["Qwen/Qwen2-7B-Instruct"]="qwen2-7b-instruct" # 接近您提到的8B
        # ["Qwen/Qwen2-0.5B-Instruct"]="qwen2-0.5b-instruct" # 一个更小的版本作为补充
        # ["Qwen/Qwen3-8B"]="qwen3-8b"
        # ["Qwen/Qwen3-4B"]="qwen3-4b"
        ["Qwen/Qwen3-1.7B"]="qwen3-1.7b"
    )

    # 遍历并下载模型
    for model_id in "${!models[@]}"; do
        local_name=${models[$model_id]}
        target_dir="$MODEL_SAVE_PATH/$local_name"

        echo "Downloading model: $model_id to $target_dir"

        # 检查目录是否存在，并据此决定是克隆还是更新
        if [ -d "$target_dir" ]; then
            echo "Directory $target_dir already exists. Checking for completeness..."
            # 进入目录并使用 git lfs pull 来下载任何缺失的大文件
            # 这可以修复中断的下载并显示进度
            (cd "$target_dir" && git lfs pull)
            echo "Model $model_id is now complete."
        else
            echo "Cloning new model: $model_id..."
            # 1. 先克隆仓库结构，跳过LFS大文件，速度很快
            GIT_LFS_SKIP_SMUDGE=1 git clone "https://hf-mirror.com/$model_id" "$target_dir"
            # 2. 进入目录，再用 git lfs pull 拉取所有大文件，会显示进度条
            (cd "$target_dir" && git lfs pull)
            echo "Model $model_id downloaded successfully."
        fi
    done

else
    # --- MODELSCOPE 下载逻辑 ---
    echo "Source: ModelScope selected."

    # 3a. 定义 ModelScope 模型列表
    MODEL_LIST="LLM-Research/Meta-Llama-3.1-8B-Instruct deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/Qwen3-8B"
    # 您可以添加更多模型，用空格隔开:
    # MODEL_LIST="model1 model2 model3"

    for model_id in $MODEL_LIST; do
        echo "--- Preparing to download model: $model_id ---"
        
        python3 download_model.py "$model_id" "$MODEL_SAVE_PATH"
        
        if [ $? -ne 0 ]; then
            echo "Failed to download $model_id. See error above."
        fi
    done
fi



echo "========================================="
echo "All tasks completed!"
echo "Models are located in: $MODEL_SAVE_PATH"
echo "========================================="
