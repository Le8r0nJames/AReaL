#!/bin/bash
# TerminalBench LoRA SFT Training Script
# 使用 GPT-5 采集的成功轨迹训练 Qwen2.5-7B-Instruct

set -e

echo "=========================================="
echo "TerminalBench LoRA SFT Training"
echo "Model: Qwen2.5-7B-Instruct"
echo "Dataset: GPT-5 successful trajectories"
echo "=========================================="
echo ""

# 设置 PyTorch CUDA 内存管理，解决 OOM 问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 清理之前的 GPU 缓存
echo "清理 GPU 缓存..."
python3 << 'PYEOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"✓ GPU 缓存已清理")
    print(f"✓ 可用 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("警告: CUDA 不可用")
PYEOF

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

# Configuration paths (modify these for your setup)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"  # HuggingFace model or local path
TRAIN_DATA="${TRAIN_DATA:-terminalbench_sft_data/harbor_gpt5_positive_rewards.jsonl}"
VALID_DATA="${VALID_DATA:-$TRAIN_DATA}"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Train data: $TRAIN_DATA"
echo "  Valid data: $VALID_DATA"
echo ""

# 运行训练
cd "$(dirname "$0")"  # Go to scripts directory

# Check if local override config exists
if [ -f ../configs/local_override.yaml ]; then
    echo "使用本地配置覆盖: ../configs/local_override.yaml"
    python3 train_sft_lora.py \
        --config ../configs/sft_lora.yaml \
        --config ../configs/local_override.yaml
else
    echo "未找到本地配置，使用环境变量: MODEL_PATH=$MODEL_PATH"
    python3 train_sft_lora.py \
        --config ../configs/sft_lora.yaml \
        --actor.path "$MODEL_PATH" \
        --train_dataset.path "$TRAIN_DATA" \
        --valid_dataset.path "$VALID_DATA"
fi

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
