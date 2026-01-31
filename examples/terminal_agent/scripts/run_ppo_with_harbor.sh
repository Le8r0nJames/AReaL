#!/bin/bash
# Harbor + AReaL PPO LoRA Training Launcher for Qwen2.5-7B
#
# This script configures vLLM to be accessible from Harbor containers
# by binding to 0.0.0.0 with a fixed port.
#
# Harbor containers will access vLLM via dev container IP (172.17.0.8)

set -e

# Configuration paths (modify these for your setup)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"  # Or SFT checkpoint path

# Configure vLLM for Harbor accessibility
export VLLM_HOST=0.0.0.0       # Bind to all interfaces
export VLLM_PORT=10001         # Fixed port
export HARBOR_VLLM_HOST=172.17.0.8  # Dev container IP (not gateway!)
export HARBOR_VLLM_PORT=10001  # Same port, accessible from Harbor containers

# Suppress LiteLLM warnings about missing model pricing info
# These are harmless - Harbor's LiteLLM just can't find pricing for custom models
export LITELLM_LOG=ERROR

# CUDA memory configuration to help with fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
echo "========================================"
echo "TerminalBench Qwen2.5-7B PPO LoRA Training"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Mode: GRPO (Group Relative Policy Optimization)"
echo "LoRA: rank=16, alpha=32"
echo ""
echo "vLLM Configuration:"
echo "  Bind to: ${VLLM_HOST}:${VLLM_PORT}"
echo "  Harbor access: http://${HARBOR_VLLM_HOST}:${HARBOR_VLLM_PORT}/v1"
echo ""
echo "Starting training..."
echo "========================================"
echo ""

# Run training from scripts directory
cd "$(dirname "$0")"

# Check if local override config exists
if [ -f ../configs/local_override.yaml ]; then
    echo "Using local override config: ../configs/local_override.yaml"
    python3 train_ppo_lora.py \
        --config ../configs/ppo_lora.yaml \
        --config ../configs/local_override.yaml \
        "$@" 2>&1 | grep -v "Failed to retrieve model info for"
else
    echo "No local override found, using MODEL_PATH=$MODEL_PATH"
    python3 train_ppo_lora.py \
        --config ../configs/ppo_lora.yaml \
        --actor.path "$MODEL_PATH" \
        "$@" 2>&1 | grep -v "Failed to retrieve model info for"
fi
