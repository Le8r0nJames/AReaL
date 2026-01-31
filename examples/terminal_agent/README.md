# Terminal Agent RL Training with TerminalBench

A complete example of training terminal-use agents using Reinforcement Learning with the
AReaL framework and TerminalBench dataset.

## Overview

This example demonstrates:

- **SFT (Supervised Fine-Tuning)**: Train on expert trajectories from TerminalBench
- **PPO RL Training**: Improve agent performance through reinforcement learning with
  Harbor evaluation
- **LoRA**: Memory-efficient training using Low-Rank Adaptation

## Quick Start

### Prerequisites

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Download TerminalBench dataset**:

```bash
# The dataset will be automatically downloaded from Harbor registry
# terminal-bench-sample@2.0 (10 tasks) for quick validation
# terminal-bench@2.0 (full dataset) for production training
```

3. **Set up environment**:

```bash
export VLLM_HOST=0.0.0.0
export VLLM_PORT=10001
export HARBOR_VLLM_HOST=172.17.0.8  # Docker bridge IP
export HARBOR_VLLM_PORT=10001
```

### Stage 1: SFT Training

Train the model on expert demonstrations:

```bash
cd examples/terminal_agent/scripts
bash run_sft.sh
```

**Configuration**: `configs/sft_lora.yaml`

- Model: Qwen2.5-7B
- LoRA: rank=16, alpha=32
- Dataset: GPT-5 trajectories from TerminalBench
- Training: 1 epoch with FSDP on 4 GPUs

**Output**: Checkpoint saved to `/home/min/Qwen2.5-7B-TerminalBench-LoRA-epoch1`

### Stage 2: PPO RL Training

Improve the SFT model through RL:

```bash
cd examples/terminal_agent/scripts
bash run_ppo_with_harbor.sh
```

**Configuration**: `configs/ppo_lora.yaml`

- Base model: SFT checkpoint from Stage 1
- Algorithm: GRPO (Group Relative Policy Optimization)
- LoRA: same as SFT (rank=16, alpha=32)
- Reward: Partial rewards based on test pass rate (0.0-1.0)
- Dataset: terminal-bench-sample@2.0 for validation

**Key features**:

- **Action-only training**: Only trains on agent actions (completion tokens), skipping
  environment outputs
- **Partial rewards**: Rewards based on test pass ratio instead of binary pass/fail
- **Harbor integration**: Evaluates agent performance in Docker containers with
  Terminus-2

## Architecture

### Dataset

- **Source**: TerminalBench 2.0 via Harbor registry
- **Integration**: `areal/dataset/terminalbench.py` and
  `areal/dataset/terminalbench_sft.py`
- **Format**: Multi-turn terminal interactions with bash commands and outputs

### Workflow

- **Class**: `HarborRolloutWorkflow` in `areal/workflow/harbor_rollout.py`
- **Agent**: Terminus-2 (OpenAI-compatible terminal agent)
- **Evaluation**: Harbor containerized sandbox with Docker environments
- **Token extraction**: Extracts only completion tokens (agent actions) to reduce
  sequence length

### Reward Function

- **Module**: `areal/reward/terminalbench_harbor.py`
- **Type**: Partial rewards (continuous 0.0-1.0)
- **Calculation**: `reward = passed_tests / total_tests`
- **Verifier**: Custom test scripts that run pytest and parse results

## Key Improvements

### 1. Action-Only Training

**Problem**: Traditional prompt+completion training leads to:

- Extremely long sequences (60k+ tokens)
- KV cache saturation (98% utilization)
- Training on irrelevant environment outputs

**Solution**: Extract only completion tokens (agent actions):

```python
# Only train on agent's actions
token_ids = completion_tokens  # Skip prompt tokens
mask_ids = [1] * len(completion_tokens)  # All trainable
```

**Result**:

- Sequence length: 60k+ → 1k-5k tokens
- KV cache usage: 98% → normal
- Training focused on agent behavior

### 2. Partial Reward Verifier

**Problem**: Binary pass/fail rewards provide sparse learning signal

**Solution**: Reward based on test pass ratio:

```bash
# Harbor verifier calculates partial reward
passed_tests=7
total_tests=10
reward=0.7  # 70% of tests passed
```

**Result**: Continuous feedback enables gradual improvement

### 3. GRPO Mode (No Reference Model)

**Problem**: Loading 7B reference model requires extra GPU memory

**Solution**: Set `kl_ctl: 0.0` to use GRPO with group-based normalization:

```yaml
actor:
  kl_ctl: 0.0  # Disable KL divergence with reference model
  reward_norm:
    mean_level: group
    std_level: group
    group_size: 4  # Sample 4 completions per prompt
```

**Result**: Single model training, saving ~23GB GPU memory

## Configuration Details

### SFT Configuration (`configs/sft_lora.yaml`)

```yaml
experiment_name: terminalbench-sft-lora
actor:
  path: "Qwen/Qwen2.5-7B-Instruct"
  dtype: float16
  use_lora: true
  lora_rank: 16
  lora_alpha: 32

train_dataset:
  path: "terminalbench_sft"
  type: "sft"
  batch_size: 1
  max_length: 8192
```

### PPO Configuration (`configs/ppo_lora.yaml`)

```yaml
experiment_name: terminalbench-qwen25-ppo-lora
actor:
  path: "/home/min/Qwen2.5-7B-TerminalBench-LoRA-epochx"
  kl_ctl: 0.0  # GRPO mode
  lora_rank: 16
  lora_alpha: 32

gconfig:
  n_samples: 4  # Group size for GRPO
  max_new_tokens: 1024  # Limit output length

train_dataset:
  path: "terminalbench:terminal-bench-sample@2.0"
  type: "rl"
```

## Troubleshooting

### Verifier Not Running

If you see `verifier_result: None` in logs:

1. Check Harbor task cache:

```bash
ls ~/.cache/harbor/tasks/*/tests/test.sh
```

2. Verify test.sh is executable and returns reward

1. Check trial result.json for errors:

```bash
cat ~/.cache/harbor/trials/<trial_name>/result.json
```

### Sequence Too Long

If sequences exceed 8192 tokens despite action-only training:

1. Reduce `max_new_tokens` in config (e.g., 512 or 768)
1. Limit number of agent turns in Harbor config
1. Check for nested token structures in logs

### KV Cache Saturation

If vLLM shows high KV cache usage:

1. Reduce `n_concurrent_trials` (default: 2)
1. Reduce `gpu_memory_utilization` (default: 0.7)
1. Increase `max_model_len` if needed

## Results

### Expected Metrics

**SFT Training**:

- Loss: ~2.5 → ~1.2 after 1 epoch
- Perplexity: ~12 → ~3.5

**PPO Training**:

- Initial reward (SFT model): 0.1 - 0.3
- After 10 epochs: 0.4 - 0.7 (target)
- Token length: 1k - 5k per rollout

### Sample Agent Trajectory

```bash
# Task: Count files in /tmp
$ ls /tmp | wc -l
42

# Agent learns to:
1. Use correct bash commands
2. Handle errors gracefully
3. Verify results
```

## File Structure

```
examples/terminal_agent/
├── README.md                    # This file
├── configs/
│   ├── sft_lora.yaml           # SFT training config
│   └── ppo_lora.yaml           # PPO RL training config
├── scripts/
│   ├── train_sft_lora.py       # SFT training script
│   ├── train_ppo_lora.py       # PPO training script
│   ├── run_sft.sh              # SFT launcher
│   └── run_ppo_with_harbor.sh  # PPO launcher with Harbor setup
└── docs/
    ├── setup.md                # Detailed setup instructions
    ├── sft_training.md         # SFT training guide
    └── ppo_training.md         # PPO training guide
```

## References

- **AReaL Framework**: https://github.com/InclusionAI/AReaL
- **TerminalBench**: https://arxiv.org/abs/2412.12047
- **Harbor**: Containerized task evaluation platform
- **Terminus-2**: OpenAI-compatible terminal agent

## Citation

If you use this example in your research, please cite:

```bibtex
@article{terminalbench2024,
  title={TerminalBench: Benchmarking Terminal Use Capabilities of Autonomous Agents},
  year={2024}
}

@software{areal2024,
  title={AReaL: Asynchronous Reinforcement Learning Framework},
  year={2024}
}
```

## License

This example follows the same license as the AReaL framework.
