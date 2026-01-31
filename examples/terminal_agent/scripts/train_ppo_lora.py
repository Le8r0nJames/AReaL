#!/usr/bin/env python3
"""TerminalBench PPO LoRA training for Qwen2.5-7B.

Continue training from SFT checkpoint with LoRA adapters using Harbor rollout workflow.

This script:
1. Loads the SFT-trained checkpoint (or base model)
2. Applies additional LoRA adapters (rank=16, alpha=32)
3. Uses Harbor's Terminus-2 for rollout generation
4. Gets partial rewards from Harbor verifier (0-1 based on test pass rate)
5. Optimizes policy with PPO to maximize task completion rate

Usage:
    # Check that your checkpoint is ready (if using SFT checkpoint):
    ls -la /path/to/checkpoint

    # Then run training from examples/terminal_agent directory:
    cd examples/terminal_agent
    bash scripts/run_ppo_with_harbor.sh

    # Or with custom model path:
    MODEL_PATH=/path/to/checkpoint bash scripts/run_ppo_with_harbor.sh

    # Monitor training:
    tail -f /tmp/areal/experiments/terminalbench-qwen25-ppo-lora/trial0/default/logs/train.log

Override config values via CLI:
    python3 scripts/train_ppo_lora.py --config configs/ppo_lora.yaml \
        --actor.optimizer.lr 1e-5 \
        --train_dataset.batch_size 2
"""

import logging
import os
import sys

# Suppress LiteLLM model info warnings
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

from areal.api.cli_args import GRPOConfig, load_expr_config  # noqa: E402
from areal.dataset import get_custom_dataset  # noqa: E402
from areal.experimental.trainer import PPOTrainer  # noqa: E402
from areal.utils.hf_utils import load_hf_tokenizer  # noqa: E402


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(valid_dataset)} validation samples")

    # Harbor rollout workflow configuration
    harbor_workflow_kwargs = dict(
        # Harbor dataset - using terminal-bench from registry or local cache
        # This path will be auto-discovered by Harbor
        harbor_dataset_path=None,  # Auto-discover from Harbor registry
        # Model endpoint for Terminus-2 to call
        # LiteLLM requires openai/ prefix
        # The model name should match vllm.served_model_name
        model_name="openai/qwen2.5-7b-lora",  # Matches vllm.served_model_name in config
        api_base="http://WILL_BE_DISCOVERED/v1",  # Auto-discovered from vLLM engine
        api_key="EMPTY",
        # Harbor settings
        n_concurrent_trials=2,  # REDUCED: 降低并发避免KV cache爆满
        timeout_sec=600.0,  # Increased timeout for 7B model
        environment_type="docker",
        # AReaL settings
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        # Enable verifier for partial rewards
        use_registry=True,  # Use Harbor registry for terminal-bench@2.0
    )

    # Evaluation uses slightly higher temperature for diversity
    eval_workflow_kwargs = {
        **harbor_workflow_kwargs,
        "gconfig": config.gconfig.new(temperature=0.7),
    }

    print("\n" + "=" * 60)
    print("PPO LoRA Training Configuration:")
    print("=" * 60)
    print(f"  Base Model: {config.actor.path}")
    print(f"  LoRA Rank: {config.actor.lora_rank}")
    print(f"  LoRA Alpha: {config.actor.lora_alpha}")
    print(f"  Learning Rate: {config.actor.optimizer.lr}")
    print(f"  Batch Size: {config.train_dataset.batch_size}")
    print(f"  KL Coefficient: {config.actor.kl_ctl}")
    print()
    print("  Workflow: HarborRolloutWorkflow")
    print("  Reward System: Harbor Partial Reward Verifier")
    print(f"  Model API: {harbor_workflow_kwargs['api_base']}")
    print(f"  Concurrent Trials: {harbor_workflow_kwargs['n_concurrent_trials']}")
    print(f"  Timeout: {harbor_workflow_kwargs['timeout_sec']}s")
    print("=" * 60 + "\n")

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.harbor_rollout.HarborRolloutWorkflow",
            workflow_kwargs=harbor_workflow_kwargs,
            eval_workflow="areal.workflow.harbor_rollout.HarborRolloutWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
