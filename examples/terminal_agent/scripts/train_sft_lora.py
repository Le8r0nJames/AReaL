#!/usr/bin/env python3
"""
TerminalBench SFT LoRA training for Qwen2.5-7B-Instruct.
Trains on GPT-5 collected trajectories (10 samples).
"""

import sys

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import SFTTrainer
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
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

    with SFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
