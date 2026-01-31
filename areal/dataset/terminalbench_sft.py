"""TerminalBench SFT dataset loader for GPT-5.1 collected trajectories."""

import json
import logging
from pathlib import Path

logger = logging.getLogger("Dataset")


def get_terminalbench_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    **kwargs,
):
    """Load TerminalBench SFT dataset from GPT-5.1 collected trajectories.

    Args:
        path: Path to the JSONL file (e.g., "./terminalbench_sft_data/train.jsonl")
        split: "train" or "test" (for now, we'll use same file for both)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        **kwargs: Additional arguments

    Returns:
        List of formatted samples for SFT training
    """
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"TerminalBench SFT data not found: {data_path}\n"
            f"Please run collect_gpt51_trajectories.sh first to collect training data."
        )

    logger.info(f"Loading TerminalBench SFT data from {data_path}")

    samples = []
    with open(data_path) as f:
        for line_idx, line in enumerate(f):
            try:
                example = json.loads(line)

                # Extract messages
                messages = example.get("messages", [])
                if not messages:
                    logger.warning(f"Line {line_idx}: missing 'messages'")
                    continue

                # Format as chat template
                # Tokenizer.apply_chat_template expects messages format
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                # Tokenize
                encodings = tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=max_length if max_length else tokenizer.model_max_length,
                    padding=False,
                )

                # Create loss mask - for SFT, we typically mask the prompt tokens
                # and only compute loss on the assistant's response
                # For simplicity, we'll compute loss on all tokens (mask all 1s)
                input_ids = encodings["input_ids"]
                loss_mask = [1] * len(input_ids)  # Compute loss on all tokens

                # Create sample - only include tensors needed for training
                sample = {
                    "input_ids": input_ids,
                    "attention_mask": encodings["attention_mask"],
                    "loss_mask": loss_mask,
                }

                samples.append(sample)

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_idx}: invalid JSON - {e}")
                continue
            except Exception as e:
                logger.warning(f"Line {line_idx}: error processing sample - {e}")
                continue

    logger.info(f"Loaded {len(samples)} samples from {data_path}")

    logger.info(
        f"Using all {len(samples)} samples for both train and validation (overfitting mode)"
    )

    return samples
