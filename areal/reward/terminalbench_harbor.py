"""TerminalBench reward function with Harbor container integration.

This reward function evaluates terminal command solutions by running them
in isolated Harbor containers, which provides accurate task verification
through Harbor's testing infrastructure.

The implementation supports two modes:
1. CLI mode: Calls Harbor CLI as subprocess (simple, isolated)
2. Direct mode: Uses Harbor Python API directly (faster, requires Harbor installed)
"""

import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from areal.utils import logging

logger = logging.getLogger("TerminalBenchHarborReward")


def extract_bash_code(completion: str) -> str:
    """Extract bash code from model completion.

    Handles various formats:
    - Plain bash commands
    - Markdown code blocks (```bash, ```sh, ```)
    - Multiple code blocks (concatenates them)

    Args:
        completion: The model's generated text

    Returns:
        Extracted bash commands as a string
    """
    solution = completion.strip()

    # Try to extract from markdown code blocks
    if "```" in solution:
        # Try bash/sh tagged blocks first
        code_blocks = re.findall(r"```(?:bash|sh)\n(.*?)```", solution, re.DOTALL)
        if code_blocks:
            return "\n".join(code_blocks)

        # Fallback to generic code blocks
        code_blocks = re.findall(r"```\n(.*?)```", solution, re.DOTALL)
        if code_blocks:
            return "\n".join(code_blocks)

    # Return as-is if no code blocks found
    return solution


async def run_harbor_cli_evaluation(
    task_name: str,
    solution_code: str,
    harbor_dataset_path: str,
    timeout_sec: float = 300.0,
) -> tuple[float, dict[str, Any]]:
    """Run Harbor evaluation using CLI subprocess.

    This creates a temporary Harbor task with the solution and runs it
    through Harbor's CLI interface.

    Args:
        task_name: Name of the TerminalBench task
        solution_code: Bash commands to evaluate
        harbor_dataset_path: Path to Harbor dataset containing the task
        timeout_sec: Maximum time to wait for evaluation

    Returns:
        Tuple of (reward, metadata_dict)
        - reward: 0.0 or 1.0 based on test results
        - metadata: Additional information about the evaluation
    """
    task_dir = Path(harbor_dataset_path) / task_name

    if not task_dir.exists():
        logger.warning(f"Task directory not found: {task_dir}")
        return 0.0, {"error": "task_not_found", "task_dir": str(task_dir)}

    # Create temporary directory for this evaluation
    with tempfile.TemporaryDirectory(prefix=f"harbor_{task_name}_") as temp_dir:
        temp_path = Path(temp_dir)

        # Copy task structure to temp directory
        temp_task = temp_path / task_name
        shutil.copytree(task_dir, temp_task)

        # Write solution to the expected location
        solution_dir = temp_task / "solution"
        solution_dir.mkdir(exist_ok=True)
        solution_file = solution_dir / "solution.sh"
        solution_file.write_text(solution_code)
        solution_file.chmod(0o755)

        # Create Harbor config file
        config = {
            "jobs_dir": str(temp_path / "jobs"),
            "n_attempts": 1,
            "timeout_multiplier": timeout_sec / 300.0,
            "orchestrator": {
                "type": "local",
                "n_concurrent_trials": 1,
                "quiet": True,
            },
            "environment": {
                "type": "docker",
                "force_build": False,
                "delete": True,
            },
            "agents": [
                {
                    "name": "oracle",
                }
            ],
            "datasets": [
                {
                    "path": str(temp_path),
                    "task_names": [task_name],  # Changed from task_ids to task_names
                }
            ],
        }

        config_file = temp_path / "harbor_config.yaml"
        import yaml

        config_file.write_text(yaml.dump(config))

        # Run Harbor CLI evaluation
        try:
            # Use harbor run command with config file
            cmd = [
                "harbor",
                "run",
                "--config",
                str(config_file),
            ]

            logger.debug(f"Running Harbor CLI: {' '.join(cmd)}")

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(),
                    timeout=timeout_sec,
                )
            except TimeoutError:
                logger.warning(f"Harbor evaluation timeout for task {task_name}")
                result.kill()
                await result.wait()
                return 0.0, {
                    "error": "timeout",
                    "timeout_sec": timeout_sec,
                }

            # Parse results from jobs directory
            jobs_dir = temp_path / "jobs"
            if jobs_dir.exists():
                # Look for reward in result files
                for result_file in jobs_dir.rglob("reward.*"):
                    if result_file.suffix == ".txt":
                        try:
                            reward_value = float(result_file.read_text().strip())
                            return reward_value, {
                                "success": True,
                                "return_code": result.returncode,
                            }
                        except (OSError, ValueError) as e:
                            logger.warning(f"Failed to parse reward file: {e}")
                    elif result_file.suffix == ".json":
                        try:
                            reward_data = json.loads(result_file.read_text())
                            reward_value = float(reward_data.get("reward", 0.0))
                            return reward_value, {
                                "success": True,
                                "return_code": result.returncode,
                                "metadata": reward_data,
                            }
                        except (OSError, ValueError, json.JSONDecodeError) as e:
                            logger.warning(f"Failed to parse reward JSON: {e}")

            # If no reward file found, assume failure
            logger.warning(
                f"No reward file found for task {task_name}. "
                f"Return code: {result.returncode}"
            )
            # Log stdout/stderr for debugging
            logger.warning(
                f"Harbor STDOUT:\n{stdout.decode('utf-8', errors='replace')}"
            )
            logger.warning(
                f"Harbor STDERR:\n{stderr.decode('utf-8', errors='replace')}"
            )
            return 0.0, {
                "error": "no_reward_file",
                "return_code": result.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[:500],
                "stderr": stderr.decode("utf-8", errors="replace")[:500],
            }

        except FileNotFoundError:
            logger.error(
                "Harbor CLI not found. Please install Harbor: "
                "pip install harbor or uv tool install harbor"
            )
            return 0.0, {"error": "harbor_not_installed"}
        except Exception as e:
            logger.warning(
                f"Exception during Harbor CLI evaluation for {task_name}: {e}",
                exc_info=True,
            )
            return 0.0, {"error": "exception", "message": str(e)}


def terminalbench_harbor_reward_fn(
    prompt,  # noqa: ARG001
    completions,
    prompt_ids,  # noqa: ARG001
    completion_ids,  # noqa: ARG001
    task_name: str,
    task_dir: str,
    harbor_dataset_path: str | None = None,
    timeout_sec: float = 300.0,
    **kwargs,  # noqa: ARG001
) -> float:
    """Evaluate TerminalBench solution using Harbor container.

    This is the synchronous wrapper for async Harbor evaluation.
    It extracts bash commands from the model's completion and runs them
    through Harbor's containerized testing infrastructure.

    Args:
        prompt: The task prompt (unused)
        completions: The model's generated solution
        prompt_ids: Tokenized prompt (unused)
        completion_ids: Tokenized completion (unused)
        task_name: Name of the TerminalBench task
        task_dir: Path to the task directory (AReaL format)
        harbor_dataset_path: Path to Harbor dataset (defaults to parent of task_dir)
        timeout_sec: Maximum evaluation time
        **kwargs: Additional arguments

    Returns:
        Reward value between 0.0 and 1.0
    """
    try:
        # Extract bash code from completion
        solution_code = extract_bash_code(str(completions))

        if len(solution_code.strip()) < 10:
            logger.debug(f"Solution too short for task {task_name}")
            return 0.0

        # Determine Harbor dataset path
        if harbor_dataset_path is None:
            # Default: assume task_dir parent is the dataset
            harbor_dataset_path = str(Path(task_dir).parent)

        # Run async evaluation
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                reward, metadata = executor.submit(
                    lambda: asyncio.run(
                        run_harbor_cli_evaluation(
                            task_name=task_name,
                            solution_code=solution_code,
                            harbor_dataset_path=harbor_dataset_path,
                            timeout_sec=timeout_sec,
                        )
                    )
                ).result()
        except RuntimeError:
            # No event loop running, create one
            reward, metadata = asyncio.run(
                run_harbor_cli_evaluation(
                    task_name=task_name,
                    solution_code=solution_code,
                    harbor_dataset_path=harbor_dataset_path,
                    timeout_sec=timeout_sec,
                )
            )

        logger.debug(
            f"Harbor evaluation for {task_name}: reward={reward}, metadata={metadata}"
        )

        return reward

    except Exception as e:
        logger.warning(
            f"Exception in terminalbench_harbor_reward_fn for task {task_name}: {e}",
            exc_info=True,
        )
        return 0.0


# Alias for backward compatibility
terminalbench_reward_fn = terminalbench_harbor_reward_fn
