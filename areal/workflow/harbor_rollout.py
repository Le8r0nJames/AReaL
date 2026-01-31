"""Harbor Rollout Workflow for AReaL RL Training.

This workflow integrates Harbor containerized evaluation with AReaL RL training.
It uses Terminus-2 agent to generate rollouts and extracts tokens/logprobs/rewards
for training.

Key features:
1. Terminus-2 generates rollouts via Harbor sandbox
2. Collects token IDs, logprobs from agent metadata
3. Collects rewards from verifier results
4. Converts Harbor outputs to AReaL training tensors

Token Format Support:
- Priority 1: Harbor RL format (token_ids + mask_ids from metadata)
- Priority 2: Legacy format (rollout_details with prompt/completion separation)

TaskConfig Path Options:
1. Local task directories (current implementation):
   - Path format: /path/to/dataset/<task_name>/
   - Each task must have Harbor task structure:
     - task.toml (task metadata)
     - instruction.md (task description)
     - environment/Dockerfile (container setup)
     - tests/ (verification scripts)

2. Registry-based (recommended for Terminal-Bench 2.0):
   - Use Harbor registry: terminal-bench@2.0
   - Automatically resolves task structure
   - Better for production deployment

   Example using registry:
   ```python
   from harbor.models.trial.config import DatasetConfig

   dataset_config = DatasetConfig(
       registry={},
       name="terminal-bench",
       version="2.0",
   )
   # Then use dataset_config.get_tasks() to get TaskConfigs
   ```

See Harbor Adapters documentation for task directory structure details.
"""

from pathlib import Path
from typing import Any

import torch
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import (
    AgentConfig,
    JobConfig,
    OrchestratorConfig,
    RegistryDatasetConfig,
)
from harbor.models.registry import RemoteRegistryInfo
from harbor.models.trial.config import EnvironmentConfig, TaskConfig, VerifierConfig

from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("HarborRollout")


class HarborRolloutWorkflow(RolloutWorkflow):
    """Rollout workflow using Harbor + Terminus-2 for generation and evaluation.

    This workflow:
    1. Takes task names from dataset
    2. Launches Harbor Job with Terminus-2 agent
    3. Terminus-2 calls your vLLM/SGLang endpoint to generate solutions
    4. Harbor verifier evaluates solutions in containers
    5. Extracts tokens, logprobs, rewards from trial results
    6. Returns AReaL-compatible training tensors
    """

    def __init__(
        self,
        gconfig,
        tokenizer,
        reward_fn=None,  # Not used - Harbor provides rewards
        stats_scope: str = "harbor_rollout",
        dump_dir: str | None = None,
        enable_thinking: bool = False,
        # Harbor-specific parameters
        use_registry: bool = True,  # Use Harbor registry (recommended)
        dataset_name: str = "terminal-bench",
        dataset_version: str = "2.0",
        task_names: list[str] | None = None,  # Specific tasks to run (None = all)
        n_tasks: int | None = None,  # Number of tasks to sample (None = all)
        # Legacy local path support (deprecated)
        harbor_dataset_path: str | None = None,
        # Agent configuration
        model_name: str = "hosted_vllm/model",
        base_url: str | None = None,  # Legacy parameter, use api_base instead
        api_base: str | None = None,  # Harbor LiteLLM uses api_base
        api_key: str = "EMPTY",
        n_concurrent_trials: int = 4,
        timeout_sec: float = 300.0,
        environment_type: str = "docker",
        **kwargs,
    ):
        super().__init__()

        self.gconfig = gconfig
        self.tokenizer = tokenizer if isinstance(tokenizer, str) else tokenizer
        self.stats_scope = stats_scope
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.enable_thinking = enable_thinking

        # Harbor dataset configuration
        self.use_registry = use_registry
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.task_names = task_names
        self.n_tasks = n_tasks
        self.harbor_dataset_path = (
            Path(harbor_dataset_path) if harbor_dataset_path else None
        )

        # Agent configuration
        self.model_name = model_name
        # Support both api_base (preferred for Harbor) and base_url (legacy)
        self.api_base = (
            api_base
            if api_base is not None
            else (base_url or "http://localhost:8000/v1")
        )
        self.api_key = api_key
        self.n_concurrent_trials = n_concurrent_trials
        self.timeout_sec = timeout_sec

        # Convert environment_type string to enum
        self.environment_type = (
            EnvironmentType.DOCKER
            if environment_type == "docker"
            else EnvironmentType(environment_type)
        )

        if use_registry:
            logger.info(
                f"Initialized HarborRolloutWorkflow with registry: "
                f"dataset={dataset_name}@{dataset_version}, "
                f"task_names={task_names}, n_tasks={n_tasks}, "
                f"model={model_name}, api_base={self.api_base}"
            )
        else:
            logger.info(
                f"Initialized HarborRolloutWorkflow with local path: "
                f"path={harbor_dataset_path}, "
                f"model={model_name}, api_base={self.api_base}"
            )

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run rollout episode using Harbor + Terminus-2.

        Args:
            engine: AReaL inference engine (not used - Terminus-2 does generation)
            data: Batch data containing task_name, task_dir, etc.

        Returns:
            Dict with training tensors:
            - input_ids: [batch, seq_len]
            - logprobs: [batch, seq_len]
            - loss_mask: [batch, seq_len]
            - rewards: [batch]
            - attention_mask: [batch, seq_len]
        """
        # Extract task information from batch
        # Support both task_id (from TerminalBench dataset) and task_name (legacy)
        task_ids = data.get("task_id", data.get("task_name", []))
        if isinstance(task_ids, str):
            task_ids = [task_ids]

        batch_size = len(task_ids)

        logger.info(f"Starting Harbor rollout for {batch_size} tasks: {task_ids}")

        # Get actual vLLM server address
        # Priority:
        # 1. From engine (if initialized and has addresses)
        # 2. From environment variables (VLLM_HOST, VLLM_PORT, HARBOR_VLLM_HOST, HARBOR_VLLM_PORT)
        # 3. From self.base_url (configured default)
        import os

        actual_base_url = None

        # Try to get from engine first
        if hasattr(engine, "_engine") and hasattr(engine._engine, "addresses"):
            if engine._engine.addresses:
                # Engine has been initialized and has server addresses
                actual_address = engine._engine.addresses[0]
                host, port = actual_address.split(":")

                # Check if we're using fixed vLLM port (via environment variable)
                vllm_host = os.getenv("VLLM_HOST", host)
                vllm_port = os.getenv("VLLM_PORT", port)

                # Harbor may need a different port (e.g., when using port mapping)
                harbor_vllm_port = os.getenv("HARBOR_VLLM_PORT", vllm_port)

                # If vLLM binds to 0.0.0.0, Harbor containers need to use host-accessible IP
                if vllm_host == "0.0.0.0":
                    import socket

                    # Check if user specified explicit host IP for Harbor
                    harbor_vllm_host = os.getenv("HARBOR_VLLM_HOST")
                    if harbor_vllm_host:
                        harbor_accessible_host = harbor_vllm_host
                        logger.info(
                            f"Using HARBOR_VLLM_HOST={harbor_vllm_host} for Harbor access"
                        )
                    else:
                        # When vLLM runs inside a container (not on host), Harbor containers
                        # should access it via the container's IP, not the gateway
                        # Try to detect if we're inside a container
                        try:
                            # Check if /.dockerenv exists (we're in a container)
                            with open("/.dockerenv"):
                                # We're in a container, use the actual bind address from engine
                                # This is the container's IP on docker bridge
                                harbor_accessible_host = host
                                logger.info(
                                    f"Detected running in container, Harbor will access "
                                    f"container IP {host}:{harbor_vllm_port}"
                                )
                        except FileNotFoundError:
                            # Not in a container, try standard host resolution
                            try:
                                socket.gethostbyname("host.docker.internal")
                                harbor_accessible_host = "host.docker.internal"
                            except socket.gaierror:
                                # Fall back to docker0 gateway on Linux
                                harbor_accessible_host = "172.17.0.1"

                    actual_base_url = (
                        f"http://{harbor_accessible_host}:{harbor_vllm_port}/v1"
                    )
                    logger.info(
                        f"vLLM binds to 0.0.0.0:{vllm_port}, Harbor will access via "
                        f"{actual_base_url} (from engine)"
                    )
                else:
                    actual_base_url = f"http://{vllm_host}:{vllm_port}/v1"
                    logger.info(
                        f"Using vLLM server address: {actual_base_url} (from engine)"
                    )

        # If engine doesn't have addresses yet, try environment variables
        if actual_base_url is None:
            vllm_host = os.getenv("VLLM_HOST")
            vllm_port = os.getenv("VLLM_PORT")

            if vllm_host and vllm_port:
                # User configured via environment variables
                harbor_vllm_port = os.getenv("HARBOR_VLLM_PORT", vllm_port)

                if vllm_host == "0.0.0.0":
                    import socket

                    harbor_vllm_host = os.getenv("HARBOR_VLLM_HOST")
                    if harbor_vllm_host:
                        harbor_accessible_host = harbor_vllm_host
                        logger.info(
                            f"Using HARBOR_VLLM_HOST={harbor_vllm_host} from env"
                        )
                    else:
                        # When running in container, detect and use container network
                        try:
                            with open("/.dockerenv"):
                                # We're in a container - need to figure out our IP
                                # For now, fall back to gateway as we don't have engine address yet
                                harbor_accessible_host = (
                                    "172.17.0.8"  # Dev container IP
                                )
                                logger.info(
                                    "Detected running in container, using dev container IP "
                                    f"{harbor_accessible_host}:{harbor_vllm_port}"
                                )
                        except FileNotFoundError:
                            try:
                                socket.gethostbyname("host.docker.internal")
                                harbor_accessible_host = "host.docker.internal"
                            except socket.gaierror:
                                harbor_accessible_host = "172.17.0.1"

                    actual_base_url = (
                        f"http://{harbor_accessible_host}:{harbor_vllm_port}/v1"
                    )
                    logger.info(
                        f"vLLM will bind to 0.0.0.0:{vllm_port}, Harbor will access via "
                        f"{actual_base_url} (from env vars)"
                    )
                else:
                    actual_base_url = f"http://{vllm_host}:{harbor_vllm_port}/v1"
                    logger.info(f"Using vLLM address from env: {actual_base_url}")
            else:
                # Fall back to configured api_base
                actual_base_url = self.api_base
                logger.info(
                    f"Engine not initialized yet and no env vars set, "
                    f"using configured api_base: {self.api_base}"
                )

        # Create Harbor Job with unique directory to avoid conflicts
        import time

        unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"

        # Choose between registry-based or local path-based task loading
        if self.use_registry:
            # Use Harbor registry (recommended for terminal-bench 2.0)
            logger.info(
                f"Using Harbor registry: {self.dataset_name}@{self.dataset_version} "
                f"with task_names={task_ids if len(task_ids) < 10 else f'{len(task_ids)} tasks'}"
            )

            job_config = JobConfig(
                jobs_dir=Path(f"/tmp/harbor_rollout_jobs/job_{unique_id}"),
                n_attempts=1,
                timeout_multiplier=self.timeout_sec / 300.0,
                orchestrator=OrchestratorConfig(
                    type="local",
                    n_concurrent_trials=self.n_concurrent_trials,
                    quiet=True,
                ),
                environment=EnvironmentConfig(
                    type=self.environment_type,
                    force_build=False,
                    delete=True,  # Clean up containers
                ),
                agents=[
                    AgentConfig(
                        name="terminus-2",
                        model_name=self.model_name,
                        kwargs={
                            "api_base": actual_base_url,
                            "api_key": self.api_key,
                            "temperature": self.gconfig.temperature,
                            "max_tokens": self.gconfig.max_new_tokens,
                            "collect_rollout_details": True,
                        },
                    )
                ],
                verifier=VerifierConfig(
                    disable=False,  # Enable verifier to get partial rewards from tests
                ),
                datasets=[
                    RegistryDatasetConfig(
                        registry=RemoteRegistryInfo(),  # Uses default Harbor registry
                        name=self.dataset_name,
                        version=self.dataset_version,
                        task_names=task_ids,  # Only run tasks from current batch
                    )
                ],
            )

            logger.info(
                f"Launching Harbor job with {len(task_ids)} tasks from registry"
            )
        else:
            # Legacy local path mode (deprecated)
            logger.info(f"Using local path mode: {self.harbor_dataset_path}")

            # Prepare task configs for Harbor
            task_configs = []
            for task_id in task_ids:
                task_path = self.harbor_dataset_path / task_id
                if not task_path.exists():
                    logger.warning(
                        f"Task path not found: {task_path}. "
                        f"Make sure harbor_dataset_path points to a directory containing Harbor task directories."
                    )
                    continue

                # Validate basic task structure
                required_files = ["task.yaml", "task.toml"]
                has_task_file = any((task_path / f).exists() for f in required_files)
                if not has_task_file:
                    logger.warning(
                        f"Task {task_id} missing task.yaml or task.toml. "
                        f"This may not be a valid Harbor task directory."
                    )
                    continue

                task_configs.append(
                    TaskConfig(
                        path=str(task_path),
                    )
                )

            if not task_configs:
                logger.error("No valid tasks found!")
                return self._empty_rollout(batch_size)

            job_config = JobConfig(
                jobs_dir=Path(f"/tmp/harbor_rollout_jobs/job_{unique_id}"),
                n_attempts=1,
                timeout_multiplier=self.timeout_sec / 300.0,
                orchestrator=OrchestratorConfig(
                    type="local",
                    n_concurrent_trials=self.n_concurrent_trials,
                    quiet=True,
                ),
                environment=EnvironmentConfig(
                    type=self.environment_type,
                    force_build=False,
                    delete=True,  # Clean up containers
                ),
                agents=[
                    AgentConfig(
                        name="terminus-2",
                        model_name=self.model_name,
                        kwargs={
                            "api_base": actual_base_url,
                            "api_key": self.api_key,
                            "temperature": self.gconfig.temperature,
                            "max_tokens": self.gconfig.max_new_tokens,
                            "collect_rollout_details": True,
                        },
                    )
                ],
                verifier=VerifierConfig(
                    disable=False,  # Enable verifier to get partial rewards from tests
                ),
                tasks=task_configs,
            )

            logger.info(
                f"Launching Harbor job with {len(task_configs)} tasks from local path"
            )

        # Log the complete JobConfig structure for debugging
        logger.info("=" * 80)
        logger.info("Harbor JobConfig structure (input to Harbor):")
        logger.info(f"  jobs_dir: {job_config.jobs_dir}")
        logger.info(f"  n_attempts: {job_config.n_attempts}")
        logger.info(f"  timeout_multiplier: {job_config.timeout_multiplier}")
        logger.info(f"  orchestrator.type: {job_config.orchestrator.type}")
        logger.info(
            f"  orchestrator.n_concurrent_trials: {job_config.orchestrator.n_concurrent_trials}"
        )
        logger.info(f"  environment.type: {job_config.environment.type}")
        logger.info(f"  environment.delete: {job_config.environment.delete}")
        logger.info(f"  verifier.disable: {job_config.verifier.disable}")
        logger.info(f"  agents: {len(job_config.agents)} agent(s)")
        for i, agent in enumerate(job_config.agents):
            logger.info(f"    Agent {i}: name={agent.name}, model={agent.model_name}")
            logger.info(f"      kwargs keys: {list(agent.kwargs.keys())}")
            logger.info(f"      api_base: {agent.kwargs.get('api_base')}")
            logger.info(f"      temperature: {agent.kwargs.get('temperature')}")
            logger.info(f"      max_tokens: {agent.kwargs.get('max_tokens')}")
            logger.info(
                f"      collect_rollout_details: {agent.kwargs.get('collect_rollout_details')}"
            )

        if self.use_registry and job_config.datasets:
            logger.info(f"  datasets: {len(job_config.datasets)} dataset(s)")
            for i, dataset in enumerate(job_config.datasets):
                logger.info(f"    Dataset {i}: name={dataset.name}@{dataset.version}")
                logger.info(
                    f"      task_names: {dataset.task_names if len(dataset.task_names) < 10 else f'{len(dataset.task_names)} tasks'}"
                )
        elif not self.use_registry and job_config.tasks:
            logger.info(f"  tasks: {len(job_config.tasks)} task(s)")
            for i, task in enumerate(job_config.tasks[:5]):  # Show first 5
                logger.info(f"    Task {i}: path={task.path}")
        logger.info("=" * 80)

        # Run Harbor Job asynchronously
        job = Job(config=job_config)
        result = await job.run()

        # Log the complete result structure for debugging
        logger.info("=" * 80)
        logger.info("Harbor Job Result (output from Harbor):")
        logger.info(f"  Total trials: {len(result.trial_results)}")
        logger.info(f"  n_total_trials (from stats): {result.n_total_trials}")
        logger.info(f"  Result object type: {type(result).__name__}")
        logger.info(f"  Result attributes: {list(vars(result).keys())}")

        # WORKAROUND: Harbor Job.run() doesn't populate trial_results in the returned object
        # We need to manually load trial results from the trial directories
        if len(result.trial_results) == 0 and result.n_total_trials > 0:
            logger.warning("  ⚠️  trial_results is EMPTY - Harbor bug detected!")
            logger.info("  Loading trial results manually from trial directories...")

            from harbor.models.trial.result import TrialResult

            trial_results = []
            jobs_dir = job_config.jobs_dir
            # Harbor creates a timestamped subdirectory
            for subdir in jobs_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("job_"):
                    # Found the timestamped directory
                    logger.info(f"  Scanning trial directory: {subdir}")
                    for trial_dir in subdir.iterdir():
                        if trial_dir.is_dir():
                            result_path = trial_dir / "result.json"
                            if result_path.exists():
                                logger.info(f"    Loading trial: {trial_dir.name}")
                                trial_result = TrialResult.model_validate_json(
                                    result_path.read_text()
                                )
                                trial_results.append(trial_result)

            logger.info(f"  Manually loaded {len(trial_results)} trial results")
            result.trial_results = trial_results

        logger.info(f"  Final trial_results count: {len(result.trial_results)}")
        logger.info("=" * 80)

        # Extract rollout data from trial results
        rollouts = []
        for idx, trial_result in enumerate(result.trial_results):
            logger.info("-" * 80)
            logger.info(f"Processing Trial {idx + 1}/{len(result.trial_results)}:")
            logger.info(f"  trial_name: {trial_result.trial_name}")
            logger.info(f"  TrialResult attributes: {list(vars(trial_result).keys())}")

            rollout = self._extract_rollout_from_trial(trial_result, trial_idx=idx)
            rollouts.append(rollout)

            # Log the extracted rollout data
            logger.info(f"  Trial {idx + 1} - Extracted Rollout:")
            logger.info(f"    token_ids length: {len(rollout['token_ids'])}")
            logger.info(f"    mask_ids length: {len(rollout['mask_ids'])}")
            logger.info(f"    logprobs length: {len(rollout['logprobs'])}")
            logger.info(f"    reward: {rollout['reward']}")
            if len(rollout["token_ids"]) > 0 and len(rollout["token_ids"]) <= 20:
                logger.info(f"    token_ids sample: {rollout['token_ids'][:20]}")
                logger.info(f"    mask_ids sample: {rollout['mask_ids'][:20]}")
            logger.info("-" * 80)

        # Convert to AReaL training tensors
        return self._convert_to_tensors(rollouts, batch_size)

    def _extract_rollout_from_trial(self, trial_result, trial_idx: int = 0) -> dict:
        """Extract rollout data from Harbor trial result.

        Supports two formats:
        1. Harbor RL format (recommended): token_ids + mask_ids
        2. Legacy format: rollout_details with prompt/completion separation

        Args:
            trial_result: Harbor TrialResult object
            trial_idx: Trial index for logging

        Returns:
            Dict with:
            - token_ids: List[int] (combined prompt+completion)
            - mask_ids: List[int] (0=prompt, 1=completion for loss)
            - logprobs: List[float] (optional, may be empty)
            - reward: float
        """
        rollout = {
            "token_ids": [],
            "mask_ids": [],
            "logprobs": [],
            "reward": 0.0,
        }

        # Extract reward from verifier result
        logger.debug(f"Trial {trial_idx}: Processing verifier result")

        # Check for exceptions
        if hasattr(trial_result, "exception_info") and trial_result.exception_info:
            exception_str = str(trial_result.exception_info)
            logger.warning(f"Trial {trial_idx}: Exception - {exception_str[:200]}")

        if not trial_result.verifier_result:
            logger.debug(
                f"Trial {trial_idx}: verifier_result is None, trying fallback extraction"
            )

            # Fallback: Extract reward from test-stdout.txt
            # Custom verifier scripts can write reward with HARBOR_REWARD_START/END markers
            try:
                import re
                from pathlib import Path
                from urllib.parse import urlparse

                trial_uri = getattr(trial_result, "trial_uri", None)
                if trial_uri:
                    # Convert file:// URI to path
                    if trial_uri.startswith("file://"):
                        trial_path = Path(urlparse(trial_uri).path)
                    else:
                        trial_path = Path(trial_uri)

                    test_stdout_path = trial_path / "verifier" / "test-stdout.txt"

                    if test_stdout_path.exists():
                        stdout_content = test_stdout_path.read_text()

                        # Try multiple reward extraction patterns

                        # Pattern 1: HARBOR_REWARD markers (preferred)
                        match = re.search(
                            r"HARBOR_REWARD_START\s+([\d.]+)\s+HARBOR_REWARD_END",
                            stdout_content,
                        )
                        if match:
                            extracted_reward = float(match.group(1))
                            logger.info(
                                f"Trial {trial_idx}: Extracted reward={extracted_reward:.4f} "
                                f"from HARBOR_REWARD markers"
                            )
                            trial_result._extracted_reward = extracted_reward
                        else:
                            # Pattern 2: "Reward: 0.1234 (12.3%)" from Test Summary
                            match = re.search(
                                r"Reward:\s+([\d.]+)\s+\(", stdout_content
                            )
                            if match:
                                extracted_reward = float(match.group(1))
                                logger.info(
                                    f"Trial {trial_idx}: Extracted reward={extracted_reward:.4f} "
                                    f"from Test Summary line"
                                )
                                trial_result._extracted_reward = extracted_reward
                            else:
                                logger.debug(
                                    f"Trial {trial_idx}: No reward patterns found in stdout"
                                )
                    else:
                        logger.debug(f"Trial {trial_idx}: test-stdout.txt not found")

            except Exception as e:
                logger.debug(
                    f"Trial {trial_idx}: Could not extract reward from stdout: {e}"
                )

        # Extract reward using multiple fallback strategies
        reward = 0.0

        # Priority 1: Fallback extraction from test-stdout.txt
        if hasattr(trial_result, "_extracted_reward"):
            reward = trial_result._extracted_reward

        # Priority 2: Standard verifier_result extraction
        elif trial_result.verifier_result:
            vr = trial_result.verifier_result

            # Try multiple reward sources
            if hasattr(vr, "rewards") and vr.rewards and isinstance(vr.rewards, dict):
                reward = float(vr.rewards.get("reward", 0.0))
            elif hasattr(vr, "score") and vr.score is not None:
                reward = float(vr.score)
            elif hasattr(vr, "passed") and vr.passed is not None:
                reward = 1.0 if vr.passed else 0.0

        rollout["reward"] = reward
        logger.info(f"Trial {trial_idx}: reward={reward:.4f}")

        # Extract tokens from agent result
        logger.debug(f"Trial {trial_idx}: Extracting tokens from agent_result")

        # Early exit if no agent result
        if not trial_result.agent_result:
            logger.warning(f"Trial {trial_idx}: No agent_result found")
            return rollout

        # Priority 1: rollout_details (Harbor's official RL format)
        if (
            hasattr(trial_result.agent_result, "rollout_details")
            and trial_result.agent_result.rollout_details
        ):
            rollout_details = trial_result.agent_result.rollout_details

            logger.info(
                f"  Extracting from rollout_details ({len(rollout_details)} turns)"
            )

            # IMPROVEMENT 3: Only train on completion tokens (actions), not prompts
            # This drastically reduces sequence length and focuses learning on agent actions
            all_completion_tokens = []
            all_logprobs = []

            for turn_idx, detail in enumerate(rollout_details):
                # Each turn has completion tokens (agent's actions)
                completion_tokens = detail.get("completion_token_ids", [])
                turn_logprobs = detail.get("logprobs", [])

                # DEBUG: Print nested structure for first turn
                if turn_idx == 0:
                    logger.info(
                        f"    Turn {turn_idx}: completion_tokens type={type(completion_tokens)}, "
                        f"len={len(completion_tokens) if completion_tokens else 0}"
                    )
                    if completion_tokens:
                        logger.info(
                            f"      completion_tokens[0] type={type(completion_tokens[0])}, "
                            f"is_list={isinstance(completion_tokens[0], list)}"
                        )
                        if isinstance(completion_tokens[0], list):
                            logger.info(
                                f"      NESTED! completion_tokens[0] len={len(completion_tokens[0])}"
                            )

                # Flatten if nested (multi-turn format)
                if completion_tokens and isinstance(completion_tokens[0], list):
                    completion_tokens = [
                        token for sublist in completion_tokens for token in sublist
                    ]
                if turn_logprobs and isinstance(turn_logprobs[0], list):
                    turn_logprobs = [lp for sublist in turn_logprobs for lp in sublist]

                all_completion_tokens.extend(completion_tokens)
                all_logprobs.extend(turn_logprobs)

            # Build token_ids: ONLY completion tokens (agent actions)
            rollout["token_ids"] = all_completion_tokens
            rollout["mask_ids"] = [1] * len(
                all_completion_tokens
            )  # Train on all actions

            # Pad logprobs to match token_ids
            if len(all_logprobs) < len(rollout["token_ids"]):
                # Pad with 0.0 if logprobs are missing
                rollout["logprobs"] = all_logprobs + [0.0] * (
                    len(rollout["token_ids"]) - len(all_logprobs)
                )
            else:
                rollout["logprobs"] = all_logprobs[: len(rollout["token_ids"])]

            logger.info(
                f"  ✓ Extracted {len(rollout['token_ids'])} action tokens across {len(rollout_details)} turns"
            )
            logger.info(
                "    (Skipped prompts to focus on agent actions and reduce sequence length)"
            )
            return rollout

        # Priority 2: Check metadata for Harbor RL format (token_ids + mask_ids)
        metadata = trial_result.agent_result.metadata
        if not metadata:
            logger.warning(f"No metadata for trial {trial_result.trial_name}")
            return rollout

        # Priority 1: Harbor RL format (token_ids + mask_ids)
        # This is the official format from Harbor RL documentation
        if "token_ids" in metadata and "mask_ids" in metadata:
            rollout["token_ids"] = metadata["token_ids"]
            rollout["mask_ids"] = metadata["mask_ids"]

            # Logprobs may or may not be present
            if "logprobs" in metadata:
                rollout["logprobs"] = metadata["logprobs"]
            else:
                # If no logprobs, use zeros (PPO can still work with just rewards)
                rollout["logprobs"] = [0.0] * len(rollout["token_ids"])

            logger.debug(
                f"Extracted Harbor RL format: {len(rollout['token_ids'])} tokens, "
                f"{sum(rollout['mask_ids'])} completion tokens"
            )

        # Priority 2: Legacy rollout_details format (fallback)
        elif "rollout_details" in metadata:
            rollout_details = metadata["rollout_details"]

            # Terminus-2 provides multi-turn rollout details
            # For single-turn tasks, take the last turn
            if rollout_details:
                last_turn = rollout_details[-1]

                prompt_token_ids = []
                completion_token_ids = []

                # Extract prompt tokens
                if "prompt_token_ids" in last_turn:
                    prompt_tokens = last_turn["prompt_token_ids"]
                    # Handle nested lists (multi-turn)
                    if prompt_tokens and isinstance(prompt_tokens[0], list):
                        prompt_token_ids = [
                            token for turn in prompt_tokens for token in turn
                        ]
                    else:
                        prompt_token_ids = prompt_tokens

                # Extract completion tokens
                if "completion_token_ids" in last_turn:
                    completion_tokens = last_turn["completion_token_ids"]
                    if completion_tokens and isinstance(completion_tokens[0], list):
                        completion_token_ids = [
                            token for turn in completion_tokens for token in turn
                        ]
                    else:
                        completion_token_ids = completion_tokens

                # Combine into token_ids and create mask_ids
                rollout["token_ids"] = prompt_token_ids + completion_token_ids
                rollout["mask_ids"] = (
                    [0] * len(prompt_token_ids)  # Don't train on prompt
                    + [1] * len(completion_token_ids)  # Train on completion
                )

                # Extract logprobs if available
                if "logprobs" in last_turn:
                    logprobs = last_turn["logprobs"]
                    if logprobs and isinstance(logprobs[0], list):
                        rollout["logprobs"] = [lp for turn in logprobs for lp in turn]
                    else:
                        rollout["logprobs"] = logprobs

                    # Pad logprobs to match token_ids
                    if len(rollout["logprobs"]) < len(rollout["token_ids"]):
                        rollout["logprobs"] = [0.0] * len(prompt_token_ids) + rollout[
                            "logprobs"
                        ]
                else:
                    rollout["logprobs"] = [0.0] * len(rollout["token_ids"])

                logger.debug(
                    f"Extracted legacy format: {len(rollout['token_ids'])} tokens, "
                    f"{len(completion_token_ids)} completion tokens"
                )

        else:
            logger.warning(
                f"No token data in agent metadata for trial {trial_result.trial_name}. "
                f"Expected 'token_ids'+'mask_ids' (Harbor RL format) or 'rollout_details' (legacy). "
                f"Available keys: {list(metadata.keys())}"
            )

        return rollout

    def _convert_to_tensors(
        self, rollouts: list[dict], batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Convert Harbor rollouts to AReaL training tensors.

        Args:
            rollouts: List of rollout dicts from Harbor trials
                Each dict should have:
                - token_ids: List[int]
                - mask_ids: List[int] (0=no loss, 1=compute loss)
                - logprobs: List[float]
                - reward: float
            batch_size: Expected batch size

        Returns:
            Dict of tensors for PPO training:
            - input_ids: [batch, seq_len]
            - logprobs: [batch, seq_len]
            - loss_mask: [batch, seq_len]
            - rewards: [batch]
            - attention_mask: [batch, seq_len]
        """
        if not rollouts:
            return self._empty_rollout(batch_size)

        # Prepare lists for batching
        all_input_ids = []
        all_logprobs = []
        all_loss_masks = []
        all_rewards = []

        for rollout in rollouts:
            # Use token_ids (combined prompt+completion)
            token_ids = rollout["token_ids"]
            mask_ids = rollout["mask_ids"]
            logprobs = rollout["logprobs"]

            # Truncate to training-friendly length (avoid FP16 overflow & numerical instability)
            # Keep TAIL for terminal tasks (final commands/results are at the end)
            max_allowed_length = 4096  # Match training max_length for stability
            if len(token_ids) > max_allowed_length:
                logger.warning(
                    f"Truncating sequence from {len(token_ids)} to {max_allowed_length} tokens (keeping TAIL)"
                )
                token_ids = token_ids[-max_allowed_length:]
                mask_ids = mask_ids[-max_allowed_length:]
                logprobs = logprobs[-max_allowed_length:]

            if not token_ids:
                logger.warning("Empty token_ids, using dummy data")
                # Use dummy data
                token_ids = (
                    [self.tokenizer.bos_token_id]
                    if hasattr(self.tokenizer, "bos_token_id")
                    else [0]
                )
                mask_ids = [1]
                logprobs = [0.0]

            # Ensure mask_ids matches token_ids length
            if len(mask_ids) != len(token_ids):
                logger.warning(
                    f"mask_ids length ({len(mask_ids)}) != token_ids length ({len(token_ids)}), "
                    f"adjusting mask_ids"
                )
                # Pad or truncate mask_ids
                if len(mask_ids) < len(token_ids):
                    mask_ids = mask_ids + [1] * (len(token_ids) - len(mask_ids))
                else:
                    mask_ids = mask_ids[: len(token_ids)]

            # Ensure logprobs matches token_ids length
            if len(logprobs) != len(token_ids):
                if len(logprobs) < len(token_ids):
                    logprobs = logprobs + [0.0] * (len(token_ids) - len(logprobs))
                else:
                    logprobs = logprobs[: len(token_ids)]

            all_input_ids.append(torch.tensor(token_ids, dtype=torch.long))
            all_logprobs.append(torch.tensor(logprobs, dtype=torch.float))
            all_loss_masks.append(torch.tensor(mask_ids, dtype=torch.int32))
            all_rewards.append(rollout["reward"])

        # Pad sequences to same length using torch.nn.utils.rnn.pad_sequence
        from torch.nn.utils.rnn import pad_sequence

        # Get proper pad token ID from tokenizer
        if (
            hasattr(self.tokenizer, "pad_token_id")
            and self.tokenizer.pad_token_id is not None
        ):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            # Fallback: use 0 if tokenizer doesn't have pad_token_id
            pad_token_id = 0
            logger.debug(f"Tokenizer has no pad_token_id, using {pad_token_id}")

        # Pad sequences
        input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=pad_token_id
        )
        logprobs = pad_sequence(all_logprobs, batch_first=True, padding_value=0.0)
        loss_mask = pad_sequence(all_loss_masks, batch_first=True, padding_value=0)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = input_ids != pad_token_id

        # Rewards: [batch]
        rewards = torch.tensor(all_rewards, dtype=torch.float)

        logger.info(
            f"Converted rollouts to tensors: "
            f"input_ids={input_ids.shape}, rewards={rewards.tolist()}"
        )

        return {
            "input_ids": input_ids,
            "logprobs": logprobs,
            "loss_mask": loss_mask,
            "rewards": rewards,
            "attention_mask": attention_mask,
        }

    def _empty_rollout(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Create empty rollout for error cases."""
        seq_len = 32  # Minimum sequence length

        return {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
            "logprobs": torch.zeros((batch_size, seq_len), dtype=torch.float),
            "loss_mask": torch.zeros((batch_size, seq_len), dtype=torch.float),
            "rewards": torch.zeros(batch_size, dtype=torch.float),
            "attention_mask": torch.zeros((batch_size, seq_len), dtype=torch.float),
        }
