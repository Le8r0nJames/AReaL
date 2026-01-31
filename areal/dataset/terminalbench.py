"""TerminalBench dataset loader for AReaL.

TerminalBench is a benchmark for terminal-based agent tasks.
For RL training with Harbor, we provide task IDs that Harbor will use.

Supports two modes:
1. Harbor Registry (recommended): Use "terminalbench:terminal-bench@2.0"
   - Automatically fetches task list from Harbor registry
   - No local files needed
2. Local path (legacy): Use "terminalbench:/path/to/terminal-bench-core/0.1.1"
   - Scans local directory for task folders
"""

from pathlib import Path


def get_terminalbench_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    **kwargs,
):
    """Get TerminalBench dataset for RL training.

    Args:
        path: Path in one of these formats:
            - "terminalbench:terminal-bench@2.0" (Harbor registry, recommended)
            - "terminalbench:/path/to/terminal-bench-core/0.1.1" (local path, legacy)
        split: Dataset split ("train" or "test")
        tokenizer: Tokenizer (unused for Harbor workflow but required by interface)
        max_length: Max length (unused for Harbor workflow)

    Returns:
        A list of dict samples with {"task_id": task_name}
    """
    # Extract the actual path/spec after "terminalbench:"
    if ":" in path:
        _, path_spec = path.split(":", 1)
    else:
        path_spec = path

    # Check if using Harbor registry format (contains @)
    if "@" in path_spec:
        # Harbor registry mode: terminal-bench@2.0
        dataset_name, version = path_spec.split("@", 1)

        # Import Harbor registry client
        try:
            from harbor.registry.client.harbor.harbor import HarborRegistryClient
        except ImportError:
            raise ImportError(
                "Harbor registry client not found. "
                "Please install Harbor: pip install harbor-agent"
            )

        # Fetch task list from Harbor registry
        client = HarborRegistryClient()
        dataset_spec = client._get_dataset_spec(dataset_name, version)

        all_tasks = [
            {
                "task_id": task.name,
            }
            for task in dataset_spec.tasks
        ]

        print(
            f"Loaded {len(all_tasks)} tasks from Harbor registry: {dataset_name}@{version}"
        )
    else:
        # Local path mode (legacy)
        dataset_root = Path(path_spec)

        if not dataset_root.exists():
            raise FileNotFoundError(
                f"TerminalBench dataset not found at {dataset_root}. "
                f"Please ensure the dataset is downloaded to this location, "
                f"or use Harbor registry format: terminalbench:terminal-bench@2.0"
            )

        # Get all task directories (subdirectories containing task.yaml)
        all_tasks = []
        for task_dir in sorted(dataset_root.iterdir()):
            if task_dir.is_dir() and (task_dir / "task.yaml").exists():
                task_name = task_dir.name
                all_tasks.append(
                    {
                        "task_id": task_name,
                    }
                )

        if not all_tasks:
            raise ValueError(
                f"No valid tasks found in {dataset_root}. "
                f"Each task should be a directory containing task.yaml"
            )

        print(f"Loaded {len(all_tasks)} tasks from local path: {dataset_root}")

    # Simple split: use first 80% for train, last 20% for test
    n_tasks = len(all_tasks)
    n_train = int(n_tasks * 0.8)

    if split == "train":
        tasks = all_tasks[:n_train]
    elif split == "test":
        tasks = all_tasks[n_train:]
    else:
        # If split not specified or is "all", return all tasks
        tasks = all_tasks

    # Convert to a simple list-like dataset
    # Harbor workflow will iterate over this and use task_id
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def __iter__(self):
            return iter(self.data)

    dataset = SimpleDataset(tasks)

    return dataset
