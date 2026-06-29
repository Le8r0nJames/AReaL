# Bailing flash25 SWE-RL

This directory keeps the Bailing flash25 CC SWE-RL config, launchers, and the
SGLang radix-cache patch together.

Files:

- `bailing_flash25_cc_grpo.yaml`: 16-node Bailing flash25 + CC GRPO config.
- `run_bailing_flash25_cc.sh`: direct controller launch after syncing this repo
  to `/storage`.
- `submit_bailing_flash25_cc.sh`: login-node Slurm submit wrapper.
- `apply_sgl_radix_cache_patch.py`: patches SGLang so
  `BailingMoeV2_5ForCausalLM` can use radix/prefix cache.

Default run shape:

- model: `/storage/openpsi/users/wanghaitao.wht/project/swe-rl-alian/ring_2_5_flash_add_new100wdata_basedsft_1e4_128k_0204`
- data: `/storage/openpsi/users/fenghui/projects/AWEAgent_DEV/AWEAgent/src/data/swe_bench_verified_rl.jsonl`
- AWEAgent: `/storage/openpsi/users/chucai.dzq/codes/ant-code/AWEAgent`
- `n_samples=8`, `batch_size=8`, `total_train_steps=1000`
- `should_accept_fn=areal.examples.filter_function.filter_function`
- `max_head_offpolicyness=2`
- controller image: `/storage/openpsi/images/areal-dev.sif`
- worker image: `/storage/openpsi/images/areal-dev-sglang-20260401.sif`

Submit:

```bash
export WANDB_API_KEY=<wandb key>
bash examples/swe/rl/flash/submit_bailing_flash25_cc.sh
```

Useful overrides:

```bash
SYNC_MODE=flash \
TRIAL_NAME=0629_cc_flash25_verified_1000 \
EXCLUDE_NODES=slurmd-74,slurmd-97 \
WANDB_API_KEY=<wandb key> \
bash examples/swe/rl/flash/submit_bailing_flash25_cc.sh
```

`SYNC_MODE=full` syncs the whole repo to `/storage` and is the default.
`SYNC_MODE=flash` syncs only this directory, which is useful when the storage
copy already has the current AReaL code and only launch/config files changed.
`SYNC_MODE=none` skips syncing. `DRY_RUN=1` writes the Slurm scripts without
submitting.

Reward checks:

- Watch `rollout/reward` and `ppo_actor/task_reward/avg` in W&B.
- If reward is all zero, first check AWEAgent imports, AEnv reachability, and
  whether `apply_sgl_radix_cache_patch.py` ran in worker startup logs.
- W&B base URL defaults to `http://8.150.1.98:8080`; the API key is read from
  the environment and is intentionally not stored in this repo.
