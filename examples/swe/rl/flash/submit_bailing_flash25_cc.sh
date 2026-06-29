#!/bin/bash
set -euo pipefail

# Submit Bailing flash25 + CC SWE-RL from a login node.
#
# The controller runs in a general AReaL image and starts actor/rollout workers
# through AReaL's Slurm scheduler. Workers use the SGLang image required by
# Bailing flash25.

AREAL_SRC=${AREAL_SRC:-"/home/admin/zjw531248/inclusionAI/AReaL/swe-rl-areal"}
AREAL_STORAGE=${AREAL_STORAGE:-"/storage/openpsi/users/chucai.dzq/codes/swe-rl-areal/AReaL"}
CONFIG=${CONFIG:-"examples/swe/rl/flash/bailing_flash25_cc_grpo.yaml"}
CONTROLLER_IMAGE=${CONTROLLER_IMAGE:-${IMAGE:-"/storage/openpsi/images/areal-dev.sif"}}
WORKER_IMAGE=${WORKER_IMAGE:-"/storage/openpsi/images/areal-dev-sglang-20260401.sif"}

MODEL_PATH=${MODEL_PATH:-"/storage/openpsi/users/wanghaitao.wht/project/swe-rl-alian/ring_2_5_flash_add_new100wdata_basedsft_1e4_128k_0204"}
TRAIN_DATA=${TRAIN_DATA:-"/storage/openpsi/users/wanghaitao.wht/project/swe-rl-alian/swe_bench_verified_rl.jsonl"}
AWEAGENT_ROOT=${AWEAGENT_ROOT:-"/storage/openpsi/users/public/projects/AWEAgent"}
SWE_AGENT_ROOT=${SWE_AGENT_ROOT:-"${AWEAGENT_ROOT}"}
FLASH_LINEAR_ATTENTION=${FLASH_LINEAR_ATTENTION:-"/storage/openpsi/users/public/projects/flash-linear-attention"}
AENV_SYSTEM_URL=${AENV_SYSTEM_URL:-"http://33.180.184.68"}

EXP_NAME=${EXP_NAME:-"chucai-bailing-cc-rl"}
TRIAL_NAME=${TRIAL_NAME:-"$(date +%m%d)_cc_flash25_verified_1000"}
N_NODES=${N_NODES:-16}
N_GPUS=${N_GPUS:-8}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1145}
TOTAL_TRAIN_EPOCHS=${TOTAL_TRAIN_EPOCHS:-1000}
N_SAMPLES=${N_SAMPLES:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_HEAD_OFFPOLICYNESS=${MAX_HEAD_OFFPOLICYNESS:-1}
MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS:-720}

AGENT_TYPE=${AGENT_TYPE:-cc}
AGENT_CONFIG=${AGENT_CONFIG:-""}
CC_AGENT_CONFIG=${CC_AGENT_CONFIG:-"train_cc_time3600"}
LLM_MODEL=${LLM_MODEL:-"bailing-flash25"}
OPENAI_MODEL=${OPENAI_MODEL:-"${LLM_MODEL}"}

RESERVATION=${RESERVATION:-swe-rl}
EXCLUDE_NODES=${EXCLUDE_NODES:-""}
FILEROOT=${FILEROOT:-"/storage/openpsi/users/chucai.dzq/areal-data/experiments"}
NFS_ROOT=${NFS_ROOT:-"/storage/openpsi/users/chucai.dzq/areal-data/name_resolve/${EXP_NAME}"}
LAUNCH_DIR=${LAUNCH_DIR:-"/storage/openpsi/users/chucai.dzq/areal-data/launch/${EXP_NAME}_${TRIAL_NAME}"}

# full: sync the whole repo, flash: sync only this directory, none: do not sync.
# SKIP_RSYNC=1 is kept as a compatibility alias for SYNC_MODE=none.
SYNC_MODE=${SYNC_MODE:-full}
if [[ "${SKIP_RSYNC:-0}" == "1" ]]; then
  SYNC_MODE=none
fi

WANDB_MODE=${WANDB_MODE:-online}
WANDB_BASE_URL=${WANDB_BASE_URL:-"http://8.150.1.98:8080"}
WANDB_DIR=${WANDB_DIR:-"/storage/openpsi/users/chucai.dzq/areal-data/wandb"}
export WANDB_BASE_URL WANDB_DIR
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

REMOTE_PROXY_SERVICE_URL=${REMOTE_PROXY_SERVICE_URL:-""}
REMOTE_PROXY_API_KEY=${REMOTE_PROXY_API_KEY:-""}
export REMOTE_PROXY_SERVICE_URL REMOTE_PROXY_API_KEY

if [[ -z "${REMOTE_PROXY_SERVICE_URL}" || -z "${REMOTE_PROXY_API_KEY}" ]]; then
echo "WARN: REMOTE_PROXY_SERVICE_URL or REMOTE_PROXY_API_KEY is not set; CC remote proxy rollout may fail" >&2
fi

SWE_RL_ADMIN_API_KEY=${SWE_RL_ADMIN_API_KEY:-"swerl-${USER:-admin}-${TRIAL_NAME}"}
export SWE_RL_ADMIN_API_KEY

if [[ "${WANDB_MODE}" != "disabled" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_MODE=${WANDB_MODE} but WANDB_API_KEY is not set; wandb online login may fail" >&2
fi

WORKER_PYTHONPATH="${AREAL_STORAGE}:${AWEAGENT_ROOT}:${FLASH_LINEAR_ATTENTION}"
CACHE_HOME="/storage/openpsi/users/chucai.dzq/areal-data/cache/home"

mkdir -p "${CACHE_HOME}" "${LAUNCH_DIR}" \
  /storage/openpsi/users/chucai.dzq/areal-data/cache/{flashinfer,hf,xdg} \
  "${FILEROOT}" "${NFS_ROOT}" "${WANDB_DIR}"

sync_sources() {
  case "${SYNC_MODE}" in
    full)
      echo "==> rsync full repo: ${AREAL_SRC} -> ${AREAL_STORAGE}"
      mkdir -p "${AREAL_STORAGE}"
      rm -rf \
        "${AREAL_STORAGE}/examples/swe/rl" \
        "${AREAL_STORAGE}/examples/swe/sweagent"
      rsync -a --delete \
        --exclude '.git/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.venv/' \
        --exclude 'outputs/' \
        --exclude '*.sif' \
        "${AREAL_SRC}/" "${AREAL_STORAGE}/"
      ;;
    flash)
      echo "==> rsync flash dir only: ${AREAL_SRC}/examples/swe/rl/flash -> ${AREAL_STORAGE}/examples/swe/rl/flash"
      mkdir -p "${AREAL_STORAGE}/examples/swe/rl/flash"
      rsync -a --delete \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "${AREAL_SRC}/examples/swe/rl/flash/" \
        "${AREAL_STORAGE}/examples/swe/rl/flash/"
      ;;
    none)
      echo "==> SYNC_MODE=none; using existing ${AREAL_STORAGE}"
      ;;
    *)
      echo "ERROR: SYNC_MODE must be full, flash, or none; got '${SYNC_MODE}'" >&2
      exit 2
      ;;
  esac
}

sync_sources

if [[ ! -f "${AREAL_STORAGE}/${CONFIG}" ]]; then
  echo "ERROR: config not found after sync: ${AREAL_STORAGE}/${CONFIG}" >&2
  exit 2
fi

grep -q "${AREAL_STORAGE}" "${AREAL_STORAGE}/${CONFIG}" \
  || echo "WARN: ${CONFIG} worker PYTHONPATH does not mention ${AREAL_STORAGE}" >&2

TRAIN_ARGS=(
  "--config" "${CONFIG}"
  "experiment_name=${EXP_NAME}"
  "trial_name=${TRIAL_NAME}"
  "total_train_steps=${TOTAL_TRAIN_STEPS}"
  "total_train_epochs=${TOTAL_TRAIN_EPOCHS}"
  "tokenizer_path=${MODEL_PATH}"
  "actor.path=${MODEL_PATH}"
  "cluster.n_nodes=${N_NODES}"
  "cluster.n_gpus_per_node=${N_GPUS}"
  "cluster.fileroot=${FILEROOT}"
  "cluster.name_resolve.nfs_record_root=${NFS_ROOT}"
  "gconfig.n_samples=${N_SAMPLES}"
  "train_dataset.path=${TRAIN_DATA}"
  "valid_dataset.path=${TRAIN_DATA}"
  "train_dataset.batch_size=${BATCH_SIZE}"
  "valid_dataset.batch_size=${BATCH_SIZE}"
  "rollout.consumer_batch_size=${BATCH_SIZE}"
  "rollout.max_head_offpolicyness=${MAX_HEAD_OFFPOLICYNESS}"
  "rollout.max_concurrent_rollouts=${MAX_CONCURRENT_ROLLOUTS}"
  "actor.prox_logp_method=recompute"
  "rollout.agent.admin_api_key=${SWE_RL_ADMIN_API_KEY}"
  "econfig.agent_type=${AGENT_TYPE}"
  "econfig.agent_config=${AGENT_CONFIG}"
  "econfig.cc_agent_config=${CC_AGENT_CONFIG}"
  "econfig.agent_root=${AWEAGENT_ROOT}"
  "econfig.swe_agent_root=${SWE_AGENT_ROOT}"
  "econfig.llm_model=${LLM_MODEL}"
  "actor.scheduling_spec.0.image=${WORKER_IMAGE}"
  "actor.scheduling_spec.0.env_vars.AREAL_STORAGE=${AREAL_STORAGE}"
  "actor.scheduling_spec.0.env_vars.FLASH_LINEAR_ATTENTION=${FLASH_LINEAR_ATTENTION}"
  "actor.scheduling_spec.0.env_vars.PYTHONPATH=${WORKER_PYTHONPATH}"
  "actor.scheduling_spec.0.env_vars.SWE_AGENT_ROOT=${SWE_AGENT_ROOT}"
  "actor.scheduling_spec.0.env_vars.AWEAGENT_ROOT=${AWEAGENT_ROOT}"
  "actor.scheduling_spec.0.env_vars.AENV_SYSTEM_URL=${AENV_SYSTEM_URL}"
  "actor.scheduling_spec.0.env_vars.OPENAI_MODEL=${OPENAI_MODEL}"
  "actor.scheduling_spec.0.env_vars.LLM_MODEL=${LLM_MODEL}"
  "actor.scheduling_spec.0.env_vars.WANDB_DIR=${WANDB_DIR}"
  "stats_logger.wandb.mode=${WANDB_MODE}"
  "stats_logger.wandb.wandb_base_url=${WANDB_BASE_URL}"
)

if [[ -n "${EXCLUDE_NODES}" ]]; then
  TRAIN_ARGS+=("actor.scheduling_spec.0.exclude='${EXCLUDE_NODES}'")
fi
if (($#)); then
  TRAIN_ARGS+=("$@")
  printf '==> extra trainer args:'
  printf ' %q' "$@"
  printf '\n'
fi

CTRL_RUN_SH="${LAUNCH_DIR}/controller_run.sh"
{
  echo "#!/bin/bash"
  echo "set -euo pipefail"
  printf 'cd %q\n' "${AREAL_STORAGE}"
  echo "TRAIN_ARGS=("
  for arg in "${TRAIN_ARGS[@]}"; do
    printf '  %q\n' "${arg}"
  done
  echo ")"
  echo 'python3 examples/swe/train_swe_rl.py "${TRAIN_ARGS[@]}"'
} > "${CTRL_RUN_SH}"
chmod +x "${CTRL_RUN_SH}"

CTRL_SH="${LAUNCH_DIR}/controller.sbatch"
cat > "${CTRL_SH}" <<EOF
#!/bin/bash
#SBATCH --job-name=chucai-bailing-cc
#SBATCH --reservation=${RESERVATION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --no-requeue
#SBATCH --output=${LAUNCH_DIR}/controller.log
#SBATCH --open-mode=append

if [[ -n "\${WANDB_API_KEY:-}" ]]; then
  export APPTAINERENV_WANDB_API_KEY="\${WANDB_API_KEY}"
fi
if [[ -n "\${WANDB_BASE_URL:-}" ]]; then
  export APPTAINERENV_WANDB_BASE_URL="\${WANDB_BASE_URL}"
fi
export APPTAINERENV_WANDB_DIR="\${WANDB_DIR:-${WANDB_DIR}}"

set -x
for v in \$(compgen -v | grep '^SLURM_'); do unset "\$v"; done

exec singularity exec --no-home --writable-tmpfs \\
  --bind /storage:/storage \\
  --bind /etc/slurm:/etc/slurm \\
  --bind /etc/passwd:/etc/passwd \\
  --bind /etc/group:/etc/group \\
  --bind /run/munge:/run/munge \\
  --bind /usr/bin/sbatch:/usr/bin/sbatch \\
  --bind /usr/bin/squeue:/usr/bin/squeue \\
  --bind /usr/bin/scontrol:/usr/bin/scontrol \\
  --bind /usr/bin/scancel:/usr/bin/scancel \\
  --bind /usr/bin/srun:/usr/bin/srun \\
  --bind /usr/lib64/slurm:/usr/lib64/slurm \\
  --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2 \\
  --bind /usr/lib64/libslurm.so.40:/usr/lib64/libslurm.so.40 \\
  --env PYTHONPATH=${WORKER_PYTHONPATH} \\
  --env AWEAGENT_ROOT=${AWEAGENT_ROOT} \\
  --env SWE_AGENT_ROOT=${SWE_AGENT_ROOT} \\
  --env FLASH_LINEAR_ATTENTION=${FLASH_LINEAR_ATTENTION} \\
  --env SWE_RL_ADMIN_API_KEY=\${SWE_RL_ADMIN_API_KEY} \\
  --env WANDB_MODE=${WANDB_MODE} \\
  --env WANDB_API_KEY=\${WANDB_API_KEY:-} \\
  --env WANDB_BASE_URL=${WANDB_BASE_URL} \\
  --env WANDB_DIR=\${WANDB_DIR:-${WANDB_DIR}} \\
  --env HOME=${CACHE_HOME} \\
  --env SLURM_CONF=/etc/slurm/slurm.conf \\
  --env AENV_SYSTEM_URL=${AENV_SYSTEM_URL} \\
  ${CONTROLLER_IMAGE} \\
  bash ${CTRL_RUN_SH}
EOF

echo "==> controller sbatch written: ${CTRL_SH}"
echo "==> controller run script written: ${CTRL_RUN_SH}"
echo "==> log will be at: ${LAUNCH_DIR}/controller.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "==> DRY_RUN=1; not submitting"
  exit 0
fi

JID=$(sbatch --parsable "${CTRL_SH}")
echo "==> submitted controller job: ${JID}"
echo "    tail -f ${LAUNCH_DIR}/controller.log"
echo "    workers: ${EXP_NAME}_${TRIAL_NAME}:{actor,rollout,...}"
