#!/bin/bash
set -euo pipefail

# Direct controller launch for Bailing flash25 + CC SWE-RL.
# For login-node submission, use submit_bailing_flash25_cc.sh; this script is
# what the controller ultimately runs after the repo is synced to /storage.

AREAL_SRC=${AREAL_SRC:-"/home/admin/chucai.dzq/inclusionAI/AReaL"}
AREAL_STORAGE=${AREAL_STORAGE:-"/storage/openpsi/users/chucai.dzq/codes/swe-rl-areal/AReaL"}
SYNC_MODE=${SYNC_MODE:-full}

CONFIG_PATH=${CONFIG_PATH:-"examples/swe/rl/flash/bailing_flash25_cc_grpo.yaml"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"examples/swe/train_swe_rl.py"}
IMAGE=${IMAGE:-"/storage/openpsi/images/areal-dev-sglang-20260401.sif"}

N_NODES=${N_NODES:-16}
N_GPUS=${N_GPUS:-8}
FILEROOT=${FILEROOT:-"/storage/openpsi/users/chucai.dzq/areal-data/experiments"}

MODEL_PATH=${MODEL_PATH:-"/storage/openpsi/users/wanghaitao.wht/project/swe-rl-alian/ring_2_5_flash_add_new100wdata_basedsft_1e4_128k_0204"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${MODEL_PATH}"}
TRAIN_DATA=${TRAIN_DATA:-"/storage/openpsi/users/fenghui/projects/AWEAgent_DEV/AWEAgent/src/data/swe_bench_verified_rl.jsonl"}

AWEAGENT_ROOT=${AWEAGENT_ROOT:-"/storage/openpsi/users/chucai.dzq/codes/ant-code/AWEAgent"}
SWE_AGENT_ROOT=${SWE_AGENT_ROOT:-"${AWEAGENT_ROOT}"}
FLASH_LINEAR_ATTENTION=${FLASH_LINEAR_ATTENTION:-"/storage/openpsi/users/public/projects/flash-linear-attention"}
AENV_SYSTEM_URL=${AENV_SYSTEM_URL:-"http://33.180.184.68"}

EXP_NAME=${EXP_NAME:-"chucai-bailing-cc-rl"}
TAG=${TAG:-""}
TRIAL_NAME=${TRIAL_NAME:-""}
if [[ -z "${TRIAL_NAME}" ]]; then
  TRIAL_NAME="$(date +%m%d)_cc_flash25_verified_1000"
  [[ -n "${TAG}" ]] && TRIAL_NAME="${TRIAL_NAME}_${TAG}"
fi

TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1000}
TOTAL_TRAIN_EPOCHS=${TOTAL_TRAIN_EPOCHS:-1000}
N_SAMPLES=${N_SAMPLES:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_HEAD_OFFPOLICYNESS=${MAX_HEAD_OFFPOLICYNESS:-2}
MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS:-720}

AGENT_TYPE=${AGENT_TYPE:-cc}
AGENT_CONFIG=${AGENT_CONFIG:-""}
CC_AGENT_CONFIG=${CC_AGENT_CONFIG:-"train_cc_time3600_nooverfit"}
LLM_MODEL=${LLM_MODEL:-"bailing-flash25"}
OPENAI_MODEL=${OPENAI_MODEL:-"${LLM_MODEL}"}

WANDB_MODE=${WANDB_MODE:-online}
WANDB_BASE_URL=${WANDB_BASE_URL:-"http://8.150.1.98:8080"}
WANDB_API_KEY=${WANDB_API_KEY:-""}
WANDB_DIR=${WANDB_DIR:-"/storage/openpsi/users/chucai.dzq/areal-data/wandb"}

NFS_ROOT=${NFS_ROOT:-"/storage/openpsi/users/chucai.dzq/areal-data/name_resolve/${EXP_NAME}"}
SWE_RL_ADMIN_API_KEY=${SWE_RL_ADMIN_API_KEY:-"swerl-${USER:-admin}-${TRIAL_NAME}"}
export SWE_RL_ADMIN_API_KEY SWE_AGENT_ROOT AWEAGENT_ROOT WANDB_BASE_URL WANDB_DIR

WORKER_PYTHONPATH="${AREAL_STORAGE}:${AWEAGENT_ROOT}:${FLASH_LINEAR_ATTENTION}${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> experiment: ${EXP_NAME}/${TRIAL_NAME}"
echo "==> model: ${MODEL_PATH}"
echo "==> data: ${TRAIN_DATA}"
echo "==> agent: ${AGENT_TYPE}, cc_config=${CC_AGENT_CONFIG}"
echo "==> image: ${IMAGE}"

if [[ "${WANDB_MODE}" != "disabled" && -z "${WANDB_API_KEY}" ]]; then
  echo "WARN: WANDB_MODE=${WANDB_MODE} but WANDB_API_KEY is not set; wandb online login may fail" >&2
fi

echo "==> rsync ${AREAL_SRC} -> ${AREAL_STORAGE}"
case "${SYNC_MODE}" in
  full)
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

grep -q "${AREAL_STORAGE}" "${AREAL_STORAGE}/${CONFIG_PATH}" \
  || echo "WARN: ${CONFIG_PATH} worker PYTHONPATH does not mention ${AREAL_STORAGE}" >&2

mkdir -p \
  /storage/openpsi/users/chucai.dzq/areal-data/cache/{home,flashinfer,hf,xdg} \
  "${FILEROOT}" "${NFS_ROOT}" "${WANDB_DIR}"

cd "${AREAL_STORAGE}"
export PYTHONPATH="${WORKER_PYTHONPATH}"

AGENT_ARGS=(
  "econfig.agent_type=${AGENT_TYPE}"
  "econfig.agent_config=${AGENT_CONFIG}"
  "econfig.cc_agent_config=${CC_AGENT_CONFIG}"
  "econfig.agent_root=${AWEAGENT_ROOT}"
  "econfig.swe_agent_root=${SWE_AGENT_ROOT}"
  "econfig.llm_model=${LLM_MODEL}"
)

WANDB_API_KEY=${WANDB_API_KEY} \
WANDB_BASE_URL=${WANDB_BASE_URL} \
WANDB_DIR=${WANDB_DIR} \
python3 "${TRAIN_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  experiment_name="${EXP_NAME}" \
  trial_name="${TRIAL_NAME}" \
  total_train_steps="${TOTAL_TRAIN_STEPS}" \
  total_train_epochs="${TOTAL_TRAIN_EPOCHS}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  actor.path="${MODEL_PATH}" \
  cluster.n_nodes="${N_NODES}" \
  cluster.n_gpus_per_node="${N_GPUS}" \
  cluster.fileroot="${FILEROOT}" \
  cluster.name_resolve.nfs_record_root="${NFS_ROOT}" \
  gconfig.n_samples="${N_SAMPLES}" \
  train_dataset.path="${TRAIN_DATA}" \
  valid_dataset.path="${TRAIN_DATA}" \
  train_dataset.batch_size="${BATCH_SIZE}" \
  valid_dataset.batch_size="${BATCH_SIZE}" \
  rollout.consumer_batch_size="${BATCH_SIZE}" \
  rollout.max_head_offpolicyness="${MAX_HEAD_OFFPOLICYNESS}" \
  rollout.max_concurrent_rollouts="${MAX_CONCURRENT_ROLLOUTS}" \
  rollout.agent.admin_api_key="${SWE_RL_ADMIN_API_KEY}" \
  actor.scheduling_spec.0.image="${IMAGE}" \
  actor.scheduling_spec.0.env_vars.AREAL_STORAGE="${AREAL_STORAGE}" \
  actor.scheduling_spec.0.env_vars.FLASH_LINEAR_ATTENTION="${FLASH_LINEAR_ATTENTION}" \
  actor.scheduling_spec.0.env_vars.PYTHONPATH="${WORKER_PYTHONPATH}" \
  actor.scheduling_spec.0.env_vars.SWE_AGENT_ROOT="${SWE_AGENT_ROOT}" \
  actor.scheduling_spec.0.env_vars.AWEAGENT_ROOT="${AWEAGENT_ROOT}" \
  actor.scheduling_spec.0.env_vars.AENV_SYSTEM_URL="${AENV_SYSTEM_URL}" \
  actor.scheduling_spec.0.env_vars.OPENAI_MODEL="${OPENAI_MODEL}" \
  actor.scheduling_spec.0.env_vars.LLM_MODEL="${LLM_MODEL}" \
  actor.scheduling_spec.0.env_vars.WANDB_DIR="${WANDB_DIR}" \
  stats_logger.wandb.mode="${WANDB_MODE}" \
  stats_logger.wandb.wandb_base_url="${WANDB_BASE_URL}" \
  "${AGENT_ARGS[@]}" \
  "$@"
