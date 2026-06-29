#!/bin/bash
set -euo pipefail

# =============================================================================
# SWE-bench agent RL (GRPO) 启动脚本 —— Qwen3-Coder-30B-A3B-Instruct + AWEAgent。
#
# 关键点：/home 是节点本地 ext4，worker 容器跨节点看不到；/storage 才是共享 CPFS。
# 因此本脚本会先把“当前分支”的 AReaL 代码 rsync 到 /storage（AREAL_STORAGE），
# 再从该共享副本启动 controller，并让 worker 的 PYTHONPATH 指向同一份代码。
#
# 配置见 examples/swe/qwen3_30b_a3b_grpo_4n_2t2r.yaml。
# scheduler.type=slurm，AReaL 内部拉起 rollout / actor / proxy worker。
#
# 运行前确认：
#   1. AENV 沙箱服务（AENV_SYSTEM_URL）对所选 instance_id 有可用镜像。
#   2. areal-dev.sif 镜像可用；如需 reservation，用 SLURM_RESERVATION 覆盖。
#   3. AWEAgent 独立仓库可见；worker 会把 aenvironment 安装到 areal-dev.sif 的 /opt/.venv。
# =============================================================================

# ---- 源码位置 ----
# 当前（/home 本地）仓库 —— 即开发分支所在。
AREAL_SRC=${AREAL_SRC:-"/home/admin/chucai.dzq/inclusionAI/AReaL"}
# 共享 /storage 上的目标副本（worker 与 controller 共用，必须与 yaml 里 PYTHONPATH 一致）。
AREAL_STORAGE=${AREAL_STORAGE:-"/storage/openpsi/users/chucai.dzq/codes/swe-rl-areal/AReaL"}

# ---- 集群 ----
N_NODES=${N_NODES:-4}
N_GPUS=${N_GPUS:-8}

# ---- 模型 & 数据 ----
MODEL_PATH=${MODEL_PATH:-"/storage/openpsi/models/Qwen__Qwen3-Coder-30B-A3B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/storage/openpsi/users/miumiu.zwh/dataset_traj/0519_filtered_2k_uniform_pr0p19_0p81/filtered.jsonl"}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}

# ---- AWEAgent / AEnv ----
SWE_AGENT_ROOT=${SWE_AGENT_ROOT:-${AWEAGENT_ROOT:-"/storage/openpsi/users/chucai.dzq/codes/ant-code/AWEAgent"}}
AWEAGENT_ROOT=${AWEAGENT_ROOT:-"${SWE_AGENT_ROOT}"}

# ---- 实验 ----
EXP_NAME=${EXP_NAME:-"swe-rl-qwen3-30b-a3b"}
AGENT_TYPE=${AGENT_TYPE:-"swe"}
AGENT_CONFIG=${AGENT_CONFIG:-""}
if [[ -z "${AENV_SYSTEM_URL:-}" ]]; then
    if [[ "${AGENT_TYPE}" == "cc" ]]; then
        AENV_SYSTEM_URL="http://33.180.184.68"
    else
        AENV_SYSTEM_URL="http://api-service-k8s.aenv.svc.et15-02-aidc.sh.s-aidc.local:8080"
    fi
fi
TAG=${TAG:-""}
TRIAL_NAME=${TRIAL_NAME:-""}
if [ -z "${TRIAL_NAME}" ]; then
    TIMESTAMP=$(date +%m%d)
    TOTAL_GPUS=$(( N_NODES * N_GPUS ))
    TRIAL_NAME="${TIMESTAMP}_${AGENT_TYPE}_qwen3_30b_a3b_g${TOTAL_GPUS}"
    [ -n "${TAG}" ] && TRIAL_NAME="${TRIAL_NAME}_${TAG}"
fi
echo "==> agent_type: ${AGENT_TYPE}, trial_name: ${TRIAL_NAME}"

SWE_RL_ADMIN_API_KEY=${SWE_RL_ADMIN_API_KEY:-"swerl-${USER:-admin}-${TRIAL_NAME}"}
export SWE_RL_ADMIN_API_KEY

# ---- 配置 ----
CONFIG_PATH=${CONFIG_PATH:-"examples/swe/qwen3_30b_a3b_grpo_4n_2t2r.yaml"}
TRAIN_SCRIPT="examples/swe/train_swe_rl.py"

# ---- 存储 ----
FILEROOT=${FILEROOT:-"/storage/openpsi/users/chucai.dzq/areal-data/experiments"}
NFS_ROOT=${NFS_ROOT:-"/storage/openpsi/users/chucai.dzq/areal-data/name_resolve/${EXP_NAME}"}

# ---- WandB（可选）----
WANDB_MODE=${WANDB_MODE:-"online"}
WANDB_API_KEY=${WANDB_API_KEY:-""}
WANDB_BASE_URL=${WANDB_BASE_URL:-"http://8.150.1.98:8080"}

# =============================================================================
# 1) 把当前分支代码同步到共享 /storage（worker 跨节点必须能看到）
# =============================================================================
echo "==> rsync ${AREAL_SRC} -> ${AREAL_STORAGE}"
mkdir -p "${AREAL_STORAGE}"
# These paths existed in older layouts. Remove them explicitly because rsync
# excludes __pycache__, and excluded files can keep deleted directories alive.
rm -rf \
    "${AREAL_STORAGE}/examples/swe/rl" \
    "${AREAL_STORAGE}/examples/swe/sweagent"
rsync -a --delete \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.venv/' \
    --exclude 'outputs/' \
    "${AREAL_SRC}/" "${AREAL_STORAGE}/"

# 校验：yaml 里 worker PYTHONPATH 写死的 AReaL 路径必须等于 AREAL_STORAGE。
if ! grep -q "${AREAL_STORAGE}" "${AREAL_STORAGE}/${CONFIG_PATH}"; then
    echo "WARN: ${CONFIG_PATH} 里的 worker PYTHONPATH 未包含 ${AREAL_STORAGE}；" >&2
    echo "      worker 可能 import 到旧代码，请同步修改 yaml。" >&2
fi

# 预建缓存 / 产物目录。
mkdir -p \
    /storage/openpsi/users/chucai.dzq/areal-data/cache/{home,flashinfer,hf,xdg} \
    "${FILEROOT}" "${NFS_ROOT}"

# =============================================================================
# 2) 从共享副本启动 controller
# =============================================================================
cd "${AREAL_STORAGE}"
export PYTHONPATH=${AREAL_STORAGE}:${AWEAGENT_ROOT}:${PYTHONPATH:-}
export SWE_AGENT_ROOT AWEAGENT_ROOT

AGENT_ARGS=(
      "econfig.agent_type=${AGENT_TYPE}"
      "econfig.agent_root=${AWEAGENT_ROOT}"
      "econfig.swe_agent_root=${SWE_AGENT_ROOT}"
      "++actor.scheduling_spec.0.env_vars.SWE_AGENT_ROOT=${SWE_AGENT_ROOT}"
      "++actor.scheduling_spec.0.env_vars.AWEAGENT_ROOT=${AWEAGENT_ROOT}"
)
if [ -n "${AGENT_CONFIG}" ]; then
    AGENT_ARGS+=("econfig.agent_config=${AGENT_CONFIG}")
fi

WANDB_API_KEY=${WANDB_API_KEY} \
WANDB_BASE_URL=${WANDB_BASE_URL} \
python ${TRAIN_SCRIPT} \
      --config ${CONFIG_PATH} \
      experiment_name=${EXP_NAME} \
      trial_name=${TRIAL_NAME} \
      tokenizer_path=${TOKENIZER_PATH} \
      actor.path=${MODEL_PATH} \
      cluster.n_nodes=${N_NODES} \
      cluster.n_gpus_per_node=${N_GPUS} \
      cluster.fileroot=${FILEROOT} \
      cluster.name_resolve.nfs_record_root=${NFS_ROOT} \
      train_dataset.path=${TRAIN_DATA} \
      valid_dataset.path=${TRAIN_DATA} \
      rollout.agent.admin_api_key=${SWE_RL_ADMIN_API_KEY} \
      stats_logger.wandb.mode=${WANDB_MODE} \
      "++actor.scheduling_spec.0.env_vars.AENV_SYSTEM_URL=${AENV_SYSTEM_URL}" \
      "${AGENT_ARGS[@]}" \
      "$@"
