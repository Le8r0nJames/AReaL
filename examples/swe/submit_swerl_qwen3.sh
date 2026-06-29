#!/bin/bash
set -euo pipefail

# =============================================================================
# SWE-bench agent RL (Qwen3-Coder-30B-A3B-Instruct + AWEAgent) 提交脚本 —— 登录节点运行。
#
# 背景 / 为什么这么做（见踩坑记录）：
#  - /home 是节点本地盘，计算节点跨节点看不到；/storage 是共享 CPFS。
#    => 先把当前分支 rsync 到 /storage 暂存，worker PYTHONPATH 指向它。
#  - 登录节点没有可用的 areal python 环境，且登录节点 apptainer 起不来
#    (/dev/loop0 无权限)。areal-dev.sif 里有全部依赖。
#    => controller 作为 sbatch 作业，在 swe-rl 预留的计算节点 sif 容器里跑。
#  - 容器里没有 slurm client：把 host 的 sbatch/squeue/scontrol/scancel +
#    /usr/lib64/slurm + libmunge/libslurm + /etc/slurm + /etc/passwd|group +
#    /run/munge 绑进去，controller 即可在容器内 sbatch 拉起 worker（已实测 OK）。
#  - controller 跑在 sbatch 作业里会继承 SLURM_* 环境，污染它给 worker 的提交。
#    => python 前 unset 所有 SLURM_*（worker 的 --reservation 等由 scheduler 从
#       config 显式生成，不依赖继承）。
#  - AEnv / exclusive / NCCL 等值对齐正在跑通的 swe-qwen35 实验。
#
# 资源占位协议（仅在“出问题需重启实验”时用，首次提交不需要）：
#  1) 先 sbatch --reservation=swe-rl --gres=gpu:8 -N4 -t UNLIMITED \
#       --job-name=chucai-colocation --wrap="sleep infinity" 占住资源；
#  2) 再 scancel 旧实验、查问题；
#  3) 提交新实验后，再 scancel 占位作业。
# =============================================================================

# ---- 路径 ----
AREAL_SRC=${AREAL_SRC:-/home/admin/chucai.dzq/inclusionAI/AReaL}
AREAL_STORAGE=${AREAL_STORAGE:-/storage/openpsi/users/chucai.dzq/codes/swe-rl-areal/AReaL}
SWE_AGENT_ROOT=${SWE_AGENT_ROOT:-${AWEAGENT_ROOT:-/storage/openpsi/users/chucai.dzq/codes/ant-code/AWEAgent}}
AWEAGENT_ROOT=${AWEAGENT_ROOT:-${SWE_AGENT_ROOT}}
# 注意：不要把 miumiu 的 areal-deps/aenv 目录加进 PYTHONPATH —— 它捆绑了 starlette 1.2.1，
# 会遮蔽 venv 的 starlette 0.50.0，导致 sglang http_server 启动时 FastAPI 报
# "Router.__init__() got an unexpected keyword argument 'on_startup'"。
# aenv 改为在 worker 的 additional_bash_cmds 里 `uv pip install aenvironment` 干净装进 venv。
IMAGE=${IMAGE:-/storage/openpsi/images/areal-dev.sif}
CONFIG=${CONFIG:-examples/swe/qwen3_30b_a3b_grpo_4n_2t2r.yaml}

# ---- 实验 ----
EXP_NAME=${EXP_NAME:-chucai-swe-demo-qwen}
TRIAL_NAME=${TRIAL_NAME:-demo1}
AGENT_TYPE=${AGENT_TYPE:-swe}
AGENT_CONFIG=${AGENT_CONFIG:-}
# 默认跑 50 step 做长跑验证；快速冒烟可用 TOTAL_TRAIN_STEPS=1 覆盖。
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-50}
WANDB_MODE=${WANDB_MODE:-online}
RESERVATION=${RESERVATION:-swe-rl}
WANDB_BASE_URL=${WANDB_BASE_URL:-http://8.150.1.98:8080}
WANDB_DIR=${WANDB_DIR:-/storage/openpsi/users/chucai.dzq/areal-data/wandb}
export WANDB_BASE_URL WANDB_DIR
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
if [[ -z "${AENV_SYSTEM_URL:-}" ]]; then
  if [[ "${AGENT_TYPE}" == "cc" ]]; then
    AENV_SYSTEM_URL=http://33.180.184.68
  else
    AENV_SYSTEM_URL=http://api-service-k8s.aenv.svc.et15-02-aidc.sh.s-aidc.local:8080
  fi
fi
SWE_RL_ADMIN_API_KEY=${SWE_RL_ADMIN_API_KEY:-"swerl-${USER:-admin}-${TRIAL_NAME}"}
export SWE_RL_ADMIN_API_KEY

echo "==> agent_type: ${AGENT_TYPE}, trial_name: ${TRIAL_NAME}"

if [[ "${WANDB_MODE}" != "disabled" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_MODE=${WANDB_MODE} but WANDB_API_KEY is not set; wandb online login may fail" >&2
fi

EXTRA_ARGS_STR=""
if (($#)); then
  printf -v EXTRA_ARGS_STR ' %q' "$@"
  echo "==> extra trainer args:${EXTRA_ARGS_STR}"
fi

# ---- 缓存 / 产物目录 ----
CACHE_HOME=/storage/openpsi/users/chucai.dzq/areal-data/cache/home
LAUNCH_DIR=/storage/openpsi/users/chucai.dzq/areal-data/launch/${EXP_NAME}_${TRIAL_NAME}
mkdir -p "$CACHE_HOME" "$LAUNCH_DIR" \
  /storage/openpsi/users/chucai.dzq/areal-data/cache/{flashinfer,hf,xdg} \
  /storage/openpsi/users/chucai.dzq/areal-data/experiments \
  "$WANDB_DIR"

# ---- 1) rsync 当前分支到 /storage（计算节点可见） ----
echo "==> rsync ${AREAL_SRC} -> ${AREAL_STORAGE}"
mkdir -p "${AREAL_STORAGE}"
# These paths existed in older layouts. Remove them explicitly because rsync
# excludes __pycache__, and excluded files can keep deleted directories alive.
rm -rf \
  "${AREAL_STORAGE}/examples/swe/rl" \
  "${AREAL_STORAGE}/examples/swe/sweagent"
rsync -a --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.venv/' --exclude 'outputs/' --exclude '*.sif' \
  "${AREAL_SRC}/" "${AREAL_STORAGE}/"

# 校验 worker PYTHONPATH 与暂存路径一致
grep -q "${AREAL_STORAGE}" "${AREAL_STORAGE}/${CONFIG}" \
  || echo "WARN: ${CONFIG} 里 worker PYTHONPATH 未包含 ${AREAL_STORAGE}" >&2

# ---- 2) 生成 controller sbatch 脚本 ----
CTRL_SH="${LAUNCH_DIR}/controller.sbatch"
cat > "${CTRL_SH}" <<EOF
#!/bin/bash
#SBATCH --job-name=chucai-swerl-controller
#SBATCH --reservation=${RESERVATION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --no-requeue
#SBATCH --output=${LAUNCH_DIR}/controller.log
#SBATCH --open-mode=append

# Forward wandb auth without writing secrets into the generated sbatch file or
# xtrace logs. Slurm exports the submit environment by default.
if [[ -n "\${WANDB_API_KEY:-}" ]]; then
  export APPTAINERENV_WANDB_API_KEY="\${WANDB_API_KEY}"
fi
if [[ -n "\${WANDB_BASE_URL:-}" ]]; then
  export APPTAINERENV_WANDB_BASE_URL="\${WANDB_BASE_URL}"
fi
export APPTAINERENV_WANDB_DIR="\${WANDB_DIR:-${WANDB_DIR}}"

set -x
# controller 在 sbatch 作业里继承的 SLURM_* 会污染它给 worker 的 sbatch；全部清掉。
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
  --env PYTHONPATH=${AREAL_STORAGE}:${AWEAGENT_ROOT} \\
  --env AWEAGENT_ROOT=${AWEAGENT_ROOT} \\
  --env SWE_AGENT_ROOT=${SWE_AGENT_ROOT} \\
  --env SWE_RL_ADMIN_API_KEY=\${SWE_RL_ADMIN_API_KEY} \\
  --env HOME=${CACHE_HOME} \\
  --env SLURM_CONF=/etc/slurm/slurm.conf \\
  --env AENV_SYSTEM_URL=${AENV_SYSTEM_URL} \\
  ${IMAGE} \\
  bash -lc "cd ${AREAL_STORAGE} && \\
    python3 examples/swe/train_swe_rl.py \\
      --config ${CONFIG} \\
      experiment_name=${EXP_NAME} \\
      trial_name=${TRIAL_NAME} \\
      total_train_steps=${TOTAL_TRAIN_STEPS} \\
      econfig.agent_type=${AGENT_TYPE} \\
      econfig.agent_config=${AGENT_CONFIG} \\
      econfig.agent_root=${AWEAGENT_ROOT} \\
      econfig.swe_agent_root=${SWE_AGENT_ROOT} \\
      rollout.agent.admin_api_key=\${SWE_RL_ADMIN_API_KEY} \\
      actor.scheduling_spec.0.env_vars.SWE_AGENT_ROOT=${SWE_AGENT_ROOT} \\
      actor.scheduling_spec.0.env_vars.AWEAGENT_ROOT=${AWEAGENT_ROOT} \\
      actor.scheduling_spec.0.env_vars.AENV_SYSTEM_URL=${AENV_SYSTEM_URL} \\
      stats_logger.wandb.mode=${WANDB_MODE}${EXTRA_ARGS_STR}"
EOF

echo "==> controller sbatch written: ${CTRL_SH}"
echo "==> log will be at: ${LAUNCH_DIR}/controller.log"

# ---- 3) 提交 ----
JID=$(sbatch --parsable "${CTRL_SH}")
echo "==> submitted controller job: ${JID}"
echo "    tail -f ${LAUNCH_DIR}/controller.log"
echo "    worker 作业名: ${EXP_NAME}_${TRIAL_NAME}:{actor,rollout,...}"
