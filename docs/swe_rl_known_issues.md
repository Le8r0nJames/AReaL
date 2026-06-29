# SWE-RL (Qwen3-Coder-30B-A3B + AWEAgent) 踩坑与修复记录

记录在 AReaL(`chucai.dzq/feat-swe-rl-sft` 分支,基于 gh/main 重构)上把 SWE-RL
(GRPO)端到端跑通过程中遇到的全部问题、根因与修复,供后续复现与排查参考。

## 环境

- **模型**:Qwen3-Coder-30B-A3B-Instruct(`Qwen3MoeForCausalLM`,32 attn heads /
  4 kv heads / 48 层 / 128 experts top-8)
- **训练**:AReaL,actor 用 MegatronEngine `(attn:d2p1t4c2|ffn:d2p1e8)`(DP2 PP1
  TP4 CP2 / EP8,16 GPU),rollout 用 SGLang `d2t8`(2 副本 × TP8)
- **Agent**:AWEAgent(`chucai.dzq/strip-to-mini-swe`,仅 mini-swe),原生 OpenAI
  tool call;数据 `miumiu.zwh/.../0519_filtered_2k.../filtered.jsonl`(含
  eval_script + faas-image)
- **镜像**:`/storage/openpsi/images/areal-dev.sif`;Slurm `swe-rl` reservation

> 端到端要跑通需要解决以下问题(#1-8 为首跑阶段;#9-11 为切到 k8s + scaleswe 数据集后新增)。
> 按"先 rollout 起来 → reward 真实 → 训练数值正常 → checkpoint 可存"的顺序排列。

______________________________________________________________________

## 1. SGLang server 起不来:`Router.__init__() got an unexpected keyword argument 'on_startup'`

- **现象**:rollout worker 的 `launch_server` 子进程 import
  `sglang.srt.entrypoints.http_server` 即崩,`server launch failed`,300s 超时。
- **根因**:worker 的 `PYTHONPATH` 混入了 `miumiu.zwh/areal-deps/aenv`,该目录
  捆绑 `starlette 1.2.1`,排在 venv 的 `0.50.0` 之前被优先 import;镜像里的
  `fastapi 0.124.4` 仍向 `starlette.Router` 传 `on_startup`,而 starlette 1.x 已
  删除该参数 → `TypeError`。(`starlette 0.50.0` 只 deprecate、不报错。)
- **排除项**:`uv pip install aenvironment` 本身**不会**升级 starlette
  (诊断作业实测:aenvironment 0.1.7 对 starlette 约束宽松,仍解析到 0.50.0)。
- **修复**:worker `PYTHONPATH` 只保留 `AWEAgent` + 本仓,**绝不**加 miumiu 的
  aenv 目录;aenv 用 `uv pip install -q aenvironment` 干净装进 venv。

## 2. rollout worker 排队超时:`WorkerTimeoutError (waited 300.0s)`

- **现象**:actor 已就绪,rollout slurm 作业排队 >300s,整体 `WorkerTimeoutError`。
- **根因**:`rollout.setup_timeout=300` 偏小。`swe-rl` reservation 有 76 节点,
  容量不是问题,但 worker `exclusive: true` 需要整空节点,集群繁忙时瞬时空闲整机
  不足,排队会超 300s。
- **修复**:`rollout.setup_timeout: 300 → 900`。

## 3. `prox_logp_method` 取值无效:`'reuse_train_logp' is not a valid ProxLogpMethod`

- **现象**:rollout 全部正常,`trainer.train()` 一进入就 `ValueError`。
- **根因**:gh 重构后 `ProxLogpMethod` 枚举只接受 `recompute/loglinear/metrics`,
  旧值 `reuse_train_logp` 已删除;dataclass 不校验 `choices`,所以拖到 train() 调
  `ProxLogpMethod(...)` 才报错。
- **修复**:`actor.prox_logp_method: reuse_train_logp → recompute`(标准 decoupled
  PPO,无近似)。

## 4. (引擎 bug)CP>1 时 `compute_logp` 崩:`split_with_sizes ... sum to N but got [...]`

- **现象**:`compute_logp` 在 actor worker 报 `split_with_sizes` 不匹配,
  split_sizes 之和 ≈ `cp_size × tensor`。
- **根因**:context parallel(CP=2)下 forward 输出是 **CP-local**(每 rank 1/cp_size
  token)。loss 路径有 `reassemble_cp_packed_logprobs` 把它 gather 回全长;但
  `forward_batch`(forward_only,即 compute_logp/compute_values)走的
  `_compute_forward_result` **缺这一步 gather**,却用全局 `output_seqlens` 去
  `unpack_sequence` → 维度差 cp_size 倍。
- **修复(port 自 `chucai.dzq/colocation-swe`)**:`areal/engine/megatron_engine.py`
  - `forward_backward_batch` 加 `gather_cp_output: bool = False` 参数;
  - `forward_step` 里 `cp_local = cp_size > 1 and not gather_cp_output`;
  - `forward_batch` 调用时传 `gather_cp_output=True`(CP-local 输出经
    `postprocess_packed_seqs_context_parallel(gather_output=True)` 的 all_gather
    回全长)。loss 路径不传该参数、保持 CP-local 不变。

## 5. (agent 配置)`no_tool_call_limit: 0` 导致 ~87% rollout 被判废、reward 虚假为 0

- **现象**:rollout "异常地快",264 条 traj 里 231 条 reward=0、89 条(34%)
  completion 仅 59 字符且 0 tool_call,内容是复读 prompt 的 `<format_example>` 占位符。
- **根因**:econfig 用的 `swe_agent_config=min-swe-agent-train-top1`(AWEAgent)其
  `no_tool_call_limit: 0` 本是给**离线筛 top1 轨迹**用(无合法 tool_call 即判废);
  误用到 RL rollout 后,`aweagent/agents/swe.py` 里 `no_tool_call(=1) <= 0` 为
  False,模型首轮一旦没产 tool_call 就 `Agent ends!!!`、episode 当场结束 reward=0,
  一次重试都不给。
- **修复**:`no_tool_call_limit: 0 → 20`(对齐原版 min-swe-agent),
  `multiple_tools_limit: 3 → 20`;模型首轮格式不对时会收到 `no_tool_warning` 并重试。

## 6. (模型选择,决定性)Instruct 复读占位符、reward 仅 4% → 换 Coder

- **现象**:Qwen3-30B-A3B-**Instruct** 指令遵循弱,即使放开 no_tool_call_limit,仍
  18% 首轮复读 `<format_example>` 占位符,reward>0 仅 4%、accept 极慢。
- **修复**:
  - **换 Qwen3-Coder-30B-A3B-Instruct**(`/storage/openpsi/models/Qwen__Qwen3-Coder-30B-A3B-Instruct`;
    head 配置与 Instruct 完全相同 → **并行拓扑不变**);
  - `rollout.agent.tool_call_parser` `qwen25 → qwen3_coder`(Coder 是 XML 工具格式
    `<tool_call><function=...>`,非 hermes JSON);
  - `<format_example>` 从占位符改成"具体示例 + 明确禁止照抄"。
  - 结果:0% 复读、0% 无 tool_call、reward>0 ~54%、task_reward avg ~0.42、正负均衡。
  - **结论:SWE-RL 必须用 Coder,不是 Instruct。**

## 7. (引擎 bug)advantage 爆炸到 1e5、grad_norm 飙到几千

- **现象**:训练 step 的 `advantages/max` 恒定 `1e5`、`min` 正常(-2.4),
  `grad_norm` 2500-4500(正常应 <1);`task_reward` 暂时没崩(grad clip 兜底)。
  关键线索:`n_seqs` = 70/82 等**不是 group_size(8)的整数倍**。
- **根因**:GRPO 的 reward 按 group 归一 `(reward-mean)/(std+eps)`。SWE rollout 有
  样本失败/被过滤,导致部分组 <8(变长组)。但 `areal/utils/data.py` 的
  `Normalization` 用固定 `for i in range(bs // group_size)` 切片:
  ① 当 bs 非 group_size 整数倍时,**尾部序列没被循环覆盖**,`std=torch.zeros_like`
  保持 0;② 变长组下固定切片还会 straddle 两个组。`std≈0` 时
  `(reward-mean)/(0+eps=1e-5)` → 成功样本 advantage 爆 `1/1e-5=1e5`(失败样本
  baseline 有差异、std>0,所以 min 正常,呈不对称)。
- **修复(port 自 `swe/main`)**:让 `Normalization` 支持变长组。
  - `areal/utils/data.py`:加 `_build_group_slices(bs, group_boundaries)`,
    `__call__` 加 `group_boundaries` 参数,mean/std 两个循环改用 `for s in
    group_slices`(组大小 `s.stop - s.start`);
  - `areal/trainer/ppo/actor.py`:`compute_advantages` 在本路径注入
    `batched["_traj_group_sizes"] = meta.traj_group_sizes`(`concat_batch` 已记录
    每个 traj 组的实际样本数),`_compute_advantages` 里 `data.pop("_traj_group_sizes")`
    作为 `group_boundaries` 传给 `reward_norm`(pop 掉避免污染 forward / split_batch)。

## 8. (引擎/配置)checkpoint save 崩:`ShardedTensor.flattened_range is not supported`

- **现象**:训练到 `saver/recover.freq_steps`(=10)存 checkpoint 时,actor worker
  报 `Engine method 'save' failed: ShardedTensor.flattened_range is not supported`,
  controller 退出。
- **根因**:megatron-core 0.16.0 的 `ShardedTensor.validate_metadata_integrity()`
  移除了对 `flattened_range` 的支持,但分布式 optimizer 默认 sharding type
  `fully_sharded_model_space` 仍会设 `flattened_range` → 保存 optimizer state 即崩。
  `use_distributed_optimizer=true` + 存 optimizer(`recover.no_save_optim=false`)触发。
  (saver 本身 `with_optim=False`,只存模型权重,不触发;崩的是 recover save。)
- **修复(port 自 `swe/main` 的 checkpointer.py,真修复而非规避)**:给
  `areal/engine/megatron_utils/checkpointer.py` 里 `self.optimizer.sharded_state_dict()`
  传 `metadata={"distrib_optim_sharding_type": "dp_reshardable"}`——`dp_reshardable`
  不使用 `flattened_range`,save optimizer state 正常工作。于是 `recover.no_save_optim`
  保持 `false`(正常存/载 optimizer state、支持断点续训),无需靠不存 optimizer 规避。
  > 备注:`no_save_optim=true` 是更早的规避方案(能跑通但丢失续训能力);dp_reshardable
  > 才是根治。

## 9. (切换 k8s / 依赖)aenv 0.1.8rc2 装错解释器 + 卸坏 pydantic,16 worker 全崩

> 切换到 k8s 环境(`AENV_SYSTEM_URL` 改为 k8s service DNS、aenv 改 `0.1.8rc2`)后新增。

- **现象**:actor worker 启动 ~35s 后 singularity step `CANCELLED`,16 个 worker 全
  `exit 1`,controller 等 worker ready 300s 超时退出。actor.log:
  `ImportError: cannot import name 'BaseModel' from 'pydantic' (unknown location)`
  + `ModuleNotFoundError: No module named 'aenv'`。(改 `--no-deps` 后 pydantic 不崩、
  worker 起得来,但 aenv 自检仍 `No module named 'aenv'`——暴露出更底层的解释器错位。)
- **根因(两层,核心是解释器错位)**:
  1. **裸 pip 装错解释器**:worker 的裸 `pip`/`python`/`python3` 全是 **miniconda
     (`/root/miniconda3`,python3.10)**,而 areal worker 实际跑在 **`/opt/.venv`
     (python3.12,其 site-packages 在 worker PYTHONPATH 里、pydantic 2.12.5 在那)**。
     `additional_bash_cmds` 的裸 `pip install` 把 aenv 装进 miniconda3.10 的 site,
     venv 的 python3.12 自然 `import aenv` 失败(`Location:
     /root/miniconda3/lib/python3.10/site-packages`)。
  2. **顺带卸坏 pydantic**:完整 `pip install aenvironment==0.1.8rc2` 时,miniconda 的
     pip 顺着 PYTHONPATH 看到 venv 的 pydantic 2.12.5,又因 aenv 声明 `pydantic<2.12`
     而卸载它(`Uninstalling pydantic-2.12.5`),再因 fastmcp 等重装回 2.12.5——root
     下卸载+重装把 pydantic 装残(`unknown location`),`areal.infra.rpc.serialization`
     的 `from pydantic import BaseModel` 崩,areal 整个 import 失败、worker 全挂。
- **诊断要点**(srun 计算节点实测,登录节点 pypi 网段隔离):
  - 裸 `python/python3/pip` → `/root/miniconda3/...`(3.10);areal worker → `/opt/.venv`
    (3.12,原生 pydantic 2.12.5,默认无 pip)。
  - aenv 0.1.8rc2 顶层模块是 `aenv`(+`cli`),`from aenv import Environment` 兼容;
    其运行依赖 venv 已基本齐全(httpx/pydantic_settings/anyio/mcp/uvicorn/starlette/
    docker 等),**唯缺 fastmcp**;aenv 硬 import `from fastmcp import Client`。
- **修复(显式用 venv 的 python + `--no-deps`)**:`additional_bash_cmds` 改为——
  - `/opt/.venv/bin/python -m ensurepip --upgrade`(venv 默认无 pip,从镜像内置 wheel 补);
  - `/opt/.venv/bin/python -m pip install --no-deps aenvironment==0.1.8rc2`(装进 venv、
    且 `--no-deps` 不触发 pydantic 卸载/重装);
  - `/opt/.venv/bin/python -m pip install 'fastmcp<3'`(aenv 要 `fastmcp<3,>=2.13`,
    装到 2.14.7;**3.x 改了包结构 `from fastmcp import Client` 失败,必须锁 `<3`**;
    带依赖装不会卸 pydantic);
  - `/opt/.venv/bin/python -c 'import aenv, aweagent'` 自检。
- **实测(smoke3)**:16 actor + 2 rollout worker 全 ready,aenv 自检不再 WARN(对比
  smoke2 满屏 `No module named 'aenv'`),pydantic 保持 2.12.5、areal 正常。
  > 备注:openai 显示 2.33.0 与 sglang 的 `openai==2.6.1` 约束不符,是镜像原有状态
  > (fastmcp 未改动 openai),无害。

## 10. (切换 k8s / 数据集)repo_dir 仍指 /testbed,scaleswe 代码在 /workspace/<repo>,reward 全 0

> aenv 安装/连通修好后暴露(#9 之后);属同一轮 k8s + top64(scaleswe)迁移。

- **现象**:aenv 连 k8s、沙箱创建、agent 多轮都正常,但 rollout **全部 reject**
  (`accepted: 0, rejected: 34+`)。merged.log(rollout)显示:
  `custom_init ... 'bash: cd: /testbed: No such file or directory'`、
  `AEnv custom_init failed: Failed to move original .git: mv: cannot stat
  '/testbed/.git'`;每条 `Finished SWE episode: reward=0.0`,并跟随
  `OpenAIProxyClient ERROR: [400] Error setting reward` + `No interactions in session`。
- **根因**:scaleswe(top64)数据集每条的代码根目录在记录的 `workdir`(`/workspace/<repo>`,
  如 `/workspace/beets`),**不是** SWE-bench 默认的 `/testbed`。而 `AenvSWE` 的
  `repo_dir` 默认 `/testbed`;生产路径 `lifecycle.run_agent_with_reward` 构造
  `AenvSWE(data_id=, data=data, faas_image=)` 传了 `data`(含 `workdir`)却**没传
  `repo_dir`**,且 `__init__` 没从 `data['workdir']` 取 repo_dir(只有测试函数
  `_test_tools_on_instance` 显式 `repo_dir=workdir`,生产路径漏接)。于是 custom_init
  的 `cd {repo_dir}=/testbed`、`mv /testbed/.git` 全失败 → agent 在错误目录、拿不到
  代码 → reward 全 0 → 组内 reward 全同 → `should_accept_fn` 判为无学习信号全 reject。
  (reward=0 时 `setting reward 400` 是 `No interactions` 的下游,非独立 bug。)
- **修复(ant-code/AWEAgent `envs/swe.py` `AenvSWE.__init__`,真修复)**:config 构造后、
  init_cmd retarget 前,若 `repo_dir` 未显式传且 `config.data['workdir']` 存在,则
  `self.config.repo_dir = workdir`。已有的 retarget 逻辑随后把 init_cmd 内的 `/testbed`
  全替换为 workdir,使 custom_init / git-hide / baseline / eval 一致指向真实 repo。
- **验证(srun 单测)**:data 带 workdir → `repo_dir=/workspace/beets` 且 init_cmd 无
  `/testbed` 残留;data 无 workdir → 仍 `/testbed`(默认布局不破坏);显式传 `repo_dir`
  → 不被 workdir 覆盖。
- **附带**:部分 traj `Aenv create failed`(沙箱创建 ~20s 快速失败),疑似 20 并发创建
  沙箱的瞬时压力;待 repo_dir 修复后观察其比例,必要时降并发 / 加重试。

## 11. (切换 k8s / 数据集)system prompt 硬编码 /testbed,模型在错目录操作,reward 稀疏

> #10 修好 repo_dir 后暴露:reward 能算但偏稀疏。

- **现象**:repo_dir 修好后 reward 能算(出现 `reward=1.0`、`accepted>0`),但多数 traj 仍
  reward=0,merged.log 频繁 `bash: cd: /testbed: No such file or directory`。
- **根因**:agent 的 system prompt(`config["instruction"]`,如 `min-swe-agent-train-top1.yaml`)
  硬编码 "MODIFY: ... in /testbed (this is the working directory for all your subsequent
  commands)",模型据此 `cd /testbed` / 用相对路径,但 scaleswe 代码在 `/workspace/<repo>`。
  且 `SWEAgent` 的 system prompt 直接用 `config["instruction"]`(**不经 render_template**),
  /testbed 死文本不被替换;`execute_bash` 工具也不加 cwd 前缀(命令 cwd 全靠模型)。
- **修复(ant-code/AWEAgent `agents/swe.py` `SWEAgent.run`)**:messages 构造后,把
  system+user content 里的 `/testbed` 替换为 `environment.config.repo_dir`(仅当
  repo_dir != /testbed)。与 #10 的 init_cmd retarget 同思路(字符串 replace,避免改 Jinja
  模板触发 StrictUndefined)。
- **备注**:`_verify_clean_baseline` 代码实际已用 `self.config.repo_dir`(git status 目标对,
  docstring 文本写 /testbed 无害);`swe_tools` 的 `find /testbed` 残留 returncode 0 不致命。

______________________________________________________________________

## 健康指标参考(20 step 长跑达标标准)

- `rollout/reward` / `task_reward/avg`:非 0(Coder 在该数据集冷启动 ~0.4),正负均衡
- `advantages/max`、`min`:有界(GRPO 归一后 ~±2),**不应出现 1e5**
- `update/grad_norm`:O(0.1~1),**不应几千**
- `update/importance_weight/avg`、`behave_imp_weight/avg`:≈ 1.00x / 0.99x(on-policy)
- checkpoint 在 `freq_steps` 正常落盘,不报 flattened_range

## 相关改动落点

| 问题 | 改动文件 | 仓库/分支 |
| --- | --- | --- |
| 4 CP compute_logp | `areal/engine/megatron_engine.py` | AReaL `feat-swe-rl-sft` |
| 7 advantage 变长组 | `areal/utils/data.py`、`areal/trainer/ppo/actor.py` | 同上 |
| 8 save optimizer (dp_reshardable) | `areal/engine/megatron_utils/checkpointer.py` | 同上 |
| 2/3/6 配置 | `examples/swe/rl/qwen3_30b_a3b_grpo_4n_2t2r.yaml` | 同上 |
| 1/5/6 agent | `aweagent/configs/1_0_0/min-swe-agent-train-top1.yaml` | AWEAgent `strip-to-mini-swe` |
| 9 aenv 装法 (--no-deps + fastmcp<3) | `examples/swe/rl/qwen3_30b_a3b_grpo_4n_2t2r.yaml` (additional_bash_cmds) | 同上 |
