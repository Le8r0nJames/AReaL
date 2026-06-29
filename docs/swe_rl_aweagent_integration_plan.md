# SWE-RL Agent 集成方案:把开源 AweAgent 精简进 AReaL(方案B)

目标:用开源 `aweai-team/AweAgent`(现代架构、原生 f2p/p2p 评测、inference==RL 同一套代码)
替换当前外部依赖 `ant-code/AWEAgent`,**精简后放进 AReaL 一个文件夹**,作为 SWE-RL 的 agent。

参考来源(开源):`/storage/openpsi/users/chucai.dzq/projects/swedemo/aweai-team/AweAgent`
内部现状(被替换):`/storage/openpsi/users/chucai.dzq/codes/ant-code/AWEAgent`

______________________________________________________________________

## 0. 背景:方案A vs 方案B

| | 方案A(已落地) | 方案B(本文) |
| --- | --- | --- |
| agent | 内部 `ant-code/AWEAgent`(外部仓,PYTHONPATH 引入) | 开源 AweAgent **精简进 AReaL 文件夹** |
| reward | 数据集预处理生成 `eval_script`(P2P 多→300s timeout 风险) | **原生 f2p/p2p 评测,无需预处理** |
| RL 数据 | 走 AReaL OpenAI proxy 收 token | 同样走 AReaL proxy(见下) |
| 工作量 | 已完成 | 中等(runtime 适配 + 精简) |
| 符合"进 AReaL/代码少" | 否 | 是 |

> 方案A 可作为兜底(先跑通 k8s+top64 流程)。方案B 是更干净的目标态。

## 1. 三个关键利好(验证后确认)

1. **走 AReaL proxy 收 token**:`LLMClient` 默认 `backend="openai"` + `base_url`/`api_key`,
   直接指向 AReaL 的 OpenAI proxy 即可——token/logprob 由 AReaL proxy 收集(与
   ant-code/AWEAgent 同机制),**不需要它给 Slime 的 `TrainingState`/`integrations/slime`**。
2. **原生 f2p/p2p reward**:`ScaleSWEEvaluator`(复用 `BeyondSWEEvaluator._eval_beyondswe`)
   直接吃数据集的 `f2p_patch`/`f2p_script`/`FAIL_TO_PASS`/`PASS_TO_PASS`:apply f2p_patch →
   上传 f2p_script 为 `test_fail_to_pass.py` → pytest 合并跑 F2P+P2P → 全 pass 即 reward=1。
   **方案B 不需要方案A 的 eval_script 预处理**。
3. **接口干净**:`AgentLoop.run(task_prompt) -> AgentResult`,inference/RL 同一套(RL 仅多一个
   `AgentContext.training` 开关,本方案走 proxy 收 token,可不用该开关)。

## 2. 集成架构(参考内部 `examples/swe/rl/agent.py` 的 SWEAgentWorkflow)

```
AReaL SWEAgentWorkflow.run(data, base_url, api_key):
  ctx = AgentContext(
      llm     = LLMClient(backend="openai", base_url=<AReaL proxy>, api_key=<per-rollout>),
      session = AEnvRuntimeSession(...),          # ← 新写的适配层(见 §3)
      tools   = SearchSWEAgent(...).get_tools(),
      max_steps = ...,
  )
  patch  = await AgentLoop(agent, ctx).run(problem_statement)   # token 由 AReaL proxy 收
  result = await ScaleSWEEvaluator().run_tests(instance, ctx.session)  # f2p/p2p → score
  return result.score                                          # reward 0/1
```

## 3. runtime 适配(方案B 关键路径,已验证可控)

AweAgent 的 `core/runtime/protocol.py::RuntimeSession` 是 ABC,**只需实现 5 个抽象方法**;
其余(`apply_patch`/`get_patch`/`_update_gitignore`)是基于这 5 个的 concrete 方法,免实现。

| RuntimeSession 抽象方法 | AEnv(`aenv.Environment`)映射 | 说明 |
| --- | --- | --- |
| `execute(command, cwd, timeout)` | `env.call_tool(f"{ver}/run", {command, timeout})` | 核心;cwd 用 `cd {cwd} && {command}` 拼 |
| `upload_file(remote_path, content)` | `execute("cat > {path} <<'EOF'\n{content}\nEOF")` | 用 execute 写文件 |
| `download_file(remote_path)` | `execute("cat {path}")` 取 stdout | |
| `list_files(path, recursive)` | `execute("find {path}" 或 "ls")` | |
| `close()` | `env.release()` | |

- 参考实现:内部 `ant-code/AWEAgent/aweagent/envs/swe.py` 已有 AEnv 全套用法
  (`from aenv import Environment`、`env.call_tool(f"{ver}/run", ...)`、`env.release()`、
  环境创建/`wait_for_ready`/persistent-bash 版本等),直接抄改成 `AEnvRuntimeSession`。
- 估计:**~100-200 行**一个类。

## 4. 精简范围(整仓 ~23865 行 → 估 ~6-8k)

| 保留(SWE-RL 必需) | 去掉 |
| --- | --- |
| `core/agent`(loop/context)、`core/llm`(client/config)、`core/eval`(isolation/utils,f2p/p2p)、`core/tool`(execute_bash/search_replace/task_done)、`core/task`(types)、`tasks/scale_swe`(task/evaluator)、`scaffold/search_swe`(agent)、`core/runtime`(protocol/types + 新增 AEnv 适配) | `cli.py`、`integrations/slime`、`tasks/{browsecomp,nl2repo,terminal_bench}`、`scaffold/{deepsearch,iter_research,terminus_2}`、`condenser`(可选)、`plugins`、`core/runtime/docker.py`(用 AEnv 替代) |

依赖裁剪:保留 `openai/pydantic/aiohttp/tiktoken/pyyaml/tenacity/structlog/json5`;
去掉 `docker/pandas/pyarrow/sandbox_fusion/anthropic/huggingface_hub` 等。

放置:`examples/swe/rl/aweagent/`(或 `areal/swe_agent/`),worker `PYTHONPATH` 指过去。

## 5. 实施步骤

1. 把 AweAgent 的精简子集(§4 保留列)copy 进 `examples/swe/rl/aweagent/`。
2. 写 `AEnvRuntimeSession(RuntimeSession)`(§3),底层用 `aenv.Environment`。
3. 改 `examples/swe/rl/agent.py` 的 `SWEAgentWorkflow.run`,改调 AweAgent 的
   `AgentLoop` + `ScaleSWEEvaluator`(§2),返回 `EvalResult.score` 作为 reward。
4. 数据集用**原始** `processed_to_upload_top64.jsonl`(无需 `_with_eval` 版,因原生 f2p/p2p)。
5. 改 yaml:`SWE_AGENT_ROOT`/`PYTHONPATH` 指向 AReaL 内的 aweagent;`tool_call_parser`
   按 SearchSWEAgent 的 `codeact_xml` 风格确认(可能需对应的 sglang parser)。
6. 裁剪依赖,确保 worker venv 能 import(去掉重依赖)。

## 6. 风险 / 待验证

1. **token 收集**:确认 `LLMClient(backend=openai)` 经 AReaL proxy 的 chat/completions
   能被正确收 token/logprob(`codeact_xml` 工具格式不影响 token 收集,但要实测一轮)。
2. **tool_call 格式**:SearchSWEAgent 用 `codeact_xml`(非 OpenAI tool call);需确认
   sglang 侧 parser / AReaL proxy 对该格式的 token 切分与 mask 正确。
3. **PASS_TO_PASS 时长**:原生 evaluator 跑 F2P+P2P(P2P 均 157),需确认 AEnv 内
   pytest 时长与 evaluator 的 timeout 设置(AweAgent 用较大的 `_BEYONDSWE_TIMEOUT`)。
4. **AEnv 版本**:k8s 环境用 `persistent-bash-env@1.1.x`;`AEnvRuntimeSession` 要对齐
   jun/k8s 适配的版本与 `AENV_SYSTEM_URL`。

## 7. 建议路径

- 先用**方案A**(已落地:ant-code/AWEAgent + eval_script 预处理数据集)把 k8s+top64 的 RL
  流程跑通、确认 reward 能算 → 作为 baseline 与兜底。
- 并行推进**方案B**:先落 `AEnvRuntimeSession`(关键路径,~100-200 行)并单测它对 AEnv 的
  execute/upload/download/close,再做精简 copy + SWEAgentWorkflow 改造。
