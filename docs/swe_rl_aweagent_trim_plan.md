# AWEAgent SWE-RL 精简方案(可执行)

> 仓库根:`/storage/openpsi/users/chucai.dzq/codes/ant-code/AWEAgent/`
> 目标:精简为 SWE-RL 训练最小集 = 运行时闭包 + `post_exit_hook` 的 `clean_instances`
> + `requirements.txt` + **单一 config**。
> **执行时机:必须等长跑 933640 结束后再删**(worker 会反复 import aweagent,
> 文件被 `git rm` 后磁盘消失 → worker 下次 import 崩溃)。

本方案由 4 路并行调研 + 综合裁定产出(Workflow `aweagent-trim-analysis`),
已在代码里逐一核实并修正调研间的冲突。

______________________________________________________________________

## 0. 关键裁定(修正调研冲突,易踩的坑)

| 项 | 裁定 | 实测证据 |
| --- | --- | --- |
| `tools/search_replace.py` | **必须保留** | `envs/swe.py:30-44` `_build_tools_save_cmds()` 遍历 `tools/*.py` 全部 `cat` 进沙箱 `/root/swe_tools/`;`swe.py:502` 执行 `python3 /root/swe_tools/search_replace.py`。是**沙箱运行时脚本(非 Python import)**,极易误删 |
| `maintenance/clean_instances.py` | **必须保留** | 两个 active YAML 的 post_exit_hook 调 `python -m aweagent.maintenance.clean_instances` |
| `tools/file_edit.py` | **可删** | 沙箱内无任何调用(只 search_replace.py 在 L502 执行);`_build_tools_save_cmds` 只 `cat` 不执行,删它只是少注入一个无用脚本 |
| active config 是否有 include | **无,完全自包含** | `1_0_0/min-swe-agent-train-top1.yaml` grep 无 include/extends/inherit |

______________________________________________________________________

## 1. 保留清单

### 1.1 运行时闭包(import 链:`AReaL agent.py → lifecycle.run_agent_with_reward → AenvSWE + SWEAgent + OpenAIModel`)

- `aweagent/__init__.py`、`aweagent/lifecycle.py`、`aweagent/runner.py`(标准 CLI 入口,体积小保守保留)
- `aweagent/agents/{__init__,swe,base}.py`(SWEAgent 主循环 + `load_config_from_dir` + jinja 渲染)
- `aweagent/envs/{__init__,swe}.py`(AenvSWE + `_build_tools_save_cmds`)
- `aweagent/models/{__init__,openai_model}.py`(OpenAIModel LLM 客户端)
- `aweagent/tools/{__init__,schema,search_replace}.py`(get_tool_schema + 沙箱脚本)
- `aweagent/common/{__init__,io,logging}.py`

### 1.2 post_exit_hook

- `aweagent/maintenance/{__init__,clean_instances}.py`

### 1.3 配置(只留一个)

- `aweagent/configs/1_0_0/min-swe-agent-train-top1.yaml`(AReaL 唯一引用,自包含)

### 1.4 包/依赖元数据

- `requirements.txt`(post_exit_hook 的 `pip install -r` 用)、`pyproject.toml`、`README.md`、`LEGAL.md`、`.gitignore`

______________________________________________________________________

## 2. 删除清单

| 路径 | 把握度 |
| --- | --- |
| `tests/`(整目录) | 确定 |
| `scripts/stress_test_api.py`、`scripts/test_thinking_in_context.py` | 确定 |
| `docs/`(5 md)、`REFACTOR_PLAN.md` | 确定 |
| `aweagent/data/swe_bench_verified*.jsonl`(7 个,~24MB 测试数据集) | 确定 |
| `aweagent/maintenance/{filter_by_reward,process_traindata}.py`(离线 SFT 工具) | 确定 |
| `aweagent/tools/file_edit.py` | 确定 |
| `aweagent/configs/{train,train_v2,train_v2_top1,train_fc,mini-swe-agent}.yaml` | 确定 |
| `aweagent/configs/1_0_0/min-swe-agent.yaml`(active 的旧版副本) | 确定 |
| `aweagent/configs/live_swe_agent.yaml` | 需确认(若 live 推理流程依赖则留) |
| `aweagent/configs/prompts/issue_rewrite_prompt.md` | 需确认(全仓无引用) |
| `**/__pycache__/`(6 处) | 确定 |

> 用户要"配置留一个就行" → 倾向激进:连 `live_swe_agent.yaml` + `prompts/` 一并删,
> 真正只留 `min-swe-agent-train-top1.yaml`。这两项全仓 grep 无运行时引用,删除安全;
> 仅当存在本仓之外的 live 推理流程引用它们时才需保留。

______________________________________________________________________

## 3. 验证脚本(精简后、commit 前,在镜像 venv 跑)

保存为 `verify_min_closure.py`,跑通后删除:

```python
import sys, traceback

def check(label, fn):
    try:
        fn(); print(f"[OK]   {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}"); traceback.print_exc(); sys.exit(1)

check("import aweagent", lambda: __import__("aweagent"))

def _lifecycle():
    from aweagent.lifecycle import run_agent_with_reward
    assert callable(run_agent_with_reward)
check("from aweagent.lifecycle import run_agent_with_reward", _lifecycle)

def _classes():
    from aweagent.envs.swe import AenvSWE
    from aweagent.agents.swe import SWEAgent
    from aweagent.models.openai_model import OpenAIModel
    from aweagent.tools.schema import get_tool_schema
    env = AenvSWE(data={"workdir": "/testbed"}, faas_image="dummy")  # 不调 check_env
    assert env is not None
check("construct AenvSWE / import SWEAgent+OpenAIModel+get_tool_schema", _classes)

def _config():
    from aweagent.agents.base import load_config_from_dir
    cfg = load_config_from_dir("1_0_0/min-swe-agent-train-top1")
    assert cfg
check("load_config_from_dir('1_0_0/min-swe-agent-train-top1')", _config)

def _clean():
    import importlib
    importlib.import_module("aweagent.maintenance.clean_instances")
check("import aweagent.maintenance.clean_instances", _clean)

def _sandbox_script():
    import os, aweagent
    p = os.path.join(os.path.dirname(aweagent.__file__), "tools", "search_replace.py")
    assert os.path.isfile(p), f"missing sandbox script: {p}"
check("tools/search_replace.py exists on disk (sandbox injection)", _sandbox_script)

print("\nALL CHECKS PASSED — 精简闭包完整")
```

> 若无 aenv server 时构造 `AenvSWE` 报网络错,把第 3 项改为"仅 import 不构造"。

______________________________________________________________________

## 4. 执行命令草案(长跑 933640 结束后)

```bash
# 前置门禁:确认 933640 不在 RUNNING
squeue -j 933640 --format="%.18i %.30j %.8T %.10M %.6D %R"

cd /storage/openpsi/users/chucai.dzq/codes/ant-code/AWEAgent
git tag pre-slim-full-configs          # 回滚锚点(保留参数调优历史)

# 确定项
git rm -r tests docs
git rm scripts/stress_test_api.py scripts/test_thinking_in_context.py REFACTOR_PLAN.md
git rm aweagent/data/swe_bench_verified*.jsonl
git rm aweagent/maintenance/filter_by_reward.py aweagent/maintenance/process_traindata.py
git rm aweagent/tools/file_edit.py
git rm aweagent/configs/train.yaml aweagent/configs/train_v2.yaml \
       aweagent/configs/train_v2_top1.yaml aweagent/configs/train_fc.yaml \
       aweagent/configs/mini-swe-agent.yaml aweagent/configs/1_0_0/min-swe-agent.yaml

# 配置留一个(用户要求):一并删 live + prompts
git rm aweagent/configs/live_swe_agent.yaml
git rm -r aweagent/configs/prompts

find . -type d -name __pycache__ -prune -exec rm -rf {} +

python verify_min_closure.py           # 必须 ALL CHECKS PASSED
rm -f verify_min_closure.py
git add -A
git commit -m "refactor(swe-rl): slim AWEAgent to minimal runtime closure"
# 回滚:git reset --hard pre-slim-full-configs
```

______________________________________________________________________

## 5. 精简效果预估

| 维度 | 节省 |
| --- | --- |
| 示例数据集 `data/*.jsonl` | -24 MB |
| `tests/` | -7 文件 |
| `scripts/`(2)、`docs/`(5)、REFACTOR_PLAN | -8 文件 |
| maintenance 4→2、tools 4→3 | -3 文件 |
| configs 9 yaml + 1 prompt → 1 yaml | -8 yaml(-1 md) |
| `__pycache__` | -304 KB |
| **总计** | **≈ -24.4 MB,删 20+ 文件 / 3 目录** |

**核心运行时代码(~2 MB)完全不动**,精简集中在数据/测试/文档/多余配置。
