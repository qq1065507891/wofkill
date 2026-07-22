# Replace Kimi with DeepSeek-V4-Pro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将活动路由中的 `Kimi-K2.6` 完整替换为 `DeepSeek-V4-Pro`，同时保留原采样参数、MiniMax reflection/judge 覆盖和用户未跟踪文件。

**Architecture:** 只修改 YAML 路由契约与其测试。复用现有 `ark_deepseek_v4_pro` 主 profile，删除 `ark_kimi_k26`，把所有原 Kimi 引用重接到主 DeepSeek profile；运行时、超时机制与未跟踪探测脚本均不改动。

**Tech Stack:** Python 3、PyYAML、pytest、PowerShell、现有 `ModelRouter.from_yaml` 路由解析器。

---

## 文件结构与责任边界

- `config/models.yaml`：唯一生产配置变更；定义模型 profile、任务覆盖、fallback 和玩家分配。
- `tests/model_gateway/test_minimax_provider_routing.py`：刻画并验证 YAML 路由契约；通过 `ModelRouter.from_yaml` 验证最终解析结果。
- `scripts/probe_all_endpoints.py`：用户未跟踪文件；只读保留，不修改、不暂存、不提交。

实现期间使用 `@test-driven-development` 保证先看到预期失败，再改生产配置；使用 `@karpathy-guidelines` 保持变更最小；提交前使用 `@verification-before-completion` 重新运行完整验证。

### Task 1: 锁定必须保留的既有路由

**Files:**
- Modify: `tests/model_gateway/test_minimax_provider_routing.py:1-80`
- Modify: `tests/model_gateway/test_minimax_provider_routing.py:174-188`
- Test: `tests/model_gateway/test_minimax_provider_routing.py`

- [ ] **Step 1: 同步测试模块头部并增加公共路由 fixture**

把模块 docstring 更新为中文当前职责说明，补上作者和修改日期；保留现有编码声明。加入 `ModelRouter` import 和模块级 router fixture：

```python
# -*- coding: utf-8 -*-
"""
验证 config/models.yaml 的 provider、模型 profile、任务覆盖和玩家路由契约。

作者: Project contributors
修改日期: 2026-07-23
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from werewolf_agent.model_gateway.router import ModelRouter


@pytest.fixture(scope="module")
def model_router() -> ModelRouter:
    """按生产入口加载模型路由，验证最终解析行为。"""
    return ModelRouter.from_yaml(CONFIG_PATH)
```

只更新模块职责描述，不给其他旧测试批量翻译或重排格式。

- [ ] **Step 2: 添加 MiniMax 覆盖路由的 characterization test**

在现有 reflection 测试附近加入：

```python
@pytest.mark.parametrize(
    "task_type",
    [
        "judge_vote_calling",
        "judge_skill_guide",
        "judge_vote_tally",
        "judge_exile",
        "judge_sheriff",
    ],
)
def test_judge_task_overrides_remain_on_minimax(
    model_router: ModelRouter,
    task_type: str,
) -> None:
    """显式 judge 任务必须继续使用原生 MiniMax 覆盖。"""
    config, _fallback = model_router.resolve_config("judge", task_type)

    assert config.provider == "minimax"
    assert config.model == "MiniMax-M2.7"
```

同时扩展现有 secondary profile 测试，精确锁定批准设计中的采样参数：

```python
def test_ark_deepseek_v4_pro_secondary_model_profile_exists(
    yaml_config: dict,
) -> None:
    """secondary DeepSeek 通道必须保留独立且固定的采样参数。"""
    profile = yaml_config["model_profiles"]["ark_deepseek_v4_pro_secondary"]

    assert profile["provider"] == "openai"
    assert profile["model"] == "DeepSeek-V4-Pro"
    assert profile["temperature"] == 0.6
    assert profile["top_p"] == 0.95
```

- [ ] **Step 3: 运行 characterization test，确认旧配置本来就是绿色**

Run:

```powershell
if (-not (Test-Path -LiteralPath '.tmp')) {
    New-Item -ItemType Directory -Path '.tmp' | Out-Null
}
$env:TMP=(Resolve-Path '.tmp').Path
$env:TEMP=$env:TMP
python -m pytest -n 0 --basetemp .tmp\pytest-deepseek-characterization -o addopts='' tests/model_gateway/test_minimax_provider_routing.py::test_minimax_default_reflection_keeps_native_endpoint tests/model_gateway/test_minimax_provider_routing.py::test_judge_task_overrides_remain_on_minimax tests/model_gateway/test_minimax_provider_routing.py::test_ark_deepseek_v4_pro_secondary_model_profile_exists -q
```

Expected: PASS。这里是对非目标行为的基线刻画，不是新功能的 RED。

### Task 2: 写出 Kimi 替换的失败契约

**Files:**
- Modify: `tests/model_gateway/test_minimax_provider_routing.py:43-80`
- Modify: `tests/model_gateway/test_minimax_provider_routing.py:163-188`
- Test: `tests/model_gateway/test_minimax_provider_routing.py`

- [ ] **Step 1: 更新 default route 测试，使其要求主 DeepSeek profile**

将 `test_minimax_default_default_provider_is_openai` 的断言更新为：

```python
def test_minimax_default_default_provider_is_openai(yaml_config: dict) -> None:
    """minimax_default 的主流量必须通过 Ark DeepSeek-V4-Pro。"""
    config = yaml_config["llm_profiles"]["minimax_default"]["default"]

    assert config == {
        "provider": "openai",
        "model_profile": "ark_deepseek_v4_pro",
    }
```

- [ ] **Step 2: 增加活动引用与最终解析测试**

加入下列测试：

```python
def test_minimax_default_tasks_use_primary_deepseek_profile(
    yaml_config: dict,
) -> None:
    """原 Kimi 的任务覆盖必须统一指向主 DeepSeek profile。"""
    profile = yaml_config["llm_profiles"]["minimax_default"]

    assert yaml_config["llm_profiles"]["ark_deepseek"]["default"] == {
        "provider": "openai",
        "model_profile": "ark_deepseek_v4_pro",
    }

    assert profile["default"]["model_profile"] == "ark_deepseek_v4_pro"
    for task_type in ("speech", "deception", "night_action"):
        assert profile["tasks"][task_type] == {
            "provider": "openai",
            "model_profile": "ark_deepseek_v4_pro",
        }


@pytest.mark.parametrize("agent_id", ["p02", "p04", "p09", "p12"])
@pytest.mark.parametrize(
    "task_type",
    ["vote", "speech", "deception", "night_action"],
)
def test_replaced_players_resolve_to_deepseek_v4_pro(
    model_router: ModelRouter,
    agent_id: str,
    task_type: str,
) -> None:
    """原 Kimi 玩家在 default 和显式任务上必须解析为 DeepSeek。"""
    config, _fallback = model_router.resolve_config(agent_id, task_type)

    assert config.provider == "openai"
    assert config.model == "DeepSeek-V4-Pro"
```

`vote` 用来覆盖未显式列出的 default route；其他三项覆盖任务级路由。

- [ ] **Step 3: 用主 DeepSeek profile 测试替换旧 Kimi profile 测试**

删除 `test_ark_kimi_k26_model_profile_exists`，改为：

```python
def test_primary_deepseek_profile_replaces_kimi(yaml_config: dict) -> None:
    """活动配置必须删除 Kimi，并保留原主通道采样参数。"""
    profiles = yaml_config["model_profiles"]

    assert "ark_kimi_k26" not in profiles
    assert all(
        profile.get("model") != "Kimi-K2.6"
        for profile in profiles.values()
    )
    primary = profiles["ark_deepseek_v4_pro"]
    assert primary["provider"] == "openai"
    assert primary["model"] == "DeepSeek-V4-Pro"
    assert primary["temperature"] == 0.5
    assert primary["top_p"] == 0.9
    assert primary["timeout"] == 120
```

- [ ] **Step 4: 修正 fallback 测试的错误语义**

把 `test_minimax_default_fallback_uses_different_provider` 重命名为
`test_minimax_default_fallback_uses_different_model_profile`，并替换为：

```python
def test_minimax_default_fallback_uses_different_model_profile(
    yaml_config: dict,
) -> None:
    """主备均走 Ark/OpenAI，但必须使用不同模型 profile。"""
    route = yaml_config["llm_profiles"]["minimax_default"]
    primary = route["default"]
    fallback = route["fallback"]

    assert primary["provider"] == fallback["provider"] == "openai"
    assert primary["model_profile"] == "ark_deepseek_v4_pro"
    assert fallback["model_profile"] == "ark_deepseek_v4_flash"
```

- [ ] **Step 5: 运行 replacement tests，确认它们因旧 Kimi 配置而失败**

Run:

```powershell
$env:TMP=(Resolve-Path '.tmp').Path
$env:TEMP=$env:TMP
python -m pytest -n 0 --basetemp .tmp\pytest-deepseek-red -o addopts='' tests/model_gateway/test_minimax_provider_routing.py::test_minimax_default_default_provider_is_openai tests/model_gateway/test_minimax_provider_routing.py::test_minimax_default_tasks_use_primary_deepseek_profile tests/model_gateway/test_minimax_provider_routing.py::test_replaced_players_resolve_to_deepseek_v4_pro tests/model_gateway/test_minimax_provider_routing.py::test_primary_deepseek_profile_replaces_kimi tests/model_gateway/test_minimax_provider_routing.py::test_minimax_default_fallback_uses_different_model_profile -q
```

Expected: FAIL；失败应明确显示旧值 `ark_kimi_k26` / `Kimi-K2.6`。若失败来自 import、fixture 或拼写错误，先修正测试，直到只因旧生产配置而失败。

### Task 3: 最小修改生产路由使测试转绿

**Files:**
- Modify: `config/models.yaml:14-74`
- Modify: `config/models.yaml:156-188`
- Modify: `config/models.yaml:213-272`
- Test: `tests/model_gateway/test_minimax_provider_routing.py`

- [ ] **Step 1: 合并主模型 profile**

在 `model_profiles` 中：

1. 删除完整的 `ark_kimi_k26` block。
2. 保留 `ark_deepseek_v4_pro` 键，将其 `model` 改为精确的
   `DeepSeek-V4-Pro`。
3. 保持 `temperature: 0.5`、`top_p: 0.9`、`timeout: 120`、reasoning 和
   structured_output 不变。
4. 保持 `ark_deepseek_v4_pro_secondary` 的 `0.6/0.95` 不变。
5. 更新 Ark 顶部注释，使其准确描述“flash + 两个 DeepSeek-V4-Pro 采样通道”，
   不再列出 Kimi 或“四个 Ark model ID”。

目标主 profile：

```yaml
  ark_deepseek_v4_pro:
    provider: openai
    model: DeepSeek-V4-Pro
    temperature: 0.5
    top_p: 0.9
    timeout: 120
    reasoning:
      level: high
    allow_text_tool_fallback: true
    structured_output:
      mode: text_json
      fallback_modes: []
```

- [ ] **Step 2: 重接所有活动 Kimi 引用**

把以下五个 `model_profile: ark_kimi_k26` 改成
`model_profile: ark_deepseek_v4_pro`：

```yaml
llm_profiles:
  ark_deepseek:
    default:
      provider: openai
      model_profile: ark_deepseek_v4_pro

  minimax_default:
    default:
      provider: openai
      model_profile: ark_deepseek_v4_pro
    tasks:
      speech:
        provider: openai
        model_profile: ark_deepseek_v4_pro
      deception:
        provider: openai
        model_profile: ark_deepseek_v4_pro
      night_action:
        provider: openai
        model_profile: ark_deepseek_v4_pro
```

reflection 和五类 `judge_*` blocks 不得修改。

- [ ] **Step 3: 同步直接相关的当前态注释**

只更新本次触及区域中的当前态注释：

- `minimax_default` 主路由说明改为 Ark `DeepSeek-V4-Pro`。
- fallback 说明改为“同一 Ark/OpenAI provider 下不同 model profile”，不再声称
  是 different provider。

不修改 `PROGRESS.md`、旧规格、运行时历史注释或未跟踪探测脚本。

- [ ] **Step 4: 重跑 replacement tests，确认转绿**

Run: 与 Task 2 Step 5 完全相同的 pytest 命令，改用
`--basetemp .tmp\pytest-deepseek-green`。

Expected: PASS。

- [ ] **Step 5: 运行整个路由测试文件**

Run:

```powershell
$env:TMP=(Resolve-Path '.tmp').Path
$env:TEMP=$env:TMP
python -m pytest -n 0 --basetemp .tmp\pytest-deepseek-routing -o addopts='' tests/model_gateway/test_minimax_provider_routing.py -q
```

Expected: 全部 PASS，且无 YAML 加载或路由验证错误。

### Task 4: 回归验证并提交实现

**Files:**
- Verify: `config/models.yaml`
- Verify: `tests/model_gateway/test_minimax_provider_routing.py`
- Preserve: `scripts/probe_all_endpoints.py`

- [ ] **Step 1: 搜索活动文件中的 Kimi 残留**

Run:

```powershell
$kimiHits = rg -n "Kimi-K2\.6|ark_kimi_k26" config/models.yaml tests/model_gateway/test_minimax_provider_routing.py
if ($LASTEXITCODE -eq 0) {
    $kimiHits
    throw "活动路由仍包含 Kimi"
}
if ($LASTEXITCODE -ne 1) {
    throw "rg 搜索失败，exit code: $LASTEXITCODE"
}
```

Expected: 无输出，整段 PowerShell 命令成功结束。

- [ ] **Step 2: 运行相关路由回归**

Run:

```powershell
$env:TMP=(Resolve-Path '.tmp').Path
$env:TEMP=$env:TMP
python -m pytest -n 0 --basetemp .tmp\pytest-deepseek-related -o addopts='' tests/model_gateway/test_minimax_provider_routing.py tests/model_gateway/test_router.py tests/agents/test_model_router.py -q
```

Expected: 全部 PASS。

- [ ] **Step 3: 运行 model_gateway 全量回归**

Run:

```powershell
$env:TMP=(Resolve-Path '.tmp').Path
$env:TEMP=$env:TMP
python -m pytest -n 0 --basetemp .tmp\pytest-deepseek-model-gateway -o addopts='' tests/model_gateway -q
```

Expected: 命令完整结束，0 failed。超时或没有最终 pytest summary 不计作通过。

- [ ] **Step 4: 运行静态验证**

Run:

```powershell
python -m compileall -q werewolf_agent tests/model_gateway
git diff --check
```

Expected: 两条命令 exit code 0，无错误输出。

- [ ] **Step 5: 检查提交边界**

Run:

```powershell
git status --short
git diff -- config/models.yaml tests/model_gateway/test_minimax_provider_routing.py
```

Expected: 本功能只有这两个 tracked 文件发生变化；
`scripts/probe_all_endpoints.py` 仍显示为原有未跟踪文件，未被修改或暂存。

- [ ] **Step 6: 仅暂存并提交两个实施文件**

Run:

```powershell
git add -- config/models.yaml tests/model_gateway/test_minimax_provider_routing.py
git diff --cached --check
git diff --cached --name-status
git commit -m "fix: replace Kimi with DeepSeek V4 Pro"
```

Expected: staged/committed 文件只有 `config/models.yaml` 和
`tests/model_gateway/test_minimax_provider_routing.py`。

- [ ] **Step 7: 提交后做最终确认**

Run:

```powershell
git show --stat --oneline HEAD
git status --short --branch
```

Expected: 最新提交只包含两个实施文件；用户原有未跟踪文件保持存在且未被纳入提交。

## 完成后的运维说明

- 不停止当前游戏，也不声称当前进程已热更新。
- 新路由只在新建 `ModelRouter` 的进程/游戏中生效；当前游戏结束后需重启游戏进程。
- 完成报告必须给出精确 pytest 结果、实现 commit ID、实际改动文件和仍未跟踪的
  `scripts/probe_all_endpoints.py` 现状。
