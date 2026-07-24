# Speech Repair and Prompt Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让发言修复约束在同一次动作生命周期内单调累积，准确标记确定性 `safe_fallback` 未调用 provider，并让 Anthropic-compatible provider 对还原后的逻辑 system 文本执行最终提示词合同校验。

**Architecture:** 在 agents 层新增纯内存 `RepairConstraintState`，只聚合质量错误、语义原因码和稳定失败历史，并由 `player_action_flow` 显式把状态传给重试提示与语义审计。在 model gateway 层继续保留实际 cache text-block payload，仅把观察器的 `system_bytes` 改为按 block 顺序还原的逻辑文本；`provider_payload_bytes` 仍是实际请求体的 canonical JSON。两个责任边界分开实现、分开提交。

**Tech Stack:** Python 3.11+、Pydantic、pytest、PowerShell、现有 ModelRouter/provider fake client、CodeGraph。

---

## 实施约束

- DNS、网络连通性、endpoint 预检、熔断和退避不在本计划范围内。
- 不增加模型调用次数，不改变 `max_retries`、structured-output mode 推进或 provider 路由。
- `unsupported_public_claim`、说话人归属与否定关系校验继续 fail-closed。
- 不把被拒发言原文、完整 persona、私有身份或 provider 原始响应写入 trace、事件或日志。
- 修改 Python 文件时遵守仓库中文模块头与中文注释规范；如果职责描述不再准确，同步模块 docstring，并把非平凡修改日期更新为 `2026-07-24`。
- 工作区已有未跟踪文件属于用户；每次提交只 `git add` 本任务列出的路径。
- 每个任务开始前运行 `codegraph.cmd explore "<涉及符号>"` 确认当前调用路径没有漂移。

## Task 1: 还原 Anthropic-compatible 逻辑 system 文本

**Files:**

- Modify: `tests/model_gateway/test_final_prompt_observer.py`
- Modify: `werewolf_agent/model_gateway/providers/anthropic.py:217-243`
- Modify: `werewolf_agent/model_gateway/final_prompt_observer.py:26-38`
- Verify: `werewolf_agent/model_gateway/providers/minimax.py:107-123`
- Verify: `tests/model_gateway/test_anthropic_provider.py:186-262`
- Verify: `tests/model_gateway/test_providers.py:141-166`

- [ ] **Step 1: 用 CodeGraph 复核共享 helper 与两个 provider 的调用路径**

Run:

```powershell
codegraph.cmd explore "_system_bytes_for_observer and its AnthropicProvider MiniMaxProvider callers"
```

Expected: `_system_bytes_for_observer` 定义于 `providers/anthropic.py`，Anthropic 与 MiniMax 都在构造 `FinalPromptAssembly` 时调用它。

- [ ] **Step 2: 先把现有 Anthropic 观察器断言改成逻辑文本语义**

在 `test_anthropic_observer_reports_top_level_system_with_no_message_index` 中删除“list 则 canonical JSON”的分支，固定断言：

```python
    assert assemblies[0].system_bytes == b"rules\npersona-final"
    assert assemblies[0].provider_payload_bytes == canonical_provider_payload(
        client.payload
    )
```

并在该测试的局部 import 中加入 `canonical_provider_payload`。

同样把 `test_router_observes_each_real_provider_assembly_in_fallback_chain` 的 Anthropic 断言改为：

```python
    assert assemblies[1].system_bytes == b"rules\npersona-final"
    assert assemblies[1].provider_payload_bytes == canonical_provider_payload(
        fallback_client.payload
    )
```

- [ ] **Step 3: 添加特殊字符、多 block 与非法 block 的失败测试**

在 `tests/model_gateway/test_final_prompt_observer.py` 添加：

```python
def test_anthropic_observer_reconstructs_exact_logical_system_text() -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        canonical_provider_payload,
    )
    from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    system_prompt = '规则：\n“保持原话”\\路径\n人格：冷静分析'
    client = _Client()
    observed = []

    AnthropicProvider(
        api_key="k", base_url="https://example.test", http_client=client,
    ).generate(
        "user",
        ModelConfig(provider="anthropic", model="m"),
        system_prompt=system_prompt,
        final_prompt_observer=observed.append,
    )

    assert client.payload is not None
    assert observed[0].system_bytes == system_prompt.encode("utf-8")
    assert observed[0].provider_payload_bytes == canonical_provider_payload(
        client.payload
    )
    assert client.payload["system"] == [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]


def test_system_bytes_for_observer_concatenates_text_blocks_without_separator() -> None:
    from werewolf_agent.model_gateway.providers.anthropic import (
        _system_bytes_for_observer,
    )

    blocks = [
        {"type": "text", "text": "第一段\n"},
        {"type": "text", "text": '第二段\\"quoted"'},
    ]

    assert _system_bytes_for_observer(blocks) == (
        '第一段\n第二段\\"quoted"'.encode("utf-8")
    )


@pytest.mark.parametrize(
    "system_value",
    (
        [{"type": "text", "text": "valid"}, {"type": "image", "text": "bad"}],
        [{"type": "text", "text": "valid"}, {"type": "text", "text": 1}],
        [{"type": "text", "text": "valid"}, "bad"],
    ),
)
def test_system_bytes_for_observer_fails_closed_for_any_invalid_block(
    system_value: object,
) -> None:
    from werewolf_agent.model_gateway.providers.anthropic import (
        _system_bytes_for_observer,
    )

    assert _system_bytes_for_observer(system_value) == b""
```

测试中如需静态类型豁免，只在调用处使用精确的 `# type: ignore[arg-type]`，不要放宽生产签名为 `Any`。

- [ ] **Step 4: 运行测试，确认因 JSON 容器字节而失败**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-prompt-contract-red -o addopts='' tests/model_gateway/test_final_prompt_observer.py -q
```

Expected: 新的逻辑 `system_bytes` 断言失败；非法 mixed block 当前不会返回 `b""`。

- [ ] **Step 5: 最小实现逻辑 text-block 还原**

把 `werewolf_agent/model_gateway/providers/anthropic.py::_system_bytes_for_observer` 改为：

```python
def _system_bytes_for_observer(
    system_value: str | list[dict[str, Any]] | None,
) -> bytes:
    """还原 provider 最终 system 的逻辑文本，非法 block 整体失败关闭。"""
    if system_value is None:
        return b""
    if isinstance(system_value, str):
        return system_value.encode("utf-8")
    texts: list[str] = []
    for block in system_value:
        if (
            not isinstance(block, dict)
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            return b""
        texts.append(block["text"])
    return "".join(texts).encode("utf-8")
```

删除该 helper 已不再使用的 JSON 序列化逻辑；若模块内其他代码仍使用 `json`，保留 import。同步 `FinalPromptAssembly.system_bytes` 的 docstring，明确其为“按 provider block 顺序还原的逻辑 system UTF-8 字节”，而 `provider_payload_bytes` 是实际请求体 canonical JSON。

不要修改 `_wrap_system_prompt_for_cache()`、Anthropic/MiniMax 的 `payload["system"]` 赋值或 `canonical_provider_payload(payload)` 调用。

- [ ] **Step 6: 添加真实 persona 合同的 Anthropic/MiniMax provider 测试**

在 `tests/model_gateway/test_final_prompt_observer.py` 添加参数化测试，直接覆盖两个 provider：

```python
@pytest.mark.parametrize(
    ("provider_name", "provider_factory"),
    (
        (
            "anthropic",
            lambda client: __import__(
                "werewolf_agent.model_gateway.providers.anthropic",
                fromlist=["AnthropicProvider"],
            ).AnthropicProvider(
                api_key="k", base_url="https://example.test", http_client=client,
            ),
        ),
        (
            "minimax",
            lambda client: __import__(
                "werewolf_agent.model_gateway.providers.minimax",
                fromlist=["MiniMaxProvider"],
            ).MiniMaxProvider(
                api_key="k", base_url="https://example.test", http_client=client,
            ),
        ),
    ),
)
def test_anthropic_compatible_contract_validates_multiline_persona_before_http(
    provider_name: str,
    provider_factory: object,
) -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        FinalPromptContract,
        canonical_provider_payload,
        validate_final_prompt_contract,
    )
    from werewolf_agent.model_gateway.router import ModelConfig

    persona = '人格：\n- 冷静\n- 引用“公开记录”\\不臆测'
    system_prompt = f"规则区\n{persona}\n输出合同"
    contract = FinalPromptContract(
        contract_id="player-system",
        version="2026-07-24",
        required_sections=(("persona", persona.encode("utf-8")),),
    )
    client = _Client()
    observed = []
    provider = provider_factory(client)  # type: ignore[operator]

    provider.generate(
        "user",
        ModelConfig(provider=provider_name, model="m"),
        system_prompt=system_prompt,
        final_prompt_observer=lambda assembly: (
            observed.append(assembly),
            validate_final_prompt_contract(assembly, contract),
        )[-1],
    )

    assert client.post_calls == 1
    assert client.payload is not None
    assert observed[0].system_bytes == system_prompt.encode("utf-8")
    assert observed[0].provider_payload_bytes == canonical_provider_payload(
        client.payload
    )
```

为保持测试可读性，执行时可以把 lambda factory 替换为文件级小 helper/class；核心断言不得删减。另加一个真正缺少 persona 的参数化用例，断言两个 provider 均抛 `FinalPromptContractError` 且 `client.post_calls == 0`。

- [ ] **Step 7: 锁定实际 payload 的类型敏感深度相等与 cache 结构**

在上述成功测试中保存 observer 执行前后 `copy.deepcopy(client.payload)` 不可行，因为 observer 在 HTTP 前执行而 fake client 尚未收到 payload。改为在观察回调中只保存 assembly，HTTP 返回后断言：

```python
    expected_system = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]
    assert type(client.payload["system"]) is list
    assert client.payload["system"] == expected_system
    assert type(client.payload["system"][0]["text"]) is str
```

并保留 `provider_payload_bytes == canonical_provider_payload(client.payload)`。该组合同时锁定容器类型、字段值和 canonical payload 字节。

- [ ] **Step 8: 运行 provider 聚焦回归**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-prompt-contract-green -o addopts='' tests/model_gateway/test_final_prompt_observer.py tests/model_gateway/test_anthropic_provider.py tests/model_gateway/test_providers.py -q
```

Expected: PASS；persona 存在时每个 fake client 恰好 1 次 HTTP，缺失或 mixed-invalid block 时 0 次 HTTP；既有 cache-control 测试不变。

- [ ] **Step 9: 提交逻辑 system 边界修复**

```powershell
git diff --check
git add werewolf_agent/model_gateway/providers/anthropic.py werewolf_agent/model_gateway/final_prompt_observer.py tests/model_gateway/test_final_prompt_observer.py
git commit -m "fix: validate reconstructed provider system text"
```

如果执行中确实需要修改 `providers/minimax.py`、`test_anthropic_provider.py` 或 `test_providers.py`，仅在有对应 diff 时加入同一提交；不要为“触碰文件”制造无意义改动。

## Task 2: 用单元测试定义累计修复状态

**Files:**

- Create: `werewolf_agent/agents/player_repair_state.py`
- Modify: `werewolf_agent/agents/player_quality_retries.py:20-80`
- Modify: `tests/agents/test_player_retry.py`

- [ ] **Step 1: 用 CodeGraph 复核重试提示与 `RetryInfo` 结构**

Run:

```powershell
codegraph.cmd explore "build_speech_quality_retry RetryInfo semantic_repair_correction_hint"
```

Expected: 质量提示来自 `player_quality_retries.py`，语义原因码中文映射仍由 `semantic_repair_audit.py` 单独持有。

- [ ] **Step 2: 先写累计状态的单元测试**

在 `tests/agents/test_player_retry.py` 添加以下行为测试（可用小 helper 构造 `SpeechPlayerAction`）：

```python
def test_repair_constraint_state_accumulates_quality_and_semantic_constraints() -> None:
    from werewolf_agent.agents.player_repair_state import RepairConstraintState
    from werewolf_agent.agents.schemas import SpeechPlayerAction

    source = SpeechPlayerAction(
        target_id="p05", speech="我怀疑p05。", reason="公开发言", confidence=0.6,
    )
    state = RepairConstraintState()
    assert state.semantic_repair_started is False
    state.record_speech_quality(source, "缺少身份立场")
    state.record_semantic_rejection(("unsupported_public_claim",))

    retry = state.augment_retry_info(
        RetryInfo(
            attempt=2,
            max_retries=3,
            error_code="semantic_claim_retention",
            error_message="语义修复未通过",
            reason_codes=["unsupported_public_claim"],
            correction_hint="删除或改写缺少公开证据支持的事实声明。",
        ),
        rejected_speech="p07声称自己是猎人，我怀疑p05。",
    )

    assert state.source_action is source
    assert state.semantic_repair_started is True
    assert state.fact_policy == "verified_claims_only"
    assert state.failure_history == (
        "speech_quality", "semantic_claim_retention",
    )
    assert retry.error_code == "semantic_claim_retention"
    assert retry.reason_codes == ["unsupported_public_claim"]
    assert "身份立场" in retry.correction_hint
    assert "删除或改写缺少公开证据支持的事实声明" in retry.correction_hint
    assert "不得新增未经公开记录支持的事实声明" in retry.correction_hint
    assert "我倾向" in retry.correction_hint
    assert "我怀疑" in retry.correction_hint


def test_repair_constraint_state_is_monotonic_and_deduplicates_constraints() -> None:
    from werewolf_agent.agents.player_repair_state import RepairConstraintState
    from werewolf_agent.agents.schemas import SpeechPlayerAction

    first = SpeechPlayerAction(target_id="p05", speech="first", reason="r")
    later = SpeechPlayerAction(target_id="p05", speech="later", reason="r")
    state = RepairConstraintState()
    state.record_speech_quality(first, "缺少身份立场")
    state.record_semantic_rejection((
        "negation_changed", "unsupported_public_claim", "negation_changed",
    ))
    state.record_speech_quality(later, "缺少身份立场")

    assert state.source_action is first
    assert state.semantic_repair_started is True
    assert state.fact_policy == "verified_claims_only"
    assert state.quality_errors == ("缺少身份立场",)
    assert state.semantic_reason_codes == (
        "unsupported_public_claim", "negation_changed",
    )
    assert state.failure_history == (
        "speech_quality", "semantic_claim_retention", "speech_quality",
    )
```

第二个测试的语义原因码期望顺序必须与 `_ordered_reason_codes()` 的既有稳定顺序一致，而不是输入顺序。

再加隐私测试：`repr(state)`、`failure_history`、`quality_errors` 和最终 audit-ready 数据中不出现 `source.speech`；允许 `source_action` 只在内存中通过显式属性访问，不为其提供序列化方法。

- [ ] **Step 3: 运行单元测试，确认新模块尚不存在**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-repair-state-red -o addopts='' tests/agents/test_player_retry.py -q
```

Expected: FAIL，原因是 `player_repair_state` 尚不存在。

- [ ] **Step 4: 把质量提示函数提升为窄公共接口**

在 `player_quality_retries.py` 将 `_speech_quality_correction_hint` 重命名为 `speech_quality_correction_hint`，`build_speech_quality_retry` 改为调用新名字，并把它加入 `__all__`。保持所有现有中文提示和 120 字符回显上限不变。

先运行现有兼容测试：

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-quality-helper -o addopts='' tests/agents/test_player_agent.py -k "speech_quality_retry_hint" -q
```

Expected: PASS。

- [ ] **Step 5: 实现纯内存 `RepairConstraintState`**

创建 `werewolf_agent/agents/player_repair_state.py`，使用仓库要求的中文模块头。实现以下窄接口：

```python
@dataclass(repr=False)
class RepairConstraintState:
    """累计一次玩家动作生命周期内仍须满足的修复约束。"""

    _source_action: PlayerAction | None = None
    _quality_errors: list[str] = field(default_factory=list)
    _semantic_reason_codes: list[str] = field(default_factory=list)
    _failure_history: list[str] = field(default_factory=list)
    _fact_policy: Literal["normal", "verified_claims_only"] = "normal"
    _semantic_repair_started: bool = False

    @property
    def source_action(self) -> PlayerAction | None: ...

    @property
    def quality_errors(self) -> tuple[str, ...]: ...

    @property
    def semantic_reason_codes(self) -> tuple[str, ...]: ...

    @property
    def failure_history(self) -> tuple[str, ...]: ...

    @property
    def fact_policy(self) -> Literal["normal", "verified_claims_only"]: ...

    @property
    def semantic_repair_started(self) -> bool: ...

    def record_speech_quality(
        self, source: PlayerAction, speech_quality_err: str,
    ) -> None: ...

    def record_semantic_rejection(self, reason_codes: Iterable[str]) -> None: ...

    def augment_retry_info(
        self,
        latest_retry: RetryInfo,
        *,
        rejected_speech: str = "",
    ) -> RetryInfo: ...
```

实现规则：

- `record_speech_quality` 只在 `_source_action is None` 时保存首次动作；质量错误按完整错误字符串去重，但每次失败都向 `_failure_history` 追加 `speech_quality`。
- `record_semantic_rejection` 首先把 `_semantic_repair_started` 单调设置为 `True`，再使用语义模块现有稳定排序函数的公开等价接口；不要在新模块复制原因码中文映射。若包含 `unsupported_public_claim`，只允许把策略从 `normal` 推进到 `verified_claims_only`。
- `semantic_repair_started` 初始为 `False`，第一次语义拒绝后永久为 `True`；后续质量、parse/schema/illegal 或成功校验都不得把它重置。单元测试必须覆盖 `False -> True` 以及再次记录质量失败后仍为 `True`。
- `augment_retry_info` 保留 `latest_retry` 的 `attempt`、`max_retries`、`error_code`、`error_message`、`early_exit_reason` 与 `failure_category`，将其 `reason_codes` 与累计语义原因码按稳定顺序合并，并把累计约束追加到最新失败的专用提示中。
- 当最新失败是 `speech_quality` 或 `semantic_claim_retention` 时，累计状态负责重建该类提示，避免把同一质量/语义提示拼两次；当最新失败是 parse/schema/illegal/vote-quality/provider 等其他类别时，保留其专用 `correction_hint`，再追加累计质量与语义约束。
- 若尚未保存 `source_action`，`augment_retry_info` 原样返回 `latest_retry`；因此接入所有 retry 分支不会改变修复开始前的行为。
- 多个质量错误中仅最后一个提示接收 `rejected_speech`，避免重复回显；状态对象本身不保存本轮被拒原文。
- `verified_claims_only` 固定提示必须同时包含“不得新增未经公开记录支持的事实声明”和“可用我倾向/我怀疑/目前不能确定表达主观判断”。
- `repr=False`，不新增 `model_dump`/`to_dict`/JSON 序列化接口。

如果需要稳定排序，不要导入带下划线私有函数；在 `semantic_repair_audit.py` 新增一个窄公共 helper，例如 `ordered_semantic_repair_reason_codes()`，内部复用现有实现并加入 `__all__`。

- [ ] **Step 6: 运行累计状态与现有语义提示测试**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-repair-state-green -o addopts='' tests/agents/test_player_retry.py tests/agents/test_semantic_repair_invariants.py -q
```

Expected: PASS；现有语义原因码顺序与中文文案保持不变。

- [ ] **Step 7: 提交累计状态的独立单元**

```powershell
git diff --check
git add werewolf_agent/agents/player_repair_state.py werewolf_agent/agents/player_quality_retries.py werewolf_agent/agents/semantic_repair_audit.py tests/agents/test_player_retry.py
git commit -m "feat: accumulate speech repair constraints"
```

## Task 3: 把累计状态接入真实动作流和语义审计

**Files:**

- Modify: `werewolf_agent/agents/player_action_flow.py:81-709`
- Modify: `werewolf_agent/agents/semantic_repair_audit.py:145-225`
- Modify: `werewolf_agent/evaluation/acceptance_terminal_semantic_metrics.py:34-48`
- Modify: `tests/agents/test_player_agent.py:4105-4490`
- Modify: `tests/agents/test_player_retry.py:132-330`
- Modify: `tests/evaluation/test_semantic_acceptance_invariants.py:244-340`
- Modify: `tests/evaluation/test_report_acceptance_metrics.py`

- [ ] **Step 1: 先写三轮收敛的动作流测试**

在 `tests/agents/test_player_agent.py` 复用 `_SequenceJsonProvider` 与 `_semantic_speech_payload`，新增：

```python
def test_speech_quality_semantic_rejection_then_success_keeps_all_constraints(
    monkeypatch,
) -> None:
    from unittest.mock import patch

    provider = _SequenceJsonProvider([
        _semantic_speech_payload("我怀疑p05。"),
        _semantic_speech_payload("p07声称自己是猎人，我怀疑p05。"),
        _semantic_speech_payload(
            "我是好人，我怀疑p05，依据其发言矛盾，我倾向投p05。"
        ),
    ])
    router = ModelRouter(
        model_profiles={}, llm_profiles={},
        player_assignments={"p01": "default"}, providers={"mock": provider},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
    context = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH, phase="day", day_number=2,
        own_role="villager", legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
    )
    monkeypatch.setattr(
        "werewolf_agent.agents.player_action_flow.time.sleep", lambda _delay: None
    )

    with patch.object(agent, "_speech_quality_error", side_effect=[
        "缺少身份立场", None,
    ]):
        action, retry = agent.act(context)

    assert provider.calls == 3
    final_prompt = provider.prompts[-1]
    assert "身份立场" in final_prompt
    assert "删除或改写缺少公开证据支持的事实声明" in final_prompt
    assert "不得新增未经公开记录支持的事实声明" in final_prompt
    assert "我倾向" in final_prompt and "我怀疑" in final_prompt
    assert retry.reason_codes == ["unsupported_public_claim"]
    assert action.trace is not None
    assert action.trace.semantic_repair_audit is not None
    assert action.trace.semantic_repair_audit["repair_failure_history"] == [
        "speech_quality", "semantic_claim_retention",
    ]
```

这里 `_speech_quality_error` 只会在第 1 和第 3 个输出上调用；第 2 个输出必须被语义门提前拒绝。

再添加一个“修复开始后插入非语义失败”的测试，防止 parse/schema/illegal 分支覆盖累计提示：

```python
def test_parse_failure_after_speech_repair_keeps_quality_constraint(
    monkeypatch,
) -> None:
    from unittest.mock import patch

    provider = _SequenceJsonProvider([
        _semantic_speech_payload("我怀疑p05。"),
        "not-json",
        _semantic_speech_payload(
            "我是好人，我怀疑p05，依据其发言矛盾，我倾向投p05。"
        ),
    ])
    router = ModelRouter(
        model_profiles={}, llm_profiles={},
        player_assignments={"p01": "default"}, providers={"mock": provider},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
    context = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH, phase="day", day_number=2,
        own_role="villager", legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
    )
    monkeypatch.setattr(
        "werewolf_agent.agents.player_action_flow.time.sleep", lambda _delay: None
    )

    with patch.object(
        agent, "_speech_quality_error", side_effect=["缺少身份立场", None]
    ):
        action, retry = agent.act(context)

    assert provider.calls == 3
    assert "身份立场" in provider.prompts[-1]
    assert retry.error_code == "parse_error"
    assert action.trace is not None
    assert action.trace.semantic_repair_audit is not None
    assert action.trace.semantic_repair_audit["repair_failure_history"] == [
        "speech_quality",
    ]
```

如果实际 parser 对 `not-json` 使用更具体的稳定错误码，以现有断言为准调整 `retry.error_code`；关键验收是第三轮 prompt 仍有“身份立场”，且累计状态没有被非语义 `RetryInfo` 覆盖。

- [ ] **Step 2: 添加终退历史与隐私测试**

扩展 `test_semantic_repair_terminal_log_trace_privacy`，断言：

```python
    audit = action.trace.semantic_repair_audit
    assert audit is not None
    assert audit["repair_failure_history"] == [
        "speech_quality", "semantic_claim_retention",
    ]
    serialized_audit = json.dumps(audit, ensure_ascii=False)
    assert rejected_speech not in serialized_audit
    assert private_sentinel not in serialized_audit
```

再扩展现有成功 audit 精确字典断言，在 `fallback_kind` 前加入：

```python
        "repair_failure_history": ["speech_quality"],
```

直接成功且未进入修复的既有测试继续断言 `semantic_repair_audit is None`。

新增真正三次失败后发布 terminal fallback 的端到端测试：

```python
def test_three_failed_speech_repairs_publish_only_sanitized_terminal_fallback(
    monkeypatch,
) -> None:
    from unittest.mock import patch

    unsupported = "p07声称自己是猎人"
    provider = _SequenceJsonProvider([
        _semantic_speech_payload("我怀疑p05。"),
        _semantic_speech_payload(f"{unsupported}，我怀疑p05。"),
        _semantic_speech_payload("我仍怀疑p05。"),
    ])
    router = ModelRouter(
        model_profiles={}, llm_profiles={},
        player_assignments={"p01": "default"}, providers={"mock": provider},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
    context = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH, phase="day", day_number=2,
        own_role="villager", legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"], public_claim_ledger=[],
    )
    monkeypatch.setattr(
        "werewolf_agent.agents.player_action_flow.time.sleep", lambda _delay: None
    )

    with patch.object(
        agent,
        "_speech_quality_error",
        side_effect=["缺少身份立场", "缺少身份立场"],
    ):
        action, _retry = agent.act(context)

    assert isinstance(action, FallbackAction)
    assert provider.calls == 3
    assert unsupported not in action.speech
    assert action.trace is not None
    audit = action.trace.semantic_repair_audit
    assert audit is not None
    assert audit["success"] is False
    assert audit["introduced_claim_count"] == 0
    assert audit["unsupported_public_claim_count"] == 0
    assert audit["repair_failure_history"] == [
        "speech_quality", "semantic_claim_retention", "speech_quality",
    ]
    assert action.trace.raw_text == ""
    assert action.trace.parsed_action is None
```

该测试必须使用现有 `build_task_terminal_fallback()` 与 `preserve_verified_claim_in_fallback()` 路径，不允许为测试新增专用 fallback。

- [ ] **Step 3: 先写 audit builder 与验收配对测试**

在 `tests/agents/test_player_retry.py` 中新增命名明确的 `test_semantic_repair_audit_records_stable_failure_history`：

```python
    audit = build_semantic_repair_audit(
        context,
        source,
        final,
        success=True,
        repair_failure_history=("speech_quality", "semantic_claim_retention"),
    )
    assert audit["repair_failure_history"] == [
        "speech_quality", "semantic_claim_retention",
    ]
```

在 `tests/evaluation/test_semantic_acceptance_invariants.py::test_paired_semantic_audit_reconciliation_compares_every_decisive_field` 的 `nested_updates` 参数中加入：

```python
        {"repair_failure_history": ["speech_quality"]},
```

并增加独立命名用例 `test_paired_semantic_audit_repair_history_is_type_sensitive`：standalone 为 `repair_failure_history=["speech_quality"]`、nested 为 tuple 或字符串时，语义指标必须 fail-closed。JSON 输入只允许 list；tuple 用于锁定 Python 内存输入也不能因宽松相等而通过。

- [ ] **Step 4: 运行测试，确认当前流程丢失早期约束/历史**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-repair-flow-red -o addopts='' tests/agents/test_player_agent.py -k "speech_quality_semantic_rejection_then_success or parse_failure_after_speech_repair or three_failed_speech_repairs or semantic_repair_terminal_log_trace_privacy or speech_quality_retry_records_real_semantic_repair_audit" -q
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-repair-audit-red -o addopts='' tests/agents/test_player_retry.py::test_semantic_repair_audit_records_stable_failure_history tests/evaluation/test_semantic_acceptance_invariants.py::test_paired_semantic_audit_repair_history_is_type_sensitive tests/evaluation/test_semantic_acceptance_invariants.py::test_paired_semantic_audit_reconciliation_compares_every_decisive_field -q
```

Expected: 两条命令均 FAIL。动作流组显示第三轮 prompt 丢失早期质量约束、三失败路径 audit 缺失完整 history；audit/acceptance 组显示 builder 尚不接受该参数或 reconciliation 尚未比较新字段。

- [ ] **Step 5: 扩展语义 audit API，保持旧调用兼容**

把 `build_semantic_repair_audit` 改为：

```python
def build_semantic_repair_audit(
    context: AgentContext,
    source: PlayerAction,
    final: PlayerAction | FallbackAction,
    *,
    success: bool,
    generic_template_used: bool = False,
    repair_failure_history: Iterable[str] = (),
) -> dict[str, Any]:
```

返回字典增加：

```python
        "repair_failure_history": [
            code for code in repair_failure_history
            if code in {"speech_quality", "semantic_claim_retention"}
        ],
```

扩展 `validate_semantic_repair`，加入同名 kw-only 默认参数，并原样传给 builder。默认空元组保证所有旧直接调用兼容。

不要把任意 `RetryInfo.error_message`、reason 文本或 speech 写进 audit。

- [ ] **Step 6: 在 `player_action_flow` 用状态替换单一 source 变量**

实施要点：

```python
    repair_state = RepairConstraintState()
```

- 所有 `semantic_repair_source is not None` 改为显式读取 `repair_state.source_action`。
- 语义校验调用：

```python
    source_action = repair_state.source_action
    if source_action is not None:
        semantic_validation = validate_semantic_repair(
            context,
            source_action,
            action,
            repair_failure_history=repair_state.failure_history,
        )
```

- 语义拒绝时先 `repair_state.record_semantic_rejection(reason_codes)`，构造现有语义专用 `RetryInfo`，再调用 `repair_state.augment_retry_info(retry)`；保留现有 warning、repeat-signature 和 `reject_latest_output()` 顺序。
- 发言质量失败时先 `repair_state.record_speech_quality(action, speech_quality_err)`，先调用现有 `build_speech_quality_retry(...)`，再调用 `repair_state.augment_retry_info(retry, rejected_speech=action.speech)`；保留“不做 repeat-signature 短路”的现有逻辑。
- terminal fallback 中 `preserve_verified_claim_in_fallback` 使用首次 source，`build_semantic_repair_audit(..., repair_failure_history=repair_state.failure_history)`。
- 成功语义 audit 必须包含成功前的完整历史；如果中间语义通过但随后质量失败，该中间 audit 不得成为最终 audit。
- 在函数内增加窄 helper，例如 `_with_repair_constraints(retry_info, *, rejected_speech="")`，并让所有会产生下一轮 prompt 或写入 terminal trace 的 retry 分支在创建其专用 `RetryInfo` 后统一经过它：empty response、parse error、schema validation、missing tool call、illegal action、vote quality、speech quality、semantic rejection，以及现有 provider/structured-output 终退分支。状态尚未激活时 helper 是 no-op；激活后保留最新失败字段并追加全部累计约束。
- parse/schema/illegal/vote-quality 的校验与专用提示内容保持不变；改变的只是它们在 repair 已开始后不再覆盖早期约束。

- [ ] **Step 7: 让 acceptance reconciliation 比较新稳定字段**

在 `acceptance_terminal_semantic_metrics.py::_SEMANTIC_AUDIT_FIELDS` 加入 `repair_failure_history`。不要把它加入旧数据的“必须存在”条件：双方都缺失时仍兼容旧 trace；一侧缺失、类型不同或值不同必须 fail-closed。

在 `tests/evaluation/test_report_acceptance_metrics.py` 用 `_semantic_pair` 构造带相同 history 的 standalone/nested 对，断言既有成功率、安全率和 supported 标志不变；再构造两侧 history 不同的对，断言 `semantic_repair_metrics_supported is False`。

- [ ] **Step 8: 运行 agent 与 acceptance 聚焦回归**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-repair-flow-green -o addopts='' tests/agents/test_player_retry.py tests/agents/test_player_agent.py tests/agents/test_semantic_repair_invariants.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py -q
```

Expected: PASS；三轮收敛、成功/终退历史、隐私与 V1/V2 配对一致性全部通过。

- [ ] **Step 9: 提交动作流与 audit 接线**

```powershell
git diff --check
git add werewolf_agent/agents/player_action_flow.py werewolf_agent/agents/semantic_repair_audit.py werewolf_agent/evaluation/acceptance_terminal_semantic_metrics.py tests/agents/test_player_agent.py tests/agents/test_player_retry.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py
git commit -m "fix: preserve cumulative speech repair constraints"
```

## Task 4: 明确 terminal safe fallback 未调用 provider

**Files:**

- Modify: `werewolf_agent/model_gateway/generation_attempt_context.py:73-98`
- Modify: `tests/model_gateway/test_generation_attempt_context.py:48-73`
- Modify: `scripts/run_real_game_reports.py:150-173,197-268`
- Modify: `tests/scripts/test_run_real_game.py:810-1039`
- Verify: `werewolf_agent/runtime/decision_outcomes.py:318-329`

- [ ] **Step 1: 先写有/无 reasoning 两类 terminal fallback 测试**

把 `test_terminal_fallback_preserves_structured_route_failure_reason` 扩展为参数化测试，覆盖：

```python
@pytest.mark.parametrize(
    ("requested", "expected_status", "expected_evidence"),
    (
        (ReasoningLevel.HIGH, ReasoningStatus.FALLBACK_DISABLED,
         EvidenceKind.FALLBACK_DISABLED),
        (ReasoningLevel.NONE, ReasoningStatus.NOT_REQUESTED, EvidenceKind.NONE),
    ),
)
```

每种情况断言：

```python
    terminal = context.attempts[-1]
    assert terminal.route_kind is RouteKind.SAFE_FALLBACK
    assert terminal.provider_attempted is False
    assert terminal.attempt_outcome is AttemptOutcome.FAILURE
    assert terminal.root_cause is RootCause.INVALID_OUTPUT
    assert terminal.normalized_reasoning_status is expected_status
    assert terminal.evidence_kind is expected_evidence
    assert terminal.reasoning_token_count == 0
    assert context.attempts[-2].provider_attempted is True
```

构造 `ReasoningLevel.NONE` 的源 attempt 时必须使用 `NOT_REQUESTED`/`NONE`，满足 `AttemptExecutionRecord` 不变量。

- [ ] **Step 2: 先写报告投影和文本展示测试**

更新 `test_reasoning_evidence_summary_is_allowlisted_and_has_exact_denominators` 的字段集合，加入 `provider_attempted`，并断言普通尝试为 `True`。

再添加 legacy/malformed mapping 测试，锁定与 `runtime/decision_outcomes.py::_provider_attempted` 相同的边界：

```python
def test_reasoning_summary_defaults_missing_legacy_provider_attempted_to_true() -> None:
    from scripts.run_real_game_reports import _reasoning_evidence_summary

    legacy = {
        "opaque_request_id": "game_legacy1234",
        "ordinal": 1,
        "provider": "openai",
        "model": "m",
        "requested_reasoning_level": "none",
        "normalized_reasoning_status": "not_requested",
        "reasoning_token_count": 0,
        "evidence_kind": "none",
        "route_kind": "primary",
        "root_cause": "none",
        "attempt_outcome": "attempt_success",
    }

    summary = _reasoning_evidence_summary([], action_attempts=(legacy,))

    assert summary["attempts"][0]["provider_attempted"] is True


@pytest.mark.parametrize("malformed", (0, 1, "false", None))
def test_reasoning_summary_rejects_non_boolean_provider_attempted(
    malformed: object,
) -> None:
    from scripts.run_real_game_reports import _reasoning_evidence_summary

    attempt = {
        "opaque_request_id": "game_legacy1234",
        "ordinal": 1,
        "provider": "openai",
        "model": "m",
        "requested_reasoning_level": "none",
        "normalized_reasoning_status": "not_requested",
        "reasoning_token_count": 0,
        "evidence_kind": "none",
        "route_kind": "primary",
        "root_cause": "none",
        "attempt_outcome": "attempt_success",
        "provider_attempted": malformed,
    }

    with pytest.raises(TypeError, match="provider_attempted must be a bool"):
        _reasoning_evidence_summary([], action_attempts=(attempt,))
```

新增一个 summary 用例，传入 `GenerationAttemptContext.append_terminal_fallback()` 生成的链，断言：

```python
    terminal = summary["attempts"][-1]
    assert terminal["route"] == "safe_fallback"
    assert terminal["provider_attempted"] is False
    assert summary["requested_denominator"] == 2
```

这里 denominator 保持 2：终止边界仍记录“请求的 reasoning 在安全兜底中被禁用”，只是不把它说成真实 provider 调用。

扩展 `print_usage_stats` 测试，断言输出行包含 `provider_attempted=false`（或等价的稳定 `call=skipped`；二选一后全仓统一）。计划默认使用 `provider_attempted=false`，与结构化字段同名。

- [ ] **Step 3: 运行测试，确认当前 terminal 继承了 `True`**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-safe-fallback-red -o addopts='' tests/model_gateway/test_generation_attempt_context.py tests/scripts/test_run_real_game.py -k "terminal_fallback or reasoning_evidence_summary or reasoning_summary or usage_report" -q
```

Expected: FAIL，terminal `provider_attempted` 当前仍为 `True`，报告也未投影/打印该字段；legacy/malformed mapping 测试同时锁定缺失默认 `True` 与非 bool 拒绝语义。

- [ ] **Step 4: 在 terminal clone 中显式覆盖事实字段**

在 `GenerationAttemptContext.append_terminal_fallback()` 的 `replace(...)` 中增加：

```python
            provider_attempted=False,
```

保留 provider/model 字段用于归因，保留两套 reasoning 状态分支，不改 attempt ordinal、root cause 或 outcome。

- [ ] **Step 5: 投影并打印 `provider_attempted`**

在 `_reasoning_evidence_summary()` 内新增与 runtime 相同的窄 helper：

```python
    def provider_attempted(attempt: Any) -> bool:
        missing = object()
        value = (
            attempt.get("provider_attempted", missing)
            if isinstance(attempt, dict)
            else getattr(attempt, "provider_attempted", missing)
        )
        if value is missing:
            return True
        if type(value) is not bool:
            raise TypeError("provider_attempted must be a bool")
        return value
```

attempt dict 中使用精确结果，不做 truthiness 强制转换：

```python
                "provider_attempted": provider_attempted(attempt),
```

在 `print_usage_stats()` attempt 行加入：

```python
            f"provider_attempted={str(attempt['provider_attempted']).lower()} "
```

不要过滤掉 safe fallback，不改 `requested_denominator`、`attempt_count` 或 `reasoning_fallback_disabled_count`。`runtime/decision_outcomes.py` 已按 `provider_attempted` 计算真实 provider 次数，只做回归验证，不重复实现。

- [ ] **Step 6: 运行 execution/report 回归**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-safe-fallback-green -o addopts='' tests/model_gateway/test_generation_attempt_context.py tests/model_gateway/test_execution_records.py tests/agents/test_decision_trace_v2.py tests/scripts/test_run_real_game.py -q
```

Expected: PASS；terminal 为 `provider_attempted=False`，真实尝试仍为 `True`，reasoning 分母和既有终退计数不变。

- [ ] **Step 7: 提交 terminal reporting 修复**

```powershell
git diff --check
git add werewolf_agent/model_gateway/generation_attempt_context.py scripts/run_real_game_reports.py tests/model_gateway/test_generation_attempt_context.py tests/scripts/test_run_real_game.py
git commit -m "fix: mark deterministic fallback as provider skipped"
```

## Task 5: 完整验证与交付证据

**Files:**

- Verify only: all files changed in Tasks 1-4

- [ ] **Step 1: 运行本设计的全量聚焦测试集合**

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-speech-prompt-focused -o addopts='' tests/agents/test_player_retry.py tests/agents/test_player_agent.py tests/agents/test_semantic_repair_invariants.py tests/model_gateway/test_generation_attempt_context.py tests/model_gateway/test_execution_records.py tests/model_gateway/test_final_prompt_observer.py tests/model_gateway/test_anthropic_provider.py tests/model_gateway/test_providers.py tests/agents/test_decision_trace_v2.py tests/scripts/test_run_real_game.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py -q
```

Expected: 全部 PASS。记录精确 pass count；超时只能报告“未验证”，不能视为通过。

- [ ] **Step 2: 运行 acceptance integrity 回归**

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-speech-prompt-acceptance -o addopts='' tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_final_acceptance_integrity.py tests/evaluation/test_report_acceptance_metrics.py -q
```

Expected: 全部 PASS；现有公开证据 fail-closed 和 reasoning 计数合同没有回归。

- [ ] **Step 2a: 运行审计阈值 evaluator 测试**

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'; python -m pytest -n 0 --basetemp C:\tmp\wofkill-speech-prompt-thresholds -o addopts='' tests/scripts/test_evaluate_audit_closure_thresholds.py -q
```

Expected: PASS；既有 `semantic_repair_success_rate >= 0.95`、公开证据/说话人归属/否定关系为 `1.0`、generic template 为 `0`、critical reasoning fallback-disabled 为 `0` 的阈值合同保持不变。

- [ ] **Step 3: 编译与静态 diff 检查**

```powershell
python -m compileall -q werewolf_agent tests scripts
git diff --check
git status --short
```

Expected: compileall 与 diff check 均为 exit 0；status 只包含用户原有未跟踪文件，不包含本任务未提交改动。

- [ ] **Step 4: 复核提交边界**

```powershell
git log --oneline -5
git show --stat --oneline HEAD
git show --stat --oneline HEAD~1
git show --stat --oneline HEAD~2
git show --stat --oneline HEAD~3
```

Expected: 四个小提交分别对应逻辑 system、累计状态单元、动作流/audit 接线、safe fallback reporting；没有 DNS 或无关文件。

- [ ] **Step 5: 可选真实运行验收（仅当用户要求或已有可用 endpoint）**

沿用仓库现有真实游戏启动方式运行一局，并只检查本设计范围：

- persona 实际存在时不再出现 `missing required sections: persona`；
- 已知 `speech_quality -> unsupported_public_claim -> speech_quality` 振荡不再因早期约束丢失而发生；
- terminal 输出明确打印 `provider_attempted=false`；
- DNS/网络错误排除在验收结论外。

真实运行若因外部网络失败，记录为“未验证”，不回退去修改 DNS、重试预算或 endpoint 策略。

## 完成定义

- 逻辑 `system_bytes` 精确等于 text-block 文本顺序拼接后的 UTF-8；实际 cache payload 字节不变。
- Anthropic 与 MiniMax 对多行/特殊字符 persona 均正确校验，真实缺失仍在 HTTP 前失败。
- 修复提示在三轮内保留全部累计质量与语义约束，`verified_claims_only` 单调不可逆。
- 成功与 terminal fallback audit 都只记录稳定 `repair_failure_history`，没有原文或私密信息。
- `safe_fallback.provider_attempted is False`，但既有 reasoning 状态、attempt 数与 fallback-disabled 计数语义不变。
- 所有已完成测试报告精确 pass count；任何超时或外部网络阻断均明确标记为未验证。
