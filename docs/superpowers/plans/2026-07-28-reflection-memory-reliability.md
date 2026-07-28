# Reflection Memory Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让完整局的反思生成、Schema 校验、事实与经验核验、持久化回读和下一局恢复全部达到 12/12。

**Architecture:** 为反思增加与讨论摘要相同级别的窄结构化输出合同和共享尝试上下文，以 native tool 为 MiniMax 主模式并在解析失败时按显式策略降级。反思事务先收集并核验全部 12 个候选，再原子持久化和回读；`partial` 保留诊断但不能更新 latest 快照或通过验收。

**Tech Stack:** Python 3.12、Pydantic v2、MiniMax Anthropic-compatible tool call、ModelRouter、GenerationAttemptContext、PostgreSQL/内存 repository、pytest。

## Global Constraints

- 设计依据：`docs/superpowers/specs/2026-07-28-reflection-memory-reliability-design.md`。
- 完整局反思验收必须是生成、核验、持久化和恢复四项 12/12；`partial` 永远不通过。
- 反思失败不得改变已经完成的游戏胜负或终局持久化。
- 不保存 provider 原文、Pydantic input、Prompt 或私密身份文本；只允许稳定失败码、字段路径和错误类型。
- 所有项目命令使用 `conda run -n wofkill`。
- 新 Python 文件遵守项目中文头部和注释规则，日期使用 `2026-07-28`。
- 每个任务独立执行红测、绿测和提交。

---

## File Structure

- `werewolf_agent/agents/reflection_output.py`：反思 tool schema、候选解析和安全诊断。
- `werewolf_agent/agents/player.py`：协调有限 primary/repair 调用，保留兼容方法签名。
- `werewolf_agent/runtime/agent_reflection_support.py`：构造上下文并投影安全生成审计。
- `werewolf_agent/runtime/nodes/summary.py`：汇总 12 人反思事务状态。
- `werewolf_agent/runtime/game_runner_memory.py`：先完整预检，再原子写入、回读和快照更新。
- `werewolf_agent/evaluation/acceptance_reflection_metrics.py`：拒绝 partial，并验证四段计数。
- `config/models.yaml`：反思专用低随机性和结构化输出策略。

### Task 1: 建立反思窄输出合同和显式结构化模式策略

**Files:**
- Create: `werewolf_agent/agents/reflection_output.py`
- Modify: `config/models.yaml:338-349`
- Modify: `tests/agents/test_reflection_generation.py`
- Modify: `tests/model_gateway/test_structured_output.py`
- Modify: `tests/agents/test_model_router.py`
- Modify: `tests/model_gateway/test_minimax_provider_routing.py`

**Interfaces:**
- Consumes: `ReflectionDraft.model_json_schema()` 和 provider 返回的文本/tool arguments。
- Produces: `reflection_draft_tool() -> dict[str, Any]`、`parse_reflection_draft_text(raw_text: str) -> ReflectionDraft`、`safe_reflection_validation_diagnostics(exc: ValidationError) -> tuple[tuple[str, ...], tuple[str, ...]]`。

- [ ] **Step 1: 写 native tool、文本回退和安全诊断红测**

```python
def test_reflection_tool_uses_strict_draft_schema() -> None:
    tool = reflection_draft_tool()
    assert tool["name"] == "submit_reflection_draft"
    assert tool["input_schema"] == ReflectionDraft.model_json_schema()


def test_safe_diagnostics_expose_paths_and_types_only() -> None:
    with pytest.raises(ValidationError) as raised:
        ReflectionDraft.model_validate({"claims": "PRIVATE", "lessons": []})
    paths, kinds = safe_reflection_validation_diagnostics(raised.value)
    assert paths == ("claims",)
    assert kinds == ("list_type",)
    assert "PRIVATE" not in repr((paths, kinds))
```

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/agents/test_reflection_generation.py tests/model_gateway/test_structured_output.py -q`

Expected: FAIL，因为 `reflection_output.py` 尚不存在，反思 profile 仍为 `text_json`。

- [ ] **Step 3: 实现窄输出模块并配置 native tool 主模式**

```python
def reflection_draft_tool() -> dict[str, Any]:
    return {
        "name": "submit_reflection_draft",
        "description": "提交可核验的赛后反思草稿。",
        "input_schema": ReflectionDraft.model_json_schema(),
    }


def parse_reflection_draft_text(raw_text: str) -> ReflectionDraft:
    data = json.loads(raw_text)
    return ReflectionDraft.model_validate(data)
```

将 `minimax_m27_reflection` 改为 `temperature: 0.2`、`structured_output.mode: native_tool`、`fallback_modes: [text_json]`。保留 `reasoning.level: high`；配置加载测试必须证明只有 reflection profile 改变。

- [ ] **Step 4: 运行输出合同测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/agents/test_reflection_generation.py tests/model_gateway/test_structured_output.py tests/agents/test_model_router.py tests/model_gateway/test_minimax_provider_routing.py -q`

Expected: PASS。

- [ ] **Step 5: 提交输出合同**

```bash
git add werewolf_agent/agents/reflection_output.py config/models.yaml tests/agents/test_reflection_generation.py tests/model_gateway/test_structured_output.py tests/agents/test_model_router.py tests/model_gateway/test_minimax_provider_routing.py
git commit -m "fix: add strict reflection output contract"
```

### Task 2: 实现带字段级定向修复的有限反思生成

**Files:**
- Modify: `werewolf_agent/agents/player.py:335-383`
- Modify: `tests/agents/test_reflection_generation.py`
- Modify: `tests/agents/test_prompt_mode_isolation.py`

**Interfaces:**
- Consumes: Task 1 的 tool/parser、`ModelRouter.resolve_structured_output_policy()` 和可选 `GenerationAttemptContext`。
- Produces: 向后兼容的 `PlayerAgent.generate_reflection(context, prompt, *, generation_attempt_context=None, generation_audit_sink=None) -> ReflectionDraft`。

- [ ] **Step 1: 写修复提示、共享 attempt context 和调用上限红测**

```python
def test_reflection_schema_repair_uses_safe_field_diagnostics(monkeypatch) -> None:
    provider = ReflectionProvider([
        '{"claims":"wrong","lessons":[]}',
        '{"claims":[],"lessons":[]}',
    ])
    agent = reflection_agent(provider, max_retries=2)
    attempts = GenerationAttemptContext(run_scope="p01")
    draft = agent.generate_reflection(
        reflection_context(), "复盘", generation_attempt_context=attempts,
    )
    assert draft == ReflectionDraft()
    assert "字段 claims 类型不符合 list_type" in provider.requests[1]["prompt"]
    assert "wrong" not in provider.requests[1]["prompt"]
    assert [item.route_kind for item in attempts.attempts] == [
        RouteKind.PRIMARY, RouteKind.REPAIR,
    ]
```

在同一测试文件中定义 `ReflectionProvider`、`reflection_agent()` 和 `reflection_context()`，分别返回受控响应、绑定该 provider 的 ModelRouter 和 `TaskType.REFLECTION` 上下文。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/agents/test_reflection_generation.py -k "schema_repair_uses_safe" -q`

Expected: FAIL，因为当前重试复用相同 Prompt 且不接收 attempt context。

- [ ] **Step 3: 修改生成循环**

```python
def generate_reflection(
    self,
    context: AgentContext,
    prompt: str,
    *,
    generation_attempt_context: GenerationAttemptContext | None = None,
    generation_audit_sink: dict[str, Any] | None = None,
) -> ReflectionDraft:
    attempt_context = generation_attempt_context or GenerationAttemptContext(
        run_scope=self.agent_id,
    )
    policy = self.model_router.resolve_structured_output_policy(
        self.agent_id, TaskType.REFLECTION.value,
    )
    active_mode = policy.primary_mode
    active_prompt = prompt
    for attempt in range(self.max_retries):
        result = self.model_router.generate(
            agent_id=self.agent_id,
            task_type=TaskType.REFLECTION.value,
            prompt=active_prompt,
            system_prompt=reflection_system_prompt,
            tools=[reflection_draft_tool()],
            tool_choice={"type": "tool", "name": "submit_reflection_draft"},
            structured_output_mode=active_mode.value,
            generation_attempt_context=attempt_context,
            max_provider_calls=1,
        )
```

每次调用后把 `result.structured_output_mode` 追加到局部 `used_modes`，并在返回或抛错前仅向可选 `generation_audit_sink` 写入 `attempt_count/provider_attempt_count/repair_attempt_count/structured_output_modes/field_paths/error_types`。`missing_tool_call/invalid_json/empty_response` 属于 protocol failure，按 policy 切换模式；Pydantic `ValidationError` 属于 schema failure，调用 `attempt_context.reject_latest_output()` 并只把字段路径和错误类型加入修复 Prompt；网络/provider failure 记录后抛出 `model_generation_failed`，不得伪装成 Schema 错误。总 provider 调用数不得超过 `max_retries`。

- [ ] **Step 4: 运行生成与模式隔离测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/agents/test_reflection_generation.py tests/agents/test_prompt_mode_isolation.py tests/model_gateway/test_generation_attempt_context.py -q`

Expected: PASS。

- [ ] **Step 5: 提交定向修复流程**

```bash
git add werewolf_agent/agents/player.py tests/agents/test_reflection_generation.py tests/agents/test_prompt_mode_isolation.py
git commit -m "fix: repair reflection schema failures deterministically"
```

### Task 3: 持久化安全的逐玩家生成审计

**Files:**
- Modify: `werewolf_agent/runtime/agent_reflection_support.py:64-173`
- Modify: `werewolf_agent/runtime/reflection_events.py`
- Modify: `werewolf_agent/runtime/nodes/summary.py:417-538`
- Modify: `tests/runtime/test_agent_reflection_support.py`
- Modify: `tests/runtime/test_reflection_security_contract.py`
- Modify: `tests/runtime/test_reflection_transaction.py`

**Interfaces:**
- Consumes: `GenerationAttemptContext.attempts`、Task 2 的 `generation_audit_sink` 和 `ReflectionDraftGenerationError.field_paths/error_types`。
- Produces: verification 中的 `generation_audit` 安全映射；game-level `generated_count/schema_validated_count/verified_count`。

- [ ] **Step 1: 写失败审计不泄密红测**

```python
def test_invalid_reflection_persists_safe_schema_diagnostics() -> None:
    result = run_agent_reflection_that_raises(
        ReflectionDraftGenerationError(
            "invalid_structured_draft",
            field_paths=("claims.0.claim_type",),
            error_types=("literal_error",),
        )
    )
    audit = result["reflection_verification"]["generation_audit"]
    assert audit["field_paths"] == ["claims.0.claim_type"]
    assert audit["error_types"] == ["literal_error"]
    assert "provider_response" not in repr(audit)
```

在测试中定义 `run_agent_reflection_that_raises(exc)` 为带单玩家 GameState 和抛出指定异常的 FakeAgent/Registry 调用 `_agent_reflection()`。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_agent_reflection_support.py tests/runtime/test_reflection_security_contract.py -k "safe_schema_diagnostics" -q`

Expected: FAIL，因为 `_agent_reflection()` 当前丢弃字段路径和 attempt 信息。

- [ ] **Step 3: 投影白名单审计和计数**

```python
generation_audit: dict[str, Any] = {}
draft = agent.generate_reflection(
    context,
    prompt,
    generation_attempt_context=attempt_context,
    generation_audit_sink=generation_audit,
)
safe_generation_audit = safe_reflection_generation_audit(generation_audit)
```

异常路径从 `ReflectionDraftGenerationError` 合并 `field_paths/error_types` 到同一 sink。`safe_reflection_generation_audit()` 对键、类型、枚举值、数量和字符串长度执行白名单；模式来自 Task 2 的 `GenerateResult.structured_output_mode`，不扩展 attempt 记录，也不保存原文。

- [ ] **Step 4: 运行反思安全和事务测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_agent_reflection_support.py tests/runtime/test_reflection_security_contract.py tests/runtime/test_reflection_transaction.py -q`

Expected: PASS。

- [ ] **Step 5: 提交安全生成审计**

```bash
git add werewolf_agent/runtime/agent_reflection_support.py werewolf_agent/runtime/reflection_events.py werewolf_agent/runtime/nodes/summary.py tests/runtime/test_agent_reflection_support.py tests/runtime/test_reflection_security_contract.py tests/runtime/test_reflection_transaction.py
git commit -m "fix: audit reflection generation failures safely"
```

### Task 4: 将反思持久化改为全员预检后的原子事务

**Files:**
- Modify: `werewolf_agent/runtime/game_runner_memory.py:86-300`
- Modify: `tests/runtime/test_game_runner.py:1404-2035`
- Modify: `tests/runtime/test_reflection_transaction.py`

**Interfaces:**
- Consumes: `canonical_verified_reflections()` 和 12 个 `ReviewReport`。
- Produces: 仅在 12 个候选全部可构建时写入的 V2 条目；精确的 `expected/generated/verified/persisted` 计数。

- [ ] **Step 1: 写 9/12 不写入且不更新 latest 的红测**

```python
def test_partial_reflections_do_not_write_entries_or_latest_snapshot() -> None:
    runner, repo = runner_with_verified_reflections(valid_players=9, total_players=12)
    runner._save_memory_snapshot()
    audit = latest_event(runner.state, "reflection_persistence_audit").payload
    assert audit["status"] == "partial"
    assert audit["expected_player_count"] == 12
    assert audit["generated_entry_count"] == 12
    assert audit["schema_validated_entry_count"] == 9
    assert audit["verified_entry_count"] == 9
    assert audit["persisted_entry_count"] == 0
    assert repo.load_memory_snapshot("latest") is None
```

在测试文件内添加 `runner_with_verified_reflections()` 和 `latest_event()`，使用现有 InMemoryGameRepository 与 12 人 finished GameState 构造数据。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_game_runner.py tests/runtime/test_reflection_transaction.py -k "partial_reflections_do_not_write" -q`

Expected: FAIL，因为当前实现会先写 9 个条目并记录持久化成功。

- [ ] **Step 3: 分离候选构建与写入阶段**

```python
candidates = _build_verified_reflection_candidates(
    reports=reports,
    verified_reflections=verified_reflections,
    ground_truth=ground_truth,
    memory_store=mem_store,
    game_id=self._game_id,
)
if len(candidates) != len(player_ids):
    self._append_reflection_persistence_audit(
        [],
        upstream_complete=False,
        transaction_status="partial",
        expected_player_count=len(player_ids),
        verified_entry_count=len(candidates),
    )
    return
```

把 `_build_verified_reflection_candidates(...) -> list[dict]` 作为模块级纯函数；它不调用 repository。只有数量等于 12 后才备份、逐条 `store_v2(raise_on_failure=True)`、保存 game/latest 快照并回读。

- [ ] **Step 4: 修正日志使用实际事务计数**

```python
logger.info(
    "Saved reflection transaction for game %s "
    "(generated=%d/%d verified=%d/%d persisted=%d/%d)",
    self._game_id,
    generated_count, player_count,
    verified_count, player_count,
    persisted_count, player_count,
)
```

删除使用 `len(ground_truth)` 作为 reviews 数的日志语义。

- [ ] **Step 5: 运行持久化、回滚和日志测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_game_runner.py tests/runtime/test_reflection_transaction.py tests/memory/test_reflection_repository.py -q`

Expected: PASS。

- [ ] **Step 6: 提交原子事务**

```bash
git add werewolf_agent/runtime/game_runner_memory.py tests/runtime/test_game_runner.py tests/runtime/test_reflection_transaction.py
git commit -m "fix: persist reflections only after full preflight"
```

### Task 5: 强制 acceptance 12/12 并验证下一局恢复

**Files:**
- Modify: `werewolf_agent/evaluation/acceptance_reflection_metrics.py:137-235`
- Modify: `werewolf_agent/runtime/game_runner_memory.py:67-84`
- Modify: `werewolf_agent/runtime/context_cross_game_memory.py`
- Modify: `werewolf_agent/agents/prompt_memory.py`
- Modify: `tests/evaluation/test_report_acceptance_metrics.py`
- Modify: `tests/runtime/test_context_memory_hints.py`
- Modify: `tests/memory/test_reflection_effectiveness.py`
- Modify: `tests/integration/test_post_july14_repair_closure.py`

**Interfaces:**
- Consumes: reflection complete/persistence audit、repository 回读和恢复后的 MemoryStore。
- Produces: acceptance 的 12/12 合同和 `restored_reflection_count` 安全计数。

- [ ] **Step 1: 写 partial 拒绝和恢复 12/12 红测**

```python
def test_reflection_acceptance_rejects_partial_even_when_persistence_is_consistent() -> None:
    metrics = compute_reflection_acceptance_metrics([
        game_with_reflections(valid=9, failed=3)
    ])
    assert metrics["reflection_contamination_metrics_supported"] is False
    assert metrics["reflection_contamination_metrics_unsupported_reason"] == (
        "reflection_partial"
    )


def test_next_game_restores_all_twelve_verified_reflections() -> None:
    runner = build_runner_restoring_snapshot_with_entries(12)
    assert runner.restored_reflection_count == 12
    for player_id in runner.state.players:
        prompt = build_memory_prompt_for(runner, player_id)
        assert "已核验跨局经验" in prompt
```

在 `tests/evaluation/test_report_acceptance_metrics.py` 中定义 `game_with_reflections()`，构造含 `reflection_complete.status=partial` 和 9/12 计数的投影。`build_runner_restoring_snapshot_with_entries()` 与 `build_memory_prompt_for()` 定义在内存恢复测试中，使用现有 `PersistentMemoryCoordinator`、`InMemoryGameRepository` 和 12 个 `ReflectionEntryV2` 完整构造。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/evaluation/test_report_acceptance_metrics.py tests/runtime/test_context_memory_hints.py tests/memory/test_reflection_effectiveness.py -k "rejects_partial or restores_all_twelve" -q`

Expected: FAIL，因为 acceptance 当前允许 partial，runner 不报告恢复条目数。

- [ ] **Step 3: 收紧 acceptance 并投影恢复计数**

```python
if reflection_status == "partial":
    return False, 0, "reflection_partial"
if reflection_status != "complete":
    return False, 0, "incomplete_reflection_audit"
if eligible_players != player_ids:
    return False, 0, "reflection_not_12_of_12"
```

`_restore_memory_if_configured()` 从恢复后的 `all_v2_entries()` 统计当前 latest snapshot 的唯一玩家条目，并暴露只读 `restored_reflection_count`。Prompt 测试必须检查每个玩家实际获得经验文本，而不是只检查快照对象存在。

- [ ] **Step 4: 运行反思整域测试和静态检查**

Run: `conda run -n wofkill python -m pytest -n 0 tests/agents/test_reflection_generation.py tests/runtime/test_reflection_security_contract.py tests/runtime/test_reflection_transaction.py tests/runtime/test_game_runner.py tests/evaluation/test_report_acceptance_metrics.py tests/runtime/test_context_memory_hints.py tests/memory/test_reflection_effectiveness.py tests/integration/test_post_july14_repair_closure.py -q`

Run: `conda run -n wofkill python -m ruff check werewolf_agent/agents/reflection_output.py werewolf_agent/agents/player.py werewolf_agent/runtime/agent_reflection_support.py werewolf_agent/runtime/game_runner_memory.py werewolf_agent/evaluation/acceptance_reflection_metrics.py`

Expected: 两条命令均退出 0。

- [ ] **Step 5: 运行全量测试**

Run: `conda run -n wofkill python -m pytest -n 0 -q`

Expected: PASS。

- [ ] **Step 6: 提交 12/12 合同**

```bash
git add werewolf_agent/evaluation/acceptance_reflection_metrics.py werewolf_agent/runtime/game_runner_memory.py werewolf_agent/runtime/context_cross_game_memory.py werewolf_agent/agents/prompt_memory.py tests/evaluation/test_report_acceptance_metrics.py tests/runtime/test_context_memory_hints.py tests/memory/test_reflection_effectiveness.py tests/integration/test_post_july14_repair_closure.py
git commit -m "fix: require complete reflection memory transactions"
```

### Task 6: 执行真实模型双局验收

**Files:**
- No code changes; verification only. Existing runner writes artifacts; do not commit generated logs or game JSON.

**Interfaces:**
- Consumes: 完成 Tasks 1–5 后的真实 runner 和唯一游戏 ID。
- Produces: 两局 `reflection_complete.status=complete`、12 个持久化条目以及第二局启动时 `restored_reflection_count=12` 的证据。

- [ ] **Step 1: 运行第一局**

Run: `conda run -n wofkill python scripts/run_real_game.py --game-id reflection_acceptance_a --seed 4201 --max-steps 500 --delay -1 --output-dir artifacts/reflection_acceptance_a`

Expected: 游戏正常终局；报告显示 generated/verified/persisted 均为 12/12。

- [ ] **Step 2: 运行第二局并恢复第一局 latest 快照**

Run: `conda run -n wofkill python scripts/run_real_game.py --game-id reflection_acceptance_b --seed 4202 --max-steps 500 --delay -1 --output-dir artifacts/reflection_acceptance_b`

Expected: 启动日志显示 restored=12/12；终局 generated/verified/persisted 仍为 12/12。

- [ ] **Step 3: 查询只读事务证据**

使用项目现有报告入口读取两份 game JSON，断言每局只有一个 `reflection_complete` 和一个 `reflection_persistence_audit`，状态均为 `complete`，12 个 decision ID、entry ID 和玩家 ID 一一对应。

- [ ] **Step 4: 断言报告证据字段完整**

```python
assert reflection_summary == {
    "generated": 12,
    "verified": 12,
    "persisted": 12,
    "restored": 12,
}
```

本任务不改代码、不提交产物；保留运行命令、game ID、报告路径和断言结果作为验收记录。
