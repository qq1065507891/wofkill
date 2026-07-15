# Post-July-14 Full Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 7 月 14 日后审计确认的 K1–K3 与 N1–N12，使核心规则执行、失败回退、反思事务和质量验收拥有一致、可追溯且向后兼容的 V2 契约。

**Architecture:** 采用四批次契约优先迁移：先修事件、结算、决策 trace、质量投影和终止状态，再引入确定性狼队共识与统一空刀策略，随后修复 provider fallback、反思、上下文和技能机会链，最后运行兼容、隐私、稳定性和平衡门禁。旧日志通过领域适配器读取；所有新游戏只写 V2。

**Tech Stack:** Python 3.11+、frozen dataclasses、Pydantic、LangGraph、pytest、YAML ruleset、现有 exposure audit 与模型网关。

---

## 实施约束

- 开始执行前在独立 worktree 中工作；不要在当前用户工作区直接实现。
- 每个 Python 新文件使用 UTF-8 声明和中文模块 docstring，作者使用 `Project contributors`，创建日期使用 `2026-07-15`。
- 每项行为变更严格采用 `@test-driven-development`：先写失败测试、确认失败原因、最小实现、运行聚焦测试。
- 每批结束使用 `@requesting-code-review`；所有完成声明使用 `@verification-before-completion`。
- Windows pytest 使用新的仓库内临时根，例如 `--basetemp .tmp/pytest-post-july14-t01`。若该目录出现 ACL 问题，只删除已确认位于当前 worktree 内的对应临时目录，然后以新目录重跑。
- 保留现有公共 API；V1 兼容只读。不得恢复 legacy writer、降低 quorum、随机选刀或静默吞掉失败。
- 不提交真实游戏大日志、原始 prompt、API 密钥、隐藏思维链或角色真值泄漏。

## 审计问题到任务映射

| 问题 | 主修任务 | 闭环任务 |
|---|---:|---:|
| K1 狼人合法计划未执行 | 7、8、9 | 15、16 |
| K2 夜聊上下文与证据链有损 | 7、8、13 | 15 |
| K3 动态状态与游戏语义不一致 | 13、14 | 15、17 |
| N1 单狼主备被判平票 | 8 | 15 |
| N2 连续空刀保护被旁路 | 9 | 15、16 |
| N3 递归上限留下伪完成局 | 6 | 15、16 |
| N4 reasoning 与权威支持者不一致 | 7、8 | 15 |
| N5 备刀被误解为同夜第二刀 | 13 | 15 |
| N6 成功解析和白天流程掩盖夜间失败 | 5、11 | 15–17 |
| N7 事件缺时间戳和关联 ID | 2 | 15、16 |
| N8 反思有效经验和持久化为零 | 12 | 15、16 |
| N9 死亡批次生产/消费不一致 | 3 | 15、16 |
| N10 语义重试耗尽被通用 fallback 掩盖 | 4、11 | 15–17 |
| N11 质量投影缺字段并接受空成功 | 5、12 | 15、16 |
| N12 provider fallback 未切换路由 | 10 | 15、16 |

## 文件结构锁定

### 新增生产模块

| 文件 | 单一职责 |
|---|---|
| `werewolf_agent/core/event_visibility.py` | 定义事件可见性枚举及 V2 顶层/V1 payload 兼容读取 |
| `werewolf_agent/runtime/event_metadata.py` | 为新运行事件分配 V2 ID、顺序、时间和 trace 元数据 |
| `werewolf_agent/core/resolution_batches.py` | 定义和兼容解析结构化死亡批次 |
| `werewolf_agent/evaluation/game_projection.py` | 构造完整、强类型的验收游戏投影 |
| `werewolf_agent/runtime/game_termination.py` | 构造 finished/aborted 终态和终止事件 |
| `werewolf_agent/runtime/wolf_consensus_evidence.py` | 验证狼人立场并独立聚合主刀、备刀共识 |
| `werewolf_agent/runtime/wolf_no_kill_policy.py` | 统一空刀原因、连续计数和确定性恢复 |
| `werewolf_agent/model_gateway/fallback_policy.py` | 构建、去重并验证 provider fallback 路由 |
| `werewolf_agent/runtime/reflection_transaction.py` | 表达反思阶段、局级状态和事务结果 |
| `werewolf_agent/runtime/skill_opportunity_events.py` | 构造神职/自爆机会链的私有与公共事件 |

### 新增聚焦测试与夹具

| 文件 | 单一职责 |
|---|---|
| `tests/fixtures/post_july14_contract_regressions.json` | 保存从审计样本脱敏提取的最小回归输入，不保存完整对局 |
| `tests/regression/test_post_july14_contract_cases.py` | 保证 15 个问题均有可执行回归用例 |
| `tests/runtime/test_event_metadata_v2.py` | V2 新写和 V1 只读兼容 |
| `tests/runtime/test_resolution_batches.py` | 死亡批次解析、过滤和聚合告警 |
| `tests/agents/test_decision_trace_v2.py` | attempt/retry/fallback 计数不变量 |
| `tests/evaluation/test_acceptance_game_projection.py` | 完整投影、支持性和保存/重算一致性 |
| `tests/runtime/test_game_termination.py` | finished/aborted 终止语义 |
| `tests/runtime/test_wolf_consensus_evidence.py` | 狼人立场与主备独立聚合 |
| `tests/runtime/test_wolf_no_kill_policy.py` | 空刀计数、重置和确定性恢复 |
| `tests/model_gateway/test_provider_fallback_policy.py` | 路由切换、去重和能力门禁 |
| `tests/agents/test_task_terminal_fallbacks.py` | 任务专用 fallback 与失败码保真 |
| `tests/runtime/test_reflection_transaction.py` | 反思状态机、零条目和回滚 |
| `tests/runtime/test_wolf_prompt_contract.py` | 分层上下文、备刀语义和 HMAC 送达证明 |
| `tests/runtime/test_skill_opportunity_events.py` | 自爆、猎人、预言家机会链和可见性 |
| `tests/integration/test_post_july14_repair_closure.py` | 四批功能的端到端闭环 |

---

# 批次一：测量基础与兼容契约

## Task 1: 建立审计问题回归清单

**Files:**
- Create: `tests/fixtures/post_july14_contract_regressions.json`
- Create: `tests/regression/test_post_july14_contract_cases.py`
- Reference: `docs/audits/2026-07-15-post-july14-game-log-audit.md`
- Reference: `docs/superpowers/specs/2026-07-15-post-july14-full-repair-design.md`

- [ ] **Step 1: 创建最小脱敏夹具**

夹具必须包含 15 个 case ID：`K1`、`K2`、`K3`、`N1`–`N12`。每个 case 只保留触发该缺陷所需字段，并带 `expected_contract`，例如：

```json
{
  "N1": {
    "alive_wolves": ["p03"],
    "primary": [{"wolf_id": "p03", "target_id": "p10"}],
    "backup": [{"wolf_id": "p03", "target_id": "p05"}],
    "expected_contract": "select_primary_without_tie"
  },
  "N11": {
    "winning_faction": "werewolf",
    "players": ["p01", "p02"],
    "reflection_entries": [],
    "expected_contract": "completed_game_but_reflection_no_valid_entries"
  }
}
```

- [ ] **Step 2: 写夹具结构测试**

```python
def test_every_audit_issue_has_a_regression_case() -> None:
    cases = load_cases()
    assert set(cases) == {"K1", "K2", "K3", *(f"N{i}" for i in range(1, 13))}
    assert all(case["expected_contract"] for case in cases.values())
```

- [ ] **Step 3: 运行夹具测试并确认失败**

Run: `python -m pytest tests/regression/test_post_july14_contract_cases.py -q --basetemp .tmp/pytest-post-july14-t01`

Expected: FAIL，因为夹具或 loader 尚未存在/不完整。

- [ ] **Step 4: 添加最小 loader 并使测试通过**

loader 保持在测试文件内，不创建生产期“审计夹具模块”。

- [ ] **Step 5: 运行测试**

Run: `python -m pytest tests/regression/test_post_july14_contract_cases.py -q --basetemp .tmp/pytest-post-july14-t01-green`

Expected: PASS，15 个 case 全部存在。

- [ ] **Step 6: 提交**

```bash
git add tests/fixtures/post_july14_contract_regressions.json tests/regression/test_post_july14_contract_cases.py
git commit -m "test: capture post-july14 contract regressions"
```

## Task 2: 引入 GameEvent V2 元数据并保持 V1 只读

**Files:**
- Create: `werewolf_agent/core/event_visibility.py`
- Create: `werewolf_agent/runtime/event_metadata.py`
- Modify: `werewolf_agent/core/models.py`
- Modify: `werewolf_agent/runtime/game_runner_execution.py`
- Modify: `werewolf_agent/storage/memory_store.py`
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/storage/postgres_store.py`
- Modify: `scripts/run_real_game.py`
- Modify: `werewolf_agent/storage/migrations.py`
- Modify: `werewolf_agent/runtime/replay.py`
- Modify: `werewolf_agent/runtime/exposure_audit.py`
- Modify: `werewolf_agent/runtime/public_ledger.py`
- Modify: `werewolf_agent/runtime/private_memory.py`
- Modify: `werewolf_agent/runtime/context_public_summary.py`
- Modify: `werewolf_agent/runtime/strategy/seer.py`
- Modify: `werewolf_agent/cognition/world_state.py`
- Modify: `werewolf_agent/api/routes/game_public_share.py`
- Modify: `scripts/run_real_game.py`
- Create: `tests/runtime/test_event_metadata_v2.py`
- Modify: `tests/runtime/test_event_sourcing.py`
- Modify: `tests/storage/test_postgres_store.py`
- Modify: `tests/storage/test_sqlite_migrations.py`
- Modify: `tests/storage/test_migrations.py`
- Modify: `tests/scripts/test_run_real_game.py`
- Modify: `tests/runtime/test_public_ledger.py`
- Modify: `tests/runtime/test_private_memory_sanitize.py`
- Modify: `tests/runtime/test_context_public_summary.py`

- [ ] **Step 1: 写 V2 新写失败测试**

覆盖：新 runner 事件具有唯一 `event_id`、连续 `sequence_number`、带时区 `occurred_at`、`game_id` 和 `schema_version="2"`；旧 `GameEvent(type, payload)` 经兼容读取仍可加载，但不能被误判为新写 V2。

```python
def test_stamp_new_events_assigns_stable_v2_metadata() -> None:
    before: list[GameEvent] = []
    after = [GameEvent(type="enter_night", payload={"night_number": 1})]
    stamped = stamp_new_events("g1", before, after, now=fixed_now)
    assert stamped[0].event_id == "g1:e000000"
    assert stamped[0].sequence_number == 0
    assert stamped[0].schema_version == "2"
```

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/runtime/test_event_metadata_v2.py -q --basetemp .tmp/pytest-post-july14-t02`

Expected: FAIL，V2 字段和 `stamp_new_events` 尚不存在。

- [ ] **Step 3: 实现可选存储字段和显式 stamping**

`GameEvent` 新增可选字段，默认均为 `None`，避免把旧构造器伪装为 V2：

```python
@dataclass(frozen=True)
class GameEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    visibility: EventVisibility | None = None
    event_id: str | None = None
    sequence_number: int | None = None
    occurred_at: datetime | None = None
    game_id: str | None = None
    trace_id: str | None = None
    schema_version: str | None = None
```

`EventVisibility` 统一现有 public、moderator、阵营私有和角色/actor 私有取值。V2 builder 把旧 payload 中的 visibility 提升为顶层权威字段，并在安全序列化后从 payload 删除重复值；V1 reader 仍可从 payload 读取。`occurred_at` 在内存中必须是带时区 `datetime`，JSON/数据库边界由唯一 serializer 转为 ISO-8601，禁止各 writer 自行处理。

`core/event_visibility.py` 提供唯一读取入口：

```python
def event_visibility(event: GameEvent) -> EventVisibility:
    if event.visibility is not None:
        return event.visibility
    return EventVisibility.from_legacy(event.payload.get("visibility", "public"))
```

`public_ledger.py`、`private_memory.py`、`context_public_summary.py`、seer strategy、cognition world state 和 API public share 全部改用该 helper。测试必须用“V2 顶层 private + payload 无 visibility”的重载事件证明它不会被当作 public，同时保留 V1 payload-only 行为。

在 `GameRunnerExecutionMixin._process_chunk()` 接受节点状态后，只对本次新增且未盖章的事件调用 `stamp_new_events()`；已有 V2 元数据不得重写。需要在同一节点引用 `source_event_id` 的新代码使用 `new_game_event(gs, ...)` 立即分配 ID。

- [ ] **Step 4: 更新所有存储、迁移、JSON 导出和 replay 兼容**

memory repository 直接保存完整 dataclass。SQLite migration v2 为 `events` 增加 nullable `event_json TEXT`，Postgres 初始化 SQL 执行 `ALTER TABLE events ADD COLUMN IF NOT EXISTS event_json JSONB`；新写把完整 GameEvent 写入 `event_json`，同时保留旧 `event_type/payload_json` 列供 V1 reader。SQLite fresh schema 也必须直接包含新列。旧行 `event_json IS NULL` 时由旧列构造 `schema_version=None` 的 V1 视图。

`scripts/run_real_game.py` 的 `events` 数组导出完整安全元数据，而不是只导出 type/payload；payload 仍先经过 `_safe_event_payload()`。replay 优先按 `sequence_number` 排序；V1 缺序号时保留原数组顺序。禁止 legacy writer 分支。

- [ ] **Step 5: 增加 exposure audit 测试**

确认新元数据不改变 visibility，不把私有 payload 复制到公共事件，并验证 `event_id` 不包含身份真值。

- [ ] **Step 6: 运行聚焦测试**

Run: `python -m pytest tests/runtime/test_event_metadata_v2.py tests/runtime/test_event_sourcing.py tests/storage/test_postgres_store.py tests/storage/test_sqlite_migrations.py tests/storage/test_migrations.py tests/scripts/test_run_real_game.py tests/runtime/test_exposure_audit.py tests/runtime/test_public_ledger.py tests/runtime/test_private_memory_sanitize.py tests/runtime/test_context_public_summary.py -q --basetemp .tmp/pytest-post-july14-t02-green`

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add werewolf_agent/core/event_visibility.py werewolf_agent/core/models.py werewolf_agent/runtime/event_metadata.py werewolf_agent/runtime/game_runner_execution.py werewolf_agent/storage/memory_store.py werewolf_agent/storage/sqlite_store.py werewolf_agent/storage/postgres_store.py werewolf_agent/storage/migrations.py werewolf_agent/runtime/replay.py werewolf_agent/runtime/exposure_audit.py werewolf_agent/runtime/public_ledger.py werewolf_agent/runtime/private_memory.py werewolf_agent/runtime/context_public_summary.py werewolf_agent/runtime/strategy/seer.py werewolf_agent/cognition/world_state.py werewolf_agent/api/routes/game_public_share.py scripts/run_real_game.py tests/runtime/test_event_metadata_v2.py tests/runtime/test_event_sourcing.py tests/storage/test_postgres_store.py tests/storage/test_sqlite_migrations.py tests/storage/test_migrations.py tests/scripts/test_run_real_game.py tests/runtime/test_exposure_audit.py tests/runtime/test_public_ledger.py tests/runtime/test_private_memory_sanitize.py tests/runtime/test_context_public_summary.py
git commit -m "feat: add v2 game event metadata"
```

### Task 2 Review Hardening: Upgrade Compatibility

**Files:**
- Modify: `werewolf_agent/runtime/event_metadata.py`
- Modify: `werewolf_agent/core/event_visibility.py`
- Modify: `werewolf_agent/runtime/replay.py`
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/storage/postgres_store.py`
- Test: `tests/runtime/test_event_metadata_v2.py`
- Test: `tests/runtime/test_event_sourcing.py`
- Test: `tests/storage/test_sqlite_migrations.py`
- Test: `tests/storage/test_postgres_store.py`

- [x] **Step 1: Add failing compatibility tests**

Cover legacy-column visibility for SQLite and Postgres, unknown legacy visibility
fail-closed behavior through helper/deserialization/public share, mixed V1/V2 replay
ordering, strict validation of existing V2 metadata, and nested payload copy isolation.

- [x] **Step 2: Run the focused tests and record the real failures**

Run each new node before implementation so every reported RED corresponds to the
current defect rather than an inferred failure.

- [x] **Step 3: Centralize event serialization compatibility**

Add one event-metadata helper that returns the legacy payload with canonical
`visibility`, while `serialize_game_event()` keeps V2 visibility authoritative at
the top level and removes the duplicate from `event_json.payload`. Both stores use
the shared helper for `payload_json`.

- [x] **Step 4: Harden replay and stamping**

Preserve mixed upgrade-log array order when legacy events are present, sort pure
V2 logs by trusted sequence numbers, and reject partial, duplicate, conflicting,
or game-inconsistent existing V2 metadata without rewriting it.

- [x] **Step 5: Run Task 2 gates and commit**

Run focused metadata/storage/replay/public privacy tests, the original Task 2
regression set, Ruff, and `git diff --check`; then commit with
`fix: harden event v2 upgrade compatibility`.

## Task 3: 统一死亡批次契约

**Files:**
- Create: `werewolf_agent/core/resolution_batches.py`
- Modify: `werewolf_agent/core/models.py`
- Modify: `werewolf_agent/engine/rule_exile.py`
- Modify: `werewolf_agent/engine/rule_death.py`
- Modify: `werewolf_agent/engine/rule_night.py`
- Modify: `werewolf_agent/engine/event_reducer.py`
- Modify: `werewolf_agent/runtime/directives/_shared.py`
- Modify: `werewolf_agent/storage/memory_store.py`
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/storage/postgres_store.py`
- Create: `tests/runtime/test_resolution_batches.py`
- Modify: `tests/runtime/test_directive_shared_helpers.py`
- Modify: `tests/runtime/test_night_resolution.py`
- Modify: `tests/storage/test_postgres_store.py`

- [ ] **Step 1: 写解析和过滤失败测试**

参数化覆盖 `day_3`、`day_3_vote`、`day_3_self_destruct`、`night_3` 和未知格式；确认当前日过滤不包含未来日死亡，未知 day 格式 fail closed、保留原始值并设置 `batch_parse_failed=True`，且只聚合告警一次。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/runtime/test_resolution_batches.py tests/runtime/test_directive_shared_helpers.py -q --basetemp .tmp/pytest-post-july14-t03`

Expected: FAIL，当前严格正则拒绝 `day_N_vote`。

- [ ] **Step 3: 实现结构化批次**

```python
@dataclass(frozen=True)
class ResolutionBatchV2:
    phase: Literal["day", "night"]
    number: int
    cause: Literal["vote", "self_destruct", "wolf_kill", "witch_poison", "hunter_shot", "rule_effect", "unknown"]

@dataclass(frozen=True)
class ResolutionBatchParseResult:
    batch: ResolutionBatchV2 | None
    raw_value: str | None
    batch_parse_failed: bool

def parse_resolution_batch(
    value: ResolutionBatchV2 | str | Mapping[str, object],
) -> ResolutionBatchParseResult:
    if isinstance(value, ResolutionBatchV2):
        return ResolutionBatchParseResult(value, None, False)
    if isinstance(value, Mapping):
        try:
            batch = ResolutionBatchV2(
                phase=str(value["phase"]),
                number=int(value["number"]),
                cause=str(value["cause"]),
            )
        except (KeyError, TypeError, ValueError):
            return ResolutionBatchParseResult(None, json.dumps(value, sort_keys=True), True)
        return ResolutionBatchParseResult(batch, None, False)
    match = _LEGACY_BATCH_RE.fullmatch(value)
    if match is None:
        return ResolutionBatchParseResult(None, value, True)
    phase, number, suffix = match.groups()
    cause = _LEGACY_CAUSE_MAP.get(suffix, "unknown")
    return ResolutionBatchParseResult(
        ResolutionBatchV2(phase=phase, number=int(number), cause=cause),
        value,
        False,
    )
```

`Death` 在迁移期同时接受 `ResolutionBatchV2 | str`，并增加 `resolution_batch_parse_failed: bool = False`。新生产者只写 V2 且 marker 必须为 false。memory、SQLite 和 Postgres loader 遇到旧字符串或新 dict 时都通过 parser 规范化；未知旧格式保留原始字符串并把 marker 设为 true，不能丢失失败原因，也不能把 dict 原样塞回 `Death.resolution_batch`。

`scripts/run_real_game.py` 使用同一个 `serialize_resolution_batch()` 输出 `{phase, number, cause}`；未知 V1 格式输出原始字符串并同时写 `resolution_batch_parse_failed: true`。禁止把 dataclass 对象直接交给 `json.dumps()`。旧 JSON reader 继续接受字符串。

- [ ] **Step 4: 改造生产者与消费者**

修改放逐、自爆、夜间死亡、猎人枪击等生产者。`collect_death_order()` 只消费 parser 结果；解析失败的 day-like 值不注入当前日指令，并使用 `(game_id, raw_batch)` 去重 warning。

- [ ] **Step 5: 运行聚焦测试**

Run: `python -m pytest tests/runtime/test_resolution_batches.py tests/runtime/test_directive_shared_helpers.py tests/runtime/test_night_resolution.py tests/runtime/test_hunter_flow.py tests/storage/test_postgres_store.py tests/scripts/test_run_real_game.py -q --basetemp .tmp/pytest-post-july14-t03-green`

Expected: PASS；malformed `day_N_vote` 回归测试为 0。

- [ ] **Step 6: 提交**

```bash
git add werewolf_agent/core/resolution_batches.py werewolf_agent/core/models.py werewolf_agent/engine/rule_exile.py werewolf_agent/engine/rule_death.py werewolf_agent/engine/rule_night.py werewolf_agent/engine/event_reducer.py werewolf_agent/runtime/directives/_shared.py werewolf_agent/storage/memory_store.py werewolf_agent/storage/sqlite_store.py werewolf_agent/storage/postgres_store.py scripts/run_real_game.py tests/runtime/test_resolution_batches.py tests/runtime/test_directive_shared_helpers.py tests/runtime/test_night_resolution.py tests/storage/test_postgres_store.py tests/scripts/test_run_real_game.py
git commit -m "fix: unify death resolution batch contract"
```

## Task 4: 修正 DecisionExecutionTrace V2 计数

**Files:**
- Modify: `werewolf_agent/agents/trace_schemas.py`
- Modify: `werewolf_agent/agents/trace_builder.py`
- Modify: `werewolf_agent/agents/player_action_result.py`
- Modify: `werewolf_agent/agents/player_action_flow.py`
- Modify: `werewolf_agent/runtime/decision_outcomes.py`
- Modify: `werewolf_agent/evaluation/decision_execution_audit.py`
- Modify: `werewolf_agent/runtime/exposure_audit.py`
- Create: `tests/agents/test_decision_trace_v2.py`
- Modify: `tests/runtime/test_decision_outcomes.py`
- Modify: `tests/evaluation/test_report_acceptance_metrics.py`

- [ ] **Step 1: 写计数矩阵失败测试**

测试首次成功、一次 retry 成功、一次 provider fallback 成功和 terminal fallback：

```python
@pytest.mark.parametrize(
    ("routes", "attempts", "retries", "provider_fallbacks"),
    [
        (["primary"], 1, 0, 0),
        (["primary", "retry"], 2, 1, 0),
        (["primary", "provider_fallback"], 2, 0, 1),
        (["primary", "retry", "safe_fallback"], 3, 1, 0),
    ],
)
def test_v2_attempt_counters(routes, attempts, retries, provider_fallbacks) -> None:
    records = tuple(make_attempt(ordinal, route) for ordinal, route in enumerate(routes, 1))
    counts = summarize_attempt_counts(records)
    assert counts.attempt_count == attempts
    assert counts.retry_count == retries
    assert counts.provider_fallback_count == provider_fallbacks
```

- [ ] **Step 2: 运行并确认当前字段语义失败**

Run: `python -m pytest tests/agents/test_decision_trace_v2.py tests/evaluation/test_report_acceptance_metrics.py -q --basetemp .tmp/pytest-post-july14-t04`

Expected: FAIL，首次成功仍写 `retry_count=1` 或 evaluator 按 `len(attempts)-1` 误算 provider fallback。

- [ ] **Step 3: 修改 trace schema 和唯一计数函数**

新增 `attempt_count`、`provider_fallback_count`、`generated_by`、`terminal_failure_code`。创建一个纯函数从 `execution_attempts` 推导全部计数，生产者与 evaluator 共用，删除新写 `total_retry_count_until_success`。

- [ ] **Step 4: 添加 V1 trace 兼容归一化**

旧 trace 仍可读；兼容函数根据 `execution_attempts.route_kind` 重建 V2 计数，不信任冲突的旧 `retry_count`。兼容结果带 `normalized_from_schema_version="1"`，不反写旧文件。

- [ ] **Step 5: 运行聚焦测试**

Run: `python -m pytest tests/agents/test_decision_trace_v2.py tests/runtime/test_decision_outcomes.py tests/evaluation/test_report_acceptance_metrics.py tests/runtime/test_exposure_audit.py -q --basetemp .tmp/pytest-post-july14-t04-green`

Expected: PASS，所有合法 fixture 的 `attempt_retry_consistency_error_count == 0`。

- [ ] **Step 6: 提交**

```bash
git add werewolf_agent/agents/trace_schemas.py werewolf_agent/agents/trace_builder.py werewolf_agent/agents/player_action_result.py werewolf_agent/agents/player_action_flow.py werewolf_agent/runtime/decision_outcomes.py werewolf_agent/evaluation/decision_execution_audit.py werewolf_agent/runtime/exposure_audit.py tests/agents/test_decision_trace_v2.py tests/runtime/test_decision_outcomes.py tests/evaluation/test_report_acceptance_metrics.py
git commit -m "fix: align decision attempt and retry semantics"
```

## Task 5: 建立完整 AcceptanceGameProjection 并修正最终评分时点

**Files:**
- Create: `werewolf_agent/evaluation/game_projection.py`
- Modify: `werewolf_agent/core/models.py`
- Modify: `werewolf_agent/evaluation/acceptance_audit.py`
- Modify: `werewolf_agent/evaluation/acceptance_world_metrics.py`
- Modify: `werewolf_agent/evaluation/acceptance_reflection_metrics.py`
- Modify: `werewolf_agent/evaluation/acceptance_power_metrics.py`
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Modify: `scripts/run_real_game.py`
- Create: `tests/evaluation/test_acceptance_game_projection.py`
- Modify: `tests/scripts/test_run_real_game.py`
- Modify: `tests/evaluation/test_final_acceptance_integrity.py`

- [ ] **Step 1: 写完整投影和 fail-unsupported 测试**

```python
def test_projection_requires_players_winner_and_status_for_completed_game() -> None:
    projection = AcceptanceGameProjection.from_state(state)
    assert projection.players
    assert projection.winning_faction == "werewolf"
    assert projection.status == "finished"

def test_world_metrics_are_unsupported_without_roles() -> None:
    metrics = compute_world_acceptance_metrics([legacy_projection_without_players])
    assert metrics["possible_world_metrics_supported"] is False
    assert metrics["possible_world_unique_rate"] is None
```

- [ ] **Step 2: 写保存分与离线重算一致性失败测试**

构造带 winner、players、reflection 事件的状态，调用 `save_game_log()` 后重新读取 JSON 和重算，断言逐字段一致。当前实现应因只传 `game_id/events` 而失败。

- [ ] **Step 3: 运行失败测试**

Run: `python -m pytest tests/evaluation/test_acceptance_game_projection.py tests/scripts/test_run_real_game.py tests/evaluation/test_final_acceptance_integrity.py -q --basetemp .tmp/pytest-post-july14-t05`

Expected: FAIL，stored reflection completed=0 或 possible-world 输入缺角色。

- [ ] **Step 4: 先引入终态字段，再实现强类型 projection**

在 `GameState` 增加 `status: Literal["running", "finished", "aborted"] = "running"` 和可选 `termination_reason`。本任务只建立可序列化字段和 legacy 推导：旧状态缺 status 时，winner 非空推导 finished，否则推导 running；真正的异常终止写入语义由 Task 6 实现。

```python
@dataclass(frozen=True)
class AcceptanceGameProjection:
    game_id: str
    events: tuple[dict[str, Any], ...]
    players: dict[str, dict[str, Any]]
    winning_faction: str | None
    status: Literal["running", "finished", "aborted"]
```

acceptance 函数先归一化为 projection。需要字段缺失时返回 `supported=false` 和 `unsupported_reason`，不得用空集合计算伪 0。

- [ ] **Step 5: 调整最终评分顺序**

runner 完成反思与 persistence audit 后才调用一次最终 `compute_game_quality_score()` 并保存。控制台预览可计算临时分，但不得写为最终 `quality_score`。保存函数接受已计算 final quality，避免内部再次在不同状态时点计算。

- [ ] **Step 6: 拆分发言指标**

保留兼容别名一个读取周期，但新写包含 `speech_non_empty_rate`、`speech_model_success_rate`、`speech_terminal_fallback_rate` 和 `speech_semantic_acceptance_rate`。

- [ ] **Step 7: 运行聚焦测试**

Run: `python -m pytest tests/evaluation/test_acceptance_game_projection.py tests/scripts/test_run_real_game.py tests/evaluation/test_final_acceptance_integrity.py tests/evaluation/test_report_acceptance_metrics.py -q --basetemp .tmp/pytest-post-july14-t05-green`

Expected: PASS；保存/重算差异为 0。

- [ ] **Step 8: 提交**

```bash
git add werewolf_agent/core/models.py werewolf_agent/evaluation/game_projection.py werewolf_agent/evaluation/acceptance_audit.py werewolf_agent/evaluation/acceptance_world_metrics.py werewolf_agent/evaluation/acceptance_reflection_metrics.py werewolf_agent/evaluation/acceptance_power_metrics.py werewolf_agent/evaluation/balance_audit.py scripts/run_real_game.py tests/evaluation/test_acceptance_game_projection.py tests/scripts/test_run_real_game.py tests/evaluation/test_final_acceptance_integrity.py tests/evaluation/test_report_acceptance_metrics.py
git commit -m "fix: compute quality from complete game projections"
```

## Task 6: 统一 finished 与 aborted 终态

**Files:**
- Create: `werewolf_agent/runtime/game_termination.py`
- Modify: `werewolf_agent/core/models.py`
- Modify: `werewolf_agent/runtime/game_runner_config.py`
- Modify: `werewolf_agent/runtime/game_runner_execution.py`
- Modify: `werewolf_agent/runtime/executor.py`
- Modify: `scripts/run_real_game.py`
- Create: `tests/runtime/test_game_termination.py`
- Modify: `tests/runtime/test_game_runner.py`
- Modify: `tests/scripts/test_run_real_game.py`

- [ ] **Step 1: 写终态不变量失败测试**

覆盖正常胜利、GraphRecursionError、stuck=50、普通不可恢复异常和 `run_step()` 异常。断言 finished 必有 winner；aborted 必有 reason、last node、phase、step 和 exception type。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/runtime/test_game_termination.py tests/runtime/test_game_runner.py -q --basetemp .tmp/pytest-post-july14-t06`

Expected: FAIL，异常当前只写 warning，状态仍为 night。

- [ ] **Step 3: 实现终止结果**

复用 Task 5 已加入的 `status` 和 `termination_reason`。`game_termination.py` 提供 `finish_game()` 与 `abort_game()`，后者追加 moderator-only `game_aborted` V2 事件。

- [ ] **Step 4: 接入所有 runner 路径**

`run()`、`run_scripted()`、`run_step()` 和 executor 统一调用终止 helper。GraphRecursionError 映射 `graph_recursion_limit`，stuck/max step 映射 `step_limit`，其他异常映射 `unrecoverable_runtime_error`。aborted 仍持久化，不运行 post-game reflection。

- [ ] **Step 5: 实现持久化失败时的最小应急导出**

`GameRunnerConfig` 增加显式 `emergency_artifact_dir`，默认指向运行目录下的 `artifacts/emergency_game_aborts`，CLI 使用本次 `--output-dir` 覆盖。若 abort 状态的正常 repository/event persistence 抛异常，使用同目录临时文件 + 原子 rename 写出 `emergency_abort_<safe_game_id>.json`，只包含 game_id、status、termination_reason、last node、phase、day/night、step、exception type 和 occurred_at；不得包含 prompt、角色真值或私有事件。

测试分别模拟 repository save 失败且应急目录可写、应急目录也不可写。后一种情况必须记录 CRITICAL，并让 CLI/executor 返回非零或 error 状态，不能打印成功。

- [ ] **Step 6: 修正输出文案和统计**

`scripts/run_real_game.py` 只对 finished 记录 `GAME_COMPLETE`；aborted 记录 `GAME_ABORTED reason=...`。平衡统计排除 aborted，稳定性统计包含它。

- [ ] **Step 7: 运行聚焦测试**

Run: `python -m pytest tests/runtime/test_game_termination.py tests/runtime/test_game_runner.py tests/scripts/test_run_real_game.py -q --basetemp .tmp/pytest-post-july14-t06-green`

Expected: PASS；`winner=None GAME_COMPLETE` 为 0。

- [ ] **Step 8: 批次一回归并提交**

Run: `python -m pytest tests/regression/test_post_july14_contract_cases.py tests/runtime/test_event_metadata_v2.py tests/runtime/test_resolution_batches.py tests/agents/test_decision_trace_v2.py tests/evaluation/test_acceptance_game_projection.py tests/runtime/test_game_termination.py -q --basetemp .tmp/pytest-post-july14-batch1`

Expected: PASS。

```bash
git add werewolf_agent/core/models.py werewolf_agent/runtime/game_termination.py werewolf_agent/runtime/game_runner_config.py werewolf_agent/runtime/game_runner_execution.py werewolf_agent/runtime/executor.py scripts/run_real_game.py tests/runtime/test_game_termination.py tests/runtime/test_game_runner.py tests/scripts/test_run_real_game.py
git commit -m "fix: persist explicit aborted game outcomes"
```

---

# 批次二：狼队共识与规则执行

## Task 7: 为狼人夜聊增加结构化立场

**Files:**
- Modify: `werewolf_agent/agents/action_schemas.py`
- Modify: `werewolf_agent/agents/schemas.py`
- Modify: `werewolf_agent/runtime/agent_wolf_actions.py`
- Modify: `werewolf_agent/runtime/nodes/wolf_discussion.py`
- Modify: `werewolf_agent/runtime/wolf_discussion_directives.py`
- Modify: `werewolf_agent/runtime/exposure_audit.py`
- Modify: `tests/runtime/test_wolf_flow.py`
- Modify: `tests/runtime/test_wolf_discussion_directives.py`
- Modify: `tests/runtime/test_agent_wolf_team_plan.py`

- [ ] **Step 1: 写结构化 stance 失败测试**

测试 propose/support/oppose/abstain；非法 wolf、死亡目标、狼队友目标、abstain 带 target 均被拒绝。每个 stance 必须引用同夜 `wolf_discussion` 的 V2 event ID。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/runtime/test_wolf_flow.py tests/runtime/test_wolf_discussion_directives.py -q --basetemp .tmp/pytest-post-july14-t07`

Expected: FAIL，当前夜聊只有自由文本。

- [ ] **Step 3: 扩展狼人私有输出 schema**

在狼人夜聊 action 中加入可选 `target_stance`，其 schema 与设计文档一致。普通白天 speech 不暴露该字段。fallback speech 必须产生 `abstain`，不能正则猜目标。

- [ ] **Step 4: 写入私有事件**

先使用 `new_game_event()` 创建 `wolf_discussion`，取得 event ID，再把验证后的 stance 放入同一 werewolf-team-only payload。计划节点只读取这些结构化字段。

- [ ] **Step 5: 运行聚焦与隐私测试**

Run: `python -m pytest tests/runtime/test_wolf_flow.py tests/runtime/test_wolf_discussion_directives.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_exposure_audit.py -q --basetemp .tmp/pytest-post-july14-t07-green`

Expected: PASS；公共视图中 stance 和狼身份泄漏为 0。

- [ ] **Step 6: 提交**

```bash
git add werewolf_agent/agents/action_schemas.py werewolf_agent/agents/schemas.py werewolf_agent/runtime/agent_wolf_actions.py werewolf_agent/runtime/nodes/wolf_discussion.py werewolf_agent/runtime/wolf_discussion_directives.py werewolf_agent/runtime/exposure_audit.py tests/runtime/test_wolf_flow.py tests/runtime/test_wolf_discussion_directives.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_exposure_audit.py
git commit -m "feat: record structured wolf target stances"
```

## Task 8: 独立聚合主刀与备刀共识

**Files:**
- Create: `werewolf_agent/runtime/wolf_consensus_evidence.py`
- Modify: `werewolf_agent/agents/wolf_team_plan_schema.py`
- Modify: `werewolf_agent/runtime/wolf_team_plan_support.py`
- Modify: `werewolf_agent/runtime/nodes/wolf_discussion.py`
- Modify: `werewolf_agent/runtime/nodes/node_helpers.py`
- Create: `tests/runtime/test_wolf_consensus_evidence.py`
- Modify: `tests/agents/test_wolf_team_plan_schema.py`
- Modify: `tests/runtime/test_wolf_team_plan_support.py`
- Modify: `tests/integration/test_wolf_team_plan_e2e.py`

- [ ] **Step 1: 写共识矩阵失败测试**

至少覆盖：3 狼 2:1、3 狼 1:1:1、3 狼 1:0、全 oppose/abstain、单狼主备、stance 替换旧支持、主刀多数/备刀平票、LLM 推荐与权威聚合不一致。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/runtime/test_wolf_consensus_evidence.py -q --basetemp .tmp/pytest-post-july14-t08`

Expected: FAIL，新聚合器不存在。

- [ ] **Step 3: 实现纯聚合模型**

```python
@dataclass(frozen=True)
class WolfPriorityConsensus:
    priority: Literal["primary", "backup"]
    target_id: str | None
    status: Literal["majority", "single_wolf", "tie", "insufficient", "all_abstain"]
    supporters_by_target: dict[str, tuple[str, ...]]

@dataclass(frozen=True)
class WolfConsensusEvidenceV2:
    night_number: int
    alive_wolf_ids: tuple[str, ...]
    stances: tuple[WolfTargetStance, ...]
    quorum: int
    primary: WolfPriorityConsensus
    backup: WolfPriorityConsensus
```

实现严格遵循规格 §4.2；出现两个目标同时达到严格多数时抛/记录 `consensus_invariant_violation` 并 fail closed。

- [ ] **Step 4: 降级旧计划字段为展示信息**

`evidence_quality` 和 LLM `evidence_from_discussion` 不再控制执行。LLM 推荐与聚合不一致时写 private audit mismatch，不修改权威结果。

- [ ] **Step 5: 接入 `_planned_wolf_kill`**

主刀 status 是 majority/single_wolf 且目标合法时选择主刀；只有主刀执行前非法才读取备刀；备刀 tie/insufficient/all_abstain 进入 NoKillPolicy，不能与主刀平级比较。

- [ ] **Step 6: 运行聚焦测试**

Run: `python -m pytest tests/runtime/test_wolf_consensus_evidence.py tests/agents/test_wolf_team_plan_schema.py tests/runtime/test_wolf_team_plan_support.py tests/integration/test_wolf_team_plan_e2e.py -q --basetemp .tmp/pytest-post-july14-t08-green`

Expected: PASS；单狼主刀和合法多数执行率 100%。

- [ ] **Step 7: 提交**

```bash
git add werewolf_agent/runtime/wolf_consensus_evidence.py werewolf_agent/agents/wolf_team_plan_schema.py werewolf_agent/runtime/wolf_team_plan_support.py werewolf_agent/runtime/nodes/wolf_discussion.py werewolf_agent/runtime/nodes/node_helpers.py tests/runtime/test_wolf_consensus_evidence.py tests/agents/test_wolf_team_plan_schema.py tests/runtime/test_wolf_team_plan_support.py tests/integration/test_wolf_team_plan_e2e.py
git commit -m "fix: derive wolf consensus from authoritative stances"
```

## Task 9: 统一 NoKillPolicy 并覆盖所有出口

**Files:**
- Create: `werewolf_agent/runtime/wolf_no_kill_policy.py`
- Modify: `werewolf_agent/runtime/nodes/wolf_consensus.py`
- Modify: `werewolf_agent/runtime/nodes/node_helpers.py`
- Modify: `werewolf_agent/runtime/nodes/wolf_night_nodes.py`
- Modify: `werewolf_agent/customization/validators.py`
- Modify: `werewolf_agent/engine/ruleset_loader.py`
- Modify: `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`
- Create: `tests/runtime/test_wolf_no_kill_policy.py`
- Modify: `tests/runtime/test_wolf_kill_support.py`
- Modify: `tests/runtime/test_wolf_night_nodes.py`
- Modify: `tests/runtime/test_wolf_flow.py`

- [ ] **Step 1: 写统一空刀失败测试**

逐一覆盖 strategic_abstain、true_tie、insufficient_quorum、invalid_primary、invalid_backup、plan_generation_failed 和 provider_unavailable。断言所有路径产生相同 schema 的 `NoKillDecision`。

- [ ] **Step 2: 写计数/重置/技能抵消测试**

连续两夜结算前空刀后第三夜恢复；`wolf_kill_selected` 立即清零；女巫救或守卫保护不重新累加；没有合法目标输出 `forced_recovery_no_legal_target`。

- [ ] **Step 3: 运行失败测试**

Run: `python -m pytest tests/runtime/test_wolf_no_kill_policy.py tests/runtime/test_wolf_kill_support.py -q --basetemp .tmp/pytest-post-july14-t09`

Expected: FAIL，计划提前返回仍绕过 legacy 保护。

- [ ] **Step 4: 实现纯策略与确定性评分**

```python
score = (
    primary_positive_support[target_id],
    backup_positive_support[target_id],
    -seat_index[target_id],
)
```

默认 `max_consecutive_pre_resolution_no_kill=2`。恢复事件记录原始 reasons、计数、候选评分和最终目标。

- [ ] **Step 5: 删除分散的连续空刀分支**

`wolf_consensus.py` 和 `_planned_wolf_kill()` 不再自行构造不同 no-kill 事件。所有 `None` 出口调用 policy；保留兼容事件名时 payload 必须使用统一 reason code。

- [ ] **Step 6: 运行聚焦测试**

Run: `python -m pytest tests/runtime/test_wolf_no_kill_policy.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_wolf_night_nodes.py tests/runtime/test_wolf_flow.py -q --basetemp .tmp/pytest-post-july14-t09-green`

Expected: PASS；任何原因连续空刀不超过阈值。

- [ ] **Step 7: 批次二集成与提交**

Run: `python -m pytest tests/runtime/test_wolf_consensus_evidence.py tests/runtime/test_wolf_no_kill_policy.py tests/integration/test_wolf_team_plan_e2e.py -q --basetemp .tmp/pytest-post-july14-batch2`

Expected: PASS。

```bash
git add werewolf_agent/runtime/wolf_no_kill_policy.py werewolf_agent/runtime/nodes/wolf_consensus.py werewolf_agent/runtime/nodes/node_helpers.py werewolf_agent/runtime/nodes/wolf_night_nodes.py werewolf_agent/customization/validators.py werewolf_agent/engine/ruleset_loader.py config/rulesets/pre_witch_hunter_idiot_mixed.yaml tests/runtime/test_wolf_no_kill_policy.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_wolf_night_nodes.py tests/runtime/test_wolf_flow.py
git commit -m "fix: route every wolf no-kill through one policy"
```

---

# 批次三：回退、反思和行为质量

## Task 10: 验证 provider fallback 路由并保留失败分类

**Files:**
- Create: `werewolf_agent/model_gateway/fallback_policy.py`
- Modify: `werewolf_agent/model_gateway/router_config.py`
- Modify: `werewolf_agent/model_gateway/router_selection.py`
- Modify: `werewolf_agent/model_gateway/router.py`
- Modify: `werewolf_agent/model_gateway/generation_attempt_context.py`
- Modify: `config/models.yaml`
- Create: `tests/model_gateway/test_provider_fallback_policy.py`
- Modify: `tests/model_gateway/test_router.py`
- Modify: `tests/model_gateway/test_generation_attempt_context.py`

- [ ] **Step 1: 写启动期路由验证失败测试**

覆盖同 provider/model、重复候选、不同 provider 同 model、同 provider 不同 model、不满足 reasoning minimum 和没有备用路由。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/model_gateway/test_provider_fallback_policy.py tests/model_gateway/test_router.py -q --basetemp .tmp/pytest-post-july14-t10`

Expected: FAIL，同路由错误当前到结果翻译阶段才暴露。

- [ ] **Step 3: 实现 route policy**

`build_fallback_routes(primary, candidates, minimum_reasoning)` 返回去重后的不可变路线；同 `(provider, model)` 被拒绝。无可用路由返回结构化 `fallback_route_unavailable`，不产生非法 attempt sequence。

- [ ] **Step 4: 在运行时二次验证**

router 每次切换前验证下一路由与当前不同；动态配置异常进入受控 terminal fallback。`translate_decision_outcome()` 保留最终防御断言，但不再承担首次发现配置错误的责任。

- [ ] **Step 5: 运行聚焦测试**

Run: `python -m pytest tests/model_gateway/test_provider_fallback_policy.py tests/model_gateway/test_router.py tests/model_gateway/test_generation_attempt_context.py tests/runtime/test_decision_outcomes.py -q --basetemp .tmp/pytest-post-july14-t10-green`

Expected: PASS；same-route provider fallback 为 0。

- [ ] **Step 6: 提交**

```bash
git add werewolf_agent/model_gateway/fallback_policy.py werewolf_agent/model_gateway/router_config.py werewolf_agent/model_gateway/router_selection.py werewolf_agent/model_gateway/router.py werewolf_agent/model_gateway/generation_attempt_context.py config/models.yaml tests/model_gateway/test_provider_fallback_policy.py tests/model_gateway/test_router.py tests/model_gateway/test_generation_attempt_context.py tests/runtime/test_decision_outcomes.py
git commit -m "fix: validate provider fallback routes before execution"
```

## Task 11: 拆分任务专用 terminal fallback 和发言指标

**Files:**
- Modify: `werewolf_agent/agents/player_fallback_speech.py`
- Modify: `werewolf_agent/agents/player_failures.py`
- Modify: `werewolf_agent/agents/player_action_result.py`
- Modify: `werewolf_agent/agents/player_action_flow.py`
- Modify: `werewolf_agent/runtime/agent_action_audit.py`
- Modify: `werewolf_agent/evaluation/acceptance_terminal_semantic_metrics.py`
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Create: `tests/agents/test_task_terminal_fallbacks.py`
- Modify: `tests/runtime/test_speech_quality.py`
- Modify: `tests/evaluation/test_semantic_acceptance_invariants.py`

- [ ] **Step 1: 写任务 fallback 失败测试**

普通发言、警长发言、夜间动作、反思、遗言和狼队计划分别断言专用 fallback kind。所有终退必须保留 `original_failure_code` 与 `failure_stage`。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/agents/test_task_terminal_fallbacks.py tests/runtime/test_speech_quality.py -q --basetemp .tmp/pytest-post-july14-t11`

Expected: FAIL，当前理由被通用“结构化输出失败”压平。

- [ ] **Step 3: 实现任务分派表**

以 task type 映射 fallback builder；反思返回 not_generated，狼计划只消费 stance，夜间动作只产生合法确定性动作或明确弃权。禁止自由文本正则从失败输出猜动作。

- [ ] **Step 4: 写入 V2 trace 分类**

设置 `generated_by="terminal_fallback"`、`terminal_failure_code`、`failure_stage` 和 `fallback_kind`。原始错误正文不导出，只保留稳定码。

- [ ] **Step 5: 修正质量指标**

非空 fallback 只增加 `speech_non_empty_rate`，不得增加 `speech_model_success_rate`；terminal fallback 和 semantic acceptance 独立统计。

- [ ] **Step 6: 运行聚焦测试**

Run: `python -m pytest tests/agents/test_task_terminal_fallbacks.py tests/runtime/test_speech_quality.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_game_balance_batch.py -q --basetemp .tmp/pytest-post-july14-t11-green`

Expected: PASS，失败码覆盖率 100%。

- [ ] **Step 7: 提交**

```bash
git add werewolf_agent/agents/player_fallback_speech.py werewolf_agent/agents/player_failures.py werewolf_agent/agents/player_action_result.py werewolf_agent/agents/player_action_flow.py werewolf_agent/runtime/agent_action_audit.py werewolf_agent/evaluation/acceptance_terminal_semantic_metrics.py werewolf_agent/evaluation/balance_audit.py tests/agents/test_task_terminal_fallbacks.py tests/runtime/test_speech_quality.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_game_balance_batch.py
git commit -m "fix: preserve task-specific terminal fallback outcomes"
```

## Task 12: 建立反思事务并禁止零条目成功

**Files:**
- Create: `werewolf_agent/runtime/reflection_transaction.py`
- Modify: `werewolf_agent/runtime/nodes/summary.py`
- Modify: `werewolf_agent/runtime/reflection_events.py`
- Modify: `werewolf_agent/runtime/game_runner_memory.py`
- Modify: `werewolf_agent/evaluation/acceptance_reflection_metrics.py`
- Create: `tests/runtime/test_reflection_transaction.py`
- Modify: `tests/runtime/test_reflection_security_contract.py`
- Modify: `tests/evaluation/test_report_acceptance_metrics.py`
- Modify: `tests/evaluation/test_final_acceptance_integrity.py`

- [ ] **Step 1: 写反思状态机失败测试**

覆盖 not_requested → generated → schema_validated → facts_verified → lessons_verified → persisted；非法跃迁必须拒绝。局级状态覆盖 complete、partial、no_valid_entries、persistence_failed 和 not_run。

- [ ] **Step 2: 写零条目与回滚失败测试**

```python
def test_zero_expected_entries_is_not_complete() -> None:
    result = summarize_reflection_transaction(entries=[])
    assert result.status == "no_valid_entries"
    assert result.persistence_complete is False
```

保留现有写入失败、回读失败和 rollback 测试，将其迁移到聚焦文件，不删除原有保障。

- [ ] **Step 3: 运行失败测试**

Run: `python -m pytest tests/runtime/test_reflection_transaction.py tests/evaluation/test_final_acceptance_integrity.py -q --basetemp .tmp/pytest-post-july14-t12`

Expected: FAIL，当前 `all([])` 允许 persistence success。

- [ ] **Step 4: 实现事务模型并接入 summary**

每名玩家记录 `failure_stage/failure_code/decision_id`。`reflection_complete` 改为局级结果事件，保留每名玩家安全 verification。没有有效 lesson 时写 `reflection_no_valid_entries`。

- [ ] **Step 5: 改造 persistence audit**

`expected_entry_count=0` 时 `persistence_complete=False`、status=no_valid_entries。非空事务继续使用现有原子写入、回读和回滚；`decision_id/claim_id/entry_id` 必须全链一致。

- [ ] **Step 6: 修正 acceptance**

completed game 在 `complete` 时计 audited；`partial` 只有在至少一条 lesson 已持久化、全部失败玩家都有明确 failure stage/code、且已持久化条目全部通过污染审计时才计 audited。`no_valid_entries` 与 `persistence_failed` 不能使 `reflection_contamination_metrics_supported=True`。

- [ ] **Step 7: 运行聚焦测试**

Run: `python -m pytest tests/runtime/test_reflection_transaction.py tests/runtime/test_reflection_security_contract.py tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_final_acceptance_integrity.py -q --basetemp .tmp/pytest-post-july14-t12-green`

Expected: PASS；零条目伪成功为 0。

- [ ] **Step 8: 提交**

```bash
git add werewolf_agent/runtime/reflection_transaction.py werewolf_agent/runtime/nodes/summary.py werewolf_agent/runtime/reflection_events.py werewolf_agent/runtime/game_runner_memory.py werewolf_agent/evaluation/acceptance_reflection_metrics.py tests/runtime/test_reflection_transaction.py tests/runtime/test_reflection_security_contract.py tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_final_acceptance_integrity.py
git commit -m "fix: make reflection persistence a non-vacuous transaction"
```

## Task 13: 修复分层上下文、备刀提示和最终 prompt 送达证明

**Files:**
- Modify: `werewolf_agent/runtime/agent_wolf_actions.py`
- Modify: `werewolf_agent/runtime/wolf_discussion_directives.py`
- Modify: `werewolf_agent/agents/prompt_system.py`
- Modify: `werewolf_agent/agents/prompt_sections.py`
- Modify: `werewolf_agent/agents/player_generation_request.py`
- Modify: `werewolf_agent/model_gateway/final_prompt_observer.py`
- Modify: `werewolf_agent/runtime/exposure_audit.py`
- Create: `tests/runtime/test_wolf_prompt_contract.py`
- Modify: `tests/model_gateway/test_final_prompt_observer.py`
- Modify: `tests/agents/test_prompt_injection_fixes.py`

- [ ] **Step 1: 写分层上下文失败测试**

构造 4 狼 × 3 轮发言，断言所有结构化 stance 都可见；最近 8 条保留原文，更早条目进入确定性摘要；记录 injected event IDs、原文/摘要/截断计数。

- [ ] **Step 2: 写语义合同失败测试**

最终系统提示必须明确：备刀不是女巫救后的第二刀、死亡玩家不可作为目标、不得把系统候选描述为局内事实、队长不得伪造支持者。

- [ ] **Step 3: 写 HMAC 送达证明测试**

使用固定测试密钥捕获 provider HTTP 前 `FinalPromptAssembly.system_bytes`，断言 template ID/version、必需区块确认、长度和 HMAC-SHA256 可复算；游戏日志不含原 prompt 或密钥。

- [ ] **Step 4: 运行失败测试**

Run: `python -m pytest tests/runtime/test_wolf_prompt_contract.py tests/model_gateway/test_final_prompt_observer.py tests/agents/test_prompt_injection_fixes.py -q --basetemp .tmp/pytest-post-july14-t13`

Expected: FAIL，当前仅 persona 有最终 prompt 证明，夜聊仍截最后 8 条。

- [ ] **Step 5: 实现分层上下文和版本化合同**

结构化状态与 stance 不受文本预算影响；原文/摘要按明确预算合并。动态存活状态每夜重建，旧计划只作为历史。

- [ ] **Step 6: 扩展现有 final prompt observer**

复用现有 run-scoped HMAC，不创建第二套密钥系统。增加 `prompt_contract_id/version` 和 required section confirmations；关键区块缺失时调用前 fail closed。

- [ ] **Step 7: 运行聚焦与隐私测试**

Run: `python -m pytest tests/runtime/test_wolf_prompt_contract.py tests/model_gateway/test_final_prompt_observer.py tests/agents/test_prompt_injection_fixes.py tests/runtime/test_exposure_audit.py -q --basetemp .tmp/pytest-post-july14-t13-green`

Expected: PASS；完整原 prompt 泄漏为 0。

- [ ] **Step 8: 提交**

```bash
git add werewolf_agent/runtime/agent_wolf_actions.py werewolf_agent/runtime/wolf_discussion_directives.py werewolf_agent/agents/prompt_system.py werewolf_agent/agents/prompt_sections.py werewolf_agent/agents/player_generation_request.py werewolf_agent/model_gateway/final_prompt_observer.py werewolf_agent/runtime/exposure_audit.py tests/runtime/test_wolf_prompt_contract.py tests/model_gateway/test_final_prompt_observer.py tests/agents/test_prompt_injection_fixes.py tests/runtime/test_exposure_audit.py
git commit -m "fix: make wolf context and prompt contracts auditable"
```

## Task 14: 补齐自爆、猎人和预言家机会链

**Files:**
- Create: `werewolf_agent/runtime/skill_opportunity_events.py`
- Modify: `werewolf_agent/runtime/nodes/day_discussion.py`
- Modify: `werewolf_agent/runtime/nodes/sheriff_registration.py`
- Modify: `werewolf_agent/runtime/nodes/sheriff_vote.py`
- Modify: `werewolf_agent/runtime/nodes/sheriff_pk.py`
- Modify: `werewolf_agent/runtime/nodes/skills.py`
- Modify: `werewolf_agent/runtime/nodes/night_specialists.py`
- Modify: `werewolf_agent/runtime/exposure_audit.py`
- Modify: `werewolf_agent/evaluation/acceptance_power_metrics.py`
- Create: `tests/runtime/test_skill_opportunity_events.py`
- Modify: `tests/runtime/test_hunter_flow.py`
- Modify: `tests/runtime/test_seer_flow.py`
- Modify: `tests/integration/test_directive_role_gating.py`

- [ ] **Step 1: 写机会链失败测试**

自爆：无机会不写事件；有机会写 opportunity 后 selected/declined，再 resolved。猎人区分 selected/declined/blocked。预言家按可查验夜写 selected/repaired/skipped/resolved。

- [ ] **Step 2: 写 visibility 失败测试**

opportunity/choice 只对 actor 和 moderator 可见；自爆/枪击公开结算事件不含私有理由；预言家查验结果绝不进入 public。

- [ ] **Step 3: 运行失败测试**

Run: `python -m pytest tests/runtime/test_skill_opportunity_events.py tests/runtime/test_hunter_flow.py tests/runtime/test_seer_flow.py -q --basetemp .tmp/pytest-post-july14-t14`

Expected: FAIL，当前只有部分 selected/declined 事件。

- [ ] **Step 4: 实现事件 builder 与角色私有投影**

若 visibility 枚举不能表达 actor+moderator，权威事件保存 moderator-only，角色视图按 actor ID 投影；不得扩大为 public。

- [ ] **Step 5: 接入所有触发点并修正指标分母**

自爆在每个实际可选择节点写 opportunity；猎人在 pending shot 进入时写；预言家在存活且能力可用夜写。指标使用 opportunity 作为分母。

- [ ] **Step 6: 运行聚焦与 exposure 测试**

Run: `python -m pytest tests/runtime/test_skill_opportunity_events.py tests/runtime/test_hunter_flow.py tests/runtime/test_seer_flow.py tests/integration/test_directive_role_gating.py tests/runtime/test_exposure_audit.py -q --basetemp .tmp/pytest-post-july14-t14-green`

Expected: PASS；公共角色真值泄漏为 0。

- [ ] **Step 7: 批次三回归并提交**

Run: `python -m pytest tests/model_gateway/test_provider_fallback_policy.py tests/agents/test_task_terminal_fallbacks.py tests/runtime/test_reflection_transaction.py tests/runtime/test_wolf_prompt_contract.py tests/runtime/test_skill_opportunity_events.py -q --basetemp .tmp/pytest-post-july14-batch3`

Expected: PASS。

```bash
git add werewolf_agent/runtime/skill_opportunity_events.py werewolf_agent/runtime/nodes/day_discussion.py werewolf_agent/runtime/nodes/sheriff_registration.py werewolf_agent/runtime/nodes/sheriff_vote.py werewolf_agent/runtime/nodes/sheriff_pk.py werewolf_agent/runtime/nodes/skills.py werewolf_agent/runtime/nodes/night_specialists.py werewolf_agent/runtime/exposure_audit.py werewolf_agent/evaluation/acceptance_power_metrics.py tests/runtime/test_skill_opportunity_events.py tests/runtime/test_hunter_flow.py tests/runtime/test_seer_flow.py tests/integration/test_directive_role_gating.py
git commit -m "feat: audit power-role action opportunities"
```

---

# 批次四：闭环验证与平衡回归

## Task 15: 建立全问题端到端闭环测试

**Files:**
- Create: `tests/integration/test_post_july14_repair_closure.py`
- Modify: `tests/regression/test_post_july14_contract_cases.py`
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Modify: `werewolf_agent/evaluation/acceptance_audit.py`
- Modify: `werewolf_agent/runtime/exposure_audit.py`

- [ ] **Step 1: 写端到端失败测试**

场景至少包含：

```text
三狼 2:1 共识 → 主刀 → 女巫救 → 平安夜归因为技能抵消
单狼主备 → 主刀执行
主刀非法 + 备刀多数 → 备刀执行
连续两夜任意结算前空刀 → 第三夜确定性恢复
递归上限 → aborted JSON
终局 → 有效/无效 reflection → 最终 quality
V1 日志 → 兼容投影 → unsupported/normalized 标识
```

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/integration/test_post_july14_repair_closure.py -q --basetemp .tmp/pytest-post-july14-t15`

Expected: FAIL，直到前 14 项全部接通。

- [ ] **Step 3: 补齐跨模块接线，不新增旁路**

只修复集成遗漏；如果需要新增业务规则，返回对应任务和规格审查，不在集成测试中临时打补丁。

- [ ] **Step 4: 加入 15 项回归映射断言**

每个 K/N case 必须映射至少一个测试节点 ID，测试报告输出缺失映射时失败。

- [ ] **Step 5: 运行集成与隐私测试**

Run: `python -m pytest tests/integration/test_post_july14_repair_closure.py tests/regression/test_post_july14_contract_cases.py tests/runtime/test_exposure_audit.py tests/evaluation/test_final_acceptance_integrity.py -q --basetemp .tmp/pytest-post-july14-t15-green`

Expected: PASS；15/15 覆盖，公共敏感字段泄漏为 0。

- [ ] **Step 6: 提交**

```bash
git add tests/integration/test_post_july14_repair_closure.py tests/regression/test_post_july14_contract_cases.py werewolf_agent/evaluation/balance_audit.py werewolf_agent/evaluation/acceptance_audit.py werewolf_agent/runtime/exposure_audit.py
git commit -m "test: close every post-july14 audit issue"
```

## Task 16: 扩展自动验收门禁与精确十局 soak

**Files:**
- Modify: `scripts/evaluate_audit_closure_thresholds.py`
- Modify: `scripts/run_audit_closure_soak.ps1`
- Modify: `scripts/run_real_game.py`
- Modify: `tests/scripts/test_evaluate_audit_closure_thresholds.py`
- Modify: `tests/scripts/test_run_audit_closure_soak.py`
- Create: `docs/audits/2026-07-15-post-july14-repair-verification.md`

- [ ] **Step 1: 写新门禁失败测试**

新增并断言：retry consistency=0、malformed batch=0、same-route fallback=0、saved/offline diff=0、winner-null complete=0、abort coverage=100%、合法多数与单狼执行=100%、空反思成功=0、source ID 可追溯=100%、公共泄漏=0。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/scripts/test_evaluate_audit_closure_thresholds.py tests/scripts/test_run_audit_closure_soak.py -q --basetemp .tmp/pytest-post-july14-t16`

Expected: FAIL，新指标尚未接入 gate。

- [ ] **Step 3: 实现门禁和隔离产物目录**

soak 必须显式接收 10 个固定 seed，输出到新的唯一 artifact root；运行前确认目录为空，不读取根目录旧 `game_*.json`。中止局保留但不补跑成“恰好十份 JSON”；gate 明确报告 finished/aborted。

- [ ] **Step 4: 运行聚焦测试**

Run: `python -m pytest tests/scripts/test_evaluate_audit_closure_thresholds.py tests/scripts/test_run_audit_closure_soak.py -q --basetemp .tmp/pytest-post-july14-t16-green`

Expected: PASS。

- [ ] **Step 5: 运行批次一至三的完整自动测试**

Run: `python -m pytest tests/agents tests/runtime tests/model_gateway tests/evaluation tests/integration tests/scripts -q --basetemp .tmp/pytest-post-july14-full`

Expected: pytest 正常退出且 0 failed。若超时，即为未验证，必须拆分重跑直到每个分组都有最终 summary。

- [ ] **Step 6: 运行精确十局稳定性 soak**

Run: `pwsh -NoProfile -Command "& './scripts/run_audit_closure_soak.ps1' -Seeds (714001..714010) -ArtifactRoot 'artifacts/post_july14_repair_soak'"`

Expected: 10 次启动记录；每次结果明确 finished 或 aborted；所有硬门禁全绿。任何 P1 失败立即停止平衡测试。

- [ ] **Step 7: 写验证报告**

报告逐项记录命令、退出码、pass count、10 局 artifact root、finished/aborted 数、K/N 门禁结果和任何未满足项。不得只写“看起来正常”。

- [ ] **Step 8: 提交**

```bash
git add scripts/evaluate_audit_closure_thresholds.py scripts/run_audit_closure_soak.ps1 scripts/run_real_game.py tests/scripts/test_evaluate_audit_closure_thresholds.py tests/scripts/test_run_audit_closure_soak.py docs/audits/2026-07-15-post-july14-repair-verification.md
git commit -m "test: enforce post-july14 repair closure gates"
```

## Task 17: 执行独立 30 局平衡批次

**Files:**
- Create: `scripts/run_balance_batch.ps1`
- Modify: `scripts/analyze_recent_balance.py`
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Create: `config/evaluation/post_july14_balance_seeds.txt`
- Create: `tests/scripts/test_run_balance_batch.py`
- Modify: `tests/scripts/test_analyze_recent_balance.py`
- Create: `docs/audits/2026-07-15-post-july14-balance-recheck.md`

- [ ] **Step 1: 写分层平衡指标失败测试**

指标必须区分狼刀选择、实际死亡、女巫/守卫抵消、主动空刀、系统空刀、白天放逐、技能机会/选择、terminal fallback 和 aborted。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/scripts/test_analyze_recent_balance.py tests/evaluation/test_game_balance_batch.py -q --basetemp .tmp/pytest-post-july14-t17`

Expected: FAIL，现有报告缺少部分 V2 分层。

- [ ] **Step 3: 实现分层聚合和置信区间**

aborted 只进稳定性，不进胜率。输出阵营胜率、Wilson 区间、平均天数、异常长局、狼刀转化、白天/夜间死亡占比、预言家有效查验夜、猎人/自爆机会率和 fallback 与胜率关联。

- [ ] **Step 4: 运行聚焦测试**

Run: `python -m pytest tests/scripts/test_analyze_recent_balance.py tests/evaluation/test_game_balance_batch.py -q --basetemp .tmp/pytest-post-july14-t17-green`

Expected: PASS。

- [ ] **Step 5: 实现并测试独立平衡批次编排器**

`run_balance_batch.ps1` 与“精确十局验收 soak”分离：读取 seed file，持续运行直到得到 30 个 finished 或候选 seed 耗尽；aborted 保留在稳定性产物中但不计 finished。seed file 至少准备 60 个与 714001..714010 不重叠的唯一 seed。成功时脚本必须输出唯一一行 `Balance batch artifact: <absolute-path>`，供下一步精确消费。

Run: `python -m pytest tests/scripts/test_run_balance_batch.py -q --basetemp .tmp/pytest-post-july14-t17-batch`

Expected: PASS，fake runner 场景证明 aborted 会消耗 seed、不会污染 finished 计数，产物路径不能逃逸独立目录。

- [ ] **Step 6: 运行 30 局独立批次**

使用与十局 soak 不重叠的固定 seeds，输出到新的唯一目录。必须获得 30 个 finished 样本；aborted 单列并补充新 seed，不能把 aborted 混入胜率分母。

Run: `$batchOutput = & pwsh -NoProfile -File scripts/run_balance_batch.ps1 -SeedFile config/evaluation/post_july14_balance_seeds.txt -RequiredFinished 30 -ArtifactRoot artifacts/post_july14_balance; $artifact = (($batchOutput | Select-String '^Balance batch artifact: ').Line -split ': ', 2)[1]; if (-not $artifact) { throw 'balance artifact path missing' }`

Expected: 30 个 finished 样本，结构门禁保持全绿。

- [ ] **Step 7: 生成平衡复核报告**

Run: `python scripts/analyze_recent_balance.py --input-dir $artifact --output docs/audits/2026-07-15-post-july14-balance-recheck.md`

报告不得因 30 局点估计直接宣称永久平衡；置信区间过宽时建议扩样。只有此阶段允许提出数值平衡调整，且需要新设计审批。

- [ ] **Step 8: 最终验证**

Run: `git diff --check`

Run: `python -m pytest tests/agents tests/runtime tests/model_gateway tests/evaluation tests/integration tests/scripts -q --basetemp .tmp/pytest-post-july14-final`

Expected: `git diff --check` 无输出且 pytest 0 failed；所有长测试均有最终 summary。

- [ ] **Step 9: 提交**

```bash
git add scripts/run_balance_batch.ps1 scripts/analyze_recent_balance.py werewolf_agent/evaluation/balance_audit.py tests/scripts/test_run_balance_batch.py tests/scripts/test_analyze_recent_balance.py config/evaluation/post_july14_balance_seeds.txt docs/audits/2026-07-15-post-july14-balance-recheck.md
git commit -m "docs: report post-july14 balance recheck"
```

---

## 最终完成清单

- [ ] K1–K3、N1–N12 每项至少有一个失败后转绿的自动化回归测试。
- [ ] 新游戏事件、死亡批次、决策 trace、终态和质量分全部使用 V2 写契约。
- [ ] V1 JSON、事件、死亡批次和 trace 可只读兼容，且带 normalized/unsupported 标识。
- [ ] `attempt_retry_consistency_error_count == 0`。
- [ ] malformed death batch 与跨日死亡注入均为 0。
- [ ] same-route provider fallback 为 0。
- [ ] 合法多数和单狼主刀执行率 100%。
- [ ] 连续结算前空刀不超过 ruleset 阈值。
- [ ] 零有效反思不再被判成功。
- [ ] 保存 quality 与完整 JSON 离线重算逐字段一致。
- [ ] finished/aborted 终态不变量全部成立。
- [ ] public exposure audit 对所有新增事件全绿。
- [ ] 精确十局稳定性 soak 硬门禁全绿。
- [ ] 30 个 finished 样本的平衡报告完成，aborted 未污染胜率。
- [ ] 每批有独立提交和代码审查证据。
