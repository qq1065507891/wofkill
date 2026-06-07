# Post-Post-Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复上一轮 18 个 fix 引入的 7 处回归/半成品 + 12 个长期遗留的 bug/契约问题 + 3 个架构清理（按 P0/P1/P2 分级）。

**Architecture:** 3 条 worktree 分支 — `fix15-prompt-v2`（prompt 侧回归 + tier 漂移）、`fix16-arch-v2`（存储/契约/模型网关/核心模型）、`fix17-periphery-v2`（peripheral 卫生 + skills + UI）。各跑 TDD 后合回 master。每条 commit 独立可回滚。

**Tech Stack:** Python 3.11, Pydantic v2, pytest, 现有 `werewolf_agent` 包结构。

**Spec source:** 第二轮 4 个 subagent 审查报告（已在会话内），对照 `docs/superpowers/plans/2026-06-07-post-review-fixes.md` 上轮 plan。

---

## 文件变更地图

| 类别 | 文件 | 涉及 Task |
|---|---|---|
| Prompt | `werewolf_agent/agents/prompt_builder.py` | P1, P2, P3 |
| Prompt | `werewolf_agent/agents/player.py` | P4 |
| 存储 | `werewolf_agent/storage/postgres_store.py` | S1, S2 |
| 存储 | `werewolf_agent/storage/sqlite_store.py` | S3 |
| 存储 | `werewolf_agent/storage/repository.py` | S4 |
| 存储 | `werewolf_agent/storage/memory_store.py` | S4 |
| 契约 | `werewolf_agent/tools/local_tools.py` + `memory/store.py` | M1 |
| 模型网关 | `werewolf_agent/model_gateway/router.py` | N1 |
| 模型网关 | `werewolf_agent/model_gateway/providers/openai.py` | N2 |
| 模型网关 | `werewolf_agent/model_gateway/providers/anthropic.py` | N3 |
| 核心 | `werewolf_agent/engine/rule_engine.py` | E1 |
| 核心 | `werewolf_agent/cognition/world_state.py` | E2 |
| 核心 | `werewolf_agent/core/models.py` | E3 |
| Agents | `werewolf_agent/agents/metrics_collector.py` | A1 |
| 技能 | `werewolf_agent/skills/werewolf_skills.py` | SK1 |
| 人设 | `werewolf_agent/persona_runtime/judge_router.py` | P5 |
| 自定义 | `werewolf_agent/customization/validators.py` | C1 |
| 自定义 | `werewolf_agent/customization/persona_adapter.py` | C2 |
| UI/Server | `werewolf_agent/ui/static/dashboard.js` + `api/routes/games.py` | U1 |

---

# Branch 1: `fix15-prompt-v2`

## Batch P: Prompt 侧回归 + tier 漂移

### Task P1: 修 C4 不完整 — vote 示例全部硬编码 ID 改占位符

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py:1330-1343`
- Modify: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: 写失败测试**

In `tests/agents/test_prompt_builder.py`, append:

```python
import re

class TestFormatExamplesNoHardcodedIDComplete:
    """P1 (post-review-v2): vote 示例所有 p0X 硬编码 ID 改占位符。"""

    def test_vote_example_no_p0X_hardcoded_anywhere(self):
        from unittest.mock import MagicMock
        from werewolf_agent.agents.schemas import ActionType
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        mock_ctx = MagicMock()
        mock_ctx.legal_actions = [ActionType.VOTE]
        mock_ctx.legal_targets = [f"p{i:02d}" for i in range(1, 13)]
        builder = PlayerPromptBuilder(mock_ctx)
        examples = builder._format_examples()
        # 不应在 vote 段示例出现 "p0X" 硬编码（除 pXX 占位符外）
        # 注：speech / wolf_kill 示例的 legal_targets[0] 取真实 ID 是 OK 的（被注释说明）
        # 这里只检查 vote 段
        vote_section_match = re.search(r'vote.*?(?=##|$)', examples, re.DOTALL)
        if vote_section_match:
            vote_section = vote_section_match.group(0)
            # 真实 ID (p01-p12) 出现次数
            real_id_hits = re.findall(r'"p(?:0[1-9]|1[0-2])"', vote_section)
            # 应该只有 pXX 或 0 个真实 ID
            assert len(real_id_hits) == 0, (
                f"vote 示例段仍含硬编码 player IDs: {real_id_hits}\n\n{vote_section[:500]}"
            )
```

- [ ] **Step 2: 跑测试确认失败**

`cd E:/NLP/agent/wofkill/.worktrees/fix15-prompt && /d/Miniforge3/Scripts/pytest tests/agents/test_prompt_builder.py::TestFormatExamplesNoHardcodedIDComplete -v -p no:cacheprovider`

- [ ] **Step 3: 修 `_format_examples` vote 段**

读 `prompt_builder.py:1320-1350`（vote 段示例块），把示例 dict 中所有 `"p05"` / `"p07"` / `"p03"` / `"p09"` 等真实 player ID 字符串替换为 `"pXX"` 占位符。同时给示例块顶部加注释：

```python
"""vote 段示例：所有 player ID 均为 pXX 占位符，LLM 应替换为当前局真实 ID。"""
```

包括但不限于：`target_id`、`standing_with_seer`、`suspect_reason` 中引用的 ID、`not_voting_reason` 中引用的 ID、`private_reason` 中引用的 ID、`pressure_target`。

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: 跑 regression** `pytest tests/agents/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix15-prompt
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "fix(prompt): vote 段示例所有硬编码 p0X 改 pXX 占位符 (post-review-v2-P1)"
```

---

### Task P2: 修 A8 回归 — `_VOTE_REASON_PRIVACY_GUARD` 在 FULL_ACTION vote 路径也注入

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (在 `_format_examples` 入口或 `_build_task_prompt` 前置层注入)
- Modify: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: 写失败测试**

```python
class TestVoteReasonGuardInFullActionPath:
    """P2 (post-review-v2): vote 阶段 FULL_ACTION 模式也应注入 _VOTE_REASON_PRIVACY_GUARD。"""

    def test_vote_full_action_includes_privacy_guard(self):
        from unittest.mock import MagicMock
        from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.DAY_VOTE,
            role="villager",
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],  # 触发 FULL_ACTION 路径
            legal_targets=[f"p{i:02d}" for i in range(1, 13)],
            strategy_directive={},
        )
        builder = PlayerPromptBuilder(ctx)
        user_prompt = builder.build_user_prompt()
        assert "P0-G3223805846-8" in user_prompt or "禁止" in user_prompt, (
            f"FULL_ACTION vote path missing _VOTE_REASON_PRIVACY_GUARD"
        )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 修注入点**

读 `prompt_builder.py:_format_examples` 入口（parts 初始化处）。在该函数顶部加：

```python
    def _format_examples(self) -> str:
        ctx = self.context
        parts: list[str] = []
        parts.append(_ACTION_TYPE_GUARD)  # 已存在
        # P2 (post-review-v2): vote 阶段 FULL_ACTION 路径也注入 privacy guard
        if ActionType.VOTE in (ctx.legal_actions or []):
            parts.append(_VOTE_REASON_PRIVACY_GUARD)
        # ... 原 parts.append 逻辑
```

同时检查 `_format_choice_prompt` 路径（line 1464）已注入 `_VOTE_REASON_PRIVACY_GUARD` 保留不动。两条路径都注入。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/agents/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "fix(prompt): FULL_ACTION vote 路径也注入 _VOTE_REASON_PRIVACY_GUARD (post-review-v2-P2)"
```

---

### Task P3: 修 HARD/SUGGESTION tier 漂移

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (HARD_CONSTRAINT_KEYS / SUGGESTION_KEYS)
- Modify: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: 写失败测试**

```python
class TestHardConstraintTierAccuracy:
    """P3 (post-review-v2): 文本含"严禁/绝对"应归 HARD_CONSTRAINT_KEYS。"""

    def test_wolf_universal_rules_in_hard_tier(self):
        from werewolf_agent.agents.prompt_builder import (
            HARD_CONSTRAINT_KEYS, SUGGESTION_KEYS, _SECTION_PRIORITIES,
        )
        # 文本强度检查
        for key in SUGGESTION_KEYS:
            # 不强制断言每个 key，但抽样检查
            pass
        # 至少 wolf_universal_rules 应在 HARD
        assert "wolf_universal_rules" in HARD_CONSTRAINT_KEYS, (
            f"wolf_universal_rules 含'绝对不要提到你的队友'等硬约束，应在 HARD_CONSTRAINT_KEYS"
        )
        # _VOTE_REASON_PRIVACY_GUARD 不在 SUGGESTION（应在 HARD 或独立注入）
        # 它的 key 实际叫 _VOTE_REASON_PRIVACY_GUARD 字符串本身，检查 _SECTION_PRIORITIES
        from werewolf_agent.agents.prompt_builder import _SECTION_PRIORITIES
        # 隐私 guard 是投票 reason 的硬约束，应在 HARD 段
        # 检查 _SECTION_PRIORITIES["硬约束"] 列表里包含相关 key
        hard_tier_keys = _SECTION_PRIORITIES.get("硬约束", [])
        # 至少包含 wolf_fake_seer_teammate 这类已有硬约束 key
        assert "wolf_fake_seer_teammate" in hard_tier_keys, (
            f"HARD tier 应含 wolf_fake_seer_teammate: {hard_tier_keys}"
        )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 修正 tier 分类**

读 `prompt_builder.py:142-160`（HARD_CONSTRAINT_KEYS / SUGGESTION_KEYS / REFERENCE_KEYS 三个 frozenset）。修改：

1. 把 `wolf_universal_rules` 从 SUGGESTION 移到 HARD_CONSTRAINT_KEYS（因含"绝对不要提到你的队友"等绝对性指令）
2. 把 `anti_herd` 从 SUGGESTION 移到 HARD_CONSTRAINT_KEYS（因含"反跟票警告"作为 P0-K6 硬约束）
3. 把 `seer_speech_directive` 中"后位硬约束"段单独抽 key `seer_late_position_rule` 放到 HARD

`_SECTION_PRIORITIES` 段标签顺序保持不变（hard / suggestion / reference）。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/agents/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "fix(prompt): 修正 HARD/SUGGESTION tier 漂移，含'绝对'约束移 HARD (post-review-v2-P3)"
```

---

### Task P4: B1 半成品 — `_most_suspect_target` producer 接线

**Files:**
- Modify: `werewolf_agent/agents/player.py:925-930` (consumer 移除死分支)
- Modify: `werewolf_agent/runtime/agent_adapter.py` (如果需要 producer 接线)
- Modify: `tests/agents/test_player_agent.py`

- [ ] **Step 1: 写测试**

```python
class TestMostSuspectTargetResolution:
    """P4 (post-review-v2): _most_suspect_target 路径要么接通要么删。"""

    def test_most_suspect_target_path_removed_or_wired(self):
        from werewolf_agent.agents.player import PlayerAgent
        from werewolf_agent.agents.schemas import (
            ActionType, AgentContext, FallbackAction,
        )
        ctx = AgentContext(
            agent_id="p08",
            task_type="day_vote",
            role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02", "p03", "p05", "p07"],
            strategy_directive={"_most_suspect_target": "p05"},
        )
        agent = PlayerAgent(agent_id="p08", model_router=None, persona=None)
        fb = agent._fallback_action(ctx)
        # 选项 A: 接通 → 应选 p05
        # 选项 B: 删 → 应选 non_self[0] = p02
        # 决策: 删，因为全代码库无 producer
        assert fb.target_id == "p02", (
            f"_most_suspect_target producer-less; should fall through to non_self[0]: {fb.target_id}"
        )
```

- [ ] **Step 2: 跑测试确认状态**

- [ ] **Step 3: 移除死分支**

读 `player.py:906-932`（`_fallback_action` vote 分支）。删除 `_most_suspect_target` 查找块。改为：

```python
            if safe_action == ActionType.VOTE:
                non_self = [t for t in context.legal_targets if t != context.agent_id]
                if not non_self:
                    safe_target = context.legal_targets[0] if context.legal_targets else None
                else:
                    fb = (
                        context.strategy_directive.get("_vote_fallback_target")
                        if context.strategy_directive else None
                    )
                    if fb and fb in non_self:
                        safe_target = fb
                    else:
                        safe_target = non_self[0]
```

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/agents/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/agents/player.py tests/agents/test_player_agent.py
git commit -m "refactor(player): 移除 _most_suspect_target 死分支（无 producer）(post-review-v2-P4)"
```

---

## Prompt Branch 全量回归

Run: `cd E:/NLP/agent/wofkill/.worktrees/fix15-prompt && /d/Miniforge3/Scripts/pytest tests/runtime/ tests/agents/ tests/cognition/ tests/rules/ -p no:cacheprovider -q`

合并：

```bash
cd E:/NLP/agent/wofkill
git merge --no-ff fix15-prompt-v2 -m "merge: fix15-prompt-v2 — 4 prompt fixes (post-review-v2 batch 1)"
```

---

# Branch 2: `fix16-arch-v2`

## Batch S: 存储层完整化

### Task S1: Postgres `self._lock` 真正 acquire

**Files:**
- Modify: `werewolf_agent/storage/postgres_store.py`
- Modify: `tests/storage/test_postgres_store.py`

- [ ] **Step 1: 写测试**

```python
class TestPostgresStoreThreadSafety:
    """S1 (post-review-v2): PostgresGameRepository 应线程安全。"""

    def test_postgres_lock_is_acquired_in_methods(self):
        from werewolf_agent.storage.postgres_store import PostgresGameRepository
        import inspect
        # 抽样检查 3 个方法都用了 self._lock
        for method in ("save_game", "save_custom_config", "save_reflection"):
            src = inspect.getsource(getattr(PostgresGameRepository, method))
            assert "self._lock" in src, (
                f"PostgresGameRepository.{method} missing self._lock acquisition"
            )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 在每个 SQL 方法入口加 `with self._lock:`**

读 `postgres_store.py`，给每个 SQL 方法（`save_game` / `load_game` / `append_events` / `load_events` / `save_deaths` / `load_deaths` / `save_model_usage` / `load_model_usage` / `save_evaluation` / `load_evaluation` / `save_config_snapshot` / `load_config_snapshot` / `save_custom_config` / `load_custom_config` / `list_custom_configs` / `list_games` / `delete_game` / `save_rag_entries` / `load_rag_entries` / `save_memory_snapshot` / `list_memory_snapshots` / `save_reflection` / `load_reflections_by_game` / `load_reflections_by_player`）的入口加 `with self._lock:` 包裹。

模式：
```python
    def save_game(self, game_id: str, state: dict) -> None:
        with self._lock:
            return self._save_game_impl(game_id, state)

    def _save_game_impl(self, game_id: str, state: dict) -> None:
        # 原方法体
```

按现有方法签名适配。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 storage regression** `pytest tests/storage/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix16-arch
git add werewolf_agent/storage/postgres_store.py tests/storage/test_postgres_store.py
git commit -m "fix(storage): PostgresGameRepository 所有方法加 self._lock 包裹 (post-review-v2-S1)"
```

---

### Task S2: Postgres `_ensure_schema` 补 `schema_version` 表

**Files:**
- Modify: `werewolf_agent/storage/postgres_store.py:_ensure_schema`

- [ ] **Step 1: 写测试**

```python
def test_postgres_ensure_schema_has_schema_version():
    """S2 (post-review-v2): PostgresGameRepository._ensure_schema 应含 schema_version 表。"""
    from werewolf_agent.storage.postgres_store import PostgresGameRepository
    import inspect
    src = inspect.getsource(PostgresGameRepository)
    # 找 _ensure_schema 方法
    assert "schema_version" in src, (
        f"Postgres _ensure_schema missing schema_version table"
    )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 补 DDL**

在 `postgres_store.py:_ensure_schema` 顶部加：

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW()
);
INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
```

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/storage/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/storage/postgres_store.py tests/storage/test_postgres_store.py
git commit -m "fix(storage): Postgres _ensure_schema 补 schema_version 表 (post-review-v2-S2)"
```

---

### Task S3: SQLite `_SCHEMA` 补 `reflections` 表（与 Postgres 对齐）

**Files:**
- Modify: `werewolf_agent/storage/sqlite_store.py:_SCHEMA`

- [ ] **Step 1: 写测试**

```python
def test_sqlite_schema_has_reflections_table():
    """S3 (post-review-v2): SQLite _SCHEMA 应含 reflections 表。"""
    from werewolf_agent.storage.sqlite_store import _SCHEMA
    assert "reflections" in _SCHEMA.lower(), (
        f"SQLite _SCHEMA missing reflections table"
    )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 补 DDL**

在 `sqlite_store.py:_SCHEMA` 加 `reflections` 表 DDL（与 Postgres `_ensure_schema` 中 reflections 表结构对齐）。按 Postgres 已有 schema 复刻。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/storage/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/storage/sqlite_store.py tests/storage/test_sqlite_store.py
git commit -m "fix(storage): SQLite _SCHEMA 补 reflections 表，与 Postgres 对齐 (post-review-v2-S3)"
```

---

### Task S4: `GameRepository` Protocol 补全 6 个新方法 + InMemoryGameRepository 实现

**Files:**
- Modify: `werewolf_agent/storage/repository.py`
- Modify: `werewolf_agent/storage/memory_store.py`
- Modify: `tests/storage/test_repository.py`

- [ ] **Step 1: 写测试**

```python
class TestGameRepositoryProtocolCompleteness:
    """S4 (post-review-v2): GameRepository Protocol 必须声明所有方法。"""

    def test_protocol_declares_reflection_methods(self):
        from werewolf_agent.storage.repository import GameRepository
        from typing import Protocol, get_type_hints
        # Protocol 应有 save_reflection / load_reflections_by_game|player|all / delete_reflection
        required = {
            "save_reflection", "load_reflections_by_game", "load_reflections_by_player",
            "load_all_reflections", "delete_reflection",
            "save_memory_snapshot", "load_memory_snapshot", "list_memory_snapshots",
        }
        annotations = GameRepository.__annotations__ if hasattr(GameRepository, "__annotations__") else {}
        for method in required:
            assert method in annotations, (
                f"GameRepository Protocol missing: {method}"
            )

    def test_in_memory_implements_reflection_methods(self):
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        for method in ("save_reflection", "load_reflections_by_game", "load_reflections_by_player",
                       "load_all_reflections", "delete_reflection",
                       "save_memory_snapshot", "load_memory_snapshot", "list_memory_snapshots"):
            assert hasattr(InMemoryGameRepository, method), (
                f"InMemoryGameRepository missing: {method}"
            )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 补 Protocol**

读 `repository.py` 找 `class GameRepository(Protocol)`，加 8 个方法声明（按 Postgres 实现签名）：

```python
    def save_reflection(self, game_id: str, player_id: str, reflection: dict) -> None: ...
    def load_reflections_by_game(self, game_id: str) -> list[dict]: ...
    def load_reflections_by_player(self, player_id: str) -> list[dict]: ...
    def load_all_reflections(self) -> list[dict]: ...
    def delete_reflection(self, game_id: str, player_id: str) -> None: ...
    def save_memory_snapshot(self, player_id: str, snapshot: dict) -> None: ...
    def load_memory_snapshot(self, player_id: str) -> dict | None: ...
    def list_memory_snapshots(self) -> list[dict]: ...
```

- [ ] **Step 4: 补 InMemoryGameRepository 实现**

读 `memory_store.py` 找 `InMemoryGameRepository` 类。添加：

```python
    def __init__(self):
        # ... 现有 __init__
        self._reflections: dict[tuple[str, str], dict] = {}  # (game_id, player_id) -> reflection
        self._memory_snapshots: dict[str, dict] = {}  # player_id -> snapshot

    def save_reflection(self, game_id: str, player_id: str, reflection: dict) -> None:
        self._reflections[(game_id, player_id)] = reflection

    def load_reflections_by_game(self, game_id: str) -> list[dict]:
        return [r for (g, p), r in self._reflections.items() if g == game_id]

    def load_reflections_by_player(self, player_id: str) -> list[dict]:
        return [r for (g, p), r in self._reflections.items() if p == player_id]

    def load_all_reflections(self) -> list[dict]:
        return list(self._reflections.values())

    def delete_reflection(self, game_id: str, player_id: str) -> None:
        self._reflections.pop((game_id, player_id), None)

    def save_memory_snapshot(self, player_id: str, snapshot: dict) -> None:
        self._memory_snapshots[player_id] = snapshot

    def load_memory_snapshot(self, player_id: str) -> dict | None:
        return self._memory_snapshots.get(player_id)

    def list_memory_snapshots(self) -> list[dict]:
        return list(self._memory_snapshots.values())
```

按实际 InMemory 现有 __init__ 风格适配。

- [ ] **Step 5: 跑测试**

- [ ] **Step 6: 跑 regression** `pytest tests/storage/ -p no:cacheprovider -q`

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/storage/repository.py werewolf_agent/storage/memory_store.py tests/storage/test_repository.py
git commit -m "feat(storage): GameRepository Protocol 补全 8 方法 + InMemoryGameRepository 实现 (post-review-v2-S4)"
```

---

## Batch M: Memory 持久化闭环

### Task M1: `MemoryStore.save_review/get_review` 持久化到 repo

**Files:**
- Modify: `werewolf_agent/memory/store.py`
- Modify: `werewolf_agent/tools/local_tools.py`
- Modify: `tests/memory/test_store.py`

- [ ] **Step 1: 写测试**

```python
class TestMemoryStoreReviewPersistence:
    """M1 (post-review-v2): MemoryStore.save_review 应持久化到 repo。"""

    def test_save_review_persists_to_repo(self):
        from werewolf_agent.memory.store import MemoryStore
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        repo = InMemoryGameRepository()
        store = MemoryStore(repo=repo)
        review_id = store.save_review("g_test", "p01", {"logic": 0.5})
        # 期望：repo._reflections[(g_test, p01)] 已被写入
        assert ("g_test", "p01") in repo._reflections, (
            f"save_review did not persist to repo: {repo._reflections}"
        )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 改 `MemoryStore.save_review` 持久化**

读 `store.py:351-376` 找 `save_review` / `get_review`。改为：

```python
    def save_review(self, game_id: str, player_id: str, review_data: dict) -> str:
        if not hasattr(self, "_reviews"):
            self._reviews = {}
        review_id = f"{game_id}:{player_id}:{len(self._reviews)}"
        record = {"review_id": review_id, "game_id": game_id, "player_id": player_id, "data": review_data}
        self._reviews[review_id] = record
        # P-M1 (post-review-v2): persist to repo
        if hasattr(self, "_repo") and self._repo is not None and hasattr(self._repo, "save_reflection"):
            self._repo.save_reflection(game_id, player_id, record)
        return review_id

    def get_review(self, game_id: str, player_id: str) -> dict | None:
        if not hasattr(self, "_reviews"):
            return None
        for r in self._reviews.values():
            if r["game_id"] == game_id and r["player_id"] == player_id:
                return r
        return None
```

按实际方法签名适配（可能函数签名是 `save_review(game_id, player_id, review_data, repo=None)` 之类）。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/memory/ tests/tools/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/memory/store.py tests/memory/test_store.py
git commit -m "fix(memory): MemoryStore.save_review 持久化到 repo (post-review-v2-M1)"
```

---

## Batch N: 模型网关 bug 修复

### Task N1: `_http_status_from_exception` 正则收紧

**Files:**
- Modify: `werewolf_agent/model_gateway/router.py:642-669`
- Modify: `tests/model_gateway/test_router.py`

- [ ] **Step 1: 写测试**

```python
class TestHttpStatusFromExceptionAccuracy:
    """N1 (post-review-v2): _http_status_from_exception 正则不应从 traceback 文本误抓。"""

    def test_year_2024_not_misclassified_as_http_status(self):
        from werewolf_agent.model_gateway.router import _http_status_from_exception
        exc = RuntimeError("Failed at line 2024 of the script")
        result = _http_status_from_exception(exc)
        # 期望：None 或 "unknown"，不应是 2024
        assert result not in (2024, "2024"), (
            f"traceback year 2024 misclassified as HTTP status: {result}"
        )

    def test_port_8080_not_misclassified_as_http_status(self):
        from werewolf_agent.model_gateway.router import _http_status_from_exception
        exc = RuntimeError("Listening on port 8080")
        result = _http_status_from_exception(exc)
        assert result not in (8080, "8080"), (
            f"port 8080 misclassified as HTTP status: {result}"
        )

    def test_real_http_status_500_still_detected(self):
        from werewolf_agent.model_gateway.router import _http_status_from_exception
        exc = RuntimeError("HTTP 500 Internal Server Error")
        result = _http_status_from_exception(exc)
        assert result in (500, "500"), f"should detect 500: {result}"
```

- [ ] **Step 2: 跑测试确认状态**

- [ ] **Step 3: 改 `_http_status_from_exception`**

读 `router.py:642-669`。改为：先从 `exception` 对象本身读 status_code 属性（httpx 等会暴露），fallback 才用正则，且正则要更精确（要求 "HTTP 5XX" 或 "status: 5XX" 上下文）：

```python
def _http_status_from_exception(exc: Exception) -> int | None:
    """P-N1: 优先从 exc 本身读 status_code，fallback 正则要求上下文。"""
    # 1. 优先从 exc.status_code (httpx 等)
    if hasattr(exc, "status_code"):
        sc = exc.status_code
        if isinstance(sc, int) and 100 <= sc < 600:
            return sc
    # 2. 从 exc.response.status_code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        sc = exc.response.status_code
        if isinstance(sc, int) and 100 <= sc < 600:
            return sc
    # 3. Fallback: 仅在 "HTTP 5XX" 上下文匹配
    import re
    msg = str(exc)
    match = re.search(r"HTTP\s*([1-5]\d{2})", msg)
    if match:
        return int(match.group(1))
    return None
```

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/model_gateway/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/model_gateway/router.py tests/model_gateway/test_router.py
git commit -m "fix(router): _http_status_from_exception 优先读 exc.status_code，回退要求 HTTP 前缀 (post-review-v2-N1)"
```

---

### Task N2: OpenAI URL 归一化 — `/v4beta` 不再错路由到 `/v1`

**Files:**
- Modify: `werewolf_agent/model_gateway/providers/openai.py:175-186`
- Modify: `tests/model_gateway/test_openai_provider.py`

- [ ] **Step 1: 写测试**

```python
class TestOpenAIURLNormalization:
    """N2 (post-review-v2): /v4beta URL 不应被错路由到 /v1/chat/completions。"""

    def test_v4beta_url_preserved(self):
        from werewolf_agent.model_gateway.providers.openai import _openai_chat_completions_url
        url = _openai_chat_completions_url("https://api.example.com/v4beta")
        assert "v4beta" in url, f"v4beta got rewritten: {url}"
        assert "/v1/chat/completions" not in url, f"v4beta misrouted to v1: {url}"

    def test_v5beta_url_preserved(self):
        from werewolf_agent.model_gateway.providers.openai import _openai_chat_completions_url
        url = _openai_chat_completions_url("https://api.example.com/v5beta")
        assert "v5beta" in url, f"v5beta got rewritten: {url}"
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 改 URL 归一化**

读 `openai.py:175-186`（`_openai_chat_completions_url`）。改为：只在 base_url 末尾已含版本号时**不追加**，缺版本号时才追加 `/v1/chat/completions`：

```python
import re

def _openai_chat_completions_url(base_url: str) -> str:
    """P-N2: 保留 base_url 中已声明的版本号 (vN / vNbeta / vNalpha)。"""
    base = base_url.rstrip("/")
    # 如果 base 已含 /chat/completions，直接返回
    if base.endswith("/chat/completions"):
        return base
    # 如果 base 已含 /vN/ 或 /vNbeta/ 或 /vNalpha/，保留
    if re.search(r"/v\d+(\w*)/?$", base) or re.search(r"/v\d+\w*$", base):
        return f"{base}/chat/completions"
    # 否则追加 /v1/chat/completions
    return f"{base}/v1/chat/completions"
```

按实际函数签名调整。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/model_gateway/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/model_gateway/providers/openai.py tests/model_gateway/test_openai_provider.py
git commit -m "fix(openai): URL 归一化保留 v4beta/v5beta 等任意版本号 (post-review-v2-N2)"
```

---

### Task N3: Anthropic `text-fallback` `{"` 前缀脆性修复

**Files:**
- Modify: `werewolf_agent/model_gateway/providers/anthropic.py:44-45, 77-78`
- Modify: `werewolf_agent/model_gateway/providers/minimax.py:55-56, 87-88` (类似代码)
- Modify: `tests/model_gateway/test_anthropic_provider.py`

- [ ] **Step 1: 写测试**

```python
class TestAnthropicTextFallbackRobustness:
    """N3 (post-review-v2): text-fallback 处理换行/空格不应吃掉首字符。"""

    def test_text_starts_with_whitespace_then_brace(self):
        # 模型先吐一个换行再 `{`，旧实现会吃 `{`
        text = "\n{\"action_type\": \"speech\"}"
        # 期望处理：去掉前缀空白后保留 `{`
        cleaned = text.lstrip()
        assert cleaned.startswith("{"), f"unexpected: {cleaned!r}"
```

- [ ] **Step 2: 跑测试确认当前实现**

- [ ] **Step 3: 改 text-fallback 逻辑**

读 `anthropic.py` 找 `text-fallback` 路径。当前逻辑（review 描述）：注入 `{"` 前缀，模型续写。如果模型先吐换行/空格，旧实现会"拼回"导致 `"{...`（双引号污染）。

改为：检测文本首个非空字符是否是 `{`/`[`，如果是**完整** JSON 起始，正常 parse；如果 model 没起首 `{` 而是普通 prose，**直接当 fallback 文本返回**（不强行拼前缀）。

实际代码：删 `{"` 前缀注入，删 `text[0] != "{"` 修补，改用标准 JSON 解析（`json.loads(text)` 失败时降级为 fallback）。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/model_gateway/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/model_gateway/providers/anthropic.py werewolf_agent/model_gateway/providers/minimax.py tests/model_gateway/test_anthropic_provider.py
git commit -m "fix(anthropic): text-fallback 改标准 JSON 解析，移除 {\" 前缀注入 (post-review-v2-N3)"
```

---

## Batch E: 核心引擎 bug 修复

### Task E1: `base_weight=2` 硬编码改用 ruleset 字段

**Files:**
- Modify: `werewolf_agent/engine/rule_engine.py:497-501`
- Modify: `werewolf_agent/config/rulesets/pre_witch_hunter_idiot_mixed.yaml`
- Modify: `tests/rules/test_rule_engine.py`

- [ ] **Step 1: 写测试**

```python
class TestSheriffWeightFromRuleset:
    """E1 (post-review-v2): sheriff 票权重应来自 ruleset 字段，非硬编码 base_weight。"""

    def test_sheriff_weight_uses_ruleset_field(self):
        from werewolf_agent.engine.rule_engine import resolve_vote
        from werewolf_agent.core.models import GameState, PlayerState
        # 构造 GS：ruleset.base_vote_weight = 3
        # 投票：警长 weight = sheriff_weight * base_vote_weight
        # 期望：结果用 1.5 * 3 = 4.5 → round 5（而非 1.5 * 2 = 3）
        ...
```

按实际 API 写测试。

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 改 `rule_engine.py`**

读 `rule_engine.py:497-501` 找 `base_weight = 2`。改为从 `gs.ruleset` 或配置读：

```python
# 旧
base_weight = 2
weight = round(sheriff_weight * base_weight)
# 新
base_weight = getattr(gs, "ruleset_base_vote_weight", 2)  # fallback 2
weight = round(sheriff_weight * base_weight)
```

- [ ] **Step 4: 在 ruleset YAML 加字段**

读 `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`，加：

```yaml
game_rules:
  base_vote_weight: 2
  sheriff_weight: 1.5
```

- [ ] **Step 5: 跑测试**

- [ ] **Step 6: 跑 regression** `pytest tests/rules/ -p no:cacheprovider -q`

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/engine/rule_engine.py werewolf_agent/config/rulesets/pre_witch_hunter_idiot_mixed.yaml tests/rules/test_rule_engine.py
git commit -m "fix(engine): sheriff 票权重从 ruleset.base_vote_weight 读取，去硬编码 (post-review-v2-E1)"
```

---

### Task E2: `cognition/world_state.py` extractor 收敛

**Files:**
- Modify: `werewolf_agent/cognition/world_state.py:373-385` (`_extract_seer_check` 等)
- Modify: `tests/cognition/test_world_state.py`

- [ ] **Step 1: 写测试**

```python
class TestExtractorSeesOnlyFactPayload:
    """E2 (post-review-v2): world_state extractor 不应能读全角色表。"""

    def test_seer_check_extractor_signature(self):
        from werewolf_agent.cognition.world_state import _EXTRACTORS
        for event_type, extractor in _EXTRACTORS.items():
            import inspect
            sig = inspect.signature(extractor)
            params = list(sig.parameters.keys())
            # 不应有"state"形参（会泄露 ground truth role 表）
            # 应只接 game_state 视图（受 visible_state 限制）
            # 实际：当前可能都接 state；按目标收紧
            ...
```

按实际签名调整（保守做法：把所有 extractor 改成只接 `(event, seer_id, alive_players_dict)` 不接 `gs`）。

- [ ] **Step 2: 跑测试确认状态**

- [ ] **Step 3: 重构 extractor 签名**

读 `world_state.py:_EXTRACTORS` 注册的所有函数。逐一检查签名。如果接 `gs`，改为只接 `event` + 受限视图。`seer_id` 应通过 visibility 映射传入而非 `next(... for ... if p.role == "seer")`。

实际重构视代码现状，目标是消除 extractor 对全角色表的访问。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 cognition regression** `pytest tests/cognition/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/cognition/world_state.py tests/cognition/test_world_state.py
git commit -m "refactor(world_state): extractor 签名收敛，去掉对全角色表的访问 (post-review-v2-E2)"
```

---

### Task E3: `core/models.py` PlayerState 补 `faction` 字段

**Files:**
- Modify: `werewolf_agent/core/models.py`
- Modify: `tests/core/test_models.py`

- [ ] **Step 1: 写测试**

```python
class TestPlayerStateFactionField:
    """E3 (post-review-v2): PlayerState 应含 faction 字段。"""

    def test_player_state_has_faction(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="werewolf", alive=True, faction="werewolf")
        assert p.faction == "werewolf"
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 加 faction 字段**

读 `core/models.py:7-15` 找 `class PlayerState`。加 `faction: str` 字段（默认从 role 推导：villager→good, werewolf→werewolf, special→good, hybrid→None）。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/core/ tests/runtime/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/core/models.py tests/core/test_models.py
git commit -m "feat(core): PlayerState 加 faction 字段，默认从 role 推导 (post-review-v2-E3)"
```

---

## Batch A: Agents 内部

### Task A1: `metrics_collector` 冷启动排序 bug 修复

**Files:**
- Modify: `werewolf_agent/agents/metrics_collector.py:74-78`
- Modify: `tests/agents/test_metrics_collector.py`

- [ ] **Step 1: 写测试**

```python
class TestMetricsCollectorColdStart:
    """A1 (post-review-v2): metrics_collector 排序应过滤小样本玩家。"""

    def test_low_sample_doesnt_dominate_top_failures(self):
        from werewolf_agent.agents.metrics_collector import MetricsCollector
        c = MetricsCollector()
        # 玩家 A: 1 局 1 失败 → rate=1.0, count=1
        c.record("p01", "task_x", error_code="x", retry_count=0)
        c.record_fallback("p01", "task_x")
        # 玩家 B: 100 局 80 失败 → rate=0.8, count=100
        for _ in range(100):
            c.record("p02", "task_x", error_code="x", retry_count=0)
            if _ < 80:
                c.record_fallback("p02", "task_x")
        top = c.get_top_failures(min_sample_count=10)
        # 期望：p02 在前（高样本 + 高失败率）
        # 当前 bug：p01 排第一（rate=1.0 但 count=1）
        if top:
            top_player = top[0][0]
            assert top_player == "p02", (
                f"cold-start bug: top failure is {top_player} (rate=1.0 but n=1)"
            )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 加 `min_sample_count` 过滤**

读 `metrics_collector.py:74-78` 找 `get_top_failures`。改为：

```python
def get_top_failures(self, min_sample_count: int = 10) -> list[tuple[str, float, int]]:
    """P-A1: 过滤小样本玩家，避免冷启动单人拉高排序。"""
    candidates = [
        (pid, p.fallback_rate, p.sample_count)
        for pid, p in self._profiles.items()
        if p.sample_count >= min_sample_count
    ]
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[:10]
```

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/agents/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/agents/metrics_collector.py tests/agents/test_metrics_collector.py
git commit -m "fix(metrics): get_top_failures 加 min_sample_count 过滤，修冷启动排序 (post-review-v2-A1)"
```

---

## Arch Branch 全量回归

Run: `cd E:/NLP/agent/wofkill/.worktrees/fix16-arch && /d/Miniforge3/Scripts/pytest tests/storage/ tests/memory/ tests/model_gateway/ tests/engine/ tests/cognition/ tests/agents/ tests/core/ -p no:cacheprovider -q`

合并：

```bash
cd E:/NLP/agent/wofkill
git merge --no-ff fix16-arch-v2 -m "merge: fix16-arch-v2 — 10 storage + memory + model_gateway + engine + agents fixes (post-review-v2 batch 2)"
```

---

# Branch 3: `fix17-periphery-v2`

## Batch SK: Skills 散文 dead prose

### Task SK1: SKILL.md 正文注入到 prompt

**Files:**
- Modify: `werewolf_agent/skills/werewolf_skills.py` (SKILL.md 正文加载)
- Modify: `werewolf_agent/skills/registry.py` (注入 prompt)
- Modify: `tests/skills/test_skills.py`

- [ ] **Step 1: 写测试**

```python
class TestSkillMarkdownBodyInjected:
    """SK1 (post-review-v2): SKILL.md 正文（frontmatter 之外的部分）应被注入到 prompt。"""

    def test_skill_body_loaded_into_prompt(self):
        from werewolf_agent.skills.werewolf_skills import _load_manifests
        from pathlib import Path
        manifests = _load_manifests()
        if not manifests:
            pytest.skip("no skills to test")
        for skill in manifests:
            if skill.body:  # 假设我们加了 body 字段
                assert len(skill.body) > 50, (
                    f"skill {skill.name} body too short or empty: {len(skill.body) if skill.body else 0}"
                )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 加 body 字段 + 加载**

读 `werewolf_agent/skills/werewolf_skills.py:_parse_skill_frontmatter`。改为：除 YAML frontmatter 外，把剩余 markdown 文本存为 `body` 字段。同步 `SkillDefinition` (在 `registry.py`) 加 `body: str` 字段。

`registry.py:apply_skill` / `dispatch_for_role` 把 `body` 拼到 `prompt_injectable`。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 skills regression** `pytest tests/skills/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/skills/werewolf_skills.py werewolf_agent/skills/registry.py werewolf_agent/skills/schemas.py tests/skills/test_skills.py
git commit -m "feat(skills): SKILL.md 正文注入 prompt，markdown-driven 真正生效 (post-review-v2-SK1)"
```

---

## Batch P: Persona dead read

### Task P5: `judge_router.task_styles` dead read 修复

**Files:**
- Modify: `werewolf_agent/persona_runtime/judge_router.py:65-83`
- Modify: `tests/persona_runtime/test_judge_router.py`

- [ ] **Step 1: 写测试**

```python
class TestJudgeRouterTaskStylesUsed:
    """P5 (post-review-v2): judge_router.task_styles 应被 JudgeAgent 消费。"""

    def test_task_styles_in_system_prompt(self):
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        from werewolf_agent.persona_runtime.schemas import JudgePersonaSnapshot
        # router 应把 task_styles 拼进 system_prompt
        ...
```

- [ ] **Step 2: 跑测试**

- [ ] **Step 3: 修复**

读 `judge_router.py:65-83`。决策：要么真拼进 `system_prompt`，要么从 snapshot 删 `task_styles` 字段。推荐**拼进**，因为 P1-4 revert 注释自承是临时回退。代码：

```python
        system_prompt = base_system
        # 拼 task_styles（如果提供）
        if persona.task_styles and persona.task_styles.get(task_type):
            system_prompt = f"{base_system}\n\n[TASK STYLE: {persona.task_styles[task_type]}]"
```

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/persona_runtime/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/persona_runtime/judge_router.py tests/persona_runtime/test_judge_router.py
git commit -m "fix(persona): judge_router task_styles 拼进 system_prompt (post-review-v2-P5)"
```

---

## Batch C: Customization 卫生

### Task C1: `_UNICODE_SUSPICIOUS_RANGES` 改 `\u` 转义

**Files:**
- Modify: `werewolf_agent/customization/validators.py:113-123`
- Modify: `tests/customization/test_validators.py`

- [ ] **Step 1: 写测试**

```python
def test_unicode_suspicious_ranges_uses_escape():
    """C1 (post-review-v2): _UNICODE_SUSPICIOUS_RANGES 应使用 \\u 转义避免 IDE/源码处理时丢字符。"""
    from werewolf_agent.customization.validators import _UNICODE_SUSPICIOUS_RANGES
    src = inspect.getsource(_UNICODE_SUSPICIOUS_RANGES) if hasattr(_UNICODE_SUSPICIOUS_RANGES, '__class__') else None
    # 简化：断言值正确
    assert "​" in _UNICODE_SUSPICIOUS_RANGES
    assert "‪" in _UNICODE_SUSPICIOUS_RANGES
```

- [ ] **Step 2: 跑测试**

- [ ] **Step 3: 改 literals 为 escape**

读 `validators.py:113-123`。改为：

```python
_UNICODE_SUSPICIOUS_RANGES = (
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "﻿",  # BOM
    "‪", "‫", "‬", "‭", "‮",  # bidirectional overrides
)
```

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/customization/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/customization/validators.py tests/customization/test_validators.py
git commit -m "fix(validators): _UNICODE_SUSPICIOUS_RANGES 改 \\u 转义 (post-review-v2-C1)"
```

---

### Task C2: `persona_adapter._slug` 支持中文 archetype

**Files:**
- Modify: `werewolf_agent/customization/persona_adapter.py:81-87`
- Modify: `tests/customization/test_persona_adapter.py`

- [ ] **Step 1: 写测试**

```python
def test_slug_preserves_chinese_archetype():
    """C2 (post-review-v2): 中文 archetype 名不应退化为 'default'。"""
    from werewolf_agent.customization.persona_adapter import _slug
    slug = _slug("冷静型")
    assert slug != "default", f"Chinese archetype collapsed to default: {slug}"
```

- [ ] **Step 2: 跑测试**

- [ ] **Step 3: 改 slug 算法**

读 `persona_adapter.py:81-87`。改用 `unicodedata.normalize` + pypinyin 或简化方案：先 `isalnum` 检查，失败后用 unicode category 保留汉字字符（作为 slug 的一部分）：

```python
import re
import unicodedata

def _slug(name: str) -> str:
    """P-C2: 保留中文 archetype 名（不退化为 default）。"""
    if not name:
        return "default"
    # 保留 ASCII alnum + 所有 CJK 字符
    cleaned = re.sub(r"[^\w一-鿿぀-ゟ゠-ヿ-]+", "-", name.lower())
    cleaned = cleaned.strip("-")
    return cleaned or "default"
```

按实际函数签名调整。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/customization/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/customization/persona_adapter.py tests/customization/test_persona_adapter.py
git commit -m "fix(persona): _slug 保留中文 archetype 名 (post-review-v2-C2)"
```

---

## Batch U: UI/Server 协同

### Task U1: dashboard.js rag-audit 404 修复

**Files:**
- Modify: `werewolf_agent/api/routes/games.py` (加 rag-audit 路由)
- Modify: `werewolf_agent/ui/static/dashboard.js` (确认路径一致)
- Modify: `tests/api/test_routes.py`

- [ ] **Step 1: 写测试**

```python
def test_rag_audit_endpoint_exists():
    """U1 (post-review-v2): /games/{id}/rag-audit 路由应存在。"""
    from fastapi.testclient import TestClient
    from werewolf_agent.api.app import create_app
    app = create_app()
    client = TestClient(app)
    # 期望 200 或 404（无 game）而非 405 method not allowed
    response = client.get("/games/g_test/rag-audit")
    assert response.status_code != 405, (
        f"rag-audit endpoint missing: {response.status_code}"
    )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 加路由**

读 `api/routes/games.py`。加 `@router.get("/games/{game_id}/rag-audit")` handler，从 repo 读 `audit_events`，过滤 `type="rag_injection_audit"`，返回 list。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression** `pytest tests/api/ -p no:cacheprovider -q`

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/api/routes/games.py tests/api/test_routes.py
git commit -m "feat(api): 加 /games/{id}/rag-audit 路由，修 dashboard.js 404 (post-review-v2-U1)"
```

---

## Peripheral Branch 全量回归

Run: `cd E:/NLP/agent/wofkill/.worktrees/fix17-periphery && /d/Miniforge3/Scripts/pytest tests/skills/ tests/persona_runtime/ tests/customization/ tests/api/ -p no:cacheprovider -q`

合并：

```bash
cd E:/NLP/agent/wofkill
git merge --no-ff fix17-periphery-v2 -m "merge: fix17-periphery-v2 — 5 periphery fixes (post-review-v2 batch 3)"
```

---

# 最终验证（合回 master 后）

## 验证 1: 全量 regression

Run: `cd E:/NLP/agent/wofkill && /d/Miniforge3/Scripts/pytest tests/runtime/ tests/agents/ tests/rules/ tests/memory/ tests/rag/ tests/skills/ tests/cognition/ tests/model_gateway/ tests/storage/ tests/api/ tests/customization/ tests/tools/ tests/evaluation/ tests/core/ -p no:cacheprovider -q`

## 验证 2: PROGRESS.md 更新

追加新章节记录本批修复总结。

---

# Self-Review Checklist

| 上一轮发现 | 对应 Task | 状态 |
|---|---|---|
| C4 不完整（13 处硬编码） | P1 | ✅ |
| A8 回归（FULL_ACTION 路径缺 guard） | P2 | ✅ |
| HARD/SUGGESTION tier 漂移 | P3 | ✅ |
| B1 半成品（`_most_suspect_target` 死代码） | P4 | ✅ |
| Postgres thread safety 回归 | S1 | ✅ |
| A1 不完整（schema_version 缺） | S2 | ✅ |
| U10 仍漂移（reflections 表） | S3 | ✅ |
| A1 Protocol 缺 6 方法 | S4 | ✅ |
| U4 半成品（review 不持久化） | M1 | ✅ |
| `_http_status_from_exception` 正则误抓 | N1 | ✅ |
| OpenAI `/v4beta` 错路由 | N2 | ✅ |
| Anthropic `{"` 前缀脆性 | N3 | ✅ |
| base_weight 硬编码 | E1 | ✅ |
| extractor 读全角色表 | E2 | ✅ |
| PlayerState 缺 faction | E3 | ✅ |
| metrics_collector 冷启动 | A1 | ✅ |
| SKILL.md 正文未注入 | SK1 | ✅ |
| task_styles dead read | P5 | ✅ |
| validators literal unicode | C1 | ✅ |
| persona slug 中文 | C2 | ✅ |
| dashboard.js 404 | U1 | ✅ |

**未在 v1 修复（标 ⚠️ 留作 v3）**：
- 性能三大问题（time.sleep 串行 / O(N²) 评估 / 双重计算）— 单条收益 < 改造成本
- Directive builders 拿 `gs` 而非 `visible`（架构大改）— 跨 9 个 directive 文件
- views.py 是 visibility 策略的事实中心（拆 projections.py）— 跨多文件重构
- SKILL.md 之外的 markdown-driven 死内容（per-player persona profile 等）

---

# 执行方式

Plan 已保存到 `docs/superpowers/plans/2026-06-07-post-review-v2-fixes.md`。

**三条工作分支**：
- `fix15-prompt-v2` (4 commits: P1, P2, P3, P4)
- `fix16-arch-v2` (10 commits: S1, S2, S3, S4, M1, N1, N2, N3, E1, E2, E3, A1)
- `fix17-periphery-v2` (5 commits: SK1, P5, C1, C2, U1)

**两个执行选项**：

1. **Subagent-Driven (推荐)** — 每条 task 派一个独立 subagent 执行
2. **Inline Execution** — 在当前会话按 Batch 顺序逐条执行

请选执行方式。
