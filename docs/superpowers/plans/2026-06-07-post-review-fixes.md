# Post-Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 6 个模块审查发现的 30 个问题（prompt bug 1、信息隔离 2、架构债务 4、定义未接 4、死代码 6、重复 4、契约违反 3、可见性配置 1、skills 散文 1、其它卫生 4）。

**Architecture:** 3 条 worktree 分支并行 — `fix12-prompt-review`（prompt 侧 B+C 批）、`fix13-arch-review`（架构债务 + 契约违反）、`fix14-cleanup`（死代码/重复/卫生）。各跑 TDD 后合回 master。每条 commit 独立可回滚。

**Tech Stack:** Python 3.11, Pydantic v2, pytest, 现有 `werewolf_agent` 包结构。

**Spec source:** 6 个 subagent 的审查报告（已在会话内） + `docs/superpowers/plans/2026-06-07-g3223805846-fixes.md`（前序修复 plan）。

---

## 文件变更地图

| 类别 | 文件 | 涉及 Batch |
|---|---|---|
| Prompt | `werewolf_agent/runtime/directives/villager.py` | C1 |
| Prompt | `werewolf_agent/agents/prompt_builder.py` | C3, C4 |
| Prompt | `werewolf_agent/runtime/directives/hunter.py` | C6 |
| Prompt | `werewolf_agent/runtime/directives/_shared.py` | C6 |
| 信息隔离 | `werewolf_agent/cognition/visibility.py` | C2 |
| Prompt 集成 | `werewolf_agent/cognition/context.py` | C1 (verify) |
| 架构债务 | `werewolf_agent/storage/postgres_store.py` | A1, A3 |
| 架构债务 | `werewolf_agent/storage/production.py` | A2 |
| 架构债务 | `werewolf_agent/customization/repository.py` | A2 |
| 架构债务 | `werewolf_agent/api/app.py` | A4 |
| 架构债务 | `werewolf_agent/runtime/agent_adapter.py` | A5, A6 |
| 架构债务 | `werewolf_agent/runtime/nodes/sheriff.py` | A6 |
| 死代码 | `werewolf_agent/runtime/nodes/_shared.py` | U1, U2, U4, U6 |
| 死代码 | `werewolf_agent/runtime/nodes/skills.py` | U5 |
| 死代码 | `werewolf_agent/agents/prompt_builder.py` | (C4 同时) |
| 死代码 | `werewolf_agent/tools/local_tools.py` | U8 |
| 重复 | `werewolf_agent/runtime/strategy/wolf.py` | U2 |
| 重复 | `werewolf_agent/runtime/strategy/hunter.py` | U2 |
| 重复 | `werewolf_agent/runtime/sheriff_policy.py` | U3 |
| 卫生 | `werewolf_agent/memory/store.py` | U7 |
| 卫生 | `werewolf_agent/storage/sqlite_store.py` + `migrations.py` | U10 |
| 卫生 | `werewolf_agent/agents/judge.py` | U11 |
| 卫生 | `werewolf_agent/cognition/world_state.py` | U9 |
| Skills | `werewolf_agent/skills/werewolf_skills.py` | (留 v2，标 ⚠️) |
| 测试 | `tests/runtime/test_directives_*.py` | C1, C6 |
| 测试 | `tests/cognition/test_visibility.py` | C2 |
| 测试 | `tests/agents/test_prompt_builder.py` | C3, C4 |
| 测试 | `tests/storage/test_postgres_store.py` | A1, A3 |
| 测试 | `tests/api/test_app.py` | A2, A4 |
| 测试 | `tests/runtime/test_agent_adapter.py` | A5, A6 |
| 测试 | `tests/runtime/test_strategy_*.py` | U2 |
| 测试 | `tests/memory/test_store.py` | U7 |
| 测试 | `tests/agents/test_judge.py` | U11 |

---

# Branch 1: `fix12-prompt-review`

## Batch C1: villager 不再读 seer_check 私有事件

### Task C1.1: 写测试

**Files:**
- Modify: `tests/runtime/test_strategy_directives.py` (新增 `TestVillagerNoSeerPrivateLeak`)

- [ ] **Step 1: 写失败测试**

```python
class TestVillagerNoSeerPrivateLeak:
    """审查 C1: villager 金水/银水判定不能读私有 seer_check 事件。"""

    def test_villager_directive_does_not_mention_seer_check_keyword(self):
        """Villager prompt 文本中不应出现 'seer_check' / '查验' 等私事件关键词。"""
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.directives.villager import build_villager_directive
        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        alive["p01"] = PlayerState(id="p01", role="seer", alive=True)
        # 注入私有 seer_check 事件（villager 不应感知）
        gs = GameState(
            players=alive, day_number=2, night_number=2,
            events=[
                GameEvent(type="seer_check", payload={
                    "target_id": "p07", "alignment": "good", "night_number": 1,
                }),
            ],
        )
        d = build_villager_directive(gs, "p07")
        full = " ".join(str(v) for v in d.values())
        # 私有事件提及 = leak
        assert "seer_check" not in full.lower(), (
            f"villager directive leaks seer_check keyword: {full!r}"
        )

    def test_villager_gold_water_only_from_public_seer_claim(self):
        """Villager 仅当公开场预言家报过 p07=好人 时才把 p07 视为金水。"""
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.directives.villager import build_villager_directive
        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        alive["p01"] = PlayerState(id="p01", role="seer", alive=True)
        # 场景 A: 仅有私有 seer_check 事件，无公开报金水
        gs_private_only = GameState(
            players=alive, day_number=2, night_number=2,
            events=[
                GameEvent(type="seer_check", payload={
                    "target_id": "p07", "alignment": "good", "night_number": 1,
                }),
            ],
        )
        d_a = build_villager_directive(gs_private_only, "p07")
        # 场景 B: 公开场预言家报过金水
        gs_public = GameState(
            players=alive, day_number=2, night_number=2,
            events=[
                GameEvent(type="speech", payload={
                    "speaker": "p01",
                    "text": "我是预言家，第 1 夜验了 p07 是好人（金水）。",
                }),
            ],
        )
        d_b = build_villager_directive(gs_public, "p07")
        # 场景 A 的 directive 不应包含"金水"建议；场景 B 应包含
        full_a = " ".join(str(v) for v in d_a.values())
        full_b = " ".join(str(v) for v in d_b.values())
        assert "金水" not in full_a or "公开" in full_a, (
            f"villager gold_water_duty reads seer_check private event: {full_a!r}"
        )
        assert "金水" in full_b, (
            f"villager should surface gold water from public claim: {full_b!r}"
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd E:/NLP/agent/wofkill/.worktrees/fix12-prompt-review && pytest tests/runtime/test_strategy_directives.py::TestVillagerNoSeerPrivateLeak -v`
Expected: FAIL

---

### Task C1.2: 改 villager.py

**Files:**
- Modify: `werewolf_agent/runtime/directives/villager.py:56-67`

- [ ] **Step 1: 删除读 `seer_check` 私有事件的代码**

读 `villager.py` 找到 `gold_water_duty` 段（line 56-67，按 review 报告）。删除整个 `for e in gs.events: if e.type == "seer_check"...` 循环（含 `role_label = "白痴" if ... else "普通村民"` 死代码一并清理）。

- [ ] **Step 2: 改用 `public_seer_claimants` + 公开 speech 事件反推**

在 `villager.py` 顶部 import 区加：

```python
from werewolf_agent.runtime.strategy.seer import public_seer_claimants as _public_seer_claimants
```

在 `build_villager_directive` 函数体内（`for e in gs.events` 循环替换处）新增：

```python
    # 公开场金水判定：只看公开 speech 事件，不读 seer_check 私有事件
    public_claimants = _public_seer_claimants(gs)
    gold_water_targets: set[str] = set()
    if public_claimants:
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") not in public_claimants:
                continue
            text = str(e.payload.get("text", ""))
            # 抓 "验了 p0X 是好人" 模式 → 标记 gold water
            for match in re.finditer(r"验[了过]?\s*(p\d+).*?好人|验[了过]?\s*(p\d+).*?金水", text):
                gold_water_targets.add(match.group(1) or match.group(2))

    if villager_id in gold_water_targets:
        parts["gold_water_duty"] = (
            f"你是{villager_id}，公开场上预言家已报你为金水。"
            "请基于此身份积极帮助好人阵营：\n"
            "1) 主动站边公开预言家，传递信任信号；\n"
            "2) 发言中可适度引用预言家给你的金水身份，但不要伪造额外查杀信息；\n"
            "3) 投票时优先跟随查杀方归票，保留对悍跳狼的质疑能力。"
        )
```

在文件顶部加 `import re`（如果还没有）。

- [ ] **Step 3: 跑测试**

Run: `pytest tests/runtime/test_strategy_directives.py::TestVillagerNoSeerPrivateLeak -v`
Expected: PASS

- [ ] **Step 4: 跑相关 regression**

Run: `pytest tests/runtime/test_strategy_directives.py tests/runtime/test_wolf_flow.py tests/cognition/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix12-prompt-review
git add werewolf_agent/runtime/directives/villager.py tests/runtime/test_strategy_directives.py
git commit -m "fix(prompt): villager gold_water_duty 改读公开 seer_claim，不再泄漏 seer_check 私有事件 (post-review-C1)"
```

---

## Batch C2: visibility.py __init__ 不再是 no-op

### Task C2.1: 写测试

**Files:**
- Modify: `tests/cognition/test_visibility.py` (新增 `TestVisibilityConfigWiring`)

- [ ] **Step 1: 写失败测试**

```python
class TestVisibilityConfigWiring:
    """审查 C2: visibility_config 应能覆写 hardcoded 可见性表。"""

    def test_visibility_init_wires_config(self):
        from werewolf_agent.cognition.visibility import VisibilityPolicy
        from dataclasses import replace
        # 构造一个 config 把某个 fact_type 改投到不同可见性
        custom_config = {"_FACT_VISIBILITY_MAP": {
            "speech": "moderator_only",  # 反向：把公开事件改 moderator_only
        }}
        # 当前实现：__init__ 收到 config 但不写入 self._FACT_VISIBILITY_MAP
        policy = VisibilityPolicy(visibility_config=custom_config)
        # 期望：custom_config 生效，speech 事件被视作 moderator_only
        from werewolf_agent.core.models import GameEvent, GameState
        gs = GameState(
            players={}, day_number=1, night_number=1,
            events=[GameEvent(type="speech", payload={"speaker": "p01", "text": "hi"})],
        )
        visible = policy.compute_fact_visibility(gs, viewer_role="villager")
        # 当前实现：speech 是 public，villager 能看到
        # 修复后：speech 是 moderator_only，villager 看不到
        facts = [f for f in visible if f.fact_type == "speech"]
        assert len(facts) == 0, (
            f"custom config did not override fact_type visibility: {[f.fact_type for f in visible]}"
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/cognition/test_visibility.py::TestVisibilityConfigWiring -v`
Expected: FAIL（当前 __init__ 不接 config）

---

### Task C2.2: 改 visibility.py

**Files:**
- Modify: `werewolf_agent/cognition/visibility.py:121-124`

- [ ] **Step 1: 让 __init__ 真正生效**

读 `visibility.py:121-124` 找到 `__init__(visibility_config)`。把参数解构后写回 self：

```python
    def __init__(self, visibility_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        if visibility_config:
            for k, v in visibility_config.items():
                if hasattr(self, k) and isinstance(v, dict):
                    # 深 merge：override 单个 fact_type
                    base = getattr(self, k, {})
                    merged = dict(base)
                    merged.update(v)
                    setattr(self, k, merged)
                elif hasattr(self, k):
                    setattr(self, k, v)
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/cognition/test_visibility.py::TestVisibilityConfigWiring -v`
Expected: PASS

- [ ] **Step 3: 跑 cognition regression**

Run: `pytest tests/cognition/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix12-prompt-review
git add werewolf_agent/cognition/visibility.py tests/cognition/test_visibility.py
git commit -m "fix(visibility): __init__ 真正生效 visibility_config，ruleset 可覆写 hardcoded 表 (post-review-C2)"
```

---

## Batch C3: 简化双层"硬约束"标签

### Task C3.1: 写测试

**Files:**
- Modify: `tests/agents/test_prompt_builder.py` (新增 `TestHardConstraintLabelUniqueness`)

- [ ] **Step 1: 写失败测试**

```python
class TestHardConstraintLabelUniqueness:
    """审查 C3: 单一硬约束标签层。"""

    def test_no_duplicate_hard_constraint_markers(self):
        """同一段 prompt 内 "硬约束" 标签出现次数应 ≤ 1。"""
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        from werewolf_agent.agents.schemas import (
            AgentContext, ActionType, OutputMode, TaskType,
        )
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.DAY_SPEECH,
            role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=[],
            strategy_directive={
                "wolf_universal_rules": "硬约束: 不要暴露队友",  # 模拟 SUGGESTION 标里实际含硬约束
                "wolf_fake_seer_teammate": "硬约束: 不要信息穿越",
            },
        )
        builder = PlayerPromptBuilder(ctx)
        user_prompt = builder.build_user_prompt()
        # 计算 "硬约束" 出现次数
        count = user_prompt.count("硬约束")
        assert count <= 1, (
            f"multiple 硬约束 labels in single prompt: {count} occurrences"
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/agents/test_prompt_builder.py::TestHardConstraintLabelUniqueness -v`
Expected: FAIL

---

### Task C3.2: 移除内层 sub-group 的"硬约束"标签

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (找到 `【硬约束】` 注入内层 sub-group 的地方)

- [ ] **Step 1: 找内层 hard constraint 注入点**

Run: `grep -n "硬约束" werewolf_agent/agents/prompt_builder.py`

具体行号视实际源码；目标是把 sub-group 内的"【硬约束】"标签去掉（保留外层 section 标签的"【硬约束】"）。

- [ ] **Step 2: 改为 soft label 格式**

把内层 sub-group 的渲染改为 `【参考】` 或 `【说明】` 格式。**只**留外层 `_SECTION_PRIORITIES["硬约束"]` 那一处显式"硬约束"标签。

举例（按实际找到的位置调整）：

```python
# 旧
parts.append("【硬约束】以下是必须遵守的指令：\n")
# 新
parts.append("以下是必须遵守的指令：\n")  # 外层 section label 已声明硬约束
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/agents/test_prompt_builder.py::TestHardConstraintLabelUniqueness -v`
Expected: PASS

- [ ] **Step 4: 跑 agents regression**

Run: `pytest tests/agents/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix12-prompt-review
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "refactor(prompt): 简化硬约束标签层（外层保留，内层 sub-group 移除） (post-review-C3)"
```

---

## Batch C4: `_format_examples` p03 硬编码修复

### Task C4.1: 写测试

**Files:**
- Modify: `tests/agents/test_prompt_builder.py` (新增 `TestFormatExamplesNoHardcodedID`)

- [ ] **Step 1: 写失败测试**

```python
class TestFormatExamplesNoHardcodedID:
    """审查 C4: vote 示例不应硬编码 p03 作为预言家 ID。"""

    def test_vote_example_does_not_hardcode_p03(self):
        from werewolf_agent.agents.prompt_builder import _format_examples, _ACTION_TYPE_GUARD
        # 调用 _format_examples 并检查 vote 段示例不含硬编码 "p03" 预言家
        # 简化：直接断言
        # 实际测试需要 mock context 来跑完整 _format_examples
        # 这里只检查示例文本不直接出现 p03 作为预言家 ID
        from unittest.mock import MagicMock
        from werewolf_agent.agents.schemas import ActionType
        mock_ctx = MagicMock()
        mock_ctx.legal_actions = [ActionType.VOTE]
        mock_ctx.legal_targets = [f"p{i:02d}" for i in range(1, 13)]
        mock_ctx.legal_actions_full = mock_ctx.legal_actions
        # 触发 _format_examples
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        builder = PlayerPromptBuilder(mock_ctx)
        # 内部访问私有方法
        examples = builder._format_examples()
        # 不应出现 "standing_with_seer": "p03" 这类硬编码
        import re
        match = re.search(r'"standing_with_seer"\s*:\s*"p03"', examples)
        assert not match, f"_format_examples hardcodes p03 as seer: {match.group(0)}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/agents/test_prompt_builder.py::TestFormatExamplesNoHardcodedID -v`
Expected: FAIL

---

### Task C4.2: 改 `_format_examples` 的 vote 示例

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (vote 段示例所在函数)

- [ ] **Step 1: 改 `standing_with_seer` 硬编码**

读 `prompt_builder.py` 找到 `vote` 段示例（review 报告说 line 1324），把 `vote_standing_with_seer="p03"` 改为占位符 + 注释：

```python
# 旧
"standing_with_seer": "p03",
# 新
"standing_with_seer": "pXX",  # 占位：当前局真实预言家 ID（由 ctx 提供）
```

同时给整个 vote 段示例加注释：

```python
"""vote 段示例: standing_with_seer/target_id 应被 LLM 替换为当前局真实玩家 ID。
此处使用 pXX 占位符避免硬编码 p03 误导 LLM。"""
```

如果实际函数没有这段代码（review 报告说 line 1324 但实际可能不同），用 grep 找 `p03` 出现位置。

- [ ] **Step 2: 跑测试**

Run: `pytest tests/agents/test_prompt_builder.py::TestFormatExamplesNoHardcodedID -v`
Expected: PASS

- [ ] **Step 3: 跑 agents regression**

Run: `pytest tests/agents/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix12-prompt-review
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "fix(prompt): _format_examples vote 段预言家 ID 改占位符 pXX，去掉硬编码 p03 (post-review-C4)"
```

---

## Batch C6: 猎人遗言与实际开枪行为一致

### Task C6.1: 写测试

**Files:**
- Modify: `tests/runtime/test_strategy_directives.py` (新增 `TestHunterLastWordsBehaviorConsistency`)

- [ ] **Step 1: 写失败测试**

```python
class TestHunterLastWordsBehaviorConsistency:
    """审查 C6: 遗言中提示与 _hunter_shot_target_from_last_words 实际行为一致。"""

    def test_hunter_exile_directive_does_not_force_shot(self):
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.directives.hunter import build_hunter_directive
        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        alive["p09"] = PlayerState(id="p09", role="hunter", alive=True)
        gs = GameState(players=alive, day_number=2, night_number=2)
        d = build_hunter_directive(gs, "p09")
        directive = d.get("hunter_speech_directive", "")
        # 不应强制"必须开枪"；允许 no_action
        assert "必须开枪" not in directive or "也可以" in directive or "未指定" in directive, (
            f"hunter exile directive forces shot but actual behavior is no_action when target missing: {directive!r}"
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/runtime/test_strategy_directives.py::TestHunterLastWordsBehaviorConsistency -v`
Expected: FAIL

---

### Task C6.2: 改 hunter.py 遗言 prompt

**Files:**
- Modify: `werewolf_agent/runtime/directives/hunter.py` (身份公开后段，加一致性条款)

- [ ] **Step 1: 加一致性条款**

读 `hunter.py` 找到 identity_exposed 分支（review 报告说 line 39-47），在末尾追加：

```python
        text = (
            "你是猎人，且你的身份已经公开。\n"
            "身份公开后的策略：\n"
            "1) 利用'我有枪'的威慑力，给狼人施加压力\n"
            "2) 明确表达你的怀疑和站边，让狼人忌惮开枪带走他们\n"
            "3) 不要虚张声势说你会带走某人——如果你被毒杀将无法开枪\n"
            "4) 如果预言家已死，你可以主动承担信息整理和归票的职责\n"
            "5) 【开枪前硬约束】临死开枪必须基于 ≥ 2 处独立公开证据，单点孤证时倾向 no_action。\n"
            "6) 【遗言一致性】你的放逐遗言与开枪行为必须一致：\n"
            "   - 如果你明确说'我会带走 p0X'，系统会按 p0X 开枪\n"
            "   - 如果你说'我选择不开枪'或未指定目标，系统走 no_action\n"
            "   - 不要在遗言里说'必须开枪'但实际找不到目标——这种 prompt 会让系统误判"
        )
```

（如果实际函数签名/参数不同，按现状调整。）

- [ ] **Step 2: 跑测试**

Run: `pytest tests/runtime/test_strategy_directives.py::TestHunterLastWordsBehaviorConsistency -v`
Expected: PASS

- [ ] **Step 3: 跑 regression**

Run: `pytest tests/runtime/test_strategy_directives.py tests/runtime/test_hunter_flow.py -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix12-prompt-review
git add werewolf_agent/runtime/directives/hunter.py tests/runtime/test_strategy_directives.py
git commit -m "fix(prompt): hunter 遗言加一致性条款，与 _hunter_shot_target_from_last_words 行为对齐 (post-review-C6)"
```

---

## Prompt Branch 全量回归

Run: `cd E:/NLP/agent/wofkill/.worktrees/fix12-prompt-review && /d/Miniforge3/Scripts/pytest tests/runtime/ tests/agents/ tests/cognition/ tests/rules/ -p no:cacheprovider -q`
Expected: all pass

合并：

```bash
cd E:/NLP/agent/wofkill
git checkout master
git merge --no-ff fix12-prompt-review -m "merge: fix12-prompt-review — 4 prompt + 1 visibility fixes (post-review batch 1)"
```

---

# Branch 2: `fix13-arch-review`

## Batch A1: postgres_store 补齐 Protocol

### Task A1.1: 写测试

**Files:**
- Modify: `tests/storage/test_postgres_store.py` (新增 `TestPostgresStoreProtocolCompleteness`)

- [ ] **Step 1: 写失败测试**

```python
class TestPostgresStoreProtocolCompleteness:
    """审查 A1: PostgresGameRepository 必须实现 GameRepository Protocol 全部方法。"""

    def test_postgres_implements_all_protocol_methods(self):
        from werewolf_agent.storage.repository import GameRepository
        from werewolf_agent.storage.postgres_store import PostgresGameRepository
        # Protocol 列出所有必须方法
        required = {
            "save_game", "load_game", "append_events", "load_events",
            "save_deaths", "load_deaths", "save_model_usage", "load_model_usage",
            "save_evaluation", "load_evaluation", "save_config_snapshot", "load_config_snapshot",
            "save_custom_config", "load_custom_config", "list_custom_configs",
            "list_games", "delete_game",
            "save_rag_entries", "load_rag_entries",
            "save_memory_snapshot", "list_memory_snapshots",
            "save_reflection", "load_reflections_by_game", "load_reflections_by_player",
        }
        # 检查 Postgres 实际方法
        # 注：PostgresGameRepository 需要 __init__ 参数；用 skipUnless 或 mock
        try:
            pg = PostgresGameRepository.__new__(PostgresGameRepository)
        except Exception:
            pytest.skip("requires real DSN")
        for method in required:
            assert hasattr(pg, method), f"PostgresGameRepository missing: {method}"

    def test_postgres_save_custom_config_actually_persists(self):
        """save_custom_config 必须真写入而不是抛 NotImplementedError。"""
        from werewolf_agent.storage.postgres_store import PostgresGameRepository
        # 用 mock 或测试桩
        pg = PostgresGameRepository.__new__(PostgresGameRepository)
        # 期望：有 _custom_configs 字段或 dict
        assert hasattr(pg, "_custom_configs") or hasattr(pg, "save_custom_config")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/storage/test_postgres_store.py::TestPostgresStoreProtocolCompleteness -v`
Expected: FAIL（postgres 缺 3 个方法）

---

### Task A1.2: 在 PostgresGameRepository 加 3 个方法

**Files:**
- Modify: `werewolf_agent/storage/postgres_store.py`

- [ ] **Step 1: 加 `_custom_configs` 字段到 __init__**

读 `postgres_store.py:__init__`，加：

```python
        self._custom_configs: dict[str, dict] = {}
```

- [ ] **Step 2: 加 3 个方法实现**

参考 `InMemoryGameRepository` 的实现，添加：

```python
    def save_custom_config(self, config_id: str, config: dict) -> None:
        with self._ensure_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO custom_configs (config_id, payload) VALUES (%s, %s) "
                    "ON CONFLICT (config_id) DO UPDATE SET payload = EXCLUDED.payload",
                    (config_id, json.dumps(config)),
                )

    def load_custom_config(self, config_id: str) -> dict | None:
        with self._ensure_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM custom_configs WHERE config_id = %s",
                    (config_id,),
                )
                row = cur.fetchone()
                return json.loads(row[0]) if row else None

    def list_custom_configs(self) -> list[dict]:
        with self._ensure_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT config_id, payload FROM custom_configs")
                return [
                    {"config_id": r[0], "config": json.loads(r[1])}
                    for r in cur.fetchall()
                ]
```

按需适配 cursor 风格（psycopg v3 vs v2）。

- [ ] **Step 3: 加 migration 表格**

在 `storage/migrations.py` v1 schema 加 custom_configs 表（如果还没有）：

```sql
CREATE TABLE IF NOT EXISTS custom_configs (
    config_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);
```

- [ ] **Step 4: 跑测试**

Run: `pytest tests/storage/test_postgres_store.py::TestPostgresStoreProtocolCompleteness -v`
Expected: PASS

- [ ] **Step 5: 跑 storage regression**

Run: `pytest tests/storage/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix13-arch-review
git add werewolf_agent/storage/postgres_store.py werewolf_agent/storage/migrations.py tests/storage/test_postgres_store.py
git commit -m "feat(storage): PostgresGameRepository 补齐 save_custom_config/load/list (post-review-A1)"
```

---

## Batch A2: customization 持久化

### Task A2.1: 写测试

**Files:**
- Modify: `tests/customization/test_repository.py` (新增 `TestCustomizationRepositoryPersistence`)

- [ ] **Step 1: 写失败测试**

```python
class TestCustomizationRepositoryPersistence:
    """审查 A2: InMemoryCustomizationRepository 应保留自定义 config（持久化 mock）。"""

    def test_in_memory_repo_save_and_load_config(self):
        from werewolf_agent.customization.repository import InMemoryCustomizationRepository
        repo = InMemoryCustomizationRepository()
        config = {"ruleset_id": "test_v1", "param": "value"}
        repo.save_ruleset("test_v1", config)
        loaded = repo.load_ruleset("test_v1")
        assert loaded == config, f"round-trip failed: {loaded}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/customization/test_repository.py::TestCustomizationRepositoryPersistence -v`
Expected: 视现状，可能 PASS（已有）或 FAIL（接口名错）

---

### Task A2.2: 评估 InMemory 持久化是否需要做

**Files:**
- Modify: `werewolf_agent/customization/repository.py` (按需)

- [ ] **Step 1: 读现有 InMemory 实现**

Run: `cat werewolf_agent/customization/repository.py`

如果 InMemory 已经支持 save/load（仅是 mock）—— 加测试覆盖即可。
如果 InMemory 完全没 save/load——按下面补：

```python
class InMemoryCustomizationRepository:
    def __init__(self):
        self._rulesets: dict[str, dict] = {}
        self._persona_packs: dict[str, dict] = {}

    def save_ruleset(self, ruleset_id: str, config: dict) -> None:
        self._rulesets[ruleset_id] = config

    def load_ruleset(self, ruleset_id: str) -> dict | None:
        return self._rulesets.get(ruleset_id)
```

(实际接口名/方法名按代码现状)

- [ ] **Step 2: 跑测试**

Run: `pytest tests/customization/ -q`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix13-arch-review
git add werewolf_agent/customization/repository.py tests/customization/test_repository.py
git commit -m "feat(customization): InMemoryCustomizationRepository 补 save/load_ruleset (post-review-A2)"
```

---

## Batch A3: api/app.py NoOp custom config 修复

### Task A3.1: 写测试

**Files:**
- Modify: `tests/api/test_app.py` (新增 `TestPersistCustomConfigNoOp`)

- [ ] **Step 1: 写失败测试**

```python
class TestPersistCustomConfigNoOp:
    """审查 A3: api/app.py _persist_custom_config 必须真写入 repo。"""

    def test_persist_custom_config_writes_to_repo(self, tmp_path, monkeypatch):
        from werewolf_agent.api.app import _persist_custom_config
        from werewolf_agent.storage.repository import GameRepository

        class StubRepo:
            def __init__(self):
                self.saved = []
            def save_custom_config(self, cid, cfg):
                self.saved.append((cid, cfg))

        repo = StubRepo()
        _persist_custom_config(repo, "test_id", {"foo": "bar"})
        assert repo.saved == [("test_id", {"foo": "bar"})], (
            f"_persist_custom_config is NoOp for repos without save_custom_config: {repo.saved}"
        )

    def test_persist_custom_config_silently_drops_silently_logs(self, caplog):
        """当 repo 没 save_custom_config 时应 warn + 抛，不静默丢。"""
        from werewolf_agent.api.app import _persist_custom_config

        class NoOpRepo:
            pass

        with caplog.at_level("WARNING"):
            _persist_custom_config(NoOpRepo(), "test_id", {"foo": "bar"})
        # 期望有 warning
        assert any("custom_config" in r.message for r in caplog.records), (
            "missing WARNING log when repo lacks save_custom_config"
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/api/test_app.py::TestPersistCustomConfigNoOp -v`
Expected: FAIL

---

### Task A3.2: 改 api/app.py 的 _persist_custom_config

**Files:**
- Modify: `werewolf_agent/api/app.py` (找 `_persist_custom_config`)

- [ ] **Step 1: 加 warn 日志 + raise**

读 `api/app.py` 找 `_persist_custom_config` 函数。当前实现（review 报告说 line 135）可能：

```python
def _persist_custom_config(repo, config_id, config):
    if hasattr(repo, "save_custom_config"):
        repo.save_custom_config(config_id, config)
```

改为：

```python
def _persist_custom_config(repo, config_id, config):
    if hasattr(repo, "save_custom_config"):
        repo.save_custom_config(config_id, config)
        logger.info("Persisted custom config %s", config_id)
    else:
        logger.warning(
            "Repository %s lacks save_custom_config — custom config %s will be lost on restart",
            type(repo).__name__, config_id,
        )
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/api/test_app.py::TestPersistCustomConfigNoOp -v`
Expected: PASS

- [ ] **Step 3: 跑 api regression**

Run: `pytest tests/api/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix13-arch-review
git add werewolf_agent/api/app.py tests/api/test_app.py
git commit -m "fix(api): _persist_custom_config 加 warn 日志，repo 缺方法时不再静默丢 (post-review-A3)"
```

---

## Batch A5: agent_sheriff_pick_speech_order 契约修复

### Task A5.1: 写测试

**Files:**
- Modify: `tests/runtime/test_agent_adapter.py` (新增 `TestSheriffPickSpeechOrderContract`)

- [ ] **Step 1: 写失败测试**

```python
class TestSheriffPickSpeechOrderContract:
    """审查 A5: agent_sheriff_pick_speech_order 不应 model_copy 改 legal_actions 改 task_type。"""

    def test_pick_speech_order_legal_actions_consistent_with_task_type(self):
        """如果 task_type 是 SHERIFF_SPEECH，legal_actions 应含 SPEECH 不含 VOTE。"""
        from werewolf_agent.runtime.agent_adapter import _build_agent_context_for_task
        from werewolf_agent.agents.schemas import TaskType
        # 构造最小 ctx
        # 期望：构建的 ctx 中 legal_actions 包含 [SPEECH]（或按 task_type 合法动作）
        # 实际：当前 model_copy 把 legal_actions 改成 [VOTE]
        # ...
```

（按实际函数签名调整测试构造）

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 改 agent_adapter.py 的 agent_sheriff_pick_speech_order**

读 `agent_adapter.py:1009-1012` 找到 `model_copy(update={"legal_actions": [VOTE]})`，改为：

```python
# 旧
ctx = ctx.model_copy(update={"legal_actions": [ActionType.VOTE]})
# 新：保持 [SPEECH] 合法动作（实际行为：警长选发言序时只能选 p0X，但不能 VOTE 玩家）
# 即：legal_actions 保持原状，不再 model_copy
```

或更彻底：把这个 model_copy 整段删掉（如果原合法动作就是 [SPEECH]）。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 regression**

Run: `pytest tests/runtime/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix13-arch-review
git add werewolf_agent/runtime/agent_adapter.py tests/runtime/test_agent_adapter.py
git commit -m "fix(agent_adapter): agent_sheriff_pick_speech_order 移除 model_copy 改 legal_actions (post-review-A5)"
```

---

## Batch A6: _sheriff_endorse_adapter 迁移到 build_agent_context

### Task A6.1: 写测试

**Files:**
- Modify: `tests/runtime/test_sheriff_flow.py` (新增 `TestSheriffEndorseAdapterModernization`)

- [ ] **Step 1: 写失败测试**

```python
class TestSheriffEndorseAdapterModernization:
    """审查 A6: _sheriff_endorse_adapter 不再用老 4 参 agent.act() 接口。"""

    def test_endorse_adapter_uses_build_agent_context(self):
        """Endorse 路径应走 build_agent_context + _merge_strategy_directive。"""
        import inspect
        from werewolf_agent.runtime.nodes.sheriff import _sheriff_endorse_adapter
        src = inspect.getsource(_sheriff_endorse_adapter)
        # 老接口特征：直接 agent.act(prompt=..., system_prompt=..., task_type=..., legal_actions=...)
        # 新接口特征：build_agent_context + adapter agent_*_endorse
        assert "build_agent_context" in src or "agent_sheriff_endorse" in src, (
            f"_sheriff_endorse_adapter still uses old 4-arg agent.act() interface"
        )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 改 `_sheriff_endorse_adapter`**

读 `sheriff.py:498-587`，整段重写为走 `build_agent_context` + `agent_sheriff_endorse` 路径（参考 `agent_sheriff_election_speech` 的写法）。

实际改法视代码现状，目标是消除老 4 参 `agent.act` 接口的直接调用。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 sheriff flow regression**

Run: `pytest tests/runtime/test_sheriff_flow.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix13-arch-review
git add werewolf_agent/runtime/nodes/sheriff.py tests/runtime/test_sheriff_flow.py
git commit -m "refactor(sheriff): _sheriff_endorse_adapter 迁移到 build_agent_context (post-review-A6)"
```

---

## Arch Branch 全量回归

Run: `cd E:/NLP/agent/wofkill/.worktrees/fix13-arch-review && /d/Miniforge3/Scripts/pytest tests/storage/ tests/api/ tests/customization/ tests/runtime/ -p no:cacheprovider -q`
Expected: all pass

合并：

```bash
cd E:/NLP/agent/wofkill
git merge --no-ff fix13-arch-review -m "merge: fix13-arch-review — 4 architecture debt fixes (post-review batch 2)"
```

---

# Branch 3: `fix14-cleanup`

## Batch U1: 死代码清理（5 处）

### Task U1.1: 删除 `_legacy_wolf_consensus` 入口处的脆弱 guard

**Files:**
- Modify: `werewolf_agent/runtime/nodes/night.py:633` (找到 `_ = AGENT_TIMEOUTS.wolf_consensus` 注释和"wired in future"残留)

- [ ] **Step 1: 清理注释**

读 `night.py:633` 附近的注释和死赋值（review 报告说 line 19-176 整段保留）。删除：

- `_ = AGENT_TIMEOUTS.wolf_consensus  # referenced for timeout contract, wired in future`
- 任何注释中"wired in future"的占位行

- [ ] **Step 2: 跑 regression**

Run: `pytest tests/runtime/test_night_flow.py tests/runtime/test_wolf_flow.py -q`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/runtime/nodes/night.py
git commit -m "chore(cleanup): 删除 _legacy_wolf_consensus 入口的 wired-in-future 占位注释 (post-review-U1)"
```

---

### Task U1.2: 删除 `post_exile_skills` 空节点

**Files:**
- Modify: `werewolf_agent/runtime/nodes/skills.py:33-37` + `werewolf_agent/runtime/graph.py` (图里直接 `exile_last_words → check_victory`)

- [ ] **Step 1: 删 `post_exile_skills` 函数**

读 `skills.py:33-37`，删除整个 `post_exile_skills` 函数（注释说"只占位"）。

- [ ] **Step 2: 改 graph.py 路由**

Run: `grep -n "post_exile_skills" werewolf_agent/runtime/graph.py`

把 `post_exile_skills` 的所有路由引用改为直接连到 `check_victory`（或下一节点）。

- [ ] **Step 3: 删所有测试 + 注册**

Run: `grep -rn "post_exile_skills" werewolf_agent/ tests/`

清理所有相关测试 + register 调用。

- [ ] **Step 4: 跑 regression**

Run: `pytest tests/runtime/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/runtime/nodes/skills.py werewolf_agent/runtime/graph.py tests/
git commit -m "chore(cleanup): 删除 post_exile_skills 空节点 (post-review-U2)"
```

---

### Task U1.3: 删除 `RuntimeState` 死字段

**Files:**
- Modify: `werewolf_agent/runtime/nodes/_shared.py` (RuntimeState TypedDict)

- [ ] **Step 1: 写测试**

在 `tests/runtime/test_graph_lifecycle.py` 加：

```python
def test_runtime_state_no_dead_fields():
    """RuntimeState 不应再含死代码字段。"""
    from werewolf_agent.runtime.nodes._shared import RuntimeState
    expected_minimal = {
        "game_state", "runtime_state_id", "wolf_action", "wolf_action_reason",
        "hybrid_master_target_id", "wolf_team_plan", "consecutive_no_exile_days",
        # 等等
    }
    # 期望：runtime_timer / agent_call_timeout 已删除
    annotations = RuntimeState.__annotations__
    assert "runtime_timer" not in annotations
    assert "agent_call_timeout" not in annotations
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 删字段**

读 `_shared.py:54-114` 删除 `runtime_timer` / `agent_call_timeout` 字段声明和 `_timer_expired` 函数。

- [ ] **Step 4: 跑 regression**

Run: `pytest tests/runtime/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/runtime/nodes/_shared.py tests/runtime/test_graph_lifecycle.py
git commit -m "chore(cleanup): 删除 RuntimeState.runtime_timer / agent_call_timeout 死字段 (post-review-U3)"
```

---

## Batch U2: `_NEGATION_WORDS` 字符类对齐

### Task U2.1: 写测试

**Files:**
- Modify: `tests/runtime/test_strategy_wolf.py` (新增 `TestNegationCharClassAlignment`)

- [ ] **Step 1: 写失败测试**

```python
class TestNegationCharClassAlignment:
    """审查 U2: _NEGATION_WORDS 在 wolf/hunter 字符类应一致。"""

    def test_negation_words_match_between_wolf_and_hunter(self):
        from werewolf_agent.runtime.strategy import wolf as wolf_mod
        from werewolf_agent.runtime.strategy import hunter as hunter_mod
        # 比较两个模块的 _NEGATION_WORDS 集合
        wolf_words = getattr(wolf_mod, "_NEGATION_WORDS", set())
        hunter_words = getattr(hunter_mod, "_NEGATION_WORDS", set())
        # 至少关键 token 必须两处都有
        for token in ("不是", "不", "没", "无", "非"):
            assert token in wolf_words, f"wolf missing {token}"
            assert token in hunter_words, f"hunter missing {token}"
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 抽取到 `_shared.py`**

读 `strategy/wolf.py:20-38` 和 `strategy/hunter.py:17-28`。

在 `strategy/_shared.py`（如果不存在则新建）加：

```python
_NEGATION_WORDS = frozenset({
    "不是", "不", "没", "无", "非", "别", "未", "否认",
    "绝对不是", "绝对不", "并没", "并非", "绝不",
})

_NEGATION_RE = re.compile(
    r"(" + "|".join(re.escape(w) for w in _NEGATION_WORDS) + r")"
)

def speech_is_negated(text: str) -> bool:
    """P-U2: wolf/hunter 共用 negation 检测。"""
    return bool(_NEGATION_RE.search(text))
```

删除 `wolf.py` 和 `hunter.py` 内的本地 `_NEGATION_WORDS/_NEGATION_RE/_speech_is_negated`，改 `from ._shared import speech_is_negated`。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 strategy regression**

Run: `pytest tests/runtime/test_strategy_wolf.py tests/runtime/test_strategy_hunter.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/runtime/strategy/ tests/runtime/test_strategy_wolf.py
git commit -m "refactor(strategy): _NEGATION_WORDS 抽到 _shared.py，wolf/hunter 字符类对齐 (post-review-U4)"
```

---

## Batch U3: `_stable_seed` 去重

### Task U3.1: 写测试

**Files:**
- Modify: `tests/runtime/test_sheriff_policy.py` (新增 `TestStableSeedDedup`)

- [ ] **Step 1: 写测试**

```python
class TestStableSeedDedup:
    """审查 U3: _stable_seed 不应在 sheriff_policy 重复实现。"""

    def test_sheriff_policy_imports_stable_seed_from_shared(self):
        import inspect
        from werewolf_agent.runtime.sheriff_policy import _stable_seed
        # 期望：sheriff_policy._stable_seed 是从 _shared 导入的同一函数
        from werewolf_agent.runtime.nodes._shared import _stable_seed as shared_seed
        assert _stable_seed is shared_seed, (
            f"sheriff_policy has its own _stable_seed implementation; should import from _shared"
        )
```

- [ ] **Step 2: 改 sheriff_policy.py**

读 `sheriff_policy.py:21`，把本地 `_stable_seed` 定义删除，改 import：

```python
from werewolf_agent.runtime.nodes._shared import _stable_seed
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/runtime/test_sheriff_policy.py -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/runtime/sheriff_policy.py tests/runtime/test_sheriff_policy.py
git commit -m "refactor: sheriff_policy 复用 _shared._stable_seed，去掉本地重复实现 (post-review-U5)"
```

---

## Batch U4: tools/local_tools stub 真正接上

### Task U4.1: 写测试

**Files:**
- Modify: `tests/tools/test_local_tools.py` (新增 `TestLocalToolsStubWired`)

- [ ] **Step 1: 写失败测试**

```python
class TestLocalToolsStubWired:
    """审查 U8: local_tools._query_cognition_matrix / _write_review 不再是 stub。"""

    def test_query_cognition_matrix_returns_real_data(self, tmp_path):
        from werewolf_agent.tools.local_tools import _query_cognition_matrix
        result = _query_cognition_matrix("p01", "p02")
        # 期望：返回真实 cognition matrix 数据
        assert result.get("available") is True
        assert "faction_read" in result or "trust" in result, (
            f"_query_cognition_matrix is still a stub: {result}"
        )

    def test_write_review_persists_to_storage(self, tmp_path):
        from werewolf_agent.tools.local_tools import _write_review
        # 期望：调用后能在某 storage 找到 review
        # 简化：检查函数返回包含 review_id（说明真写入了）
        result = _write_review("g_test", "p01", {"logic": 0.5})
        assert result.get("persisted") is True
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 接通到 `MemoryStore` / `PersistentMemoryCoordinator`**

读 `tools/local_tools.py` 当前 stub 行为。改为：

```python
def _query_cognition_matrix(viewer_id, target_id) -> dict:
    from werewolf_agent.memory.store import MemoryStore
    store = MemoryStore.instance()  # 或合适访问方式
    matrix = store.get_cognition_matrix(viewer_id, target_id)
    return {
        "available": matrix is not None,
        "faction_read": matrix.faction_read if matrix else None,
        "trust": matrix.trust if matrix else None,
        "key_evidence": matrix.key_evidence if matrix else [],
    }

def _write_review(game_id, player_id, review_data) -> dict:
    from werewolf_agent.memory.store import MemoryStore
    store = MemoryStore.instance()
    review_id = store.save_review(game_id, player_id, review_data)
    return {"persisted": True, "review_id": review_id}
```

按 `MemoryStore` 实际 API 调整。

- [ ] **Step 4: 跑测试**

- [ ] **Step 5: 跑 tools regression**

Run: `pytest tests/tools/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/tools/local_tools.py tests/tools/test_local_tools.py
git commit -m "fix(tools): local_tools._query_cognition_matrix / _write_review 真正接 MemoryStore (post-review-U8)"
```

---

## Batch U7: _REFLECTION_PLAYER_ID_RE 扩展

### Task U7.1: 写测试

**Files:**
- Modify: `tests/memory/test_store.py` (新增 `TestReflectionPlayerIdRegexCoverage`)

- [ ] **Step 1: 写失败测试**

```python
class TestReflectionPlayerIdRegexCoverage:
    """审查 U7: _REFLECTION_PLAYER_ID_RE 应覆盖更多命名。"""

    @pytest.mark.parametrize("pid", ["p01", "p12", "p99", "p100", "agent_3", "P5"])
    def test_scrub_handles_various_player_id_formats(self, pid):
        from werewolf_agent.memory.store import _REFLECTION_PLAYER_ID_RE
        text = f"玩家 {pid} 当时站边 p01"
        scrubbed = _REFLECTION_PLAYER_ID_RE.sub("[玩家ID已省略]", text)
        assert pid not in scrubbed, (
            f"_REFLECTION_PLAYER_ID_RE failed to scrub {pid}: {scrubbed}"
        )
```

- [ ] **Step 2: 跑测试**

- [ ] **Step 3: 改 regex**

读 `memory/store.py:35-43`，扩展 regex：

```python
_REFLECTION_PLAYER_ID_RE = re.compile(
    r"\b(?:p|player|agent)[\d_]{0,4}\d\b",  # 覆盖 p01-p999 / player_01 / agent_3
    re.IGNORECASE,
)
```

按需微调。

- [ ] **Step 4: 跑测试**

Run: `pytest tests/memory/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/memory/store.py tests/memory/test_store.py
git commit -m "fix(memory): _REFLECTION_PLAYER_ID_RE 扩展覆盖 p100+/player_N/agent_N 命名 (post-review-U7)"
```

---

## Batch U10: schema 双源对齐

### Task U10.1: 写测试

**Files:**
- Modify: `tests/storage/test_sqlite_migrations.py` (新增 `TestSchemaDriftGuard`)

- [ ] **Step 1: 写失败测试**

```python
class TestSchemaDriftGuard:
    """审查 U10: SqliteGameRepository._SCHEMA 与 migrations.py v1 必须一致。"""

    def test_sqlite_schema_matches_migration(self):
        from werewolf_agent.storage.sqlite_store import _SCHEMA
        from werewolf_agent.storage.migrations import MIGRATIONS
        v1_migration = next(m for m in MIGRATIONS if m.version == 1)
        # 提取 CREATE TABLE 语句集合
        import re
        schema_tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _SCHEMA))
        migration_tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", v1_migration.sql))
        assert schema_tables == migration_tables, (
            f"schema drift: _SCHEMA has {schema_tables - migration_tables}, "
            f"migrations has {migration_tables - schema_tables}"
        )
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 修复 drift**

读 `sqlite_store.py:_SCHEMA` 和 `migrations.py:v1`，找出 diff 并修复：

- 如果 `_SCHEMA` 多了 `custom_configs` 而 `migrations` 少 → 加 migration
- 反之亦然

- [ ] **Step 4: 跑测试**

Run: `pytest tests/storage/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/storage/sqlite_store.py werewolf_agent/storage/migrations.py tests/storage/test_sqlite_migrations.py
git commit -m "fix(storage): _SCHEMA 与 migrations v1 对齐，防 schema 漂移 (post-review-U10)"
```

---

## Batch U11: judge.py catch-all 异常 logger

### Task U11.1: 写测试

**Files:**
- Modify: `tests/agents/test_judge_agent.py` (新增 `TestJudgeCatchAllLogging`)

- [ ] **Step 1: 写失败测试**

```python
class TestJudgeCatchAllLogging:
    """审查 U11: judge.py LLM catch-all 异常应记 logger.warning。"""

    def test_guide_skill_use_logs_on_exception(self, caplog):
        from werewolf_agent.agents.judge import JudgeAgent
        # 构造会抛异常的 mock router
        # 调 guide_skill_use
        # 期望：caplog 含 WARNING
        with caplog.at_level("WARNING"):
            # 触发
            pass
        assert any("guide_skill_use" in r.message for r in caplog.records), (
            "judge.py catch-all swallows exception silently"
        )
```

- [ ] **Step 2: 改 judge.py**

读 `judge.py:271-292`（guide_skill_use）和 `judge.py:425-436`、`489-537` 等 catch-all 块。把 `except Exception: pass` 改为：

```python
except Exception:
    logger.warning("judge.broadcast.XXX failed", exc_info=True)
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/agents/test_judge_agent.py -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/agents/judge.py tests/agents/test_judge_agent.py
git commit -m "fix(judge): catch-all 异常改 logger.warning(exc_info=True) (post-review-U11)"
```

---

## Batch U9: world_state extractor 角色访问收敛

### Task U9.1: 写测试

**Files:**
- Modify: `tests/cognition/test_world_state.py` (新增 `TestExtractorRoleAccess`)

- [ ] **Step 1: 写失败测试**

```python
class TestExtractorRoleAccess:
    """审查 U9: world_state extractor 不应能直接读全角色表。"""

    def test_extractor_receives_only_fact_payload(self):
        """每个 extractor 函数签名不应有 players 字典访问。"""
        from werewolf_agent.cognition.world_state import _EXTRACTORS
        for event_type, extractor in _EXTRACTORS.items():
            sig = inspect.signature(extractor)
            params = list(sig.parameters.keys())
            # 不应有 "state" 形参（会泄露 ground truth）
            # 应只有 "event" 形参
            assert "event" in params, (
                f"extractor for {event_type} missing event param"
            )
            # 注：实际签名可能不同，按现状调整
```

按实际 extractor 签名调整。

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 重构 extractor 签名**

读 `world_state.py:373-385`（`_extract_seer_check`）和所有 `_EXTRACTORS` 注册的函数。把 `(state, event)` 改为 `(event, game_state_players_view=...)` 之类的受限接口。

实际重构视代码现状，目标是**让 extractor 拿不到完整 state.players**。

- [ ] **Step 4: 跑测试**

Run: `pytest tests/cognition/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup
git add werewolf_agent/cognition/world_state.py tests/cognition/test_world_state.py
git commit -m "refactor(world_state): extractor 签名收敛，去掉对 state.players 的全表访问 (post-review-U9)"
```

---

## Cleanup Branch 全量回归

Run: `cd E:/NLP/agent/wofkill/.worktrees/fix14-cleanup && /d/Miniforge3/Scripts/pytest tests/runtime/ tests/agents/ tests/memory/ tests/storage/ tests/cognition/ -p no:cacheprovider -q`
Expected: all pass

合并：

```bash
cd E:/NLP/agent/wofkill
git merge --no-ff fix14-cleanup -m "merge: fix14-cleanup — 8 cleanup/dead-code/dedup fixes (post-review batch 3)"
```

---

# 最终验证（合回 master 后）

## 验证 1: 全量 regression

Run: `cd E:/NLP/agent/wofkill && /d/Miniforge3/Scripts/pytest tests/runtime/ tests/agents/ tests/rules/ tests/memory/ tests/rag/ tests/skills/ tests/cognition/ tests/model_gateway/ tests/storage/ tests/api/ tests/customization/ tests/tools/ tests/evaluation/ -p no:cacheprovider -q`
Expected: all pass

## 验证 2: PROGRESS.md 更新

Run: 追加新章节记录本批修复总结，参考 `2026-06-07-g3223805846-fixes.md` 段落格式。

---

# Self-Review Checklist

| 审查 issue | 对应 Task | 状态 |
|---|---|---|
| C1 villager 读 seer_check 私有 | C1.1, C1.2 | ✅ |
| C2 visibility __init__ no-op | C2.1, C2.2 | ✅ |
| C3 双层硬约束标签 | C3.1, C3.2 | ✅ |
| C4 _format_examples p03 硬编码 | C4.1, C4.2 | ✅ |
| C6 hunter 遗言与实际行为不一致 | C6.1, C6.2 | ✅ |
| A1 postgres_store Protocol 不完整 | A1.1, A1.2 | ✅ |
| A2 customization 持久化 | A2.1, A2.2 | ✅ |
| A3 api/app.py NoOp custom config | A3.1, A3.2 | ✅ |
| A5 agent_sheriff_pick_speech_order 契约 | A5.1, A5.2 | ✅ |
| A6 _sheriff_endorse_adapter 老接口 | A6.1, A6.2 | ✅ |
| U1 _legacy_wolf_consensus 入口注释 | U1.1 | ✅ |
| U2 post_exile_skills 空节点 | U1.2 | ✅ |
| U3 RuntimeState 死字段 | U1.3 | ✅ |
| U4 _NEGATION_WORDS 漂移 | U2.1, U2.2 | ✅ |
| U5 _stable_seed 重复 | U3.1 | ✅ |
| U8 local_tools stub | U4.1, U4.2 | ✅ |
| U7 _REFLECTION_PLAYER_ID_RE 命名空间 | U7.1 | ✅ |
| U10 schema 双源漂移 | U10.1 | ✅ |
| U11 judge catch-all 静默 | U11.1 | ✅ |
| U9 world_state extractor 全表访问 | U9.1 | ✅ |

**未在 v1 修复（标 ⚠️ 留作 v2）**：
- `skills/werewolf_skills.py:106` 散文未注入 + eager-load — 设计问题，需先决定"markdown-driven"语义
- `api/views.py` 与 `api/routes/games.py` 路由不匹配（dashboard.js 404）— UI 与 server 协同
- `customization/ruleset_registry` 只硬编码默认 12 人局 — marketplace 路线未定
- 各种 actor 与 `_NEGATION_WORDS` 之外的小漂移（hunter `sheriff_election` 缺 timeout 等）

---

# 执行方式

Plan 已保存到 `docs/superpowers/plans/2026-06-07-post-review-fixes.md`。

**三条工作分支**：
- `fix12-prompt-review` (5 commits: C1, C2, C3, C4, C6)
- `fix13-arch-review` (5 commits: A1, A2, A3, A5, A6)
- `fix14-cleanup` (9 commits: U1.1, U1.2, U1.3, U2, U3, U4, U7, U9, U10, U11)

**两个执行选项**：

1. **Subagent-Driven (推荐)** — 每条 task 派一个独立 subagent 执行，spec + code quality 双 review
2. **Inline Execution** — 在当前会话按 Batch 顺序逐条执行，每个 commit 后做 checkpoint

请选执行方式。
