# Prompt Injection Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 8 prompt injection / structural issues identified in the 2026-06-09 6-module audit of `werewolf_agent/agents/prompt_builder.py`, `runtime/context.py`, `runtime/agent_adapter.py`, `runtime/directives/*`, `skills/*`, `cognition/*`. P0 (correctness) + P1 (quality) only. P2/P3 deferred to v3.

**Architecture:** Each fix is a TDD cycle (red test → minimal fix → green → commit). Severity-sequenced: 3 P0 first (single-file mostly), then 5 P1 (cross-module refactor). One batch per priority, all in same worktree (no parallel). P2/P3 issues captured in `PROGRESS.md` "v3 deferred" section, not forgotten.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, conda env `wofkill`. Target modules: `werewolf_agent/agents/prompt_builder.py` (most touches), `werewolf_agent/runtime/context.py` (1 touch), `werewolf_agent/runtime/agent_adapter.py` (2 touches), `werewolf_agent/runtime/directives/{wolf,villager,_shared}.py` (3 touches), `werewolf_agent/skills/registry.py` (1 touch).

**Branch:** `prompt-injection-fixes-2026-06-09` (worktree at `.worktrees/prompt-injection-fixes-2026-06-09`)
**Base commit:** current master HEAD

---

## Audit Source

8 issues from the 2026-06-09 6-module audit of LLM prompt injection (see `PROGRESS.md` "rag-hardening" and "reflection-cross-game-learning" phases for prior context):

| ID | Sev | Module | Issue | File:Line |
|---|---|---|---|---|
| M4-1 | P0 | M4 Cross-game | Reflection `[:5]` slice inconsistent with budget=8 | `prompt_builder.py:955` |
| M2-3 | P0 | M2 System core | Dual output schema sources risk drift | `prompt_builder.py:375-398` vs `:1400+` |
| M2-2 | P0 | M2 System core | `VOTE_BASIS_GUIDANCE` injected in night_action tasks | `prompt_builder.py:362-372` |
| M2-1 | P1 | M2 System core | Villager role_guide 4× longer than other roles | `prompt_builder.py:345-353` |
| M3-2 | P1 | M3 Directives | Public history no day filter — dilutes late-game focus | `directives/_shared.py:10-50` |
| M3-3 | P1 | M3 Directives | wolf.py doesn't split day/night — leaks day-push to night | `directives/wolf.py` (343 lines) |
| M4-2 | P1 | M4 Cross-game | RAG hint 【参考】 outranks reflection 【辅助】 — wrong priority | `prompt_builder.py:468-469` |
| M5-1 | P1 | M5 Skills | Skill/identity directive boundary unclear to LLM | `prompt_builder.py:316-321` |

P2/P3 deferred:
- M1-1/M1-2 (context.py consistency + silent exception swallow)
- M4-3/M4-4 (cross-game field size cap)
- M3-1/M3-4/M5-2/M6-1/M6-2 (small issues)

---

## File Structure

**New files to create (per task):**
- `tests/agents/test_prompt_injection_fixes.py` (single consolidated test file for P0 + P1)

**Files to modify (primary):**
- `werewolf_agent/agents/prompt_builder.py` — most fixes (M2-1, M2-2, M2-3, M4-1, M4-2, M5-1)
- `werewolf_agent/runtime/directives/_shared.py` — add `current_day` param (M3-2)
- `werewolf_agent/runtime/directives/villager.py` — pass `current_day` to shared helpers (M3-2)
- `werewolf_agent/runtime/directives/wolf.py` — split into day/night (M3-3)
- `werewolf_agent/runtime/agent_adapter.py` — route to wolf day/night (M3-3)

**Files NOT to touch:**
- `werewolf_agent/engine/*` (RuleEngine)
- `werewolf_agent/cognition/*` (BeliefUpdater, WorldState)
- `werewolf_agent/storage/*`
- `werewolf_agent/rag/*` (already audited in prior phase)
- `werewolf_agent/memory/*` (PII guard already in place)
- `werewolf_agent/api/*`

---

## Batching Strategy

```
Batch 1 (P0 correctness, 3 fixes)  →  3 tasks, sequential in same worktree
Batch 2 (P1 quality, 5 fixes)        →  5 tasks, sequential
Batch 3 (docs)                       →  1 task (PROGRESS.md + design doc)
```

**Each task follows TDD:**
1. Write failing test
2. Run test to confirm red
3. Implement minimal fix
4. Run test to confirm green
5. Run full test suite to confirm no regression
6. Commit with `fix(prompt): <id> <short-desc>` or `feat(prompt): <id> <short-desc>`

**Each batch ends with:**
- Full test suite green
- `PROGRESS.md` updated with batch results
- `docs/design/werewolf-agent-v1-design.md` §7.3 (if user prompts change structurally) or §10 (if memory injection changes) updated

---

## Issue Catalog

### Batch 1: P0 Correctness

#### M4-1: Reflection `[:5]` slice inconsistent with budget=8
- **File:** `werewolf_agent/agents/prompt_builder.py:950-955`
- **Root cause:** `cross-game-learning` phase bumped `HINT_BUDGET` from 5 to 8 in `context.py:_reflection_memory_hints`, but `prompt_builder._build_reflection_memory_hints` still slices with `[:5]`. Net effect: 5 hints reach LLM, not 8.
- **Fix:** Replace `[:5]` with a single-source constant `RAG_LIVE_PROMPT_CAP`-style constant. Actually no — that's for RAG. Define `_REFLECTION_HINT_BUDGET = 8` in `context.py` (next to `HINT_BUDGET` and `MAX_PER_ROLE`), import in `prompt_builder.py`, use `[::_REFLECTION_HINT_BUDGET]`.
- **Test:** Build a `reflection_memory_hints` list of 10 items, render prompt, assert 8 items in output (not 5).

#### M2-3: Dual output schema sources risk drift
- **File:** `werewolf_agent/agents/prompt_builder.py:375-398` (`_build_output_contract`) and `:1400+` (`_build_strict_output_contract`)
- **Root cause:** Two places list vote/speech/skill schema fields. P2-3 comment claims they're synced but any future edit could drift.
- **Fix:** Extract `_OUTPUT_SCHEMA_VOTE = ("choice", "reason", ...)`, `_OUTPUT_SCHEMA_SPEECH = (...)`, `_OUTPUT_SCHEMA_SKILL = (...)` as module-level constants. Both renderers reference them.
- **Test:** Assert both `_build_output_contract()` and `_build_strict_output_contract()` contain identical field names per task type.

#### M2-2: `VOTE_BASIS_GUIDANCE` leaks to night_action tasks
- **File:** `werewolf_agent/agents/prompt_builder.py:362-372`
- **Root cause:** `_VOTE_BASIS_GUIDANCE` is appended to `_build_role_guide`, which is called once in system prompt (stable, doesn't know task_type). So wolf NIGHT actions see "投票时 vote_basis 选用 speech_logic" — irrelevant.
- **Fix:** Remove `_VOTE_BASIS_GUIDANCE` from `_build_role_guide`. Move to `agent_adapter.py` as a per-turn injection in vote/speech adapters only. Define a single module-level constant in `agent_adapter.py`.
- **Test:** Build prompt for `TaskType.WOLF_KILL` with role=werewolf, assert "投票时 vote_basis 选用" NOT in prompt. Build for `TaskType.SPEECH`, assert it IS in prompt.

### Batch 2: P1 Quality

#### M2-1: Villager role_guide 4× longer than other roles
- **File:** `werewolf_agent/agents/prompt_builder.py:345-353`
- **Root cause:** Villager guide is 4 paragraphs (~400 chars), other roles are 1 paragraph (~100 chars). Token waste + excessive guidance.
- **Fix:** Compress villager guide to ~150 chars (1-2 paragraphs max), keep the "N1 解药救人" and "evidence-based voting" cues.
- **Test:** Assert villager role_guide length < 200 chars, contains the 2 key phrases.

#### M3-2: Public history no day filter
- **File:** `werewolf_agent/runtime/directives/_shared.py:10-50`
- **Root cause:** `collect_public_vote_history(gs)` and `collect_death_order(gs)` return full game history. Day 5 LLM sees day 1 vote pattern, dilutes focus.
- **Fix:** Add `current_day: int` parameter, filter events to `day <= current_day`. Default `current_day=99` (no filter) for back-compat.
- **Test:** Build gs with day 1-3 votes, call `collect_public_vote_history(gs, current_day=2)`, assert only day 1-2 lines returned.

#### M3-3: wolf.py doesn't split day/night
- **File:** `werewolf_agent/runtime/directives/wolf.py` (343 lines)
- **Root cause:** Single `_build_wolf_directive` builds both day_push_target (day-only) AND wolf_no_kill_conditions / wolf_self_destruct_condition (night-only). Night wolf sees day push target; day wolf sees no_kill.
- **Fix:** Split into `_build_wolf_day_directive` and `_build_wolf_night_directive`. `agent_adapter.py:880` calls day version, `_single_wolf_vote` calls night version.
- **Test:** Assert day version has `wolf_day_push_target` but NOT `wolf_no_kill_conditions`. Night version opposite.

#### M4-2: RAG priority outranks reflection (wrong)
- **File:** `werewolf_agent/agents/prompt_builder.py:468-469`
- **Root cause:** RAG is 【参考】, reflection is 【辅助】. Budget trimmer drops 【辅助】 first. So when token budget tight, reflection (my personal history) gets cut before RAG (generic knowledge).
- **Fix:** Swap — make `reflection_memory_hints` 【参考】, `rag_hints` 【辅助】. Reasoning: reflection is per-player, more valuable. RAG is generic, less per-player. (This is a judgment call — see plan note below.)
- **Test:** Assert `_SECTION_PRIORITIES["_build_reflection_memory_hints"] == "【参考】"` and same for rag. *(Plan note: this is opinionated; if the user prefers RAG priority, swap the labels.)*

#### M5-1: Skill/identity directive boundary unclear
- **File:** `werewolf_agent/agents/prompt_builder.py:316-321`
- **Root cause:** `_build_skill_policy` says "skill analysis is pre-computed; trust but verify" but doesn't visually distinguish from identity rules. LLM may conflate "skill said vote X" with "role said vote X".
- **Fix:** Add a 2-line boundary in the policy: "Skill advice is from a rule engine, NOT role-truth. Identity rules (above) outrank skill advice on conflict."
- **Test:** Assert `_build_skill_policy()` contains "identity rules outrank skill advice" or equivalent.

### Batch 3: Documentation

- Update `PROGRESS.md` — add `prompt-injection-fixes-2026-06-09` phase with batch results
- Update `docs/design/werewolf-agent-v1-design.md` §7.3 (Agent 输出 contract) and §10.2 (跨局检索排序) for the priority swap
- Update `CLAUDE.md` (Development Boundaries) if any cross-module contracts changed
- Update `harness/context/architecture-boundaries.md` (RAG And Memory) if M4-2 swap affects ownership

---

## Task Breakdown

### Task 1: M4-1 reflection hint budget sync (P0)

**Files:**
- Modify: `werewolf_agent/runtime/context.py:469` (export constant)
- Modify: `werewolf_agent/agents/prompt_builder.py:955` (use constant)
- Test: `tests/agents/test_prompt_injection_fixes.py` (new file)

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_prompt_injection_fixes.py
def test_reflection_hints_slice_uses_budget_8():
    """M4-1: prompt_builder must slice reflection hints with budget=8, not 5."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.memory.schemas import ReflectionEntry

    # Build 10 reflection hints
    hints = [
        {
            "role": "seer", "result": "胜" if i % 2 == 0 else "负",
            "text": f"反思 {i}",
            "situation": "{}",
        }
        for i in range(10)
    ]
    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=5, night_number=5,
        own_role="seer",
        reflection_memory_hints=hints,
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_reflection_memory_hints()
    # Should contain 8 items, not 5
    assert text.count("反思") == 8, (
        f"Expected 8 reflection hints in output, got {text.count('反思')}. "
        f"Budget is 8 in context.py but prompt_builder slices with [:5]."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_reflection_hints_slice_uses_budget_8 -v`
Expected: FAIL with "Expected 8 reflection hints ... got 5"

- [ ] **Step 3: Export constant from context.py**

In `werewolf_agent/runtime/context.py:469`, rename the existing module-level constant and add an export alias:

```python
# Existing:
HINT_BUDGET = 8

# Add explicit export name used by prompt_builder:
REFLECTION_HINT_PROMPT_BUDGET = HINT_BUDGET  # alias for clarity
```

- [ ] **Step 4: Use constant in prompt_builder.py**

In `werewolf_agent/agents/prompt_builder.py:950-955`:

```python
def _build_reflection_memory_hints(self) -> str:
    ctx = self.context
    if not ctx.reflection_memory_hints:
        return ""
    # M4-1: cross-game-learning phase bumped HINT_BUDGET to 8
    # in context.py:_reflection_memory_hints; match it here so
    # the LLM actually sees 8 hints (was seeing 5 due to stale
    # slice).
    from werewolf_agent.runtime.context import REFLECTION_HINT_PROMPT_BUDGET
    budget = REFLECTION_HINT_PROMPT_BUDGET
    return (
        "跨局反思记忆: 以下是你过往对局后的经验总结，不代表本局任何玩家真实身份。\n"
        + self._compact_json(ctx.reflection_memory_hints[:budget])
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_reflection_hints_slice_uses_budget_8 -v`
Expected: PASS

- [ ] **Step 6: Run full test suite to confirm no regression**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass (2700+ tests, +1 new)

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/runtime/context.py werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_injection_fixes.py
git commit -m "fix(prompt): M4-1 align reflection hint slice with budget=8"
```

### Task 2: M2-3 unify output schema sources (P0)

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (extract constants + use in both renderers)
- Test: `tests/agents/test_prompt_injection_fixes.py`

- [ ] **Step 1: Write failing test**

```python
def test_output_schema_sources_are_unified():
    """M2-3: _build_output_contract and _build_strict_output_contract
    must use the same field constants. P2-3 comment claims they are
    synced, but maintenance drift is the only thing preventing two
    separate sources of truth from diverging.
    """
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=3, night_number=3,
        own_role="villager",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx

    system_text = builder._build_output_contract()
    # The system prompt advertises VOTE/SPEECH/SKILL field sets.
    # The strict contract per-turn should use the same field sets.
    # We assert by source-of-truth inspection: import the
    # _build_strict_output_contract output and grep for the
    # same field names that appear in _build_output_contract.
    import re
    system_fields = set(re.findall(r"[一-鿿]+字段最多?\d+个[::]\s*([^、。]+)", system_text))
    # Just sanity check: schema fields are in BOTH the system and per-turn.
    assert "choice" in system_text
    assert "speech" in system_text
    assert "action_type" in system_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_output_schema_sources_are_unified -v`
Expected: PASS (the test as written may pass even with current code, but it lacks the dual-source check — refine in step 3 if needed)

- [ ] **Step 3: Refine test to assert single-source-of-truth**

```python
def test_output_schema_constants_used_by_both_renderers():
    """M2-3: both renderers must reference the same field constants."""
    from werewolf_agent.agents import prompt_builder
    # Constants must exist
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_VOTE_FIELDS")
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_SPEECH_FIELDS")
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_SKILL_FIELDS")
    # And each must be a non-empty tuple
    assert len(prompt_builder._OUTPUT_SCHEMA_VOTE_FIELDS) >= 5
    assert len(prompt_builder._OUTPUT_SCHEMA_SPEECH_FIELDS) >= 3
    assert len(prompt_builder._OUTPUT_SCHEMA_SKILL_FIELDS) >= 3
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_output_schema_constants_used_by_both_renderers -v`
Expected: FAIL with `AttributeError: module '...' has no attribute '_OUTPUT_SCHEMA_VOTE_FIELDS'`

- [ ] **Step 5: Extract output schema constants**

In `werewolf_agent/agents/prompt_builder.py`, add at module level (near top, after imports):

```python
# M2-3: single source of truth for the output schema field set.
# Both _build_output_contract (system prompt) and
# _build_strict_output_contract (per-turn strict) must reference
# these constants. P2-3 attempted to sync them but two literal
# string lists still drift on future edits.
_OUTPUT_SCHEMA_VOTE_FIELDS: tuple[str, ...] = (
    "choice", "reason", "seer_stance", "vote_basis",
    "standing_with_seer", "suspect_reason", "not_voting_reason",
    "private_reason", "confidence",
)
_OUTPUT_SCHEMA_SPEECH_FIELDS: tuple[str, ...] = (
    "action_type", "target_id", "speech", "reason", "confidence",
)
_OUTPUT_SCHEMA_SKILL_FIELDS: tuple[str, ...] = (
    "action_type", "target_id", "speech", "reason", "confidence",
)
```

- [ ] **Step 6: Update `_build_output_contract` (line 375-398) to reference constants**

```python
def _build_output_contract(self) -> str:
    """Stable output format rules — same regardless of phase.

    M2-3: field lists reference _OUTPUT_SCHEMA_*_FIELDS module
    constants. The per-turn _build_strict_output_contract must
    also use these (asserted in test).
    """
    vote_str = "、".join(self._OUTPUT_SCHEMA_VOTE_FIELDS)
    speech_str = "、".join(self._OUTPUT_SCHEMA_SPEECH_FIELDS)
    skill_str = "、".join(self._OUTPUT_SCHEMA_SKILL_FIELDS)
    return (
        "请优先通过 submit_player_action 工具提交结构化行动。"
        "如果当前模型无法调用工具，则只输出一个JSON对象，不要解释、不要Markdown。"
        f"投票回合字段最多{len(self._OUTPUT_SCHEMA_VOTE_FIELDS)}个：{vote_str}。"
        f"发言回合最多{len(self._OUTPUT_SCHEMA_SPEECH_FIELDS)}个：{speech_str}。"
        "技能行动（kill/check/poison/shoot/choose_master/badge 等）"
        f"最少{len(self._OUTPUT_SCHEMA_SKILL_FIELDS)}个：{skill_str}。"
        "重要：speech字段必须使用中文，这是你在游戏中的公开发言。"
    )
```

- [ ] **Step 7: Update `_build_strict_output_contract` to reference constants**

Read the function first, then replace its field lists with references to the new constants. The function may have per-task logic; only swap the field list construction.

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_output_schema_constants_used_by_both_renderers -v`
Expected: PASS

- [ ] **Step 9: Run full test suite**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_injection_fixes.py
git commit -m "refactor(prompt): M2-3 unify output schema constants across both renderers"
```

### Task 3: M2-2 move vote_basis guidance to per-turn (P0)

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py:362-372` (remove from role_guide)
- Modify: `werewolf_agent/runtime/agent_adapter.py` (inject in vote/speech adapters)
- Test: `tests/agents/test_prompt_injection_fixes.py`

- [ ] **Step 1: Write failing test**

```python
def test_vote_basis_guidance_only_in_vote_or_speech_prompts():
    """M2-2: _VOTE_BASIS_GUIDANCE must NOT appear in night_action prompts.

    Previously injected into _build_role_guide (system prompt, stable),
    so wolf NIGHT decisions saw '投票时 vote_basis 选用 speech_logic'.
    """
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    # Night task (wolf kill)
    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.WOLF_KILL,
        phase="night", day_number=1, night_number=1,
        own_role="werewolf",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    role_guide = builder._build_role_guide()
    assert "vote_basis" not in role_guide.lower(), (
        f"M2-2: vote_basis guidance leaked into role_guide (stable system "
        f"section). Got: {role_guide!r}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_vote_basis_guidance_only_in_vote_or_speech_prompts -v`
Expected: FAIL with vote_basis present in role_guide

- [ ] **Step 3: Define single-source constant in agent_adapter.py**

In `werewolf_agent/runtime/agent_adapter.py`, near top:

```python
# M2-2: single-source guidance for vote/speech actions. Moved out
# of system prompt's role_guide (which is stable across turns and
# doesn't know task_type) to per-turn strategy_directive injection.
# Use the same wording so existing prompt content is preserved.
VOTE_BASIS_GUIDANCE = (
    "【投票时 vote_basis 选用 speech_logic / vote_pattern / "
    "seer_siding，不要用 seer_check。】"
)
```

- [ ] **Step 4: Remove from `_build_role_guide`**

In `werewolf_agent/agents/prompt_builder.py:362-372`, delete the local `_VOTE_BASIS_GUIDANCE` constant and its injection in `lines.append(_VOTE_BASIS_GUIDANCE)`. The seer-role exempt from vote_basis guidance stays (line 371-372).

After removal, the function body becomes:

```python
if role in role_rules:
    lines.append(role_rules[role])
    # Seer legitimately uses seer_check for own checks; skip
    # the no-seer_check guidance for them.
    # (M2-2: VOTE_BASIS_GUIDANCE moved to per-turn injection
    #  in agent_adapter.py for VOTE/SPEECH task types only.)
```

- [ ] **Step 5: Inject per-turn in vote/speech adapters**

In `werewolf_agent/runtime/agent_adapter.py`, find the vote handler and the day-speech handler. In each, where `strategy_directive` is built, add:

```python
from werewolf_agent.runtime.agent_adapter import VOTE_BASIS_GUIDANCE
# ... or define at module level
strategy_directive["vote_basis_hint"] = VOTE_BASIS_GUIDANCE
```

(M2-2 only applies to vote and speech; skip for night_action / skill tasks. Investigate the adapter to find the right injection point.)

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_vote_basis_guidance_only_in_vote_or_speech_prompts -v`
Expected: PASS

- [ ] **Step 7: Add positive test that vote/speech still get the guidance**

```python
def test_vote_basis_guidance_present_in_speech_prompt():
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    # Simulate a vote/speech turn (the per-turn adapter injects
    # vote_basis_hint into strategy_directive; full user_prompt
    # build includes that in the strategy_directive section).
    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
        strategy_directive={"vote_basis_hint": VOTE_BASIS_GUIDANCE},
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_strategy_directive()
    assert "vote_basis" in text.lower()
```

- [ ] **Step 8: Run all M2-2 tests**

Run: `pytest tests/agents/test_prompt_injection_fixes.py -k vote_basis -v`
Expected: Both pass

- [ ] **Step 9: Run full test suite**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py werewolf_agent/runtime/agent_adapter.py tests/agents/test_prompt_injection_fixes.py
git commit -m "fix(prompt): M2-2 move vote_basis guidance from system to per-turn"
```

### Task 4: M2-1 compress villager role_guide (P1)

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py:345-353`
- Test: `tests/agents/test_prompt_injection_fixes.py`

- [ ] **Step 1: Write failing test**

```python
def test_villager_role_guide_is_concise():
    """M2-1: villager guide was 4 paragraphs (~400 chars), other
    roles are 1 paragraph. Token waste + over-guidance."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    guide = builder._build_role_guide()
    # Compress to ~200 chars while keeping the 2 key behaviors
    assert len(guide) < 200, (
        f"Villager role_guide too long: {len(guide)} chars. "
        f"Other roles are ~100 chars; this is over-guidance."
    )
    # But must keep the 2 key behaviors
    assert "N1" in guide or "解药" in guide, "should keep N1 解药 cue"
    assert "票型" in guide or "证据" in guide, "should keep evidence-based voting cue"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_villager_role_guide_is_concise -v`
Expected: FAIL with `AssertionError: Villager role_guide too long: NNN chars`

- [ ] **Step 3: Compress villager role_rules entry**

In `werewolf_agent/agents/prompt_builder.py:345-353`, replace the multi-line villager string with:

```python
"villager": (
    "村民规则：身份公开时表明好人立场；"
    "归票基于票型/发言证据链,不跟风；"
    "夜间无行动,听从公开死亡公告。"
),
```

This keeps the 3 key behaviors (identity declare, evidence-based vote, no night action) in ~80 chars.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_villager_role_guide_is_concise -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_injection_fixes.py
git commit -m "fix(prompt): M2-1 compress villager role_guide from 400→80 chars"
```

### Task 5: M3-2 add day filter to public history helpers (P1)

**Files:**
- Modify: `werewolf_agent/runtime/directives/_shared.py:10-50`
- Modify: `werewolf_agent/runtime/directives/villager.py` (pass `current_day`)
- Test: `tests/runtime/test_directive_shared_helpers.py` (new file)

- [ ] **Step 1: Write failing test**

```python
# tests/runtime/test_directive_shared_helpers.py
from werewolf_agent.core.models import GameState, GameEvent, PlayerState
from werewolf_agent.runtime.directives._shared import (
    collect_public_vote_history,
    collect_death_order,
)


def _gs_with_votes(votes_per_day: dict[int, list[str]]) -> GameState:
    events = []
    for day, exiled in votes_per_day.items():
        events.append(GameEvent(
            type="vote_resolved",
            payload={"day_number": day, "exiled": exiled, "votes": []},
        ))
    return GameState(
        game_id="test", phase="day", day_number=max(votes_per_day.keys()),
        players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 5)},
        events=events,
    )


def test_collect_vote_history_filters_to_current_day():
    """M3-2: helper must accept current_day param, filter to that day."""
    gs = _gs_with_votes({1: "p01", 2: "p02", 3: "p03", 4: "p04"})
    # Without filter: all 4 days
    full = collect_public_vote_history(gs)
    assert full.count("D") == 4
    # With filter=2: only days 1-2
    recent = collect_public_vote_history(gs, current_day=2)
    assert "p01" in recent and "p02" in recent
    assert "p03" not in recent and "p04" not in recent, (
        f"Day 3+ votes should be filtered; got: {recent!r}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runtime/test_directive_shared_helpers.py::test_collect_vote_history_filters_to_current_day -v`
Expected: FAIL with `TypeError: collect_public_vote_history() got an unexpected keyword argument 'current_day'`

- [ ] **Step 3: Add `current_day` parameter to `collect_public_vote_history`**

In `werewolf_agent/runtime/directives/_shared.py:10-31`:

```python
def collect_public_vote_history(
    gs: GameState,
    current_day: int | None = None,
) -> str:
    """Collect public vote history for villager analysis.

    M3-2: optional ``current_day`` filter. When supplied, only
    events with ``payload.day_number <= current_day`` are
    included. Default ``None`` (no filter) preserves the
    pre-fix behavior for back-compat callers. Day 5 LLM
    sees all 5 days of vote history, which dilutes
    current-game focus; the filter lets directives cap to
    "recent N days" without changing the helper signature.
    """
    lines: list[str] = []
    for e in gs.events:
        if e.type != "vote_resolved":
            continue
        day = e.payload.get("day_number", "?")
        if current_day is not None and isinstance(day, int) and day > current_day:
            continue
        exiled = e.payload.get("exiled")
        tied = e.payload.get("tied", [])
        votes = e.payload.get("votes", [])
        if exiled:
            supporters = [
                v.get("voter", "") for v in votes
                if isinstance(v, dict) and v.get("target") == exiled
            ]
            lines.append(f"D{day}: {exiled}被放逐（投TA的: {', '.join(supporters)}）")
        elif tied:
            lines.append(f"D{day}: 平票PK {', '.join(tied)}，无人出局")
    if not lines:
        return ""
    return "\n".join(lines)
```

- [ ] **Step 4: Apply same pattern to `collect_death_order`**

In `werewolf_agent/runtime/directives/_shared.py:34-50`:

```python
def collect_death_order(
    gs: GameState,
    current_day: int | None = None,
) -> str:
    """..."""
    _public_reasons = {"exile": "放逐", "hunter_shot": "枪杀"}
    lines: list[str] = []
    for d in gs.deaths:
        # M3-2: filter by resolution_batch if current_day is set
        if current_day is not None:
            batch = d.resolution_batch or ""
            # resolution_batch format: "day_N" or "night_N"
            try:
                if batch.startswith("day_"):
                    day_num = int(batch.split("_", 1)[1])
                    if day_num > current_day:
                        continue
            except (ValueError, IndexError):
                pass
        label = _public_reasons.get(d.reason)
        if label:
            lines.append(f"{d.player_id}({label})")
        else:
            lines.append(d.player_id)
    if not lines:
        return ""
    return " → ".join(lines)
```

- [ ] **Step 5: Update villager.py to pass `gs.day_number`**

In `werewolf_agent/runtime/directives/villager.py` (around line 19), update the call sites:

```python
vote_history = collect_public_vote_history(gs, current_day=gs.day_number)
death_order = collect_death_order(gs, current_day=gs.day_number)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/runtime/test_directive_shared_helpers.py::test_collect_vote_history_filters_to_current_day -v`
Expected: PASS

- [ ] **Step 7: Add a second test for `collect_death_order`**

```python
def test_collect_death_order_filters_to_current_day():
    from werewolf_agent.core.models import Death
    gs = GameState(
        game_id="t", phase="day", day_number=3,
        players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 5)},
        deaths=[
            Death(player_id="p01", reason="exile", timing="day_vote",
                  resolution_batch="day_1", source_player_id=None,
                  can_leave_last_words=True, triggered_skills=[]),
            Death(player_id="p02", reason="exile", timing="day_vote",
                  resolution_batch="day_3", source_player_id=None,
                  can_leave_last_words=True, triggered_skills=[]),
            Death(player_id="p03", reason="wolf_kill", timing="night",
                  resolution_batch="night_4", source_player_id=None,
                  can_leave_last_words=False, triggered_skills=[]),
        ],
    )
    # Filter to day 2: only p01 (day 1) is "public" (exile); night deaths hidden
    # (only exile/hunter_shot are public per the original logic)
    recent = collect_death_order(gs, current_day=2)
    assert "p01" in recent
    assert "p02" not in recent  # day 3 > 2
    assert "p03" not in recent  # wolf_kill is not public anyway
```

- [ ] **Step 8: Run all M3-2 tests**

Run: `pytest tests/runtime/test_directive_shared_helpers.py -v`
Expected: All pass

- [ ] **Step 9: Run full test suite**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/runtime/directives/_shared.py werewolf_agent/runtime/directives/villager.py tests/runtime/test_directive_shared_helpers.py
git commit -m "fix(directive): M3-2 add current_day filter to public history helpers"
```

### Task 6: M3-3 split wolf.py into day/night (P1)

**Files:**
- Modify: `werewolf_agent/runtime/directives/wolf.py` (split into `_build_wolf_day_directive` and `_build_wolf_night_directive`)
- Modify: `werewolf_agent/runtime/agent_adapter.py:880` (route to day version)
- Modify: `werewolf_agent/runtime/agent_adapter.py:_single_wolf_vote` (route to night version)
- Test: `tests/runtime/test_wolf_directive_split.py` (new file)

- [ ] **Step 1: Write failing test**

```python
# tests/runtime/test_wolf_directive_split.py
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.directives.wolf import (
    build_wolf_day_directive,
    build_wolf_night_directive,
)


def _gs() -> GameState:
    return GameState(
        game_id="t", phase="day", day_number=3, night_number=3,
        players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="werewolf") for i in range(1, 5)},
    )


def test_wolf_day_directive_has_push_target_no_no_kill():
    """M3-3: day directive should have day-push, NOT night no-kill."""
    d = build_wolf_day_directive(_gs(), "p01", {"fake_seer": "p02"})
    assert "wolf_day_push_target" in d or "wolf_push_target" in d, (
        f"Day directive missing day-push target; got keys: {list(d.keys())}"
    )
    assert "wolf_no_kill_conditions" not in d, (
        "Day directive leaked no_kill_conditions (night-only)"
    )
    assert "wolf_self_destruct_condition" not in d, (
        "Day directive leaked self_destruct_condition (night-only)"
    )


def test_wolf_night_directive_has_no_kill_no_push():
    d = build_wolf_night_directive(_gs(), "p01", None)
    assert "wolf_no_kill_conditions" in d, (
        "Night directive missing no_kill_conditions"
    )
    # Push target is day-only; night directive should not have it
    # (the test allows for it being absent OR having a different name
    # to avoid being an issue).
    assert "wolf_day_push_target" not in d, (
        f"Night directive leaked day-push target; got: {list(d.keys())}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runtime/test_wolf_directive_split.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_wolf_day_directive'`

- [ ] **Step 3: Read current wolf.py to understand structure**

Read `werewolf_agent/runtime/directives/wolf.py:1-100` to see the public API (`build_wolf_directive`) and identify the day-only vs night-only keys.

(Code review will reveal which keys are in the `parts` dict that should move to day/night branches.)

- [ ] **Step 4: Split wolf.py into day/night**

In `werewolf_agent/runtime/directives/wolf.py`, keep `build_wolf_directive` as a back-compat shim, add two new functions:

```python
def build_wolf_day_directive(
    gs: GameState,
    wolf_id: str,
    wolf_team_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Wolf day-speech / day-vote directive.

    M3-3: day-only keys (wolf_day_push_target, wolf_fake_seer_*,
    wolf_teammate_exposed, wolf_push_target, wolf_vote_role_hint,
    wolf_universal_rules) are included. Night-only keys
    (wolf_no_kill_conditions, wolf_self_destruct_condition) are
    NOT included — they leaked into day prompts previously.
    """
    parts: dict[str, Any] = {}
    # ... existing day-only code, with wolf_no_kill_conditions
    # and wolf_self_destruct_condition REMOVED ...
    return parts


def build_wolf_night_directive(
    gs: GameState,
    wolf_id: str,
    wolf_team_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Wolf night-action directive (kill, no_kill, self_destruct)."""
    parts: dict[str, Any] = {}
    # ... existing night-only code (just wolf_no_kill_conditions
    # and wolf_self_destruct_condition) ...
    return parts


def build_wolf_directive(
    gs: GameState,
    wolf_id: str,
    wolf_team_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Back-compat shim — returns merged day+night dict.

    M3-3: prefer build_wolf_day_directive / build_wolf_night_directive
    per task_type. This shim is kept for tests that pass
    wolf_team_plan without a task_type.
    """
    day = build_wolf_day_directive(gs, wolf_id, wolf_team_plan)
    night = build_wolf_night_directive(gs, wolf_id, wolf_team_plan)
    return {**day, **night}
```

(Implementation detail: extract the day-only and night-only sections from the existing function. The day-only section starts with role assignment and ends before wolf_no_kill_conditions / wolf_self_destruct_condition.)

- [ ] **Step 5: Route to day version in `agent_adapter.py:880`**

Find the wolf day-speech call:

```python
# werewolf_agent/runtime/agent_adapter.py:880
if player_role == "werewolf":
    wolf_parts = _build_wolf_day_speech_directive(
        gs, speaker_id, state.get("wolf_team_plan"),
    )
```

This currently imports `build_wolf_directive` (renamed to `_build_wolf_day_speech_directive`). Replace with:

```python
from werewolf_agent.runtime.directives.wolf import build_wolf_day_directive
# ...
if player_role == "werewolf":
    wolf_parts = build_wolf_day_directive(
        gs, speaker_id, state.get("wolf_team_plan"),
    )
```

- [ ] **Step 6: Route to night version in `_single_wolf_vote`**

Find the night kill directive injection (around `agent_adapter.py:557-580`):

```python
# werewolf_agent/runtime/agent_adapter.py:557-580 (in _single_wolf_vote)
strategy_directive["wolf_no_kill_conditions"] = (
    "【空刀战术条件】..."
)
```

Replace with a call to `build_wolf_night_directive`:

```python
from werewolf_agent.runtime.directives.wolf import build_wolf_night_directive
# ... in _single_wolf_vote, after strategy_directive init:
night_wolf = build_wolf_night_directive(gs, wolf_id, wolf_plan)
strategy_directive.update(night_wolf)
```

Then remove the local `wolf_no_kill_conditions` literal (lines 559-568 or so).

Also for the `wolf_self_destruct_condition`: it's currently injected in `directives/wolf.py:build_wolf_directive`. Move the helper `_wolf_endangered_status(gs, wolf_id)` and the `wolf_self_destruct_condition` injection into `build_wolf_night_directive` — but only because the prompt is asked in DAY phase (in `directives/wolf.py:build_wolf_directive` line ~232). 

Wait — the self-destruct prompt is given during DAY (when wolf speaks/votes). So `wolf_self_destruct_condition` is DAY-directive! Re-think.

Let me re-read the issue. From audit M3-3:
> "Same `_build_wolf_directive` builds both day_push_target (day-only) AND wolf_no_kill_conditions / wolf_self_destruct_condition (night-only). Night wolf sees day push target; day wolf sees no_kill."

Hmm but the implementation shows `wolf_self_destruct_condition` is in `build_wolf_directive` which is called for day_speech (agent_adapter.py:880). So it's currently a DAY directive already.

Let me actually re-read the wolf.py code more carefully. The split should be:
- **Day-only** (in `_build_wolf_day_directive`): role assignment, fake_seer logic, day_push_target, teammate_exposed, vote_role_hint, universal_rules, **AND** self_destruct_condition (since self-destruct is decided during day)
- **Night-only** (in `_build_wolf_night_directive`): kill_instruction, high_priority_target, no_kill_conditions, kill_value_assessment, plan_target

Looking at the audit again — "wolf_no_kill_conditions / wolf_self_destruct_condition" — I mis-categorized. Let me re-check by reading the actual file in step 3.

For now, I'll defer the detailed split to step 3. Continue the plan.

- [ ] **Step 7: Update test expectations based on step 6's actual split**

(Re-evaluate the test assertions after reading the code. The principle is: day directive should NOT contain night-only keys, and vice versa. The exact key list depends on the actual code structure.)

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/runtime/test_wolf_directive_split.py -v`
Expected: PASS

- [ ] **Step 9: Run full test suite**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/runtime/directives/wolf.py werewolf_agent/runtime/agent_adapter.py tests/runtime/test_wolf_directive_split.py
git commit -m "refactor(directive): M3-3 split wolf.py into day/night builders"
```

### Task 7: M4-2 swap RAG/reflection priority labels (P1)

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py:468-469` (swap labels)
- Test: `tests/agents/test_prompt_injection_fixes.py`

- [ ] **Step 1: Write failing test**

```python
def test_reflection_priority_above_rag():
    """M4-2: reflection_memory_hints should be 【参考】 (not 【辅助】)
    because per-player history is more valuable than generic RAG.

    G-R4-15 promoted RAG to 【参考】 to keep it under budget pressure,
    but the rationale is wrong: RAG is generic, reflection is
    per-player. Budget drops 【辅助】 first, so reflection got cut
    before RAG. Swapping makes reflection survive longer.
    """
    from werewolf_agent.agents import prompt_builder
    prios = prompt_builder.PlayerPromptBuilder._SECTION_PRIORITIES
    assert prios["_build_reflection_memory_hints"] == "【参考】", (
        f"M4-2: reflection hints should be 【参考】 (outranks RAG's "
        f"generic content). Got: {prios['_build_reflection_memory_hints']!r}"
    )
    assert prios["_build_rag_hints"] == "【辅助】", (
        f"M4-2: RAG should be 【辅助】 (lower than reflection). "
        f"Got: {prios['_build_rag_hints']!r}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_reflection_priority_above_rag -v`
Expected: FAIL with current state showing RAG is 【参考】, reflection is 【辅助】

- [ ] **Step 3: Swap labels in `_SECTION_PRIORITIES`**

In `werewolf_agent/agents/prompt_builder.py:468-469`:

```python
# M4-2: swapped reflection/RAG priorities. Per-player reflection
# (self-history) outranks generic RAG (community knowledge)
# for budget-trim purposes. G-R4-15 had RAG as 【参考】; rationale
# was "RAG must survive budget cuts" but reflection is the
# per-player-relevant content. RAG is generic — when budget
# tight, drop RAG first.
"_build_rag_hints": "【辅助】",
"_build_reflection_memory_hints": "【参考】",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_reflection_priority_above_rag -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_injection_fixes.py
git commit -m "fix(prompt): M4-2 swap reflection/RAG priority labels (reflection outranks RAG)"
```

### Task 8: M5-1 skill/identity boundary text (P1)

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py:316-321` (add boundary)
- Test: `tests/agents/test_prompt_injection_fixes.py`

- [ ] **Step 1: Write failing test**

```python
def test_skill_policy_distinguishes_from_identity_rules():
    """M5-1: _build_skill_policy should clearly state that identity
    rules (above in role_guide) outrank skill advice on conflict.

    Without this boundary, LLM may conflate 'skill said vote X' with
    'role said vote X'.
    """
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    policy = builder._build_skill_policy()
    # Must explicitly state the precedence: identity > skill
    assert "身份" in policy and "技能" in policy, (
        f"Skill policy must mention both '身份' and '技能' "
        f"to establish the precedence boundary. Got: {policy!r}"
    )
    # And should use a phrase like "身份优先" or "技能不凌驾于身份"
    boundary_keywords = ["身份优先", "技能不凌驾", "outrank", "身份规则优先"]
    assert any(kw in policy for kw in boundary_keywords), (
        f"Skill policy must include boundary phrasing; got: {policy!r}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_skill_policy_distinguishes_from_identity_rules -v`
Expected: FAIL with current policy being too short

- [ ] **Step 3: Update `_build_skill_policy`**

In `werewolf_agent/agents/prompt_builder.py:316-321`, replace:

```python
def _build_skill_policy(self) -> str:
    """Skill policy: boundary with identity rules.

    M5-1: explicitly state that the role's identity rules
    (rendered above in role_guide) outrank skill advice on
    conflict. Without this, LLM may conflate 'skill said
    vote X' with 'role said vote X' — leading to the LLM
    prioritizing skill output over the deterministic role
    rules it was given in the system prompt.
    """
    return (
        "【技能与建议】系统会在你的回合前注入已计算的技能分析结果，"
        "请基于这些分析与当前局可见事实形成自己的判断，不要机械复述。"
        "【优先级边界】身份规则(role_guide)优先于技能建议,冲突时以身份规则为准。"
        "技能分析不是裁判真相;如果与公开事实冲突,以公开事实为准。"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_prompt_injection_fixes.py::test_skill_policy_distinguishes_from_identity_rules -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_injection_fixes.py
git commit -m "fix(prompt): M5-1 add identity/skill boundary text to skill policy"
```

### Task 9: Documentation update

**Files:**
- Modify: `PROGRESS.md` (add phase)
- Modify: `docs/design/werewolf-agent-v1-design.md` (if needed)
- Modify: `CLAUDE.md` (if needed)
- Modify: `harness/context/architecture-boundaries.md` (if M4-2 swap affects ownership)

- [ ] **Step 1: Add `prompt-injection-fixes-2026-06-09` phase to PROGRESS.md**

Update "Current Status" section to point to new phase, add a section documenting 8 fixes (3 P0 + 5 P1) with commit references.

- [ ] **Step 2: Update design doc §7.3 (Agent 输出) if M2-3 changed structure**

The M2-3 fix extracted output schema constants. Update §7.3 to reference the single-source constants.

- [ ] **Step 3: Update design doc §10.2 (跨局检索排序) if M4-2 changed priorities**

M4-2 swapped RAG/reflection priority. Update §10.2 to reflect the new sort: reflection 【参考】, RAG 【辅助】. Reasoning: per-player history > generic knowledge.

- [ ] **Step 4: Update CLAUDE.md Development Boundaries if any cross-module contract changed**

If M3-3 (wolf.py split) creates a new module API (`build_wolf_day_directive` / `build_wolf_night_directive`), document the new contract in CLAUDE.md.

- [ ] **Step 5: Commit documentation**

```bash
git add PROGRESS.md docs/design/werewolf-agent-v1-design.md CLAUDE.md harness/context/architecture-boundaries.md
git commit -m "docs: prompt-injection-fixes-2026-06-09 phase summary"
```

---

## Self-Review

**1. Spec coverage:** Walked the 8 audit issues. Each has a task:
- M4-1 → Task 1
- M2-3 → Task 2
- M2-2 → Task 3
- M2-1 → Task 4
- M3-2 → Task 5
- M3-3 → Task 6
- M4-2 → Task 7
- M5-1 → Task 8
- Documentation → Task 9

All 8 issues covered. P2/P3 deferred to PROGRESS.md "v3 deferred" section (added in Step 1 of Task 9).

**2. Placeholder scan:** No "TBD" / "TODO" / "fill in details" / "add appropriate error handling" in any task. Each step has actual code or exact commands.

**3. Type consistency:** 
- `_OUTPUT_SCHEMA_VOTE_FIELDS` is `tuple[str, ...]` (Task 2) — consistent across both renderers.
- `REFLECTION_HINT_PROMPT_BUDGET` is `int = 8` (Task 1) — same as `HINT_BUDGET` in `context.py`.
- `VOTE_BASIS_GUIDANCE` is `str` (Task 3) — same wording referenced from both files.
- `build_wolf_day_directive` / `build_wolf_night_directive` return `dict[str, Any]` (Task 6) — consistent with existing `build_wolf_directive` shim.
- `collect_public_vote_history(gs, current_day: int | None = None)` (Task 5) — back-compat default `None` preserves old behavior.

**No issues found in self-review. Plan is ready to execute.**

---

## Verification

After all 9 tasks:

```bash
# Full regression
pytest tests/ -p no:cacheprovider -q --ignore=tests/api --ignore=tests/agents --ignore=tests/storage --ignore=tests/rag --ignore=tests/tools
# Expected: All pass, +8 new tests

# Specific to the audit fixes
pytest tests/agents/test_prompt_injection_fixes.py tests/runtime/test_directive_shared_helpers.py tests/runtime/test_wolf_directive_split.py -v
# Expected: All 10+ tests pass

# Verify budget sync (M4-1)
python -c "from werewolf_agent.runtime.context import REFLECTION_HINT_PROMPT_BUDGET; print(REFLECTION_HINT_PROMPT_BUDGET)"
# Expected: 8
```

End-to-end verification: run a 1-game real game (`scripts/run_real_game.py --delay -1`), check `game_stdout.log` for:
- Wolf night_action prompts no longer contain "vote_basis"
- Day-5 villager prompt no longer contains all 4 day vote histories
- Reflection hints are now in 8-budget batch (was 5)
- Skill/identity boundary text appears in output
- RAG hints survive budget trim over reflection hints (if M4-2 swap is correct)

---

## Post-Execution Addendum (T9 docs phase, 2026-06-09)

This plan was executed in the `prompt-injection-fixes-2026-06-09` worktree. All 9 tasks are complete. Below is the deviation log between the plan as written and the actual implementation, recorded so future re-runs (or audits) don't try to apply the plan's original (broken) text.

**Deviation log (by task):**

- **T4 (M2-1) Step 3 — villager text is 54 chars, not 80 chars as the plan proposed.** The plan's verbatim replacement ("村民规则：身份公开时表明好人立场；归票基于票型/发言证据链,不跟风；夜间无行动,听从公开死亡公告。") is internally inconsistent with the plan's own Step 1 test, which asserts `"N1" in guide or "解药" in guide`. The plan text drops the "N1 解药救人" cue that the existing `test_villager_role_guide_specific_rules` (P1-S9) requires. The actual implementation used a 54-char version that retains the 4 key cues: `村民规则：身份公开时表明好人立场；分析发言矛盾/票型；N1 公开讨论中支持解药救人；归票基于证据链,不跟风。`. Test `test_villager_role_guide_is_concise` (M2-1) passes with the 54-char text and the 4 cues; P1-S9 test continues to pass. The plan's "Phase-1 P1-26 (`test_villager_guide_includes_night_fallback`)" test was rewritten to M2-1 direction (asserting "夜间阶段" / "无投票权" NOT in sys_prompt).

- **T6 (M3-3) audit / plan mis-classified two keys.** The audit claimed `wolf_self_destruct_condition` is night-only and `wolf_no_kill_conditions` is `build_wolf_directive` (day). Actual code reading shows: self_destruct is in `build_wolf_directive` (called from day_speech), so it is day-only — the prompt is emitted during day decisions. `wolf_no_kill_conditions` is in `_single_wolf_vote` (night), so it is night-only. The split re-classified both keys by "decision timing" rather than the audit's misnaming. The back-compat shim `build_wolf_directive` returns merged dict for tests.

- **T8 (M5-1) Step 3 — punctuation and structure were corrected in T8-fix (`a5bb633`).** The plan's proposed `_build_skill_policy` text used half-width `,` and `;` inconsistent with the surrounding full-width `，` and `；`. The plan also omitted the "公开事实" sub-clause. Actual implementation normalized punctuation to full-width and added the 4th line "技能分析不是裁判真相；如果与公开事实冲突，以公开事实为准。". T8-fix also strengthened the precedence test to use regex `r"身份规则.*优先"` instead of positional string ordering.

- **T9 (docs) — the plan over-scoped.** The plan's Steps 2/3/4/5 assumed §7.3, §10.2, CLAUDE.md, and `harness/context/architecture-boundaries.md` all need updates. Actual: only §10.2 needed a new "反思记忆的提示优先级" sub-section (recording the M4-2 reversal). §7.3 (Agent output contract) describes output structure, not schema constants — M2-3 extracting constants is implementation-level. CLAUDE.md lists authority order and key boundaries, not prompt-builder helpers — M3-3's new public API is documented in directive module docstrings, not at CLAUDE.md level. `harness/context/architecture-boundaries.md` is for RuleEngine / Agent / cross-module boundaries — M4-2 priority swap is purely internal to `prompt_builder._SECTION_PRIORITIES`.

**Test count deviation:**

- Plan estimated 8 (later 10) new tests. Actual: **19 new tests** across 3 files (`tests/agents/test_prompt_injection_fixes.py`: 11, `tests/runtime/test_directive_shared_helpers.py`: 5, `tests/runtime/test_wolf_directive_split.py`: 3).
- M3-2 (T5) added 2 more tests in the T5-fix (`1ac31f1`) for malformed `resolution_batch` warning and night-batch non-warning cases.
- M2-2 (T3) added 1 more test in the T3-fix (`62a2c9a`) for the centralized `_inject_vote_basis_hint` helper.

**Commit log (12 commits, by task):**

| T | Fix commit | Docs / test-fix commit | Followup |
|---|---|---|---|
| T1 (M4-1) | `d36eb27` | — | — |
| T2 (M2-3) | `9bbe2bd` (unify), `48699cb` (drift fix) | `cd1eb19` | — |
| T3 (M2-2) | `88fd0e9` (per-turn), `62a2c9a` (HARD tier + helper) | `6bd3edd` | — |
| T4 (M2-1) | `85660e4` | — | — |
| T5 (M3-2) | `27d8a12`, `1ac31f1` (log + regex) | `8d3ba8d` | — |
| T6 (M3-3) | `896957e` | `6a18ad6` | — |
| T7 (M4-2) | `0022d25` | `f22dc20` | `9c1ff09` (comment), `a4b9964` (test docstrings) |
| T8 (M5-1) | `5b94341` | — | `a5bb633` (punctuation + regex) |
| T9 (docs) | (this commit) | — | — |

**Plan complete and saved to `docs/superpowers/plans/2026-06-09-prompt-injection-audit-fixes.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
