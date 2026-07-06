# SRP File Splitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按单一职责、高内聚低耦合、显式优于隐式的原则，把剩余大文件拆成聚焦模块，并保持现有公开 API 和运行行为不变。

**Architecture:** 每个阶段先保留原文件作为兼容外壳，把一类职责迁移到新模块，再从原文件显式导入并重新导出需要兼容的符号。运行时核心文件优先拆，因为它们阻塞后续维护；测试、schema 和低风险工具文件放在后面。

**Tech Stack:** Python 3, pytest, ruff, compileall, CodeGraph, PowerShell, repo-local `.tmp` pytest basetemp。

---

## 拆分原则

- 原文件先作为 facade 保留，公共导入路径不变，避免一次性破坏调用方。
- 每次只迁移一个职责组，测试通过后提交。
- 新 Python 文件必须包含 UTF-8 编码声明和中文模块 docstring。
- 注释默认中文，只有协议字段、API 名称、第三方名词和必要 literal 保持英文。
- 优先拆生产代码，再同步拆对应测试文件。
- 保留兼容导出，尤其不要误删 `AGENT_TIMEOUTS`、`_inject_skill_output` 等兼容入口。
- 每阶段结束都运行 targeted tests、`compileall`、针对 touched files 的 `ruff F401,F841`、`git diff --check`。

## 总体执行顺序

1. 稳定基线和工作区边界。
2. 继续拆 `werewolf_agent/runtime/agent_adapter.py`。
3. 拆 `werewolf_agent/runtime/context.py`。
4. 拆 `werewolf_agent/runtime/nodes/night.py`。
5. 拆 `werewolf_agent/agents/prompt_builder.py`。
6. 拆 `werewolf_agent/skills/werewolf_skills.py`。
7. 拆 `werewolf_agent/agents/output_parser.py`。
8. 拆 `werewolf_agent/agents/player.py`。
9. 拆 `werewolf_agent/memory/reflection.py`。
10. 拆 `werewolf_agent/runtime/nodes/day.py`。
11. 拆 `werewolf_agent/api/routes/games.py`。
12. 拆第二批 600 行以上但风险较低的模块。
13. 拆对应超大测试文件。
14. 最终全量验证、提交、CodeGraph 同步。

---

### Task 0: Baseline And Safety Rails

**Files:**
- Inspect: `werewolf_agent/runtime/agent_adapter.py`
- Inspect: `werewolf_agent/runtime/context.py`
- Inspect: `werewolf_agent/runtime/nodes/night.py`
- Inspect: `werewolf_agent/agents/prompt_builder.py`
- Inspect: `werewolf_agent/skills/werewolf_skills.py`
- Inspect: `werewolf_agent/agents/output_parser.py`
- Inspect: `werewolf_agent/agents/player.py`
- Inspect: `werewolf_agent/memory/reflection.py`
- Inspect: `werewolf_agent/runtime/nodes/day.py`
- Inspect: `werewolf_agent/api/routes/games.py`

- [ ] **Step 1: Confirm CodeGraph is current**

Run:

```powershell
codegraph status .
```

Expected: `[OK] Index is up to date`.

- [ ] **Step 2: Capture dirty worktree before edits**

Run:

```powershell
git status --short
```

Expected: unrelated user changes may exist; do not revert them.

- [ ] **Step 3: Run characterization suite for current refactor surface**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime tests/agents tests/skills tests/memory tests/api -q
```

Expected: current known baseline. If too broad or slow, record failures and use focused suites per task.

- [ ] **Step 4: Commit only if there are plan-only changes to save**

Run:

```powershell
git add docs/superpowers/plans/2026-07-05-srp-file-splitting.md
git commit -m "docs: plan srp file splitting"
```

Expected: optional docs commit; skip if the user wants implementation commits only.

---

### Task 1: Split Remaining Agent Adapter Runtime Responsibilities

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Create: `werewolf_agent/runtime/agent_registry.py`
- Create: `werewolf_agent/runtime/agent_action_audit.py`
- Create: `werewolf_agent/runtime/wolf_kill_support.py`
- Create: `werewolf_agent/runtime/agent_reflection_support.py`
- Test: `tests/runtime/test_agent_registry.py`
- Test: `tests/runtime/test_agent_action_audit.py`
- Test: `tests/runtime/test_wolf_kill_support.py`
- Test: `tests/runtime/test_agent_reflection_support.py`
- Existing tests: `tests/runtime/test_witch_flow.py`, `tests/runtime/test_night_flow.py`, `tests/runtime/test_day_discussion.py`, `tests/runtime/test_vote_flow.py`, `tests/runtime/test_pk_flow.py`, `tests/runtime/test_sheriff_flow.py`, `tests/runtime/test_hunter_flow.py`

- [ ] **Step 1: Write characterization tests for registry behavior**

Test `AgentRegistry` and `SimpleAgentRegistry` import from the old module and new module. Verify the same agent lookup behavior.

- [ ] **Step 2: Move registry types**

Move `AgentRegistry` and `SimpleAgentRegistry` into `agent_registry.py`. In `agent_adapter.py`, import and re-export them.

- [ ] **Step 3: Move audit helper functions**

Move `_audit_context_kwargs`, `_seer_credibility_audit_payload`, `_is_sheriff_silenced`, and vote-basis hint helpers into `agent_action_audit.py`.

- [ ] **Step 4: Move wolf kill support**

Move `_build_wolf_kill_directive` and `_single_wolf_vote` into `wolf_kill_support.py`. Keep public compatibility imports in `agent_adapter.py`.

- [ ] **Step 5: Move reflection support**

Move `_strip_in_game_directives` and `_agent_reflection` into `agent_reflection_support.py`.

- [ ] **Step 6: Run focused runtime adapter tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_reflection_support.py tests/runtime/test_witch_flow.py tests/runtime/test_night_flow.py tests/runtime/test_day_discussion.py tests/runtime/test_vote_flow.py tests/runtime/test_pk_flow.py tests/runtime/test_sheriff_flow.py tests/runtime/test_hunter_flow.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
python -m compileall -q werewolf_agent/runtime
python -m ruff check werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/agent_registry.py werewolf_agent/runtime/agent_action_audit.py werewolf_agent/runtime/wolf_kill_support.py werewolf_agent/runtime/agent_reflection_support.py tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_reflection_support.py --select F401,F841
git diff --check
git add werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/agent_registry.py werewolf_agent/runtime/agent_action_audit.py werewolf_agent/runtime/wolf_kill_support.py werewolf_agent/runtime/agent_reflection_support.py tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_reflection_support.py
git commit -m "refactor: split agent adapter support modules"
```

---

### Task 2: Split Agent Context Construction

**Files:**
- Modify: `werewolf_agent/runtime/context.py`
- Create: `werewolf_agent/runtime/context_persona.py`
- Create: `werewolf_agent/runtime/context_rag.py`
- Create: `werewolf_agent/runtime/context_memory_hints.py`
- Create: `werewolf_agent/runtime/context_skill_advice.py`
- Create: `werewolf_agent/runtime/context_strategy_directives.py`
- Create: `werewolf_agent/runtime/context_action_trace.py`
- Test: `tests/runtime/test_context_persona.py`
- Test: `tests/runtime/test_context_rag.py`
- Test: `tests/runtime/test_context_memory_hints.py`
- Test: `tests/runtime/test_context_skill_advice.py`
- Test: `tests/runtime/test_context_strategy_directives.py`
- Existing tests: `tests/runtime/test_context.py`, `tests/runtime/test_strategy_directives.py`

- [ ] **Step 1: Write characterization tests for compatibility imports**

Verify `build_agent_context` and `_inject_skill_output` remain importable from `werewolf_agent.runtime.context`.

- [ ] **Step 2: Move persona helpers**

Move `_load_persona_profile`, `_get_persona_speech_style`, and `_get_persona_task_style` into `context_persona.py`.

- [ ] **Step 3: Move RAG helpers**

Move `_rag_phase_for_task`, `_normalize_legal_actions_to_tags`, `_inject_seed_rag_hints`, `_extract_suspects`, `_extract_trusts`, `_extract_role_claim`, `_extract_vote_intent`, and `_first_sentence` into `context_rag.py`.

- [ ] **Step 4: Move memory and cognition hints**

Move `_profile_memory_hint`, `_reflection_memory_hints`, `_evidence_id_ref`, and `_cognition_matrix_hint` into `context_memory_hints.py`.

- [ ] **Step 5: Move skill advice conversion**

Move `_skill_output_to_advice_frame`, `_skill_advice_frame_to_prompt_dict`, and `_inject_skill_output` into `context_skill_advice.py`; re-export `_inject_skill_output` from `context.py`.

- [ ] **Step 6: Move strategy directive merging**

Move `_merge_strategy_directive`, `_directive_size`, and `_cap_strategy_directive` into `context_strategy_directives.py`.

- [ ] **Step 7: Move action trace payload construction**

Move `_action_trace_payload` and direct payload helpers into `context_action_trace.py`.

- [ ] **Step 8: Keep `context.py` as orchestration facade**

Leave `build_agent_context` in `context.py`, but make dependencies explicit through imports.

- [ ] **Step 9: Run focused context tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_context.py tests/runtime/test_context_persona.py tests/runtime/test_context_rag.py tests/runtime/test_context_memory_hints.py tests/runtime/test_context_skill_advice.py tests/runtime/test_context_strategy_directives.py tests/runtime/test_strategy_directives.py -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Verify and commit**

Run compileall, touched-file ruff, `git diff --check`, then commit:

```powershell
git commit -m "refactor: split runtime context builders"
```

---

### Task 3: Split Night Runtime Nodes

**Files:**
- Modify: `werewolf_agent/runtime/nodes/night.py`
- Create: `werewolf_agent/runtime/nodes/night_entry.py`
- Create: `werewolf_agent/runtime/nodes/night_specialists.py`
- Create: `werewolf_agent/runtime/nodes/night_resolution.py`
- Create: `werewolf_agent/runtime/nodes/wolf_night_nodes.py`
- Test: `tests/runtime/test_night_entry.py`
- Test: `tests/runtime/test_night_specialists.py`
- Test: `tests/runtime/test_night_resolution.py`
- Test: `tests/runtime/test_wolf_night_nodes.py`
- Existing tests: `tests/runtime/test_night_flow.py`, `tests/runtime/test_witch_flow.py`, `tests/runtime/test_wolf_flow.py`

- [ ] **Step 1: Characterize public imports from `night.py`**

Verify existing imports for `enter_night`, `night_witch`, `night_seer`, `resolve_night`, `wolf_discussion`, `wolf_team_plan_node`, and `wolf_consensus` still work.

- [ ] **Step 2: Move night entry and simple status nodes**

Move `enter_night`, `night_hunter_idiot_status`, and shared entry helpers to `night_entry.py`.

- [ ] **Step 3: Move specialist night actions**

Move `night_witch`, `night_seer`, and `first_night_hybrid_master` to `night_specialists.py`.

- [ ] **Step 4: Move night resolution**

Move `resolve_night` and death/victory-related night resolution helpers to `night_resolution.py`.

- [ ] **Step 5: Move wolf night planning and consensus**

Move `_legacy_wolf_consensus`, `wolf_discussion`, `_build_fallback_wolf_team_plan`, `wolf_team_plan_node`, and `wolf_consensus` to `wolf_night_nodes.py`.

- [ ] **Step 6: Convert `night.py` into explicit facade**

Import and re-export all public node functions. Avoid wildcard imports.

- [ ] **Step 7: Run focused night tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_night_entry.py tests/runtime/test_night_specialists.py tests/runtime/test_night_resolution.py tests/runtime/test_wolf_night_nodes.py tests/runtime/test_night_flow.py tests/runtime/test_witch_flow.py tests/runtime/test_wolf_flow.py tests/runtime/test_hybrid_master_directives.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split night runtime nodes"
```

---

### Task 4: Split Prompt Builder

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Create: `werewolf_agent/agents/prompt_sections.py`
- Create: `werewolf_agent/agents/prompt_persona.py`
- Create: `werewolf_agent/agents/prompt_memory.py`
- Create: `werewolf_agent/agents/prompt_strategy.py`
- Create: `werewolf_agent/agents/prompt_salience.py`
- Test: `tests/agents/test_prompt_sections.py`
- Test: `tests/agents/test_prompt_persona.py`
- Test: `tests/agents/test_prompt_memory.py`
- Test: `tests/agents/test_prompt_strategy.py`
- Existing tests: `tests/agents/test_prompt_builder.py`, `tests/agents/test_prompt_formatting.py`

- [ ] **Step 1: Characterize final prompt output**

Add focused snapshots or exact string assertions for system prompt, user prompt, persona hints, memory hints, and strategy directives.

- [ ] **Step 2: Move section model and section rendering**

Move `_SectionSpec`, section ordering, section formatting, and common section assembly helpers into `prompt_sections.py`.

- [ ] **Step 3: Move persona prompt rendering**

Move persona style and persona text rendering into `prompt_persona.py`.

- [ ] **Step 4: Move memory prompt rendering**

Move profile/reflection/cognition memory prompt rendering into `prompt_memory.py`.

- [ ] **Step 5: Move strategy and directive prompt rendering**

Move strategy directive rendering into `prompt_strategy.py`.

- [ ] **Step 6: Move salience helpers**

Move `_slim_salience_item` and salience-specific formatting into `prompt_salience.py`.

- [ ] **Step 7: Keep `PlayerPromptBuilder` as a small facade**

`PlayerPromptBuilder` should coordinate inputs and call focused renderers. It should not own low-level formatting logic.

- [ ] **Step 8: Run focused prompt tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_prompt_builder.py tests/agents/test_prompt_formatting.py tests/agents/test_prompt_sections.py tests/agents/test_prompt_persona.py tests/agents/test_prompt_memory.py tests/agents/test_prompt_strategy.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split player prompt builder"
```

---

### Task 5: Split Werewolf Skill Handlers

**Files:**
- Modify: `werewolf_agent/skills/werewolf_skills.py`
- Create: `werewolf_agent/skills/advice_frames.py`
- Create: `werewolf_agent/skills/skill_context.py`
- Create: `werewolf_agent/skills/good_skill_handlers.py`
- Create: `werewolf_agent/skills/wolf_skill_handlers.py`
- Create: `werewolf_agent/skills/review_skill_handlers.py`
- Create: `werewolf_agent/skills/skill_handler_registry.py`
- Test: `tests/skills/test_advice_frames.py`
- Test: `tests/skills/test_skill_context.py`
- Test: `tests/skills/test_good_skill_handlers.py`
- Test: `tests/skills/test_wolf_skill_handlers.py`
- Existing tests: `tests/skills/test_werewolf_skills.py`, `tests/skills/test_skill_manifest_loader.py`, `tests/skills/test_registry.py`

- [ ] **Step 1: Characterize `apply_skill` behavior**

Add tests for unchanged output for at least one good skill, one wolf skill, and one review correction skill.

- [ ] **Step 2: Move advice frame helpers**

Move `_cap_prompt_injectable`, `_advice_frame`, `_push_vote_advice_frame`, `_counter_claim_advice_frame`, `_hide_identity_advice_frame`, `_generic_skill_advice_frame`, and `_ensure_skill_advice_frame` into `advice_frames.py`.

- [ ] **Step 3: Move skill context extraction**

Move `_count_seer_claimants`, `_get_seer_claimants`, `_alive_wolves`, `_alive_non_wolves`, `_vote_targets_for_player`, `_seer_checks_on_target`, `_alerts_for_player`, `_belief_top_suspects`, and `_wolf_teammates_exposed` into `skill_context.py`.

- [ ] **Step 4: Move good-side handlers**

Move `bold_claim_handler`, `counter_claim_handler`, `push_vote_handler`, `swing_vote_handler`, `find_power_handler`, `protect_power_handler`, `resist_push_handler`, and `last_words_handler` into `good_skill_handlers.py` where appropriate.

- [ ] **Step 5: Move wolf-side handlers**

Move `deep_hook_handler`, `hide_identity_handler`, and `wolf_pit_handler` into `wolf_skill_handlers.py`.

- [ ] **Step 6: Move review correction handlers**

Move `review_correction_handler`, `_review_correction_wolf`, and `_review_correction_good` into `review_skill_handlers.py`.

- [ ] **Step 7: Move handler registry**

Move `register_handler`, `get_handler`, and default handler registration into `skill_handler_registry.py`. Keep `apply_skill` as the public facade in `werewolf_skills.py`.

- [ ] **Step 8: Run focused skill tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/skills/test_werewolf_skills.py tests/skills/test_advice_frames.py tests/skills/test_skill_context.py tests/skills/test_good_skill_handlers.py tests/skills/test_wolf_skill_handlers.py tests/skills/test_skill_manifest_loader.py tests/skills/test_registry.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split werewolf skill handlers"
```

---

### Task 6: Split Output Parser

**Files:**
- Modify: `werewolf_agent/agents/output_parser.py`
- Create: `werewolf_agent/agents/action_data_extraction.py`
- Create: `werewolf_agent/agents/action_normalization.py`
- Create: `werewolf_agent/agents/action_repair.py`
- Create: `werewolf_agent/agents/speech_intent_parser.py`
- Create: `werewolf_agent/agents/choice_pipeline.py`
- Test: `tests/agents/test_action_data_extraction.py`
- Test: `tests/agents/test_action_normalization.py`
- Test: `tests/agents/test_action_repair.py`
- Test: `tests/agents/test_speech_intent_parser.py`
- Existing tests: `tests/agents/test_output_parser.py`, `tests/agents/test_parse_dispatch.py`

- [ ] **Step 1: Characterize `parse_action` and public helper imports**

Verify imports from `output_parser.py` continue to work.

- [ ] **Step 2: Move raw extraction helpers**

Move `extract_parameter_tag_action`, `extract_partial_decision_data`, and `extract_decision_data` into `action_data_extraction.py`.

- [ ] **Step 3: Move normalization helpers**

Move `normalize_action_data`, `_normalize_typos`, `clean_enum_value`, `clean_reason`, and optional-private-field sanitizers into `action_normalization.py`.

- [ ] **Step 4: Move repair helpers**

Move vote, target, and speech repair helpers into `action_repair.py`.

- [ ] **Step 5: Move speech intent helpers**

Move `infer_speech_intent`, seer stance inference, vote basis inference, and speech pressure helpers into `speech_intent_parser.py`.

- [ ] **Step 6: Move choice pipeline helpers**

Move `parse_choice_action`, `parse_speech_intent_action`, `uses_choice_pipeline`, and `uses_speech_intent_pipeline` into `choice_pipeline.py`.

- [ ] **Step 7: Keep `output_parser.py` as parser facade**

`parse_action` remains the main entry point and delegates to focused modules.

- [ ] **Step 8: Run focused parser tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_output_parser.py tests/agents/test_parse_dispatch.py tests/agents/test_action_data_extraction.py tests/agents/test_action_normalization.py tests/agents/test_action_repair.py tests/agents/test_speech_intent_parser.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split agent output parser"
```

---

### Task 7: Split Player Agent Pipeline

**Files:**
- Modify: `werewolf_agent/agents/player.py`
- Create: `werewolf_agent/agents/player_failures.py`
- Create: `werewolf_agent/agents/player_latency.py`
- Create: `werewolf_agent/agents/player_generation.py`
- Create: `werewolf_agent/agents/player_retry.py`
- Test: `tests/agents/test_player_failures.py`
- Test: `tests/agents/test_player_latency.py`
- Test: `tests/agents/test_player_generation.py`
- Existing tests: `tests/agents/test_player_agent.py`, `tests/integration/test_live_runtime.py`, `tests/integration/test_live_game_flow.py`

- [ ] **Step 1: Characterize `PlayerAgent.act` behavior**

Add tests for success path, provider failure fallback, invalid action retry, and latency recording if not already covered.

- [ ] **Step 2: Move failure classification**

Move `_fallback_reason`, `_categorize_failure_category`, and related failure metadata into `player_failures.py`.

- [ ] **Step 3: Move latency helpers**

Move `_latency_from_result` and timing extraction helpers into `player_latency.py`.

- [ ] **Step 4: Move provider generation wrapper**

Move model call orchestration into `player_generation.py`, keeping dependencies explicit.

- [ ] **Step 5: Move retry/fallback pipeline helpers**

Move retry decision and fallback construction logic into `player_retry.py`.

- [ ] **Step 6: Keep `PlayerAgent` as coordinator**

`player.py` should own public class lifecycle and high-level `act`, not low-level failure parsing or prompt/generation plumbing.

- [ ] **Step 7: Run focused player tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_player_agent.py tests/agents/test_player_failures.py tests/agents/test_player_latency.py tests/agents/test_player_generation.py tests/integration/test_live_runtime.py tests/integration/test_live_game_flow.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split player agent pipeline"
```

---

### Task 8: Split Reflection Memory Module

**Files:**
- Modify: `werewolf_agent/memory/reflection.py`
- Create: `werewolf_agent/memory/reflection_sanitization.py`
- Create: `werewolf_agent/memory/reflection_quality.py`
- Create: `werewolf_agent/memory/reflection_synthesis.py`
- Create: `werewolf_agent/memory/reflection_repository.py`
- Test: `tests/memory/test_reflection_sanitization.py`
- Test: `tests/memory/test_reflection_quality.py`
- Test: `tests/memory/test_reflection_synthesis.py`
- Existing tests: `tests/memory/test_reflection_v2.py`, `tests/memory/test_memory.py`

- [ ] **Step 1: Characterize public reflection classes**

Verify `ReflectionQualityGate`, `ReflectionSynthesizer`, and `ReflectionMemory` are still importable from `reflection.py`.

- [ ] **Step 2: Move text sanitization**

Move `_iter_section_items`, `_scrub_ids`, and `_cap_source_text` into `reflection_sanitization.py`.

- [ ] **Step 3: Move quality gate**

Move `ReflectionQualityGate` and quality scoring helpers into `reflection_quality.py`.

- [ ] **Step 4: Move synthesizer**

Move `ReflectionSynthesizer` into `reflection_synthesis.py`.

- [ ] **Step 5: Move storage wrapper**

Move `ReflectionMemory` into `reflection_repository.py`.

- [ ] **Step 6: Keep `reflection.py` as compatibility facade**

Explicitly import and re-export public classes and helpers still used by callers.

- [ ] **Step 7: Run focused memory tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/memory/test_reflection_v2.py tests/memory/test_memory.py tests/memory/test_reflection_sanitization.py tests/memory/test_reflection_quality.py tests/memory/test_reflection_synthesis.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split reflection memory modules"
```

---

### Task 9: Split Day Runtime Nodes

**Files:**
- Modify: `werewolf_agent/runtime/nodes/day.py`
- Create: `werewolf_agent/runtime/nodes/day_deaths.py`
- Create: `werewolf_agent/runtime/nodes/day_discussion.py`
- Create: `werewolf_agent/runtime/nodes/day_vote.py`
- Create: `werewolf_agent/runtime/nodes/day_finish.py`
- Test: `tests/runtime/test_day_deaths.py`
- Test: `tests/runtime/test_day_vote_nodes.py`
- Existing tests: `tests/runtime/test_day_discussion.py`, `tests/runtime/test_vote_flow.py`, `tests/runtime/test_pk_flow.py`

- [ ] **Step 1: Characterize public imports from `day.py`**

Verify public node functions remain importable.

- [ ] **Step 2: Move death announcement and last words**

Move `_death_reason_label`, `announce_deaths`, `announce_deaths_with_badge_loss`, `night_death_last_words`, and `exile_last_words` into `day_deaths.py`.

- [ ] **Step 3: Move discussion node**

Move `free_discussion` into `day_discussion.py`.

- [ ] **Step 4: Move vote and exile nodes**

Move `day_vote`, `_broadcast_vote_details`, `resolve_vote`, and `resolve_exile` into `day_vote.py`.

- [ ] **Step 5: Move game finish**

Move `check_victory` and `finish_game` into `day_finish.py`.

- [ ] **Step 6: Keep `day.py` as explicit facade**

Re-export public node functions without wildcard imports.

- [ ] **Step 7: Run focused day tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_day_discussion.py tests/runtime/test_vote_flow.py tests/runtime/test_pk_flow.py tests/runtime/test_day_deaths.py tests/runtime/test_day_vote_nodes.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split day runtime nodes"
```

---

### Task 10: Split Games API Route Module

**Files:**
- Modify: `werewolf_agent/api/routes/games.py`
- Create: `werewolf_agent/api/routes/game_auth.py`
- Create: `werewolf_agent/api/routes/game_persistence.py`
- Create: `werewolf_agent/api/routes/game_cognition_views.py`
- Create: `werewolf_agent/api/routes/game_public_share.py`
- Test: `tests/api/test_game_auth_helpers.py`
- Test: `tests/api/test_game_persistence_helpers.py`
- Existing tests: `tests/api/test_api.py`, `tests/api/test_auth.py`, `tests/api/test_views.py`, `tests/api/test_pause_resume_lock.py`

- [ ] **Step 1: Characterize route registration**

Verify `create_game_router` exposes the same routes before and after refactor.

- [ ] **Step 2: Move auth helpers**

Move `_resolve_caller_role`, `_enforce_moderator_only`, and `_enforce_create_game_auth` into `game_auth.py`.

- [ ] **Step 3: Move persistence helpers**

Move `_get_game` and `_persist` into `game_persistence.py`.

- [ ] **Step 4: Move cognition view helpers**

Move `_build_cognition_data_for_viewer` and `_build_locked_config_snapshot` into `game_cognition_views.py`.

- [ ] **Step 5: Move public share helpers**

Move `_event_is_public_for_share` and `_pick_public_mvp_candidate` into `game_public_share.py`.

- [ ] **Step 6: Keep `games.py` as route declaration module**

`games.py` should define routes and dependency wiring only.

- [ ] **Step 7: Run focused API tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/api/test_api.py tests/api/test_auth.py tests/api/test_views.py tests/api/test_pause_resume_lock.py tests/api/test_game_auth_helpers.py tests/api/test_game_persistence_helpers.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split game route helpers"
```

---

### Task 11: Split Remaining Large Support Modules

**Files:**
- Modify: `werewolf_agent/evaluation/metrics.py`
- Modify: `werewolf_agent/model_gateway/router.py`
- Modify: `werewolf_agent/runtime/nodes/_shared.py`
- Modify: `werewolf_agent/runtime/game_runner.py`
- Modify: `werewolf_agent/engine/rule_engine.py`
- Modify: `werewolf_agent/runtime/nodes/sheriff.py`
- Modify: `werewolf_agent/rag/retriever.py`
- Modify: `werewolf_agent/rag/vector_store.py`
- Test: matching focused tests under `tests/evaluation`, `tests/runtime`, `tests/rules`, `tests/rag`

- [ ] **Step 1: Re-rank these files after Tasks 1-10**

Run:

```powershell
$rows = Get-ChildItem -Recurse -File -Filter *.py werewolf_agent | ForEach-Object {
    $text = [System.IO.File]::ReadAllText($_.FullName)
    [PSCustomObject]@{
        Lines = ($text -split "`r?`n").Count
        Path = $_.FullName.Substring((Get-Location).Path.Length + 1)
    }
}
$rows | Where-Object { $_.Lines -ge 600 } | Sort-Object Lines -Descending
```

Expected: updated candidate list.

- [ ] **Step 2: Split `metrics.py`**

Create `evaluation/claim_metrics.py`, `evaluation/pace_metrics.py`, and `evaluation/world_model_metrics.py`; keep `MetricsAggregator` facade.

- [ ] **Step 3: Split `model_gateway/router.py`**

Create `model_gateway/provider_call.py`, `model_gateway/retry_policy.py`, and `model_gateway/usage_records.py`; keep `ModelRouter` facade.

- [ ] **Step 4: Split `runtime/nodes/_shared.py`**

Create `runtime/nodes/runtime_state.py`, `runtime/nodes/action_audit.py`, and `runtime/nodes/node_helpers.py`; preserve compatibility imports for shared node helpers.

- [ ] **Step 5: Split `runtime/game_runner.py`**

Create `runtime/game_runner_config.py`, `runtime/game_runner_setup.py`, and `runtime/game_runner_execution.py`; keep `GameRunner` facade.

- [ ] **Step 6: Split `engine/rule_engine.py` only after rule tests are stable**

Create `engine/action_legality.py`, `engine/night_resolution.py`, and `engine/victory_conditions.py`; keep `RuleEngine` facade.

- [ ] **Step 7: Split `runtime/nodes/sheriff.py`**

Create `sheriff_registration_node.py`, `sheriff_vote_node.py`, and `sheriff_speech_node.py`; keep `sheriff.py` facade.

- [ ] **Step 8: Split RAG retriever/vector store modules**

Create focused modules for tokenization, local vector storage, embedding vector storage, and provider-specific vector stores.

- [ ] **Step 9: Run matching focused suites and commit per file group**

Commit each file group independently:

```powershell
git commit -m "refactor: split evaluation metrics"
git commit -m "refactor: split model gateway router"
git commit -m "refactor: split shared runtime node helpers"
git commit -m "refactor: split game runner helpers"
git commit -m "refactor: split rule engine responsibilities"
git commit -m "refactor: split sheriff runtime nodes"
git commit -m "refactor: split rag retrieval storage"
```

---

### Task 12: Split Oversized Test Files To Match Production Boundaries

**Files:**
- Modify: `tests/agents/test_prompt_builder.py`
- Modify: `tests/runtime/test_strategy_directives.py`
- Modify: `tests/agents/test_player_agent.py`
- Modify: `tests/runtime/test_context.py`
- Modify: `tests/rag/test_rag.py`
- Create: focused test files matching newly created production modules.

- [ ] **Step 1: Split prompt builder tests**

Move section, persona, memory, strategy, and salience tests into dedicated files created in Task 4.

- [ ] **Step 2: Split strategy directive tests**

Group tests by directive module or role domain.

- [ ] **Step 3: Split player agent tests**

Group tests by success path, fallback/retry, latency, and provider failure behavior.

- [ ] **Step 4: Split context tests**

Move tests to files created in Task 2.

- [ ] **Step 5: Split RAG tests**

Group ingestion, retrieval, vector storage, prompt rendering, and hardening tests by module.

- [ ] **Step 6: Run moved test suites**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents tests/runtime tests/rag -q
```

Expected: same pass/fail baseline as before moving tests.

- [ ] **Step 7: Verify and commit**

Commit message:

```powershell
git commit -m "test: split oversized characterization suites"
```

---

### Task 13: Final Verification And CodeGraph Sync

**Files:**
- All changed production and test files.

- [ ] **Step 1: Run broad targeted suite**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime tests/agents tests/skills tests/memory tests/api tests/evaluation tests/rag tests/rules -q
```

Expected: all relevant tests pass or known unrelated baseline failures are documented.

- [ ] **Step 2: Compile source**

Run:

```powershell
python -m compileall -q werewolf_agent
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run touched-file ruff**

Run:

```powershell
$files = git diff --name-only HEAD~1..HEAD
python -m ruff check $files --select F401,F841
```

Expected: `All checks passed!`

- [ ] **Step 4: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 5: Clean repo-local temp artifacts**

Run:

```powershell
$workspace = (Resolve-Path -LiteralPath .).Path
$tmpPath = Join-Path $workspace ".tmp"
if (Test-Path -LiteralPath $tmpPath) {
    $tmp = (Resolve-Path -LiteralPath $tmpPath).Path
    if (-not $tmp.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside workspace: $tmp"
    }
    Remove-Item -LiteralPath $tmp -Recurse -Force
}
```

Expected: `.tmp` removed.

- [ ] **Step 6: Sync CodeGraph**

Run:

```powershell
codegraph sync .
codegraph status .
```

Expected: `[OK] Index is up to date`.

- [ ] **Step 7: Final commit if needed**

Run:

```powershell
git status --short
git log --oneline -n 10
```

Expected: only intended changes committed; unrelated user changes remain untouched.

---

## Completion Criteria

- Previously identified SRP candidates are either split or explicitly documented as cohesive enough to remain.
- Every original public module path still imports successfully.
- `agent_adapter.py`, `context.py`, `night.py`, `prompt_builder.py`, `werewolf_skills.py`, `output_parser.py`, `player.py`, `reflection.py`, `day.py`, and `games.py` are reduced to focused modules or compatibility facades.
- Focused tests exist next to each new responsibility module.
- Each commit is small, reversible, and tied to one responsibility group.
- CodeGraph is synced after the final commit.
