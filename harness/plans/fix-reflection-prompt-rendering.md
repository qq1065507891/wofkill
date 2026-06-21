# Fix Plan - Reflection Prompt Rendering (supplement to fix-reflection-memory)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (or subagent-driven-development). Keep one active task in `PROGRESS.md`. This supplements `fix-reflection-memory.md` (1a+1b merged at 920649f). Verify against `docs/design/werewolf-agent-v1-design.md` §10.2.

## Relationship

Supplements `fix-reflection-memory.md`. The original 1a (allowlist) + 1b (parser/merge) are merged (`920649f`), but **real-game verification failed**: game `g_415824166` (started 2026-06-21 12:07, **post-merge code**) still has **0 approved** — all 12 `llm_self_review` contain no section headers, mistake/strength all empty. The root cause is more precise than the original plan's "visible_world_state": 1a fixed the `strategy_directive` dict, but `reflection_task` is **demoted at prompt-rendering time**, so the LLM still doesn't see section requirements as binding.

## Root Cause (verified against code + g_415824166 data, 2026-06-21)

Three independent causes:

1. **`reflection_task` rendering demoted (most upstream):** `_build_strategy_directive` (`werewolf_agent/agents/prompt_builder.py:1287-1295`) classifies `strategy_directive` keys by `HARD_CONSTRAINT_KEYS` / `SUGGESTION_KEYS` / `REFERENCE_KEYS`; unmatched keys **fallback to 【参考】**. `reflection_task` is a new key never registered in any set (`prompt_builder.py` contains no `reflection_task` string) → fallback 【参考】 (REFERENCE, "仅供决策参考"). `_compact_json` (`prompt_builder.py:2065`) then renders the whole template as a JSON value: `{"reflection_task":"你是村民...【投票错误】...","game_outcome":"..."}`. The LLM sees the section requirements as a **lowest-priority JSON field**, not a binding instruction — so it ignores them and writes free-form in-game analysis.

2. **`visible_world_state` dominates attention:** 【可见世界状态】 is a standalone `NEVER_DROP_TIER` section (`prompt_builder.py:401`); live state (alive players / speeches / vote trails) is far more salient than the 【参考】 JSON → LLM follows the live board and writes in-game speech/analysis.

3. **defect B generic default:** `error_analysis` empty → `ReviewReport.improvement_suggestions` default is a `_GENERIC_PHRASES` member ("复盘失败对局，关注关键转折点的信息缺失") → pollutes `actionable_advice` + `recommended_action` → `generic_text` (-0.25) → 22 entries stuck at score 0.2.

## Deliverables

- **PR1:** Force `reflection_task` to render as a hard/standalone directive — register in `HARD_CONSTRAINT_KEYS` (→ 【硬约束】 MUST) OR render as a dedicated top-level section when `task_type==REFLECTION` (not via `_compact_json`). This is the key fix that makes the LLM actually follow the section template.
- **PR2:** `REFLECTION` branch in `build_visible_player_state` / `build_agent_context` — post-game summary only (identity, outcome, own vote/action timeline), no live `visible_world_state`. (Original plan's step 3.)
- **PR3:** Fix `_generate_suggestions` (`werewolf_agent/memory/review.py`) generic default — must NOT be a `_GENERIC_PHRASES` member.
- Verification: prompt_builder render test (`reflection_task` NOT under 【参考】) + real-game 1a-verify (user-run).

## Required Boundary

- **PR1:** registering `reflection_task` in `HARD_CONSTRAINT_KEYS` (or adding a REFLECTION standalone section) must not perturb existing vote/speech/witch/wolf directive rendering. Add a regression assertion that a normal SPEECH/VOTE prompt still groups its directives identically.
- **PR2:** `visible_world_state` weakening applies **ONLY** to `task_type==REFLECTION`; vote/speech unchanged. Assert via test.
- `RuleEngine` untouched; `reflections` schema unchanged. Task 1's allowlist (`_POST_GAME_KEEP`) stays in place.
- `auto_verified=False` / truth-token defenses (from 1b) stay intact.

## Required Order

1. **PR1** `reflection_task` hard rendering + prompt_builder render test (assert `reflection_task` content appears under 【硬约束】 or a top-level section, NOT inside 【参考】 JSON). Regression: normal SPEECH prompt directive grouping unchanged.
2. **PR2** `visible_world_state` REFLECTION branch + test (REFLECTION context's visible state is post-game summary; SPEECH/VOTE unchanged).
3. **PR3** generic `improvement_suggestions` default fix + test (default is not a `_GENERIC_PHRASES` member; `generic_text` no longer auto-triggers from review default).
4. **1a-verify (user-run real game):** confirm LLM now outputs section headers + `reflections` table has ≥1 `approved`.

## Files

- Modify: `werewolf_agent/agents/prompt_builder.py` — `HARD_CONSTRAINT_KEYS` (add `reflection_task`) OR new REFLECTION standalone section in `_build_strategy_directive` / `build_user_prompt`.
- Modify: `werewolf_agent/runtime/context.py` + the `build_visible_player_state` helper — REFLECTION branch.
- Modify: `werewolf_agent/memory/review.py` — `_generate_suggestions` default.
- Test: `tests/agents/test_prompt_builder.py` (reflection_task render position + SPEECH regression), `tests/runtime/test_agent_reflection_context.py` (visible_world_state branch), `tests/memory/` (generic suggestions).
- Docs: `PROGRESS.md`, design doc §10.2.

## Done Means

- prompt_builder render: `reflection_task` content appears under 【硬约束】 or a dedicated top-level section, NOT inside 【参考】 JSON.
- `task_type==REFLECTION` context's visible state is a post-game summary (no live alive-list / vote trails).
- `_generate_suggestions` default is not a `_GENERIC_PHRASES` member.
- **Real game (1a-verify):** LLM outputs `【投票错误】` / `【保留的优点】` section headers; `reflections` table has ≥1 `approved`; next game's `reflection_memory_hints` non-empty.
- All new + existing reflection/prompt_builder tests pass; `RuleEngine` unaffected; `git diff --check` clean.

## Risks

- **PR1 `HARD_CONSTRAINT_KEYS` registration grows the hard section** (never trimmed) — monitor prompt length; if it busts budget, use the standalone-section approach instead.
- **PR2 `build_visible_player_state` branch is non-trivial** (final-review warned) — budget for it.
- **1a-verify needs a user-run real game (1-2h)** — code-level tests (render + synthesize) are not enough; the original failure was a rendering/behavior gap only visible in a real LLM call.
- **LLM template adherence** is still not guaranteed even after PR1/PR2; if 1a-verify still shows no sections, the fallback is a stricter template (system-prompt-level instruction) — out of scope here, escalate.
