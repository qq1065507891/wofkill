# Fix Plan - Reflection Memory Dead Chain

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement task-by-task. Keep exactly one active task in `PROGRESS.md`. Verify against `docs/design/werewolf-agent-v1-design.md` §10.2. Peer-reviewed 2026-06-21; revised to apply review findings.

## Relationship to fix-belief-cognition

This plan and `fix-belief-cognition.md` are **independent** (separate PRs, different acceptance metrics: `approved` count vs accuracy) but **cooperate on unblocking `approved`**: both can make `mistake_patterns` non-empty — 1b via LLM-parsed mistakes, defect B via deterministic `error_analysis`. The score analysis in Risks shows 1b alone may stall at 0.55 if LLM bullets are short; defect B provides longer deterministic mistakes that reliably cross 0.70. **Recommended order: 1a → 1b → defect B (from the other plan) for stable `approved > 0`.**

## Goal

Make cross-game reflection memory actually reach live game prompts. As of 2026-06-21, 168 reflection entries are persisted to the `reflections` table but **0 are `approved`**, so `query_live` (`werewolf_agent/memory/reflection.py`) returns nothing and `reflection_memory_hints` / `error_pattern_hint` are never injected into any game prompt.

## Root Cause (diagnosed 2026-06-21, peer-reviewed)

**Defect A (addressed here):** The reflection task context is not stripped of in-game decision directives. `_agent_reflection` (`werewolf_agent/runtime/agent_adapter.py:2592-2611`) calls `build_agent_context(TaskType.REFLECTION)`, then `_merge_strategy_directive` (`context.py:943`) only merges `reflection_task` as a peer key. `TaskType.REFLECTION`'s only special handling is skipping RAG retrieval (`context.py:230`). Verified: all 12 self-reviews of game `g_1416328333` are non-empty but none contain the template's section headers; werewolves wrote future-tense kill plans.

**Synthesis gap (addressed here):** `ReflectionSynthesizer` parses `【保留的优点】` into `preserved_strengths` but has no parser for mistake sections. So `mistake_patterns` only comes from `ReviewReport.error_analysis` (defect B), which is empty.

**Defect B (separate plan, `fix-belief-cognition.md`):** `_evaluate_judgments` (`review.py:106`) records only misjudgments with `best_prob > 0.3`; `role_probabilities` stay at `1/7 ≈ 0.143` for most players → `error_analysis` always empty.

**Score caveat (peer-review finding):** 1b alone may not reach `approved`. See Risks.

## Deliverables

- **1a:** Post-game context via **allowlist** (not blacklist). When `task_type == REFLECTION`, strip `strategy_directive` to **only** `{reflection_task, game_outcome}`. (Peer review showed the original 11-key blacklist was ~70% phantom — keys like `wolf_team_plan`/`vote_pressure_context`/`must_address`/`directive` don't exist in the reflection path; only `skill_tactical_advice`/`role_alerts`/`must_address_alerts`/`witch_poison_deterrent`(conditional)/`hybrid_master_dead` actually appear. Allowlist is future-proof.)
- **1a-verify:** One-game experiment confirming the LLM now emits real reflection section headers.
- **1b:** `ReflectionSynthesizer._extract_llm_mistakes` — parse **6 sections** (`【投票错误】`/`【信息缺失】`/`【神职执行】`/`【悍跳分析】`/`【暴露原因】`/`【角色分工】`) into `ReflectionMistakePattern` per the field mapping below, merged into `mistake_patterns` (deterministic first, LLM supplementary, jaccard≥0.6 dedup, cap 3).
- Regression + new unit tests for both.
- Update `PROGRESS.md` and a note in design doc §10.2.

### 1b Field Mapping Spec (peer-review required)

For each LLM mistake bullet (after `_scrub_ids`):

| `ReflectionMistakePattern` field | value |
|---|---|
| `wrong_action` | `_scrub_ids(bullet_text)` |
| `better_action` | `_default_advice(role)` (`reflection.py:419`) — LLM bullets describe the error, not the fix |
| `category` | by section header (NOT keyword-based `_category`): `【投票错误】→vote_mistake`, `【信息缺失】→info_miss`, `【神职执行】→role_execution`, `【悍跳分析】`/`【暴露原因】`/`【角色分工】→decision_mistake` |
| `trigger` | `_trigger_for_category(category, ...)` (reuse `reflection.py:408`) |
| `fact_basis` | `"llm_transferable"` |
| `auto_verified` | `False`  ← **CRITICAL, see Boundary** |
| `corrected_from_llm` | `False` |

## Required Boundary

- `reflections` table schema is stable — **no new columns** (CLAUDE.md §Reflection Memory).
- LLM self-review bullets containing truth tokens (`实际`/`真实身份`/`底牌`/`查验结果`/`死亡原因`, `_LLM_TRUTH_TOKENS`) must be dropped — spec rule 7.
- **NEW (Major safety, peer-review):** LLM-derived mistake patterns MUST set `auto_verified=False` + `fact_basis="llm_transferable"`. NEVER copy the deterministic path's `auto_verified=True` — `reflection.py:214` `if entry.prompt_card.auto_verified: return False` would **bypass the `unsafe_truth_claim` gate**. A unit test must assert `auto_verified is False` on every LLM mistake.
- Truth-token filtering has **two independent gates**: (a) per-bullet drop in `_extract_llm_mistakes`, (b) entry-level `_has_unsafe_truth_claim` (`reflection.py:213-217`). Test both.
- 1b must degrade gracefully: if no section is parseable, fall back to current behavior. Never inject noise.
- `RuleEngine` untouched. `_agent_reflection` keeps `TaskType.REFLECTION` (P0-RF1) and `_scrub_player_ids` (P0-RF2).

## Required Order

1. **1a allowlist + unit test.** After `_agent_reflection` builds context with `task_type==REFLECTION`, assert `set(strategy_directive.keys()) == {"reflection_task", "game_outcome"}`. Implement as: `kept = {k: v for k, v in directive.items() if k in {"reflection_task","game_outcome"}}` before `_merge_strategy_directive(reflection_directive)`.
2. **1a-verify (gate).** Run one real or scripted game. Inspect `source.llm_self_review` — must contain `【投票错误】` or `【保留的优点】` headers, not in-game decisions.
3. **Conditional (NON-trivial — budget for it):** if 1a-verify fails, `visible_world_state` (`context.py:1566` unconditional; `prompt_builder.py:853-855` renders unconditionally; `build_visible_player_state` does not branch on task_type) still misleads the LLM. Fix requires `build_visible_player_state` to branch on `task_type==REFLECTION` and emit a "game over" summary. Re-run verify.
4. **1b `_extract_llm_mistakes` + unit tests:** 6 sections parsed; truth-token bullet dropped; field mapping correct (`auto_verified=False`, `fact_basis="llm_transferable"`, category by header); empty/section-less input → `[]`.
5. **1b merge into `synthesize` (~L264-271) + tests:** deterministic first, LLM supplementary, jaccard≥0.6 dedup, cap 3. Plus end-to-end: a self-review with a truth-token mistake bullet → 1b drops it → gate evaluates → `quality_flags` does NOT contain `unsafe_truth_claim` (proves the per-bullet drop suffices and the entry isn't hard-rejected).
6. **Score analysis (NEW, peer-review required).** Construct a representative LLM self-review (real bullets), run `synthesize` + `ReflectionQualityGate.evaluate`, confirm whether `quality_score ≥ 0.70`. If it stalls at 0.55 (`short_prompt_card`), **document that defect B is required for stable `approved > 0`** and flag it in PROGRESS.
7. **End-to-end.** Run games until an `approved` entry appears; confirm the next game's `reflection_memory_hints` is non-empty.

## Files

- Modify: `werewolf_agent/runtime/agent_adapter.py` — `_agent_reflection` (L2592-2623); replace blacklist idea with allowlist strip before `_merge_strategy_directive`.
- Modify: `werewolf_agent/memory/reflection.py` — add `_LLM_MISTAKE_SECTION_RE` (6 sections), `_extract_llm_mistakes`, merge call in `synthesize` (~L264-271).
- Modify (conditional, step 3): `werewolf_agent/runtime/context.py` `build_visible_player_state` / `build_agent_context` (~L1566) + `werewolf_agent/agents/prompt_builder.py:853-855` — branch on `task_type==REFLECTION`.
- Test: `tests/memory/test_reflection_v2.py` — mistake parsing, truth-token drop, dedup, field mapping, end-to-end gate.
- Test: `tests/runtime/test_agent_reflection_context.py` (new) or extend `tests/runtime/test_strategy_directives.py` — allowlist assertion.
- Docs: `PROGRESS.md`; `docs/design/werewolf-agent-v1-design.md` §10.2.

## Done Means

- A fresh game's `source.llm_self_review` contains section headers (no more "今晚刀…" / "第N夜观察").
- Score analysis (step 6) shows at least one entry can reach `approved`; the `reflections` table has ≥1 `approved` after running games.
- `query_live` returns non-empty; next game's `reflection_memory_hints` is non-empty.
- All new + existing reflection tests pass; truth-token bullets rejected at both gates; LLM mistakes assert `auto_verified is False`.
- No new `reflections` columns; `RuleEngine` unaffected; `git diff --check` clean.

## Risks

- **1a allowlist may still leak `visible_world_state`.** Step 3 is the fallback; it is non-trivial (branch `build_visible_player_state`), not a one-liner — budget for it.
- **Score ceiling (key peer-review finding).** 1b adds `+0.25` (non-empty mistake) but `prompt_card` completeness (`+0.25`) and the `short_prompt_card` penalty (`-0.15`) depend on **LLM bullet length**. Optimistic path: `0.25+0.25+0.10+0.10+0.10+0.05 = 0.85` → approved. Pessimistic (short/generic bullets): `~0.45` → rejected. **1b alone does NOT guarantee `approved > 0`.** Defect B (`fix-belief-cognition.md`) provides longer deterministic mistakes (`"误判X为Y，最佳角色概率0.80"`) that reliably cross 0.70. If step-6 score analysis stalls, escalate defect B priority.
- **LLM template adherence.** 1b must be robust to missing sections; fallback to current behavior, never inject noise.
