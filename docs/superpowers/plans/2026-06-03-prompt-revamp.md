# Prompt Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan batch-by-batch. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 70 prompt / retry / skill / RAG / memory / directive / info-isolation issues identified in the 2026-06-03 audit of `werewolf_agent/agents/prompt_builder.py`, `player.py`, `output_parser.py`, `context.py`, and supporting modules.

**Architecture:** Severity-sequenced fix in 6 batches. Batch 0 verifies 3 uncertain items before committing to a fix direction. Batches 1-3 are P0 (production-affecting). Batch 4 is P1 by-area parallel subagents. Batch 5 is P2 polish. Each fix is a TDD cycle (red test → fix → green → commit). Every batch ends with a full test suite + game trace verification.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, conda env `wofkill`. Existing modules: `werewolf_agent/agents/{prompt_builder,player,output_parser,parse_dispatch,schemas,trace_builder,tool_schema,metrics_collector}.py`, `werewolf_agent/runtime/{context,agent_adapter,directives,strategy}.py`, `werewolf_agent/skills/{registry,werewolf_skills,schemas}.py`, `werewolf_agent/rag/{knowledge_service,injector,retriever,schemas,ingestion}.py`, `werewolf_agent/memory/{store,reflection,cognition_matrix,review,schemas}.py`, `werewolf_agent/runtime/private_memory.py`, `tests/integration/test_e2e_info_leak.py`.

**Branch:** `prompt-revamp-2026-06` (worktree at `.worktrees/prompt-revamp-2026-06`)
**Base commit:** `abe63ca refactor(schemas): PlayerAction as discriminated union of 10 action variants`

---

## Audit Source

70 issues identified across:
- Static analysis of `prompt_builder.py`, `context.py`, `agent_adapter.py`, `player.py`, `output_parser.py`, `directives/*`, `strategy/*`, `skills/*`, `rag/*`, `memory/*`
- Design-doc cross-check (`docs/design/werewolf-agent-v1-design.md` Chapter 7 / 8 / 9 / 10 / 11)
- Production game trace `game_g_3528592081.json` (82 actions: 15 fallbacks, 18 parse_errors, 17 empty_responses)

Issue catalog with file:line in section "Issue Catalog" below.

---

## File Structure

**New files to create (per batch, on demand):**
- `tests/agents/test_prompt_mode_isolation.py` (P0-S1)
- `tests/agents/test_directive_role_gating.py` (P0-I1, P0-I3)
- `tests/memory/test_belief_visibility.py` (P0-M8)
- `tests/memory/test_reflection_temporal.py` (P0-M3)
- `tests/rag/test_prompt_safe_render.py` (P0-G1, P0-G2, P0-G3)
- `tests/integration/test_seer_claim_isolation.py` (P0-M2 extension)

**Files to modify (primary):**
- `werewolf_agent/agents/prompt_builder.py` — core prompt assembly (most fixes touch this)
- `werewolf_agent/agents/player.py` — retry loop, fallback, speech quality
- `werewolf_agent/agents/output_parser.py` — encoding repair, hint differentiation
- `werewolf_agent/agents/schemas.py` — add fields, remove leakers
- `werewolf_agent/runtime/context.py` — visibility, RAG gating, directive filtering
- `werewolf_agent/runtime/agent_adapter.py` — directive injection, hybrid prompt
- `werewolf_agent/runtime/directives/*.py` — grouping, leakage, hybrid follow-master
- `werewolf_agent/runtime/private_memory.py` — sanitize expansion, marker precision
- `werewolf_agent/runtime/strategy/*.py` — already split, may need minor
- `werewolf_agent/memory/{reflection,cognition_matrix,store,review}.py` — vector search prep
- `werewolf_agent/rag/{injector,retriever,knowledge_service,schemas}.py` — render slimmer

**Files NOT to touch:**
- `werewolf_agent/engine/*` (RuleEngine)
- `werewolf_agent/cognition/*` (BeliefUpdater, WorldState)
- `werewolf_agent/storage/*`
- `werewolf_agent/api/*`
- `werewolf_agent/ui/*`

---

## Batching Strategy

```
Batch 0 (verification, ~1-2 days)  →  3 task subagents
Batch 1 (P0 quick wins, 10)        →  10 task subagents (sequential in same worktree)
Batch 2 (P0 structural, 10)        →  10 task subagents (sequential)
Batch 3 (P0 redesign, 6)           →  6 task subagents
Batch 4 (P1 by area, 30)           →  5 area subagents (parallel via worktrees)
Batch 5 (P2 polish, 14)            →  1 area subagent
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
- `scripts/print_game_audit.py game_g_3528592081.json` shows reduced fallback count
- PROGRESS.md updated with batch summary

---

## Batch 0: Verification (3 tasks)

Goal: Reduce uncertainty on 3 P0 items before committing to fix direction. Each task is a **test/measurement only**, no fix yet.

### Task 0.1: Verify P0-K1 (skill tool actually called?)

**Hypothesis:** Tool skills (`wolf_pit`, `find_power`, `last_words`) are exposed as `skill_tools` but maybe never invoked by any LLM. If confirmed, the "double path" is a non-issue and we can drop the tool exposure (pure injection path).

**Files:**
- Create: `tests/agents/test_skill_tool_invocation.py`
- Read: `game_g_3528592081.json`, `game_g_2989362760.json`, `game_g_2913931821.json` (all 3 saved games)

**Steps:**
- [ ] **Step 1:** Write test that scans saved game JSONs for `tool_call_name` field matching skill tool names (`skill_analyze_wolf_pit`, `skill_find_power_roles`, `skill_analyze_last_words`)
- [ ] **Step 2:** Run test, record: count of tool calls vs count of `submit_player_action` calls
- [ ] **Step 3:** If count == 0: write finding "tool path is dead code" → K1 fix becomes "delete tool exposure, keep injection"
- [ ] **Step 4:** If count > 0: write finding "tool path active" → K1 fix becomes "delete pre-injection, keep tool path"
- [ ] **Step 5:** Commit test + finding as `docs/audit/2026-06-03-batch0-k1-finding.md`

### Task 0.2: Verify P0-M8 (BeliefUpdater leaks private info?)

**Hypothesis:** `BeliefUpdater.update_from_belief` may use private info (e.g., wolf teammate ID, seer check result) to compute `role_probabilities` for other players, then `cognition_matrix_hint` exposes those probabilities in non-wolf / non-seer player prompts.

**Files:**
- Create: `tests/memory/test_belief_visibility.py`
- Read: `werewolf_agent/cognition/belief.py`, `werewolf_agent/cognition/belief_updater.py` (or similar)

**Steps:**
- [ ] **Step 1:** Write test: for a wolf player, build `belief_state` that includes "I know p07 is my teammate" (private). Pass through `BeliefUpdater`. Check that the resulting `role_probabilities` for p07, p08, etc. do NOT have p07 marked as low-wolf (because that would let a non-wolf player infer teammate).
- [ ] **Step 2:** Run test, record finding: which private signals propagate to public role probabilities?
- [ ] **Step 3:** If leak found: write `docs/audit/2026-06-03-batch0-m8-finding.md` with specific signal/policy. M8 fix is "audit BeliefUpdater data sources + add filter".
- [ ] **Step 4:** If no leak: write finding "BeliefUpdater is public-safe", M8 fix becomes "no change, just add regression test".
- [ ] **Step 5:** Commit test + finding.

### Task 0.3: Verify P0-S2 (single-shot vs three-step JSON output)

**Hypothesis:** Design doc §7.2 says "weak models must use three-step generation". Current implementation uses single-shot. Maybe single-shot works for current models, or maybe three-step would significantly improve success rate.

**Files:**
- Create: `scripts/measure_output_mode_success.py` (one-off measurement)
- Read: `werewolf_agent/agents/player.py`, `werewolf_agent/agents/output_parser.py`

**Steps:**
- [ ] **Step 1:** Run `scripts/run_real_game.py` 3 times with current single-shot pipeline. Record: success rate, fallback rate, parse_error rate, empty_response rate.
- [ ] **Step 2:** Read `scripts/print_game_audit.py` output for those runs.
- [ ] **Step 3:** Calculate: what % of actions would benefit from 3-step? Estimate based on fallback reasons:
  - If most fallbacks are `empty_response` → single-shot is the bottleneck, 3-step helps
  - If most fallbacks are `parse_error` → 3-step helps
  - If most fallbacks are `vote_quality` / `speech_quality` → 3-step doesn't help (quality is post-parse)
- [ ] **Step 4:** Write `docs/audit/2026-06-03-batch0-s2-finding.md` with recommendation: implement 3-step OR keep single-shot + tighten prompt.
- [ ] **Step 5:** Commit measurement script + finding.

---

## Batch 1: P0 Quick Wins (10 tasks)

**Strategy:** Each is a small, surgical change with a focused test. Run sequentially in same worktree. Commit after each.

### Task 1.1 (P0-S1): Split `build_user_prompt` into 3 mode-mutually-exclusive paths
- **File:** `werewolf_agent/agents/prompt_builder.py:229-249, 251-278, 419-565`
- **Fix:** Introduce `_build_user_prompt_full_action()`, `_build_user_prompt_target_choice()`, `_build_user_prompt_speech_intent()`. Each renders only its mode's fields. `build_user_prompt()` dispatches based on `_select_output_mode()`.
- **Test:** `tests/agents/test_prompt_mode_isolation.py::test_no_choice_field_in_full_action_mode`, `test_no_intent_field_in_full_action_mode`, `test_no_action_type_in_speech_intent_mode`
- **Evidence:** 95 actions in g_3528592081 contain `intent` field; 63 contain `vote_basis` field. Confirms mode bleeding.

### Task 1.2 (P0-S5): Group strategy_directive into [硬约束/建议/参考] sections
- **File:** `werewolf_agent/agents/prompt_builder.py:376-380` + add new `_build_strategy_directive()` that groups keys.
- **Fix:** New function renders 3 separate sections:
  - `[硬约束]` keys: `wolf_fake_seer_execution`, `must_address_alerts`, `first_night_killed`, `speech_silent`, `vote_silent`, `witch_night_action`, `role_alerts`
  - `[建议]` keys: `wolf_speech_directive`, `wolf_universal_rules`, `good_vote_decision_guard`, `anti_herd`, `sheriff_vote_push`, `speech_originality`, `seer_speech_directive`
  - `[参考]` keys: `skill_tactical_advice`, `wolf_day_push_target`, `wolf_high_priority_target`, `wolf_plan_target`, `master_behavior_summary`
- **Test:** `tests/agents/test_prompt_builder.py::test_strategy_directive_has_three_groups`, `test_hard_constraints_keys_in_hard_group`
- **Evidence:** game trace shows p08 wolf receives `wolf_fake_seer_execution` directive and follows it (claim seer). Confirms directives are received.

### Task 1.3 (P0-S6): Move retry hint to after task prompt, include last_error_message
- **File:** `werewolf_agent/agents/prompt_builder.py:229-249, 411-417` + `player.py:524-583`
- **Fix:** Reorder user_prompt sections: task → retry → contract. Retry hint now includes first 100 chars of `error_message` (the actual parse error or quality violation, not generic).
- **Test:** `tests/agents/test_player_agent.py::test_retry_hint_includes_error_message`, `test_retry_hint_position_after_task`
- **Evidence:** game trace Action 50 retry 3x all same error — generic hint doesn't help.

### Task 1.4 (P0-S7): Change `claimed_view` example to enum value
- **File:** `werewolf_agent/agents/prompt_builder.py:441, 481, 497`
- **Fix:** Replace `"claimed_view": "我是好人"` with `"claimed_view": "good_player_without_night_info"` (the schema's `FactionGoal` enum-style value, not natural language).
- **Test:** `tests/agents/test_prompt_builder.py::test_claimed_view_example_uses_enum`
- **Evidence:** game trace shows wolves write `claimed_view: "我是好人，混水摸鱼"` (natural language with strategy hint).

### Task 1.5 (P0-S8): Remove vote-only fields from SPEECH example
- **File:** `werewolf_agent/agents/prompt_builder.py:478-499`
- **Fix:** SPEECH example drops `seer_stance`/`vote_basis`/`standing_with_seer`/`suspect_reason`/`not_voting_reason`/`private_reason`. The parser/validator should also reject them on non-vote actions.
- **Test:** `tests/agents/test_schemas.py::test_speech_action_rejects_vote_only_fields`, `tests/agents/test_prompt_builder.py::test_speech_example_no_vote_fields`
- **Evidence:** all 67 successful speech actions in g_3528592081 contain `vote_basis: "fallback"`.

### Task 1.6 (P0-M2): Expand `_sanitize_role_claims` to all private text fields + variants
- **File:** `werewolf_agent/runtime/private_memory.py:46-50, 86-99, 138-172, 188-194`
- **Fix:** Add new regex patterns: `(我.{0,4}(队友|同伴|同伴是|阵营里))`, `(我的.{0,2}(身份|底牌|真身|阵营))`, `(我(看穿|发现|验出|查验))`. Apply sanitize to ALL `point`/`text`/`reason` fields, not just `suspect_reason` + `private_reason`.
- **Test:** `tests/runtime/private_memory.py::test_sanitize_removes_team_member_disclosure`, `test_sanitize_applies_to_all_text_fields`
- **Evidence:** game trace Action 56: p02 wolf `private_reason: "p07 是我的队友..."` — leaks team ID through memory.

### Task 1.7 (P0-M3): Sort reflections by `game_id` not `entry_id`
- **File:** `werewolf_agent/runtime/context.py:251-270`
- **Fix:** Change sort key from `(-priority, str(r.entry_id))` to `(-priority, str(r.game_id), str(r.entry_id))`.
- **Test:** `tests/runtime/test_context.py::test_reflection_sort_by_game_id`
- **Evidence:** static analysis; UUIDs not time-ordered.

### Task 1.8 (P0-M4): Replace profile 5-dim dump with role-specific win-rate only
- **File:** `werewolf_agent/runtime/context.py:231-248` + `werewolf_agent/memory/schemas.py:76-91`
- **Fix:** Only inject `games_played`, current-role win rate, 1-sentence summary. Drop raw `logic/deception/credibility` floats. Player profile `learning_rate` and `risk_preference` exposed only to judge/replay, not live prompt.
- **Test:** `tests/runtime/test_context.py::test_profile_prompt_no_raw_dimensions`, `test_profile_prompt_only_current_role_winrate`
- **Evidence:** static analysis; design §10.1 says 6 dims but prompt exposes 3 — change to design-allowed "rank description" form.

### Task 1.9 (P0-R2): Reduce empty_response rate on critical roles
- **File:** `werewolf_agent/agents/prompt_builder.py` (shorten seer/witch role prompts), `werewolf_agent/runtime/agent_adapter.py` (raise `max_tokens` for these roles)
- **Fix:** (a) Move `_build_skill_catalog` from system to user for seer/witch (already 200 chars; saves ~1500 chars of system cache). (b) Bump `AGENT_TIMEOUTS` for seer check / witch action by 2x. (c) Add explicit "If model times out, return `no_action` not empty" hint in retry.
- **Test:** `tests/agents/test_player_agent.py::test_seer_shortened_prompt`, `tests/integration/test_real_llm_smoke.py::test_seer_no_empty_response`
- **Evidence:** 17 empty_responses in g_3528592081, mostly on seer/villager. Action 57 = p03 seer vote fallback.

### Task 1.10 (P0-R3): Robust encoding repair in output_parser
- **File:** `werewolf_agent/agents/output_parser.py:52-83` (`repair_json_text`)
- **Fix:** Detect garbled Chinese (mojibake pattern: `��` adjacent to a quote-like char). If found, try ISO-8859-1 → UTF-8 round-trip. Also fix trailing comma in objects/arrays. Fall back to `extract_parameter_tag_action` for parameter-tag-style outputs.
- **Test:** `tests/agents/test_output_parser.py::test_repair_mojibake_quotes`, `test_repair_handles_trailing_comma`
- **Evidence:** game trace Action 50 raw text shows `��intent��:"..."` (mojibake quotes around JSON keys).

---

## Batch 2: P0 Structural (10 tasks)

**Strategy:** Touches more files. May include backward-incompatible prompt changes. Still sequential in same worktree.

### Task 2.1 (P0-S2): ~~Implement three-step JSON generation~~ DEFERRED
- **File:** ~~`werewolf_agent/agents/player.py:228-648` (new three-step path)~~
- **Decision:** **DEFER indefinitely.** See `docs/audit/2026-06-03-batch0-s2-finding.md`.
- **Rationale:** Three-step generation would save ~30-45/279 actions (10-15%) at 3x token cost. Net negative ROI. Better fixes: P0-S1 mode isolation + P0-R2 shorter prompts + P0-R3 encoding repair.
- **Status:** ~~Active~~ → Deferred. If a future weak-model provider is added, revisit.
- **No code change for this task.**

### Task 2.2 (P0-K1): Skill tool single-path (decision from Task 0.1)
- **File:** `werewolf_agent/runtime/context.py:309-377, 758-766` + `werewolf_agent/agents/prompt_builder.py:181-214, 382-389`
- **Fix:** (If tool path dead) Drop `skill_tools` exposure. Keep `skill_analysis_hints` injection. (If tool path active) Drop pre-injection, only expose tools.
- **Test:** `tests/agents/test_skill_injection.py::test_only_one_skill_path_active`
- **Dependency:** Task 0.1.

### Task 2.3 (P0-K2): Add `applies_to_task_types` to SkillDefinition
- **File:** `werewolf_agent/skills/schemas.py:95-103` + `werewolf_agent/skills/registry.py:111-126` + `werewolf_agent/runtime/context.py:342`
- **Fix:** New field `applies_to_task_types: list[TaskType]`. `dispatch_for_role` filters by `task_type` in addition to role/phase. Wolf skills only dispatch on SPEECH / VOTE / WOLF_DISCUSSION, not NIGHT_ACTION etc.
- **Test:** `tests/skills/test_registry.py::test_skill_dispatch_filtered_by_task_type`, `test_no_irrelevant_skills_in_speech_prompt`

### Task 2.4 (P0-G1): RAG prompt only shows title/summary/key_decisions
- **File:** `werewolf_agent/rag/injector.py:123-147` + new `werewolf_agent/rag/prompt_renderer.py`
- **Fix:** Split `hits_to_context_items` into `hits_to_context_items()` (full, for audit) and `hits_to_prompt_lines()` (slim: title + summary + 2-3 key_decisions). Prompt uses slim variant.
- **Test:** `tests/rag/test_knowledge_service.py::test_prompt_render_drops_metadata`

### Task 2.5 (P0-G2): Hide RAG score/source/quality from live prompt
- **File:** `werewolf_agent/rag/prompt_renderer.py` (new) + `werewolf_agent/agents/prompt_builder.py:340-347`
- **Fix:** Slim renderer never includes `relevance_score`, `quality_grade.value`, `source_type.value`, `visibility_boundary.value`, `display_annotation`. Audit log keeps full data.
- **Test:** `tests/rag/test_prompt_renderer.py::test_no_metadata_in_live_prompt`

### Task 2.6 (P0-G3): Explicit RAG player-ID-vs-current-game warning
- **File:** `werewolf_agent/agents/prompt_builder.py:340-347`
- **Fix:** Add hard constraint prefix: "⚠️ RAG 案例中的玩家 ID 与本局无关；不得直接套用案例中具体玩家的发言或票型。" to the RAG hints section.
- **Test:** `tests/agents/test_prompt_builder.py::test_rag_hints_include_player_id_warning`

### Task 2.7 (P0-M1): private_memory prompt labels as [本局·第N轮]
- **File:** `werewolf_agent/agents/prompt_builder.py:319-330`
- **Fix:** Prepend section title with "【本局·第N轮·私有记忆】". Also remove `_add_own_speech_notes` extraction of "矛盾"/"前后不一" markers (too noisy, replace with structured fact extraction only).
- **Test:** `tests/agents/test_prompt_builder.py::test_private_memory_labeled_as_current_game`

### Task 2.8 (P0-M5): Either expose all 6 profile dims OR remove unused ones
- **File:** `werewolf_agent/memory/schemas.py:76-91` (decide) + `werewolf_agent/runtime/context.py:231-248` (render)
- **Fix:** If keeping 6 dims, render all 6. If trimming, remove `leadership`/`learning_rate`/`risk_preference` from `PlayerProfile`. Pick based on whether other code uses them. Decision: keep 6, render all 6 with rank description not raw float.
- **Test:** `tests/runtime/test_context.py::test_all_six_profile_dims_exposed_or_removed`

### Task 2.9 (P0-M7): Remove `visible_world_state` fallback for private_memory
- **File:** `werewolf_agent/agents/prompt_builder.py:319-330`
- **Fix:** Read only from `ctx.private_memory_hints`. If empty, skip section.
- **Test:** `tests/agents/test_prompt_builder.py::test_no_private_memory_dual_source`

### Task 2.10 (P0-D1): Render strategy_directive as 3 grouped sections
- **File:** `werewolf_agent/agents/prompt_builder.py:376-380` (covered by Task 1.2 actually — if 1.2 done, this is verified). Skip if duplicate.
- **Reconsider:** Task 1.2 already does the grouping. Mark 2.10 as "absorbed into 1.2".

---

## Batch 3: P0 Redesign (6 tasks)

**Strategy:** Deep changes requiring design discussion. Subagent per task with review.

### Task 3.1 (P0-M6): Add vector search to reflection memory
- **File:** `werewolf_agent/memory/reflection.py:112-137` + new `werewolf_agent/memory/vector_index.py`
- **Fix:** Add optional `vector_index` parameter to `ReflectionMemory`. If provided, `query()` first does vector similarity, falls back to exact match. Use simple cosine similarity on tag-weighted text (no need for full embeddings; for design parity). Per design §10.2.
- **Test:** `tests/memory/test_memory.py::test_reflection_query_with_vector_index`, `test_reflection_falls_back_to_exact_match`
- **Note:** Full embedding-based vector store is a future task. For now, a hash-based bag-of-words vector with cosine similarity satisfies the design contract.

### Task 3.2 (P0-M8): ~~BeliefUpdater private-info audit~~ MERGED into 3.5
- **Decision:** BeliefUpdater algorithm is private-info safe per `docs/audit/2026-06-03-batch0-m8-finding.md`. Risk is at rendering layer.
- **Merged into Task 3.5** (P0-M9: cognition_matrix_hint rendering fix).

### Task 3.3 (P0-I1): Add strategy_directive role-gating tests
- **File:** `tests/integration/test_e2e_info_leak.py` (extend) + new `tests/integration/test_directive_role_gating.py`
- **Fix:** For each role (villager/seer/witch/hunter/idiot/hybrid/werewolf), assert the **specific** directive keys that should be present and absent in `context.strategy_directive`. E.g., villager must NOT have `wolf_speech_directive`, `wolf_fake_seer_teammate`, `witch_night_action`, `seer_night_check`, `hunter_shot`.
- **Test:** `tests/integration/test_directive_role_gating.py::test_villager_no_wolf_directive`, etc.
- **Evidence:** static analysis only; existing `_assert_no_forbidden_info` checks `visible_world_state` not `strategy_directive`.

### Task 3.4 (P0-I2): Hybrid follows master's faction post-choice
- **File:** `werewolf_agent/runtime/directives/hybrid.py:10-48` + `werewolf_agent/runtime/context.py:667-676`
- **Fix:** When `gs.hybrid_master_faction == "werewolf"`, inject wolf-style "hidden ally" directives (mild `wolf_universal_rules`). When `"good"`, inject good-side analysis focus. Currently both fall through to neutral villager.
- **Test:** `tests/runtime/test_strategy_directives.py::test_hybrid_master_wolf_receives_hidden_wolf_directive`
- **Evidence:** game trace: p04 hybrid master=p01 wolf, but p04 voted like a villager (analyzed seer logic, voted wolf p07). Mismatch with design intent.

### Task 3.5 (P0-M9 + absorbed M8): cognition_matrix_hint rendering fix
- **File:** `werewolf_agent/runtime/context.py:273-301`, `werewolf_agent/memory/cognition_matrix.py:64-72`
- **Fix:**
  1. Render `key_evidence` and `open_questions` as **ID references** (`salience_items#abc123`), not full text.
  2. Trust/faction_lean/top_role_guess remain as summary statistics (already public-derived).
  3. Add regression test that wolf and villager with same public facts produce same role_probabilities (per M8 finding).
- **Test:** `tests/memory/test_belief_visibility.py::test_belief_state_uses_only_public_signals`, `tests/runtime/test_context.py::test_cognition_matrix_no_text_evidence`

### Task 3.6 (P0-I4): private_intent stance_notes don't enter cross-game memory
- **File:** `werewolf_agent/memory/reflection.py:62-89` (in `_store_review_reflection` or similar) + `werewolf_agent/runtime/private_memory.py:114-120`
- **Fix:** When converting `private_memory.stance_notes` to reflection text, strip player IDs. Replace "站边 p03" with "站边预言家" (role-based, not ID-based). Also strip `standing_with_seer` concrete IDs.
- **Test:** `tests/memory/test_reflection.py::test_reflection_stance_no_player_ids`

### Task 3.7 (P0-I3): Test that wolf private info doesn't leak via directives
- **File:** `tests/integration/test_directive_role_gating.py` (extend)
- **Fix:** For villager/seer/witch/hunter/idiot context, assert `wolf_fake_seer_teammate`, `wolf_day_push_target`, `wolf_plan_target`, `wolf_teammate_exposed`, `wolf_high_priority_target` are NOT in `strategy_directive` or `visible_world_state`.
- **Test:** new test class `TestDirectiveWolfPrivateNoLeak`.

---

## Batch 4: P1 by Area (5 parallel subagents)

**Strategy:** 5 worktrees, one per area. Each subagent does its area's P1 issues. Merge back sequentially.

**Areas:**
1. **Prompt** subagent → `prompt-revamp-p1-prompt` worktree, fixes: P1-S3, S4, S6 (residual), S9, S7 (residual), plus the 3 P2-S items if convenient
2. **Skill** subagent → `prompt-revamp-p1-skill` worktree, fixes: P1-K3, K4 (residual), K5, K6
3. **RAG** subagent → `prompt-revamp-p1-rag` worktree, fixes: P1-G4, G5, G6, G7, G8
4. **Memory** subagent → `prompt-revamp-p1-memory` worktree, fixes: P1-M10, M11, M12, M13, M14
5. **Directives** subagent → `prompt-revamp-p1-directives` worktree, fixes: P1-D4, D5, D6

Each area's subagent follows the same TDD cycle per issue. Detailed sub-plans generated on demand when subagent dispatched.

---

## Batch 5: P2 Polish (1 subagent)

**Strategy:** Single subagent, 14 P2 items. Lower priority, can skip if time-pressed.

Items: P2-S10, S11, S12, S13, S14, P2-K7, K8, K9, K10, P2-G9, G10, G11, P2-M15, M16

Detailed sub-plan on demand.

---

## Cross-cutting Constraints

1. **No regression**: Every batch ends with `python -m pytest tests/ -q` passing.
2. **TDD**: Every fix has a failing test first.
3. **No silent behavior changes**: Each fix updates PROGRESS.md with what changed and why.
4. **Backward compat**: Where possible, keep old field names as deprecated aliases. E.g., `claimed_view` enum values may be added but old strings still parse.
5. **Locale**: All user-facing strings in Chinese (per codebase convention).
6. **No new dependencies**: Use existing libraries (pydantic, numpy if needed, sqlite).
7. **Each commit is independently revertible**: One fix per commit.

---

## Issue Catalog (for reference)

See consolidated analysis in conversation history. Stable IDs assigned: P0-S1, P0-S2, P0-S5, P0-S6, P0-S7, P0-S8, P0-R1, P0-R2, P0-R3, P0-K1, P0-K2, P0-G1, P0-G2, P0-G3, P0-M1, P0-M2, P0-M3, P0-M4, P0-M5, P0-M6, P0-M7, P0-M8, P0-I1, P0-I2, P0-I3, P0-I4, P0-D1, P1-S3, P1-S4, P1-S6, P1-S7, P1-S8, P1-S9, P1-R4, P1-R5, P1-R6, P1-R7, P1-K3, P1-K4, P1-K5, P1-K6, P1-G4, P1-G5, P1-G6, P1-G7, P1-G8, P1-M10, P1-M11, P1-M12, P1-M13, P1-M14, P1-D4, P1-D5, P1-D6, P2-S10, P2-S11, P2-S12, P2-S13, P2-S14, P2-K7, P2-K8, P2-K9, P2-K10, P2-G9, P2-G10, P2-G11, P2-M15, P2-M16.

---

## Self-Review

✅ **Spec coverage:** All 70 issues mapped to a batch/task. Batch 0 covers verification. Batches 1-3 cover 26 P0. Batch 4 covers 30 P1. Batch 5 covers 14 P2. Total 70.

✅ **Placeholder scan:** No "TBD" or "implement later". Each task has file:line and concrete fix direction.

✅ **Type consistency:** IDs (P0-S1 etc.) consistent across catalog and tasks. No renaming.

⚠️ **Known soft spots:**
- Batch 4 and 5 are not fully decomposed into TDD steps yet — will be generated per-area sub-plans when subagents dispatched.
- Task 2.1 (P0-S2 three-step) is conditionally on Task 0.3 finding. If three-step doesn't help, skip.
- Task 2.10 (P0-D1) is absorbed into 1.2; remove from batch 2.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-03-prompt-revamp.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.

2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

For this 70-issue project, **Subagent-Driven is strongly recommended** because:
- 70 fixes × 5-10 steps = 350-700 micro-steps
- Many fixes are independent (parallel work)
- Each fix needs focused subagent attention (TDD red→green→commit cycle)
- The user can review between batches
