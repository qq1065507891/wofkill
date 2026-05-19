# Real Werewolf Game Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated Werewolf games follow high-quality 12-player real-game flow, with judge-controlled sheriff/PK phases, credible seer and wolf strategies, evidence-based votes, and auditable logs.

**Architecture:** Put hard game rules in runtime/rule-engine policy code, put soft player strategy in agent context and validators, and keep audit-only reasoning out of public timeline. The runtime graph owns phase order; strategy helpers summarize claims, vote logic, wolf plans, and pressure points for agents.

**Tech Stack:** Python, LangGraph runtime, Pydantic agent schemas, pytest, existing `werewolf_agent` modules.

---

## File Structure

- Modify: `werewolf_agent/runtime/graph.py`
  - Owns sheriff routing, day speech order, PK speech/revote, wolf discussion, and event emission.
- Modify: `werewolf_agent/runtime/agent_adapter.py`
  - Builds task-specific contexts for sheriff vote, PK speech, wolf discussion, day speech, day vote, and witch decisions.
- Modify: `werewolf_agent/engine/rule_engine.py`
  - Resolves sheriff election, sheriff PK, all-player-on-sheriff no-election, exile PK, sheriff vote weight, and legal targets.
- Modify: `werewolf_agent/agents/schemas.py`
  - Adds task/action fields for sheriff vote, PK speech, wolf tactical choice, vote basis, seer claim contract, and witch poison judgment.
- Modify: `werewolf_agent/agents/player.py`
  - Strengthens prompts and validation for required speech content, role commitments, vote reasoning, and tool-call outputs.
- Modify: `werewolf_agent/cognition/world_state.py`
  - Extracts claims, seer check chains, sheriff badge flow, vote stance, PK facts, and witch-relevant public pressure.
- Modify: `werewolf_agent/cognition/contradiction.py`
  - Adds claim-chain and vote-chain contradiction alerts.
- Create: `werewolf_agent/runtime/sheriff_policy.py`
  - Small focused helper for sheriff candidates, voters, all-on-sheriff detection, sheriff PK, and speech-order decisions.
- Create: `werewolf_agent/runtime/wolf_strategy.py`
  - Builds wolf discussion round requirements, consensus summaries, tactical opportunities, and plan evidence.
- Create: `werewolf_agent/runtime/vote_quality.py`
  - Validates vote reasons and extracts evidence basis.
- Create: `werewolf_agent/runtime/speech_quality.py`
  - Validates public speeches have stance, targets, protection logic, vote leaning, and evidence.
- Modify: `werewolf_agent/model_gateway/providers.py`
  - Enforces provider-level tool/function-call requirements and records structured-output metadata.
- Test: `tests/runtime/test_sheriff_policy.py`
- Test: `tests/runtime/test_pk_flow.py`
- Test: `tests/runtime/test_wolf_strategy.py`
- Test: `tests/runtime/test_vote_quality.py`
- Test: `tests/runtime/test_speech_quality.py`
- Test: `tests/agents/test_agents.py`
- Test: `tests/cognition/test_cognition.py`

---

## Task 1: Sheriff Election Policy

**Files:**
- Create: `werewolf_agent/runtime/sheriff_policy.py`
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/engine/rule_engine.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `tests/runtime/test_sheriff_policy.py`

- [ ] **Step 1: Write failing tests for all-player-on-sheriff no-election**

Test that when every alive player is a sheriff candidate and nobody withdraws to exactly one candidate, the result is `sheriff_no_election` with reason `all_players_on_sheriff`, and `sheriff_id` remains `None`.

Run: `pytest tests/runtime/test_sheriff_policy.py::test_all_players_on_sheriff_loses_badge -v`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 2: Write failing tests for real sheriff voters**

Test that only players who did not go on sheriff are eligible voters. Players who went on sheriff and then withdrew still cannot vote.

Run: `pytest tests/runtime/test_sheriff_policy.py::test_only_off_sheriff_players_vote -v`
Expected: FAIL.

- [ ] **Step 3: Implement `sheriff_policy.py`**

Add helpers:
- `eligible_sheriff_voters(gs, candidates, withdrew) -> list[str]`
- `is_all_players_on_sheriff(gs, candidates) -> bool`
- `resolve_no_vote_sheriff_reason(gs, candidates, voters) -> str`
- `choose_no_sheriff_speech_order(gs, seed) -> list[str]`
- `choose_sheriff_led_speech_order(gs, sheriff_id, focus_players, direction) -> list[str]`

Rules:
- All alive players on sheriff plus no single remaining candidate means no sheriff.
- No candidate means no sheriff.
- One remaining candidate after withdrawal becomes sheriff without vote.
- Existing off-sheriff voters choose among remaining candidates.

- [ ] **Step 4: Wire sheriff vote agent calls**

In `agent_adapter.py`, add `agent_sheriff_vote(...)` using `TaskType.SHERIFF_VOTE` and legal targets equal to remaining candidates.

In `graph.py`, update `sheriff_vote`:
- If one candidate remains, elect directly.
- If all players are on sheriff and more than one remains, emit `sheriff_no_election`.
- If eligible voters exist and no scripted votes were provided, call `agent_sheriff_vote` for each eligible voter.
- Emit audit traces privately.

- [ ] **Step 5: Run sheriff tests**

Run: `pytest tests/runtime/test_sheriff_policy.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: implement sheriff election policy`

---

## Task 2: Speech Order With And Without Sheriff

**Files:**
- Modify: `werewolf_agent/runtime/sheriff_policy.py`
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_sheriff_policy.py`

- [ ] **Step 1: Write failing tests for no-sheriff judge order**

Test that when no sheriff exists, the judge creates a deterministic random speech order from a seed and emits `speech_order_selected` with `reason="judge_no_sheriff"`.

Run: `pytest tests/runtime/test_sheriff_policy.py::test_no_sheriff_judge_selects_speech_start -v`
Expected: FAIL.

- [ ] **Step 2: Write failing tests for sheriff-led order**

Test that when sheriff exists and there are focus players such as counterclaim seers or a black-claimed player, speech order puts those focus players early and sheriff last for归票.

Run: `pytest tests/runtime/test_sheriff_policy.py::test_sheriff_places_counterclaims_early_and_self_last -v`
Expected: FAIL.

- [ ] **Step 3: Implement speech order selection**

Update `route_after_sheriff_vote` or the day-entry setup so `free_discussion` receives `speech_order`.

Rules:
- No sheriff: judge chooses deterministic start and clockwise/counterclockwise order.
- Sheriff: choose focus players early when present, remaining players by seat order, sheriff last.
- Event must include `speech_order`, `start_player`, `direction`, and `reason`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/runtime/test_sheriff_policy.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add judge and sheriff speech order policy`

---

## Task 3: PK Flow Only Lets Tied Players Speak

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `tests/runtime/test_pk_flow.py`

- [ ] **Step 1: Write failing test for exile PK speeches**

Test that after first exile tie, only `pk_candidates` receive `pk_speech` agent calls and `tie_pk_speech` events.

Run: `pytest tests/runtime/test_pk_flow.py::test_only_pk_candidates_speak_after_exile_tie -v`
Expected: FAIL because current `tie_pk_speech` emits an empty event.

- [ ] **Step 2: Write failing test for revote voters and targets**

Test that PK revote legal targets are only tied candidates, and voters exclude candidates if the selected rules require PK台下投票.

Run: `pytest tests/runtime/test_pk_flow.py::test_pk_revote_targets_only_tied_candidates -v`
Expected: FAIL if current target filtering is incomplete.

- [ ] **Step 3: Implement `agent_pk_speech`**

Add an adapter using `TaskType.PK_SPEECH`, legal action `SPEECH`, and context containing first vote tally plus prior accusations.

- [ ] **Step 4: Update `tie_pk_speech`**

Call only PK candidates, emit public `tie_pk_speech` per candidate, and private `action_trace_audit`.

- [ ] **Step 5: Update revote context**

Ensure `day_vote` receives:
- `revote=True`
- `pk_candidates`
- prior vote tally
- PK speech summary

- [ ] **Step 6: Run tests**

Run: `pytest tests/runtime/test_pk_flow.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: enforce PK speech and revote flow`

---

## Task 4: Wolf Night Discussion Must Produce Evidence-Based Consensus

**Files:**
- Create: `werewolf_agent/runtime/wolf_strategy.py`
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `tests/runtime/test_wolf_strategy.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for no silent wolf discussion**

Test that wolf discussion legal actions exclude `NO_ACTION`, and empty `speech` is rejected/retried.

Run: `pytest tests/runtime/test_wolf_strategy.py::test_wolf_discussion_requires_non_empty_speech -v`
Expected: FAIL.

- [ ] **Step 2: Write failing test for discussion-derived plan**

Use scripted wolf discussion outputs where wolves propose targets and agree/disagree. Assert `wolf_team_plan.evidence_from_discussion` references those statements and `wolf_kill_selected.reason` includes the consensus reason.

Run: `pytest tests/runtime/test_wolf_strategy.py::test_wolf_plan_is_derived_from_discussion_evidence -v`
Expected: FAIL because current plan is deterministic by seat order.

- [ ] **Step 3: Write failing test for early stop**

If a majority agrees on target and roles in round 1, later rounds are skipped and event `wolf_discussion_ended_early` is emitted.

Run: `pytest tests/runtime/test_wolf_strategy.py::test_wolf_discussion_can_end_early_after_consensus -v`
Expected: FAIL.

- [ ] **Step 4: Implement `wolf_strategy.py`**

Add:
- `round_requirements(night_number, round_number)`
- `extract_wolf_proposal(text)`
- `summarize_wolf_consensus(discussion_events, alive_wolves)`
- `build_wolf_team_plan_from_discussion(gs, previous_plan, consensus)`
- `should_end_discussion_early(consensus, alive_wolves)`

Plan fields:
- `fake_seer`
- `pusher`
- `hooker`
- `deep_cover`
- `night_kill_primary`
- `night_kill_backup`
- `day_push_target`
- `public_story`
- `rush_vote_opportunity`
- `evidence_from_discussion`
- `unresolved_disagreements`

- [ ] **Step 5: Update wolf discussion prompts**

In `player.py`, for `TaskType.WOLF_DISCUSSION` require:
- Night 1 round 1: suspected gods and seer/witch/hunter reads.
- Night 1 round 2: propose fake seer, pusher, hooker, deep cover.
- Night 1 round 3: agree/disagree with target, backup, next-day push.
- Later nights: review vote/speech/claim outcomes and adjust plan.

No wolf discussion response may say "night does not need speech" or return empty `speech`.

- [ ] **Step 6: Update wolf kill selection**

`wolf_consensus` must choose from the discussion-derived plan and emit:
- `target_id`
- `plan_key`
- `consensus_level`
- `reason`
- `supporting_wolves`

- [ ] **Step 7: Run tests**

Run: `pytest tests/runtime/test_wolf_strategy.py tests/agents/test_agents.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 8: Commit**

Commit message: `feat: derive wolf plans from night consensus`

---

## Task 5: Seer Claim Contract And Counterclaim Memory

**Files:**
- Modify: `werewolf_agent/cognition/world_state.py`
- Modify: `werewolf_agent/cognition/contradiction.py`
- Modify: `werewolf_agent/agents/schemas.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `tests/cognition/test_cognition.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for seer claim contract extraction**

Given speech "我是预言家，昨晚验 p01 查杀，警徽流 p05 p07", assert structured facts include claimed role, check result, badge flow, and pressure target.

Run: `pytest tests/cognition/test_cognition.py::test_extracts_seer_claim_contract -v`
Expected: FAIL if badge flow/check chain is missing.

- [ ] **Step 2: Write failing test for role commitment persistence**

If p01 claimed seer in sheriff speech, later p01 day speech context must include that commitment and contradiction alert if p01 says "等预言家跳出来".

Run: `pytest tests/cognition/test_cognition.py::test_seer_claim_commitment_detects_later_contradiction -v`
Expected: FAIL.

- [ ] **Step 3: Implement claim extraction**

Extract:
- `claimed_role`
- `seer_check_claim`
- `badge_flow`
- `black_claim`
- `gold_claim`
- `counterclaim_group`

- [ ] **Step 4: Implement contradiction detection**

Detect:
- player contradicts own claimed role
- claimed seer changes check result without explanation
- claimed seer lacks badge flow in sheriff speech
- player votes against claimed logic without explanation

- [ ] **Step 5: Strengthen seer and fake-seer prompts**

For real seer and fake seer:
- Must report check result.
- Must explain why checked target.
- Must give badge flow if in sheriff phase.
- Must respond to counterclaims and black claims.

- [ ] **Step 6: Run tests**

Run: `pytest tests/cognition/test_cognition.py tests/agents/test_agents.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: track seer claims and counterclaim commitments`

---

## Task 6: Evidence-Based Vote Quality

**Files:**
- Create: `werewolf_agent/runtime/vote_quality.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `tests/runtime/test_vote_quality.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for vote basis validator**

Test that a vote reason must include at least one basis from:
- seer check
- counterclaim
- badge flow
- contradiction
- vote tally
- stance reversal
- PK speech
- prior speech quote/summary

Run: `pytest tests/runtime/test_vote_quality.py::test_vote_reason_requires_logic_basis -v`
Expected: FAIL.

- [ ] **Step 2: Write failing test for full-day vote context**

Assert `agent_day_vote` context includes all current-day speeches or a full day summary, not just last six utterances.

Run: `pytest tests/runtime/test_vote_quality.py::test_vote_context_contains_full_day_discussion_summary -v`
Expected: FAIL.

- [ ] **Step 3: Implement `vote_quality.py`**

Add:
- `build_day_discussion_summary(gs)`
- `extract_vote_basis(action)`
- `validate_vote_reason(action, context)`
- `build_vote_pressure_context(gs, voter_id, pk_candidates=None)`

- [ ] **Step 4: Integrate vote validation**

If an LLM vote has no valid logic basis, retry with correction hint. If retries fail, fallback must include a concrete basis from known facts.

- [ ] **Step 5: Add wolf rush as optional strategy**

Expose `rush_vote_opportunity` to wolves, but never force it. Agent must record:
- `wolf_rush_considered`
- `chosen_vote_strategy`: `rush`, `hook`, `split`, `abandon_teammate`, or `conservative`
- `reason`

- [ ] **Step 6: Run tests**

Run: `pytest tests/runtime/test_vote_quality.py tests/agents/test_agents.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: require evidence-based voting`

---

## Task 7: Witch Poison Pressure Policy

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/cognition/world_state.py`
- Test: `tests/agents/test_agents.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing test for witch high-pressure context**

Given public state where a seer black claim survived or an exposed god created a clear wolf candidate, assert witch night context contains `poison_pressure_targets`.

Run: `pytest tests/runtime/test_runtime.py::test_witch_context_includes_poison_pressure_targets -v`
Expected: FAIL.

- [ ] **Step 2: Write failing test for no-poison explanation**

If witch has poison and selects `no_action` while pressure targets exist, reason must explicitly explain why not poisoning those targets.

Run: `pytest tests/agents/test_agents.py::test_witch_no_poison_must_explain_pressure_targets -v`
Expected: FAIL.

- [ ] **Step 3: Implement witch pressure context**

Build pressure from:
- unresolved black claim
- self-exposed wolf-like claim
- vote bloc against seer
- player contradicted claimed role
- player pushed dead true seer after reveal if role reveal is available in test context

- [ ] **Step 4: Update witch prompt and validation**

Do not force poison. Require explicit evaluation:
- poison target
- reason to poison
- reason to hold poison
- risk of poisoning god/good player

- [ ] **Step 5: Run tests**

Run: `pytest tests/agents/test_agents.py tests/runtime/test_runtime.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add witch poison pressure reasoning`

---

## Task 8: Judge Broadcast And Audit Observability

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `scripts/print_game_audit.py`
- Test: `tests/test_game_audit.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing test for judge phase broadcasts**

Assert game events include judge-visible broadcasts for:
- enter night
- wolf discussion round
- seer action
- witch action
- hunter status
- day announce
- sheriff election
- speech order
- vote
- PK
- exile

Run: `pytest tests/runtime/test_runtime.py::test_judge_broadcasts_major_phases -v`
Expected: FAIL for missing broadcasts.

- [ ] **Step 2: Write failing test for audit output**

Assert audit output prints public text, private reasoning summary, action trace, retry/fallback reason, vote tally, and real target IDs.

Run: `pytest tests/test_game_audit.py::test_audit_report_includes_private_and_public_sections -v`
Expected: FAIL if missing sections.

- [ ] **Step 3: Implement event payloads**

Every major event must include enough payload to audit:
- actor
- target
- phase
- day/night
- tally when relevant
- reason when relevant
- visibility

Private intent stays in `action_trace_audit` only.

- [ ] **Step 4: Update audit script**

Print sections:
- Judge timeline
- Public speeches
- Wolf private chat
- Wolf plan evidence
- Votes and vote basis
- Witch/seer/hunter private actions
- Fallback/retry summary

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_game_audit.py tests/runtime/test_runtime.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: improve judge and audit observability`

---

## Task 9: Strict Tool-Call Structured Output

**Files:**
- Modify: `werewolf_agent/model_gateway/providers.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/agents/schemas.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for plain text rejection**

When `tool_choice` is set, provider output without an actual tool/function call must fail immediately instead of parsing text JSON.

Run: `pytest tests/agents/test_agents.py::test_plain_text_rejected_when_tool_call_required -v`
Expected: FAIL if plain text is still accepted.

- [ ] **Step 2: Write failing test for provider capability failure**

If a provider cannot enforce tool/function calls, agent action should fail with a clear `structured_output_unsupported` error rather than silently falling back.

Run: `pytest tests/agents/test_agents.py::test_provider_without_tool_call_support_fails_explicitly -v`
Expected: FAIL.

- [ ] **Step 3: Add structured output metadata to trace**

Extend action trace with:
- `tool_call_required`
- `tool_call_received`
- `tool_call_name`
- `parse_success`
- `parse_error`
- `retry_count`
- `structured_failure_reason`

- [ ] **Step 4: Enforce tool-call-only parsing**

In `PlayerAgent.act`, when tools are requested:
- accept only JSON arguments from the tool/function call;
- reject normal assistant text even if it contains valid JSON;
- retry on missing tool call;
- after max retries, emit fallback only for recoverable model mistakes, not provider capability failures.

- [ ] **Step 5: Run tests**

Run: `pytest tests/agents/test_agents.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: enforce strict tool-call structured output`

---

## Task 10: Judge-Controlled Night And Day Broadcasts

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/agents/judge.py`
- Test: `tests/runtime/test_runtime.py`
- Test: `tests/test_game_audit.py`

- [ ] **Step 1: Write failing test for night judge broadcasts**

Assert the event stream contains judge broadcasts before and after each night role stage:
- "天黑请闭眼"
- "狼人请睁眼"
- "狼人请统一刀人"
- "狼人请闭眼"
- "预言家请睁眼"
- "预言家请闭眼"
- "女巫请睁眼"
- "女巫请闭眼"
- "猎人状态确认"

Run: `pytest tests/runtime/test_runtime.py::test_judge_controls_night_role_sequence -v`
Expected: FAIL if broadcasts are missing.

- [ ] **Step 2: Write failing test for day judge broadcasts**

Assert day flow broadcasts include:
- daybreak
- death announcement
- sheriff election start/result
- speech order selection
- vote start
- vote result
- PK start/result
- exile result

Run: `pytest tests/runtime/test_runtime.py::test_judge_controls_day_sequence -v`
Expected: FAIL.

- [ ] **Step 3: Implement judge broadcast helper**

Add a small helper in `graph.py` that appends `judge_broadcast` events with:
- `phase`
- `message`
- `day_number`
- `night_number`
- `visibility="public"` or role-private visibility where appropriate
- relevant target/result payload.

- [ ] **Step 4: Wire broadcasts around existing nodes**

Do not change game rules in this task. Only add observable judge control around each existing node.

- [ ] **Step 5: Run tests**

Run: `pytest tests/runtime/test_runtime.py tests/test_game_audit.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add judge-controlled phase broadcasts`

---

## Task 11: Public Speech Quality Validator

**Files:**
- Create: `werewolf_agent/runtime/speech_quality.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `tests/runtime/test_speech_quality.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for filler speech rejection**

Speech such as "再观察一下" or "信息太少，先听后面" must fail validation unless it includes required logic fields.

Run: `pytest tests/runtime/test_speech_quality.py::test_rejects_filler_day_speech -v`
Expected: FAIL.

- [ ] **Step 2: Write failing test for required public speech components**

A valid day speech must include:
- identity perspective or claimed role stance;
- at least one suspicion target;
- at least one protected/trusted target or reason for not protecting anyone;
- vote leaning;
- at least one evidence basis.

Run: `pytest tests/runtime/test_speech_quality.py::test_day_speech_requires_stance_targets_and_evidence -v`
Expected: FAIL.

- [ ] **Step 3: Write failing test for stronger sheriff/PK/seer speech**

Sheriff, PK, seer, and fake-seer speeches must additionally include:
- side choice or counterclaim response;
- attack point or defense point;
- check chain or badge flow when claiming seer.

Run: `pytest tests/runtime/test_speech_quality.py::test_high_pressure_speech_requires_claim_logic -v`
Expected: FAIL.

- [ ] **Step 4: Implement `speech_quality.py`**

Add:
- `extract_speech_quality(text, phase)`
- `validate_public_speech(text, phase, context)`
- `build_speech_retry_hint(missing_fields)`
- `fallback_speech_with_basis(context)`

- [ ] **Step 5: Integrate speech retries**

In day/sheriff/PK speech adapters, retry if speech quality is insufficient. Fallback speech must still name targets and evidence.

- [ ] **Step 6: Run tests**

Run: `pytest tests/runtime/test_speech_quality.py tests/agents/test_agents.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: validate public speech quality`

---

## Task 12: Contradiction Alerts Must Be Answered

**Files:**
- Modify: `werewolf_agent/cognition/context.py`
- Modify: `werewolf_agent/cognition/contradiction.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `tests/cognition/test_cognition.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for contradiction context priority**

If a player self-exposes as wolf, contradicts a seer claim, reverses stance, or votes against their stated logic, the next player's context must include a high-priority contradiction alert.

Run: `pytest tests/cognition/test_cognition.py::test_high_priority_contradiction_reaches_next_player_context -v`
Expected: FAIL if alerts are omitted or low priority.

- [ ] **Step 2: Write failing test for required contradiction response**

When contradiction alerts exist, a player's speech must do one of:
- question the contradiction;
- take a side;
- explain why they are parking it temporarily.

Run: `pytest tests/agents/test_agents.py::test_speech_must_answer_visible_contradiction_alert -v`
Expected: FAIL.

- [ ] **Step 3: Add contradiction response directive**

Build a context field `must_address_alerts` containing top alerts with:
- alert type;
- involved players;
- public evidence;
- required response modes.

- [ ] **Step 4: Add validation and retry**

Speech validation must fail when high-priority alerts are ignored. Retry hint should name the ignored alert.

- [ ] **Step 5: Run tests**

Run: `pytest tests/cognition/test_cognition.py tests/agents/test_agents.py -q --basetemp .pytest-tmp`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: require players to address contradictions`

---

## Task 13: Integration Regression Suite

**Files:**
- Modify: `tests/runtime/test_runtime.py`
- Modify: `tests/agents/test_agents.py`
- Modify: `tests/cognition/test_cognition.py`
- Test: existing focused test suite

- [ ] **Step 1: Run focused suite**

Run:
`pytest tests/runtime/test_sheriff_policy.py tests/runtime/test_pk_flow.py tests/runtime/test_wolf_strategy.py tests/runtime/test_vote_quality.py tests/runtime/test_speech_quality.py tests/runtime/test_runtime.py tests/agents/test_agents.py tests/cognition/test_cognition.py tests/test_game_audit.py -q --basetemp .pytest-tmp`

Expected: PASS.

- [ ] **Step 2: Run broader suite**

Run:
`pytest tests/api tests/rules tests/evaluation tests/runtime tests/agents tests/cognition tests/test_game_audit.py -q --basetemp .pytest-tmp`

Expected: PASS, except known unrelated dirty log formatting is not part of pytest.

- [ ] **Step 3: Run diff check on touched code**

Run:
`git diff --check -- werewolf_agent tests scripts docs/superpowers/plans/2026-05-19-real-werewolf-game-quality.md`

Expected: PASS.

- [ ] **Step 4: Commit**

Commit message: `test: add real game quality regression coverage`

---

## Task 14: Real LLM Game Acceptance

**Files:**
- No required source changes unless tests reveal issues.
- Output: `game_stdout.log`, `game_g_<seed>.json`, audit markdown.

- [ ] **Step 1: Run one fixed-seed real game**

Use the existing project command for real LLM game startup with a fixed seed, for example seed `42` or the current acceptance seed.

Expected:
- Structure output success rate 100%.
- Fallback count 0.
- Every action uses provider-enforced tool/function calls.
- Game normally ends in 3-5 days; extreme target no more than 6 days.

- [ ] **Step 2: Inspect acceptance criteria**

Check:
- First night wolf discussion has no silent wolves unless discussion ended early by explicit consensus.
- Wolf plan cites discussion evidence.
- If seer black-claims a wolf, all voters explicitly address that black claim.
- If wolf fake-seer exists, it maintains its check chain.
- Sheriff exists unless all players went on sheriff or second sheriff tie occurs.
- No sheriff case has judge-selected speech order.
- PK only has tied players speaking.
- Votes cite concrete logic basis.
- Public speeches include stance, suspicion/protection target, vote leaning, and evidence.
- High-priority contradictions are addressed by later players.
- Judge broadcasts every major night/day phase.
- Witch explains poison or no-poison decision against pressure targets.
- Wolves may consider rush vote but are not forced to rush.

- [ ] **Step 3: Generate audit report**

Run the audit script and inspect the sections listed in Task 8.

- [ ] **Step 4: Fix any acceptance failure with new failing test first**

For each issue discovered, write the smallest failing test in the relevant test file, then implement the fix.

- [ ] **Step 5: Final verification**

Run focused suite again and record the commands/results in the final implementation summary.

- [ ] **Step 6: Commit**

Commit message: `chore: verify real werewolf game quality`

---

## Acceptance Criteria

- Sheriff election follows real rules:
  - Off-sheriff voters elect sheriff.
  - All-player-on-sheriff causes badge loss unless withdrawal leaves one candidate.
  - No sheriff means judge chooses speech order.
  - Sheriff means sheriff controls speech order and归票 placement.
- PK follows real rules:
  - Only tied players speak.
  - Revote targets are only PK candidates.
  - Second tie causes no exile or badge loss depending phase.
- Wolf night discussion is meaningful:
  - No empty/no-action wolf speeches in required discussion rounds.
  - Plan is derived from discussion evidence.
  - Discussion can end early only after explicit consensus.
- Seer and fake-seer behavior is coherent:
  - Check result, badge flow, and reason are tracked.
  - Claimed seer cannot forget or contradict its own claim without alert.
- Voting is evidence based:
  - Every vote cites a concrete logic basis.
  - Vote context includes complete current-day discussion summary.
  - Wolves see rush opportunities as optional, not mandatory.
- Public speech quality is enforced:
  - Day speech cannot be empty or filler.
  - Speech includes stance, suspicion/protection target, vote leaning, and evidence.
  - Sheriff, PK, seer, and fake-seer speeches include stronger claim logic.
- Contradictions must be answered:
  - Claim conflicts, vote conflicts, stance reversals, and self-exposure alerts reach later players.
  - Later players must question, side with or against, or explicitly park the contradiction.
- Witch decisions are explainable:
  - Poison pressure targets are surfaced.
  - Holding poison requires explicit reasoning.
- LLM output is strictly structured:
  - Tool/function call is required for every agent action.
  - Plain text JSON is rejected when tool calls are required.
  - Unsupported providers fail explicitly instead of silently degrading.
- Judge controls the flow:
  - Night and day phases have explicit judge broadcasts.
  - Broadcast payloads contain real targets, tallies, and phase metadata when public.
- Logs are审计able:
  - Judge timeline, public speech, private audit, vote tally, real targets, retries, and fallback reasons are visible.
  - Public timeline does not leak private role intent.

---

## Execution Options

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, faster and cleaner for this multi-module change.
2. **Inline Execution** - execute tasks in this session with checkpoints after each task group.
