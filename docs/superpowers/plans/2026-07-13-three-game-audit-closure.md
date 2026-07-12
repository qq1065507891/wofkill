# Three-Game Audit Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every issue found in the 2026-07-12 three-game audit, with deterministic terminal behavior, verifiable LLM reasoning execution, lower safe fallback rates, evidence-grounded context, and privacy-safe observability.

**Architecture:** Keep the existing rule engine and LangGraph runtime, but introduce narrow contracts at five boundaries: death-trigger completion before terminal commit, per-attempt model execution evidence, schema/semantic normalization before fallback, evidence-gated cognition and reflection, and final-prompt exposure auditing. Each boundary owns its events and metrics so reporting cannot silently reinterpret runtime behavior.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph runtime, pytest, YAML model routing, provider-compatible OpenAI/Anthropic HTTP adapters.

---

## File responsibility map

- `werewolf_agent/runtime/graph.py`, `graph_registration.py`: route death reactions, terminal checks, last words, and next phases.
- `werewolf_agent/runtime/nodes/day_deaths.py`, `day_finish.py`, `skills.py`: apply death-related events and atomically commit terminal state.
- `werewolf_agent/runtime/nodes/_shared.py` and the existing dispatch helper it exposes: reject in-game Agent calls after terminal state.
- `werewolf_agent/model_gateway/usage_records.py`: immutable per-attempt model execution schema and normalized reasoning evidence.
- `werewolf_agent/model_gateway/router.py`, `router_selection.py`, provider adapters: enforce task reasoning policy and preserve all attempts.
- `werewolf_agent/agents/player_generation.py`, `runtime/nodes/action_audit.py`: attach privacy-safe model execution evidence to each action.
- `scripts/run_real_game.py`, `werewolf_agent/evaluation/balance_audit.py`: aggregate exact reasoning/fallback/terminal metrics.
- `werewolf_agent/agents/wolf_team_plan_schema.py`, `runtime/wolf_team_plan_support.py` and the current wolf-plan node: shared contract, deterministic normalization, and separate fallback outcomes.
- `werewolf_agent/runtime/speech_quality.py`, `werewolf_agent/evaluation/balance_public_claims.py`: classify public facts and perform invariant-preserving semantic repair.
- Hunter/witch directive modules and their Agent action builders: require alternative comparison and friendly-fire risk evidence.
- `werewolf_agent/cognition/worlds.py`: canonical world identity, deduplication, and evidence-reference gates.
- `werewolf_agent/memory/reflection_synthesis.py`, `reflection_sanitization.py`, `reflection_repository.py`: structured reflection claims and persistence gate.
- `werewolf_agent/agents/prompt_persona.py`, `player_persona.py`, `runtime/exposure_audit.py`: privacy-safe final system-message Persona evidence.

## Shared test command prefix

All pytest commands use a writable temp directory and disable external tracing:

```powershell
$env:LANGCHAIN_TRACING_V2='false'
$env:LANGSMITH_TRACING='false'
$env:TMP='E:\NLP\agent\wofkill\.tmp'
$env:TEMP=$env:TMP
python -m pytest --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

### Task 0: Compatibility characterization and authoritative outcome taxonomy

**Files:**
- Create: `werewolf_agent/model_gateway/execution_records.py`
- Create: `werewolf_agent/runtime/decision_outcomes.py`
- Modify: `werewolf_agent/model_gateway/usage_records.py`
- Test: `tests/model_gateway/test_execution_records.py`
- Test: `tests/model_gateway/test_router_split_helpers.py`
- Test: `tests/runtime/test_decision_outcomes.py`
- Test: `tests/runtime/test_agent_adapter.py`

- [ ] **Step 1: Add compatibility characterization tests**

Add `test_usage_record_legacy_fields_remain_readable`, `test_generate_result_legacy_reasoning_fields_remain_readable`, `test_provider_package_exports_remain_stable`, `test_agent_adapter_reexports_dispatch_helpers`, and `test_shared_agent_timeouts_remain_importable`.

- [ ] **Step 2: Run compatibility tests and verify GREEN before migration**

```powershell
python -m pytest tests/model_gateway/test_execution_records.py::test_usage_record_legacy_fields_remain_readable tests/model_gateway/test_execution_records.py::test_generate_result_legacy_reasoning_fields_remain_readable tests/model_gateway/test_execution_records.py::test_provider_package_exports_remain_stable tests/runtime/test_agent_adapter.py::test_agent_adapter_reexports_dispatch_helpers tests/runtime/test_agent_adapter.py::test_shared_agent_timeouts_remain_importable -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected: characterization tests pass against the pre-migration API or reveal the exact compatibility surface that must be preserved.

- [ ] **Step 3: Add RED table tests for the authoritative taxonomy**

Add `test_outcome_taxonomy_is_mutually_exclusive` and `test_retry_semantics_for_repair_provider_fallback_and_terminal_fallback`. Expected RED: `decision_outcomes` does not exist and current fields conflate fallback transition with root cause.

- [ ] **Step 4: Implement minimal enums and event translator**

Define `RootCause`, `AttemptOutcome`, `DecisionOutcome`, and one translator that owns attempt/retry/final-outcome semantics. Add per-attempt record types while retaining read-only legacy properties on `UsageRecord` and `GenerateResult`.

- [ ] **Step 5: Run GREEN and compatibility regression**

```powershell
python -m pytest tests/model_gateway/test_execution_records.py tests/model_gateway/test_router_split_helpers.py tests/runtime/test_decision_outcomes.py tests/runtime/test_agent_adapter.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

- [ ] **Step 6: Commit exact files**

```powershell
git add werewolf_agent/model_gateway/execution_records.py werewolf_agent/model_gateway/usage_records.py werewolf_agent/runtime/decision_outcomes.py tests/model_gateway/test_execution_records.py tests/model_gateway/test_router_split_helpers.py tests/runtime/test_decision_outcomes.py tests/runtime/test_agent_adapter.py
git commit -m "refactor: establish decision outcome contracts"
```

### Task 1: P0 death-trigger completion and atomic victory

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/runtime/graph_registration.py`
- Modify: `werewolf_agent/runtime/nodes/day_deaths.py`
- Modify: `werewolf_agent/runtime/nodes/day_finish.py`
- Modify: `werewolf_agent/runtime/nodes/night_resolution.py`
- Modify if required by the established dispatch boundary: `werewolf_agent/runtime/nodes/_shared.py`
- Test: `tests/runtime/test_day_deaths.py`
- Test: `tests/runtime/test_hunter_flow.py`
- Test: `tests/runtime/test_game_runner.py`
- Test: `tests/integration/test_live_game_flow.py`

- [ ] **Step 1: Write the failing parity-after-exile test**

Create a state with three alive wolves and two alive good players, exile one good player, and assert the next route is the victory gate without invoking `exile_last_words`.

- [ ] **Step 2: Run the test and verify RED**

Add `test_exile_parity_routes_to_victory_before_last_words` in `tests/runtime/test_day_deaths.py`, then run:

```powershell
python -m pytest tests/runtime/test_day_deaths.py::test_exile_parity_routes_to_victory_before_last_words -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: current route is `exile_last_words` or the last-words registry is called once.

- [ ] **Step 3: Write failing death-trigger ordering tests**

Add `test_pending_hunter_shot_precedes_victory_commit`, `test_no_shot_hunter_checks_victory_once`, `test_last_wolf_exile_skips_last_words`, and `test_terminal_state_is_committed_at_step_boundary`. Also add `test_night_resolution_commits_victory_before_day_agents` in `tests/runtime/test_night_flow.py` and `test_poison_resolution_commits_victory_after_forced_death_reactions` in `tests/runtime/test_witch_flow.py`. Assert exactly one `victory`, no premature winner, and no in-game action trace after terminal commit.

- [ ] **Step 4: Run the new tests and verify RED for the intended ordering**

```powershell
python -m pytest tests/runtime/test_hunter_flow.py::test_pending_hunter_shot_precedes_victory_commit tests/runtime/test_hunter_flow.py::test_no_shot_hunter_checks_victory_once tests/runtime/test_day_deaths.py::test_last_wolf_exile_skips_last_words tests/runtime/test_game_runner.py::test_terminal_state_is_committed_at_step_boundary tests/runtime/test_night_flow.py::test_night_resolution_commits_victory_before_day_agents tests/runtime/test_witch_flow.py::test_poison_resolution_commits_victory_after_forced_death_reactions -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: at least exile routes through last words; night/poison paths allow a later phase or do not prove zero Agent calls.

- [ ] **Step 5: Implement a single pending-death-trigger route**

Route `resolve_exile` to pending hunter resolution when required; otherwise route to `check_victory`. Route a no-winner result to last words or the next phase, and a winner to reflection/finish. Do not duplicate victory calculation outside `RuleEngine.check_victory`.

- [ ] **Step 6: Add a dispatch defense guard**

At the common player-Agent dispatch boundary, reject in-game tasks when `winning_faction` is set. Allow only explicitly tagged post-game reflection.

- [ ] **Step 7: Run terminal, hunter, vote, sheriff, and integration tests**

```powershell
python -m pytest tests/runtime/test_day_deaths.py tests/runtime/test_hunter_flow.py tests/runtime/test_night_flow.py tests/runtime/test_witch_flow.py tests/runtime/test_graph_lifecycle.py tests/runtime/test_vote_flow.py tests/runtime/test_sheriff_flow.py tests/runtime/test_game_runner.py tests/integration/test_live_game_flow.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected GREEN: all processes exit 0; no post-terminal game action trace.

- [ ] **Step 8: Commit**

```powershell
git add werewolf_agent/runtime/graph.py werewolf_agent/runtime/graph_registration.py werewolf_agent/runtime/nodes/day_deaths.py werewolf_agent/runtime/nodes/day_finish.py werewolf_agent/runtime/nodes/night_resolution.py werewolf_agent/runtime/nodes/_shared.py tests/runtime/test_day_deaths.py tests/runtime/test_hunter_flow.py tests/runtime/test_night_flow.py tests/runtime/test_witch_flow.py tests/runtime/test_game_runner.py tests/integration/test_live_game_flow.py
git commit -m "fix: commit victory before post-exile agent calls"
```

### Task 2: P1 central reasoning policy and per-attempt execution evidence

**Files:**
- Modify: `config/models.yaml`
- Modify: `werewolf_agent/model_gateway/usage_records.py`
- Create: `werewolf_agent/model_gateway/reasoning_policy.py`
- Modify: `werewolf_agent/model_gateway/router_selection.py`
- Modify: `werewolf_agent/model_gateway/router.py`
- Modify: `werewolf_agent/model_gateway/providers/__init__.py`
- Modify: `werewolf_agent/model_gateway/providers/openai.py`
- Modify: `werewolf_agent/model_gateway/providers/anthropic.py`
- Modify: `werewolf_agent/model_gateway/providers/minimax.py`
- Modify: `werewolf_agent/model_gateway/providers/glm.py`
- Modify: `werewolf_agent/agents/player_generation.py`
- Modify: `werewolf_agent/runtime/agent_wolf_actions.py`
- Modify: `werewolf_agent/runtime/nodes/summary.py`
- Modify: `werewolf_agent/runtime/nodes/action_audit.py`
- Modify: `scripts/run_real_game.py`
- Test: `tests/model_gateway/test_reasoning_policy.py`
- Test: `tests/model_gateway/test_openai.py`
- Test: `tests/model_gateway/test_anthropic_provider.py`
- Test: `tests/model_gateway/test_providers.py`
- Test: `tests/model_gateway/test_router.py`
- Test: `tests/agents/test_model_router.py`
- Test: `tests/runtime/test_agent_wolf_team_plan.py`
- Test: `tests/runtime/test_context.py`
- Test: `tests/runtime/test_reflection_prompt.py`
- Test: `tests/runtime/test_agent_action_audit.py`
- Test: `tests/scripts/test_run_real_game.py` or the existing runner-report test module

- [ ] **Step 1: Write parameterized failing reasoning-policy tests**

In `tests/model_gateway/test_reasoning_policy.py`, add `test_every_player_task_has_minimum_reasoning_level`, `test_low_profile_fails_with_profile_name`, and `test_missing_profile_reasoning_fails_with_profile_name`. Enumerate ordinary speech/last words/votes at least medium; seer, witch, hunter, wolf plan, deception and reflection high; deterministic judge/rule work none. In `tests/runtime/test_agent_wolf_team_plan.py`, `tests/runtime/test_context.py`, and `tests/runtime/test_reflection_prompt.py`, add exact tests for direct wolf-plan attempts, LLM cognition-helper attempts and reflection dispatch.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/model_gateway/test_reasoning_policy.py tests/runtime/test_agent_wolf_team_plan.py::test_wolf_plan_attempt_enforces_high_reasoning tests/runtime/test_context.py::test_llm_cognition_helper_enforces_task_reasoning tests/runtime/test_reflection_prompt.py::test_reflection_dispatch_enforces_high_reasoning -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: no central task policy exists, missing/low profile configuration does not fail with the offending profile, or direct subcalls bypass policy.

- [ ] **Step 3: Implement the central policy and fail-closed router checks**

Resolve the required level from task type for every attempt and validate all player profiles in `config/models.yaml` at startup. Extend the routing configuration to an ordered fallback chain while preserving the legacy single-fallback YAML form. Reject a player attempt whose resolved model is below the minimum; try successive reasoning-capable fallbacks, then return an explicit safe failure if exhausted.

- [ ] **Step 4: Write failing per-attempt schema tests**

Define expected `ModelAttemptRecord` fields: opaque request ID, ordinal, provider/model, route kind, root cause, attempt outcome, requested level, normalized status, token count and evidence kind. Assert earlier failures are not overwritten by final success.

- [ ] **Step 5: Implement attempt sequence recording**

Keep `UsageRecord` compatibility accessors only where existing callers require them; make the attempt sequence the source of truth. Define `attempt_count`, `retry_count`, terminal fallback null semantics, and provider transition outcomes exactly as the spec.

- [ ] **Step 6: Tighten provider confirmation rules**

OpenAI confirms only positive reasoning tokens or an authoritative executed status. Anthropic confirms only positive reasoning tokens or a non-empty provider thinking block documented as executed reasoning. Add `test_minimax_reasoning_evidence_is_normalized_without_content` and `test_glm_reasoning_evidence_is_normalized_without_content`; either adapt their shared compatible response or explicitly return `requested_unconfirmed`/`unsupported`. Empty/echoed fields remain `requested_unconfirmed`. Never export thinking content.

- [ ] **Step 7: Write failing audit privacy and linkage tests**

Assert action audit contains allowlisted attempt summaries, opaque run-scoped IDs, no raw provider request ID, prompt, response body, raw error or hidden thinking.

- [ ] **Step 8: Implement action linkage and runner aggregation**

Export exact counts/rates, including null plus `supported=false` on zero denominators. Add `reasoning_confirmation_rate`, critical-task request coverage and `reasoning_fallback_keep_rate`.

- [ ] **Step 9: Run model-gateway, Agent-router, direct-subcall, action-audit and runner tests**

```powershell
python -m pytest tests/model_gateway tests/agents/test_model_router.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_context.py tests/runtime/test_reflection_prompt.py tests/runtime/test_agent_action_audit.py tests/scripts/test_run_real_game.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected GREEN: all exit 0; every player attempt class has an asserted level and every provider returns a normalized evidence status.

- [ ] **Step 10: Commit**

```powershell
git add config/models.yaml werewolf_agent/model_gateway/usage_records.py werewolf_agent/model_gateway/execution_records.py werewolf_agent/model_gateway/reasoning_policy.py werewolf_agent/model_gateway/router_selection.py werewolf_agent/model_gateway/router.py werewolf_agent/model_gateway/providers/__init__.py werewolf_agent/model_gateway/providers/openai.py werewolf_agent/model_gateway/providers/anthropic.py werewolf_agent/model_gateway/providers/minimax.py werewolf_agent/model_gateway/providers/glm.py werewolf_agent/agents/player_generation.py werewolf_agent/runtime/agent_wolf_actions.py werewolf_agent/runtime/nodes/summary.py werewolf_agent/runtime/nodes/action_audit.py scripts/run_real_game.py tests/model_gateway/test_reasoning_policy.py tests/model_gateway/test_openai.py tests/model_gateway/test_anthropic_provider.py tests/model_gateway/test_providers.py tests/model_gateway/test_router.py tests/agents/test_model_router.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_context.py tests/runtime/test_reflection_prompt.py tests/runtime/test_agent_action_audit.py tests/scripts/test_run_real_game.py
git commit -m "feat: prove reasoning execution per model attempt"
```

### Task 3: P1 wolf-plan contract normalization and fallback metrics

**Files:**
- Modify: `werewolf_agent/agents/wolf_team_plan_schema.py`
- Modify: `werewolf_agent/runtime/wolf_team_plan_support.py`
- Modify: `werewolf_agent/runtime/agent_wolf_actions.py`
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Modify: `scripts/run_real_game.py`
- Test: `tests/agents/test_wolf_team_plan_schema.py`
- Test: `tests/runtime/test_wolf_team_plan_support.py`
- Test: `tests/runtime/test_agent_wolf_team_plan.py`
- Test: `tests/integration/test_wolf_team_plan_e2e.py`
- Test: `tests/evaluation/test_game_balance_batch.py`

- [ ] **Step 1: Add RED cases from the three real games**

Cover a sole `night_plan` wrapper, a sole `plan` wrapper, reasoning longer than 200 Unicode characters, missing `public_story` with otherwise valid evidence, illegal target, duplicate role assignment, and missing target evidence.

- [ ] **Step 2: Verify current parser produces schema fallback for repairable cases**

```powershell
python -m pytest tests/agents/test_wolf_team_plan_schema.py::test_unwraps_single_night_plan_envelope tests/agents/test_wolf_team_plan_schema.py::test_unicode_reasoning_is_truncated_before_validation tests/runtime/test_agent_wolf_team_plan.py::test_missing_public_story_is_normalized_without_target_evidence_changes -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: repairable payloads end in `schema_validation_failed`.

- [ ] **Step 3: Implement deterministic normalization before validation**

Unwrap only one recognized wrapper, truncate only the bounded reasoning field, and synthesize only the neutral public-story field. Preserve target/evidence byte-for-byte. Illegal targets or missing evidence still fail.

- [ ] **Step 4: Add table-driven outcome/metric tests**

Verify schema terminal fallback, strategy terminal fallback, normalization success, total plans, exact denominators, and null unsupported values.

- [ ] **Step 5: Implement shared contract rendering**

Generate prompt/tool field requirements from the Pydantic contract or one shared descriptor so prompt and schema cannot drift.

- [ ] **Step 6: Run all wolf-flow and balance tests**

```powershell
python -m pytest tests/agents/test_wolf_team_plan_schema.py tests/runtime/test_wolf_team_plan_support.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_wolf_flow.py tests/integration/test_wolf_team_plan_e2e.py tests/evaluation/test_game_balance_batch.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/agents/wolf_team_plan_schema.py werewolf_agent/runtime/wolf_team_plan_support.py werewolf_agent/runtime/agent_wolf_actions.py werewolf_agent/evaluation/balance_audit.py scripts/run_real_game.py tests/agents/test_wolf_team_plan_schema.py tests/runtime/test_wolf_team_plan_support.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_wolf_flow.py tests/integration/test_wolf_team_plan_e2e.py tests/evaluation/test_game_balance_batch.py
git commit -m "fix: normalize repairable wolf plans before fallback"
```

### Task 4: P1 public-claim semantic repair and power-role evidence

**Files:**
- Modify: `werewolf_agent/runtime/speech_quality.py`
- Modify: `werewolf_agent/evaluation/balance_public_claims.py`
- Modify: `werewolf_agent/agents/player_fallback_speech.py`
- Modify: `werewolf_agent/runtime/directives/hunter.py`
- Modify: `werewolf_agent/runtime/strategy/hunter.py`
- Modify: `werewolf_agent/runtime/directives/witch.py`
- Modify: `werewolf_agent/runtime/strategy/witch.py`
- Test: `tests/runtime/test_speech_quality.py`
- Test: `tests/evaluation/test_game_balance_batch.py`
- Test: `tests/runtime/test_hunter_shot_directives.py`
- Test: `tests/runtime/test_witch_flow.py`
- Test: `tests/agents/test_player_agent.py`

- [ ] **Step 1: Add RED semantic cases**

Reproduce the p11 statement that correctly attributes p05's public claim. Add mixed valid/invalid claims, negation, attribution, unsupported system fact, target preservation, zero-valid-claim fallback and generic-template rejection.

- [ ] **Step 2: Verify RED against current quality validator**

```powershell
python -m pytest tests/runtime/test_speech_quality.py::test_attributed_public_claim_is_not_rejected_as_system_fact tests/runtime/test_speech_quality.py::test_mixed_claim_repair_preserves_target_negation_and_valid_claim tests/agents/test_player_agent.py::test_zero_valid_claim_fallback_is_target_specific_without_new_facts -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: the attributed p11-style statement is rejected, repair loses an invariant, or the fallback uses the generic information-insufficient template.

- [ ] **Step 3: Implement typed public-claim classification**

Represent system facts, player claims and current-player inference separately. Match attributed statements against public transcript; do not require player claims to be system-confirmed facts.

- [ ] **Step 4: Implement invariant-preserving repair**

Allow only attribution insertion, certainty downgrade or removal of the invalid claim. Preserve target, negation and verified claims. Introduce no new factual assertion.

- [ ] **Step 5: Emit semantic outcomes through Task 0's authoritative translator**

Emit mutually consistent root cause, attempt outcome and decision outcome. Preserve at least one verified argument when available; otherwise use a task-specific target-preserving non-factual expression.

- [ ] **Step 6: Add RED hunter/witch comparison tests**

Cover multiple legal alternatives, a single legal target with `no_legal_alternative`, no targets, retained skill, and friendly-fire risk evidence.

```powershell
python -m pytest tests/runtime/test_hunter_shot_directives.py::test_hunter_compares_legal_alternative_and_friendly_fire_risk tests/runtime/test_hunter_shot_directives.py::test_hunter_single_target_records_no_legal_alternative tests/runtime/test_witch_flow.py::test_witch_poison_records_alternative_and_retain_skill_option -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: current directives lack structured alternative/friendly-fire fields.

- [ ] **Step 7: Implement prompt fields and validation without deterministic target selection**

- [ ] **Step 8: Run speech, public-claim, hunter, witch and PlayerAgent tests**

```powershell
python -m pytest tests/runtime/test_speech_quality.py tests/evaluation/test_game_balance_batch.py tests/runtime/test_hunter_shot_directives.py tests/runtime/test_witch_flow.py tests/agents/test_player_agent.py tests/runtime/test_decision_outcomes.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

- [ ] **Step 9: Commit**

```powershell
git add werewolf_agent/runtime/speech_quality.py werewolf_agent/evaluation/balance_public_claims.py werewolf_agent/agents/player_fallback_speech.py werewolf_agent/runtime/directives/hunter.py werewolf_agent/runtime/strategy/hunter.py werewolf_agent/runtime/directives/witch.py werewolf_agent/runtime/strategy/witch.py tests/runtime/test_speech_quality.py tests/evaluation/test_game_balance_batch.py tests/runtime/test_hunter_shot_directives.py tests/runtime/test_witch_flow.py tests/agents/test_player_agent.py tests/runtime/test_decision_outcomes.py
git commit -m "fix: preserve valid evidence through semantic repair"
```

### Task 5: P2 evidence-gated possible worlds

**Files:**
- Modify: `werewolf_agent/cognition/worlds.py`
- Modify: the simulator prompt/export consumer if it accepts unknown world IDs
- Modify: the seer context/directive builder that consumes possible worlds
- Test: `tests/cognition/test_possible_worlds.py`
- Test: the existing simulator test module
- Test: `tests/runtime/test_context.py`
- Create: `tests/runtime/test_seer_flow.py`

- [ ] **Step 1: Write failing canonicalization and evidence tests**

Assert differently ordered identical assignments collapse to one world; `why=[]`, unknown evidence IDs and duplicate assignments are excluded; unknown simulator world IDs are rejected; no-evidence N1 context contains no concrete role guess.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/cognition/test_possible_worlds.py::test_duplicate_assignments_collapse_after_canonicalization tests/cognition/test_possible_worlds.py::test_world_without_valid_public_evidence_is_not_promptable tests/runtime/test_context.py::test_first_night_seer_context_has_no_unsupported_role_guess tests/runtime/test_seer_flow.py::test_seer_reason_does_not_quote_evidence_free_world_role -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: duplicate worlds remain, empty `why` is promptable, or the seer reason cites a concrete unsupported role.

- [ ] **Step 3: Implement canonical UTF-8 JSON assignment identity**

Sort player IDs, serialize only normalized assignment pairs, hash for internal identity, renormalize probabilities after deduplication, and retain deterministic ordering.

- [ ] **Step 4: Enforce evidence references and safe empty output**

Only public event/claim IDs may populate `why`. When none exist, produce an evidence-insufficient summary without concrete power-role guesses.

- [ ] **Step 5: Remove unsupported world-role claims from seer reasons**

Do not change legal target generation; only prevent unsupported cognition output from becoming a stated reason.

- [ ] **Step 6: Run cognition, context, simulator and seer tests**

```powershell
python -m pytest tests/cognition/test_possible_worlds.py tests/cognition/test_simulator.py tests/runtime/test_context.py tests/runtime/test_seer_flow.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/cognition/worlds.py werewolf_agent/cognition/simulator.py werewolf_agent/runtime/context.py werewolf_agent/runtime/directives/seer.py tests/cognition/test_possible_worlds.py tests/cognition/test_simulator.py tests/runtime/test_context.py tests/runtime/test_seer_flow.py
git commit -m "fix: gate possible worlds on unique public evidence"
```

### Task 6: P2 structured reflection memory fact gate

**Files:**
- Modify: `werewolf_agent/memory/reflection_synthesis.py`
- Modify: `werewolf_agent/memory/reflection_sanitization.py`
- Modify: `werewolf_agent/memory/reflection_repository.py`
- Modify: `werewolf_agent/runtime/reflection_prompt.py`
- Modify: `werewolf_agent/runtime/game_runner_memory.py`
- Modify: `scripts/run_real_game.py`
- Test: `tests/memory/test_reflection_synthesis.py`
- Test: `tests/memory/test_reflection_sanitization.py`
- Test: `tests/memory/test_reflection_repository.py`
- Test: `tests/runtime/test_reflection_prompt.py`
- Test: `tests/runtime/test_game_runner.py`
- Test: `tests/scripts/test_run_real_game.py`

- [ ] **Step 1: Add RED structured-claim tests from observed hallucinations**

Cover wrong role, wrong vote, wrong death cause, wrong potion target, unknown event ref, partially false lesson, valid abstraction, and prior-game player IDs rendered into a new game.

- [ ] **Step 2: Verify current free-text path accepts invalid facts**

```powershell
python -m pytest tests/memory/test_reflection_synthesis.py::test_wrong_role_claim_is_rejected tests/memory/test_reflection_sanitization.py::test_partially_false_lesson_is_not_persisted tests/memory/test_reflection_repository.py::test_cross_game_ids_are_anonymized_on_retrieval -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: free text is accepted without typed event-backed claims.

- [ ] **Step 3: Define Pydantic reflection claims and lessons**

Require typed claims with event references and lessons with claim dependencies. Keep raw draft moderator-only and out of cross-game retrieval.

- [ ] **Step 4: Implement deterministic verification and fail-closed persistence**

Validate each supported claim against final `GameState` events. Reject unknown types/refs and every lesson depending on a rejected claim. Persist verified abstractions with source game ID; anonymize player IDs on retrieval.

- [ ] **Step 5: Add rejection metrics and runner export**

Add `test_runner_exports_reflection_rejected_fact_count` and require the count to equal rejected structured claims only, not rejected lessons or free-text keyword matches. If lesson rejection observability is needed, export a separate `reflection_rejected_lesson_count`.

- [ ] **Step 6: Run all reflection/memory/runtime-memory tests**

```powershell
python -m pytest tests/memory/test_reflection_synthesis.py tests/memory/test_reflection_sanitization.py tests/memory/test_reflection_repository.py tests/runtime/test_reflection_prompt.py tests/runtime/test_game_runner.py tests/scripts/test_run_real_game.py::test_runner_exports_reflection_rejected_fact_count -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/memory/reflection_synthesis.py werewolf_agent/memory/reflection_sanitization.py werewolf_agent/memory/reflection_repository.py werewolf_agent/runtime/reflection_prompt.py werewolf_agent/runtime/game_runner_memory.py scripts/run_real_game.py tests/memory/test_reflection_synthesis.py tests/memory/test_reflection_sanitization.py tests/memory/test_reflection_repository.py tests/runtime/test_reflection_prompt.py tests/runtime/test_game_runner.py tests/scripts/test_run_real_game.py
git commit -m "fix: persist only verified reflection lessons"
```

### Task 7: P2 Persona final-prompt proof

**Files:**
- Modify: `werewolf_agent/agents/prompt_persona.py`
- Modify: `werewolf_agent/agents/player_persona.py`
- Modify: `werewolf_agent/agents/player_generation_request.py`
- Modify: `werewolf_agent/runtime/exposure_audit.py`
- Test: `tests/agents/test_prompt_persona.py`
- Test: `tests/agents/test_prompt_injection_fixes.py`
- Test: `tests/runtime/test_context_persona.py`
- Test: `tests/runtime/test_exposure_audit.py`

- [ ] **Step 1: Write RED linkage tests**

Assert every configured persona action links persona exposure to final system-message index, run-scoped opaque fingerprint, character count and confirmed injection. Cover initial request, structured retry, semantic retry and provider fallback.

- [ ] **Step 2: Add privacy RED tests**

Assert no persona text, plain SHA/MD5 digest, prompt text or global stable fingerprint appears in exported events.

```powershell
python -m pytest tests/agents/test_prompt_persona.py::test_final_system_message_has_run_scoped_persona_proof tests/agents/test_prompt_injection_fixes.py::test_persona_proof_survives_retry_and_provider_fallback tests/runtime/test_exposure_audit.py::test_persona_audit_exports_no_plain_hash_or_prompt_content -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: current injection audit reports `injected=false` or exposes only a pre-assembly snapshot without final-message proof.

- [ ] **Step 3: Implement final-message evidence at request construction**

Generate the fingerprint from a per-run secret/nonce and final message bytes. Record only the allowlisted proof after message assembly, not from the pre-assembly snapshot.

- [ ] **Step 4: Reconcile exposure and injection audit events**

Use decision identity as the join key and compute the exact confirmation rate with null unsupported behavior.

- [ ] **Step 5: Run Persona, prompt and exposure tests**

```powershell
python -m pytest tests/agents/test_prompt_persona.py tests/agents/test_prompt_injection_fixes.py tests/runtime/test_context_persona.py tests/runtime/test_exposure_audit.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

- [ ] **Step 6: Commit**

```powershell
git add werewolf_agent/agents/prompt_persona.py werewolf_agent/agents/player_persona.py werewolf_agent/agents/player_generation_request.py werewolf_agent/runtime/exposure_audit.py tests/agents/test_prompt_persona.py tests/agents/test_prompt_injection_fixes.py tests/runtime/test_context_persona.py tests/runtime/test_exposure_audit.py
git commit -m "feat: prove persona presence in final system prompts"
```

### Task 8: P3 unified taxonomy, reports, documentation, and final verification

**Files:**
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Modify: `scripts/analyze_recent_balance.py`
- Modify: `scripts/run_real_game.py`
- Modify: existing audit/report schemas as required
- Modify: `docs/superpowers/specs/2026-07-13-three-game-audit-closure-design.md` only if implementation discovers a documented contract correction
- Test: `tests/evaluation/test_game_balance_batch.py`
- Test: `tests/scripts/test_analyze_recent_balance.py`
- Test: runner report tests

- [ ] **Step 1: Write table-driven RED report-consistency tests using Task 0 taxonomy**

Cover provider timeout→retry success, provider fallback→reasoning preserved, schema normalization success, semantic repair success, terminal safe fallback and illegal action. Assert root cause, attempt outcome, decision outcome, attempt count, retry count and aggregate counters agree exactly.

```powershell
python -m pytest tests/runtime/test_decision_outcomes.py::test_runtime_events_and_report_projection_agree tests/evaluation/test_game_balance_batch.py::test_all_audit_closure_rates_have_exact_denominators tests/scripts/test_analyze_recent_balance.py::test_analyzer_reports_supported_false_for_zero_denominators tests/scripts/test_run_real_game.py::test_runner_exports_all_audit_closure_metrics -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected RED: report projections reclassify free-text reasons or required metrics/support flags are absent.

- [ ] **Step 2: Extend Task 0's single event translator with report projections**

Reports consume runtime events without reclassifying free-text reasons. Add zero-denominator support flags for every rate.

- [ ] **Step 3: Add all acceptance metrics**

Include terminal post-win calls, both wolf-plan thresholds, semantic repair invariants, world uniqueness/evidence coverage, power-role evidence completeness, reflection rejection count, Persona confirmation and reasoning coverage/confirmation/fallback keep rates.

- [ ] **Step 4: Run focused audit/report tests**

```powershell
python -m pytest tests/runtime/test_decision_outcomes.py tests/evaluation/test_game_balance_batch.py tests/scripts/test_analyze_recent_balance.py tests/scripts/test_run_real_game.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected GREEN: all report counts exactly match source events and every zero denominator has `supported=false`.

- [ ] **Step 5: Run broad module regressions**

```powershell
python -m pytest tests/agents -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
python -m pytest tests/runtime -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
python -m pytest tests/model_gateway tests/cognition tests/memory tests/evaluation tests/integration -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Expected: completed test processes with zero failures. A timeout is recorded as unverified.

- [ ] **Step 6: Run static verification**

```powershell
python -m compileall -q werewolf_agent scripts
git diff --check
git status --short
```

- [ ] **Step 7: Attempt the complete suite with external tracing disabled**

```powershell
python -m pytest -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp -o addopts=""
```

Record the exact pass/skip/fail result. Do not infer completion from percentage output before process exit.

- [ ] **Step 8: Commit final reporting changes**

```powershell
git add werewolf_agent/runtime/decision_outcomes.py werewolf_agent/evaluation/balance_audit.py scripts/analyze_recent_balance.py scripts/run_real_game.py tests/runtime/test_decision_outcomes.py tests/evaluation/test_game_balance_batch.py tests/scripts/test_analyze_recent_balance.py tests/scripts/test_run_real_game.py docs/superpowers/specs/2026-07-13-three-game-audit-closure-design.md
git commit -m "feat: report audit closure acceptance metrics"
```

- [ ] **Step 9: Provide the fixed 10-game soak command and thresholds**

Do not run paid real-model games without explicit user authorization. The handoff must list the exact 10-game command, output files, and every hard threshold from the approved design.
