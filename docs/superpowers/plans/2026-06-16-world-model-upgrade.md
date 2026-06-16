# World Model Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Werewolf agents from prompt-driven contextual play to a phased world-model architecture with persistent cognition, possible worlds, bounded simulation, and separated private decision/public dialogue.

**Architecture:** Keep the existing rule engine, runtime graph, prompt builder, RAG V2, Reflection V2, and persona router. Add a world-model stack beside the current cognition modules, then inject compact prompt-safe summaries into `AgentContext`. Each phase is independently testable and preserves the current fallback path.

**Tech Stack:** Python, Pydantic, dataclasses, pytest, LangGraph runtime, existing `MemoryStore`, `GameRepository`, `PlayerPromptBuilder`, and structured-output action contracts.

---

## Scope And Execution Rules

- Follow TDD for every production change.
- Do not change rule outcomes.
- Do not remove existing fallback behavior until the replacement is tested.
- Do not expose ground truth to live players.
- Keep RAG/reflection/persona as references or priors, never current-game
  evidence.
- Keep each phase mergeable on its own.

## Progress Ledger

- 2026-06-16: Phase 0 completed. Added prompt-boundary tests for future
  world-model context fields, added `AgentContext` placeholders, and verified
  schema/runtime/info-leak guardrails with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/agents/test_schemas.py tests/runtime/test_context.py tests/integration/test_e2e_info_leak.py -q`.
- 2026-06-16: Phase 1 completed. Added live `CognitionStateManager`, wired it
  through runtime state/context/agent adapter, updated runner chunk processing,
  and preserved live cognition matrices in memory snapshots. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/runtime/test_cognition_state.py tests/runtime/test_context.py tests/runtime/test_game_runner.py tests/runtime/test_agent_adapter.py tests/memory/test_memory.py -q`.
- 2026-06-16: Phase 2 completed. Added deterministic relation scoring,
  pressure/coalition summaries, relation metadata annotations, and weighted
  deception review ordering. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/memory/test_relation_scoring.py -q`
  and
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/memory/test_memory.py tests/memory/test_reflection_v2.py -q`.
- 2026-06-16: Phase 3 completed. Added constrained possible-world generation,
  prompt-safe possible-world cards, and runtime context injection after belief
  summary. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/cognition/test_possible_worlds.py tests/agents/test_prompt_builder.py tests/runtime/test_context.py tests/integration/test_e2e_info_leak.py -q`.
- 2026-06-16: Phase 4 completed. Added decision/dialogue planning contracts,
  explicit plan-envelope parsing for vote tasks, persona-aware private audit
  metadata, and public-dialogue leak boundaries. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/agents/test_planning.py tests/agents/test_player_agent.py tests/runtime/test_agent_adapter.py tests/runtime/test_vote_flow.py -q`.
- 2026-06-16: Phase 5 completed. Added bounded deterministic simulator,
  prompt-safe prediction cards, and runtime context injection after possible
  worlds. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/cognition/test_simulator.py tests/agents/test_prompt_builder.py tests/runtime/test_context.py -q`.
- 2026-06-16: Phase 6 completed. Added persona policy priors for planning
  thresholds/tone, enforced good-role deception boundaries, and recorded
  persona policy in private planning audit. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/persona_runtime/test_policy.py tests/agents/test_planning.py tests/agents/test_prompt_builder.py -q`.
- 2026-06-16: Phase 7 completed. Enforced current-evidence/reference
  separation, allowed historical references only under `reference_refs`, and
  changed prompt-budget tiers so RAG/reflection/profile are dropped before
  current-game belief/world/simulation summaries. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/rag/test_prompt_renderer.py tests/memory/test_reflection_v2.py tests/agents/test_prompt_builder.py tests/agents/test_planning.py -q`.
- 2026-06-16: Phase 8 completed. Added world-model evaluation metrics,
  moderator-only world-model audit view/API, and optional dashboard audit
  panel. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/evaluation/test_evaluation.py tests/api/test_views.py tests/ui/test_dashboard_enhanced.py -q`.
- 2026-06-16: Phase 8 review hardening completed. World-model audit now flows
  from real `action_trace_audit` records into API/UI/evaluation, public
  `DialoguePlan` text rejects copied private conceal content, and the dashboard
  reads nested world/prediction audit cards with HTML escaping. Verified with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/agents/test_planning.py tests/agents/test_trace_builder.py tests/api/test_views.py tests/evaluation/test_evaluation.py tests/ui/test_dashboard_enhanced.py -q`.
- 2026-06-16: Final regression completed. Verified all selected world-model
  and application suites with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/cognition tests/memory tests/agents tests/runtime tests/rag tests/persona_runtime tests/evaluation tests/api tests/ui -q`
  and integration guardrails with:
  `pytest -n0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp tests/integration/test_live_game_flow.py tests/integration/test_e2e_info_leak.py -q`.

## File Map

### New Files

- `werewolf_agent/runtime/cognition_state.py`
  - Owns in-game cognition lifecycle and per-viewer matrix updates.
- `tests/runtime/test_cognition_state.py`
  - Covers initialization, visibility-safe updates, snapshot behavior.
- `werewolf_agent/memory/relation_scoring.py`
  - Computes relation weights, pressure summaries, and coalition summaries.
- `tests/memory/test_relation_scoring.py`
  - Covers deterministic relation scoring and summaries.
- `werewolf_agent/cognition/worlds.py`
  - Generates top-k possible worlds and marginal probabilities.
- `tests/cognition/test_possible_worlds.py`
  - Covers role constraints, visibility boundaries, top-k normalization.
- `werewolf_agent/agents/planning.py`
  - Defines `DecisionPlan`, `DialoguePlan`, conversion helpers.
- `tests/agents/test_planning.py`
  - Covers plan validation and hidden-info boundaries.
- `werewolf_agent/cognition/simulator.py`
  - Produces bounded next-event predictions.
- `tests/cognition/test_simulator.py`
  - Covers prediction shape, bounds, and visibility.
- `werewolf_agent/persona_runtime/policy.py`
  - Converts sanitized persona snapshots into policy priors.
- `tests/persona_runtime/test_policy.py`
  - Covers role-safe persona priors.
- `werewolf_agent/runtime/world_model_audit.py`
  - Centralizes world-model audit sanitization and extraction from action traces.

### Modified Files

- `werewolf_agent/memory/schemas.py`
  - Add optional audit models only if needed by tests.
- `werewolf_agent/memory/cognition_matrix.py`
  - Add evidence/delta helpers if `CognitionStateManager` needs them.
- `werewolf_agent/memory/store.py`
  - Expose world-model snapshot helpers while preserving current API.
- `werewolf_agent/memory/relation_graph.py`
  - Preserve existing event shape; support weighted metadata helpers.
- `werewolf_agent/runtime/game_runner.py`
  - Initialize and save cognition state manager.
- `werewolf_agent/runtime/nodes/_shared.py`
  - Add optional runtime state keys for cognition manager/world model.
- `werewolf_agent/runtime/context.py`
  - Prefer managed cognition/world summaries when present; fallback to current
    recompute path.
- `werewolf_agent/agents/schemas.py`
  - Add optional `possible_worlds`, `simulation_predictions`, and planning audit
    fields to `AgentContext` only when needed.
- `werewolf_agent/agents/prompt_builder.py`
  - Render compact world-model sections below current facts and above
    long-term RAG/reflection.
- `werewolf_agent/agents/player.py`
  - Route selected tasks through planning helpers after tests exist.
- `werewolf_agent/runtime/agent_adapter.py`
  - Pass world-model context and planning output through selected task paths.
- `werewolf_agent/evaluation/metrics.py`
  - Add calibration and world-model metrics.
- `werewolf_agent/api/views.py` and `werewolf_agent/api/routes/games.py`
  - Add moderator-only audit views in later phases.
- `werewolf_agent/ui/static/dashboard.*`
  - Add optional world-model panels in final phase.

---

## Phase 0: Contracts And Guardrails

**Purpose:** Pin the invariants before adding behavior.

**Files:**
- Test: `tests/agents/test_schemas.py`
- Test: `tests/runtime/test_context.py`
- Test: `tests/integration/test_e2e_info_leak.py`
- Modify: `docs/superpowers/specs/2026-06-16-world-model-upgrade-design.md`

- [x] **Step 1: Write prompt-boundary tests for future world-model fields**

Add tests proving new fields, when present, are not rendered unless explicitly
whitelisted and marked prompt-safe.

Run:

```powershell
pytest tests/agents/test_schemas.py tests/runtime/test_context.py -q
```

Expected: fail because the fields do not exist yet or are not rendered.
After Phase 0 is implemented, rerunning this test should pass while still
proving the fields are not rendered by `PlayerPromptBuilder`.

- [x] **Step 2: Add `AgentContext` optional placeholders**

Add optional fields with safe defaults:

```python
possible_worlds: dict[str, Any] = Field(default_factory=dict)
simulation_predictions: dict[str, Any] = Field(default_factory=dict)
decision_plan_audit: dict[str, Any] = Field(default_factory=dict)
dialogue_plan_audit: dict[str, Any] = Field(default_factory=dict)
```

Do not render them yet except through explicit later renderers.

- [x] **Step 3: Run guardrail tests**

Run:

```powershell
pytest tests/agents/test_schemas.py tests/runtime/test_context.py tests/integration/test_e2e_info_leak.py -q
```

Expected: pass.

---

## Phase 1: Persistent In-Game Cognition

**Purpose:** Make cognition matrix updates live during the game.

**Files:**
- Create: `werewolf_agent/runtime/cognition_state.py`
- Create: `tests/runtime/test_cognition_state.py`
- Modify: `werewolf_agent/runtime/game_runner.py`
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/runtime/nodes/_shared.py`
- Modify: `werewolf_agent/memory/cognition_matrix.py`
- Modify: `werewolf_agent/memory/store.py`

**Implementation clarifications from Phase 1 review:**

- `CognitionStateManager` owns live in-game cognition. `restored_memory` remains
  cross-game/profile/reflection memory. Do not blur those meanings in prompts.
- `build_agent_context()` needs an explicit optional manager path, for example
  `cognition_state_manager=...`, because current adapter calls only pass
  `restored_memory=state.get("restored_memory")`.
- The runtime must call `update_from_events()` after game state changes,
  ideally from `GameRunner._process_chunk()` or the shared state handoff, not
  only during initialization.
- Evidence should prefer existing structured `EvidenceItem` fields
  (`source_event`, `day`, `speaker`, `claim`) where practical. String refs such
  as `event:13:speech` are acceptable only as compact prompt/audit labels.
- `_save_memory_snapshot()` must save the live manager's `MemoryStore` when
  present. Creating a fresh store at game end will erase learned matrices.

- [x] **Step 1: Write failing tests for manager initialization**

Test:

- manager creates one matrix per player;
- each matrix excludes the viewer;
- role names come from current game roles;
- no updates happen before events are processed.

Run:

```powershell
pytest tests/runtime/test_cognition_state.py::test_initializes_one_matrix_per_player -q
```

Expected: fail because `CognitionStateManager` does not exist.

- [x] **Step 2: Implement minimal `CognitionStateManager`**

API:

```python
class CognitionStateManager:
    def __init__(self, memory_store: MemoryStore) -> None: ...
    def initialize(self, game_state: GameState) -> None: ...
    def update_from_events(self, game_state: GameState) -> list[CognitionUpdateRecord]: ...
    def prompt_belief_summary(self, viewer_id: str, game_state: GameState) -> dict[str, Any]: ...
    def processed_event_count(self) -> int: ...
```

Use existing `build_world_state`, `VisibilityPolicy`, `BeliefUpdater`, and
`MemoryStore.sync_matrix`.

- [x] **Step 3: Write failing visibility test**

Test that a villager does not receive wolf teammate evidence, while a wolf can
use known teammate facts only in their own matrix.

Run:

```powershell
pytest tests/runtime/test_cognition_state.py::test_updates_respect_viewer_visibility -q
```

Expected: fail until visibility filtering is applied.

- [x] **Step 4: Apply visibility filtering**

Use `VisibilityPolicy.filter_visible_facts(world_state, viewer_id, role)`.

- [x] **Step 5: Write failing matrix-delta test**

Test that a public claim or vote changes the relevant matrix from uniform and
adds at least one evidence reference.

Run:

```powershell
pytest tests/runtime/test_cognition_state.py::test_meaningful_events_update_matrix_and_evidence -q
```

Expected: fail until evidence references are added.

- [x] **Step 6: Add evidence references**

Use event index references such as `event:13:speech`. Do not require a schema
change to `GameEvent`.

- [x] **Step 7: Integrate manager into `GameRunner`**

Initialize manager when the game state is created. Store it in runtime state as
an optional object, similar to `restored_memory`.

- [x] **Step 8: Prefer manager summary in `runtime.context`**

If a manager exists, use `prompt_belief_summary()`. Otherwise keep current
recompute path unchanged.

- [x] **Step 9: Persist updated matrices at game end**

`_save_memory_snapshot()` should reuse the live manager's `MemoryStore` when
available. It must not create a fresh store with uniform matrices unless no
manager exists.

- [x] **Step 10: Run Phase 1 tests**

Run:

```powershell
pytest tests/runtime/test_cognition_state.py tests/memory/test_memory.py tests/runtime/test_game_runner.py tests/runtime/test_context.py -q
```

Expected: pass.

---

## Phase 2: Evidence And Relation Graph Scoring

**Purpose:** Turn relation events into usable evidence for worlds and review.

**Files:**
- Create: `werewolf_agent/memory/relation_scoring.py`
- Create: `tests/memory/test_relation_scoring.py`
- Modify: `werewolf_agent/memory/relation_graph.py`
- Modify: `werewolf_agent/memory/store.py`
- Modify: `werewolf_agent/memory/review.py`

- [x] **Step 1: Write failing tests for scored relation events**

Test that imported vote, claim, attack, and defend facts receive bounded
weights and event refs.

Run:

```powershell
pytest tests/memory/test_relation_scoring.py::test_scores_relation_events_with_event_refs -q
```

Expected: fail.

- [x] **Step 2: Implement relation scoring helpers**

Keep scoring simple and deterministic:

- vote: 0.6;
- claimed role: 0.5;
- seer-check claim: 0.75;
- direct attack/defense: 0.4 to 0.7 depending on text strength.

- [x] **Step 3: Add pressure and coalition summaries**

APIs:

```python
relation_strength(graph, source, target, predicates=None, day_window=None) -> float
player_pressure_summary(graph, player_id, day_window=None) -> dict[str, Any]
coalition_edges(graph, day_window=None) -> list[dict[str, Any]]
```

- [x] **Step 4: Wire relation summaries into review**

`ReviewGenerator._analyze_deception()` should prefer stronger relation evidence
when selecting `deceived_by`.

- [x] **Step 5: Run Phase 2 tests**

Run:

```powershell
pytest tests/memory/test_relation_scoring.py tests/memory/test_memory.py tests/memory/test_reflection_v2.py -q
```

Expected: pass.

---

## Phase 3: Possible Worlds

**Purpose:** Add constrained hidden-role hypotheses.

**Files:**
- Create: `werewolf_agent/cognition/worlds.py`
- Create: `tests/cognition/test_possible_worlds.py`
- Modify: `werewolf_agent/agents/schemas.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `werewolf_agent/runtime/context.py`

- [x] **Step 1: Write failing role-count test**

Test that generated worlds respect a ruleset role distribution.

Run:

```powershell
pytest tests/cognition/test_possible_worlds.py::test_worlds_respect_role_counts -q
```

Expected: fail because engine does not exist.

- [x] **Step 2: Implement `PossibleWorld` and `PossibleWorldSet`**

Use dataclasses. Keep serialization helpers small:

```python
def to_prompt_dict(self, max_assignments: int = 4) -> dict[str, Any]
```

- [x] **Step 3: Implement candidate generation**

Start with bounded enumeration/sampling:

- fix viewer self role;
- fix role-visible private facts;
- apply known public hard facts;
- fill remaining roles by ruleset counts;
- cap candidate count before scoring.

- [x] **Step 4: Write failing scoring test**

Test that a world aligned with public claim/vote evidence ranks higher than a
contradictory one.

- [x] **Step 5: Implement scoring**

Use belief marginals and relation graph weights. Normalize top-k probabilities.

- [x] **Step 6: Write failing visibility test**

Test that villager prompt worlds render as hypotheses and never as truth, while
wolf prompt worlds may fix known teammates only in private context.

- [x] **Step 7: Add prompt-safe renderer**

Add `_build_possible_worlds()` to `PlayerPromptBuilder`. Position it after
current facts/belief and before long-term learning context.

- [x] **Step 8: Run Phase 3 tests**

Run:

```powershell
pytest tests/cognition/test_possible_worlds.py tests/agents/test_prompt_builder.py tests/runtime/test_context.py tests/integration/test_e2e_info_leak.py -q
```

Expected: pass.

---

## Phase 4: DecisionPlan And DialoguePlan

**Purpose:** Separate private choice from public expression.

**Files:**
- Create: `werewolf_agent/agents/planning.py`
- Create: `tests/agents/test_planning.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`

- [x] **Step 1: Write failing schema tests**

Test that `DecisionPlan` validates legal action, confidence, target, private
goal, and evidence refs.

Run:

```powershell
pytest tests/agents/test_planning.py::test_decision_plan_requires_legal_target_for_vote -q
```

Expected: fail.

- [x] **Step 2: Implement plan schemas**

Use Pydantic models with `extra="forbid"`.

- [x] **Step 3: Write failing dialogue safety test**

Test that `DialoguePlan` rendering refuses hidden fields and wolf teammate
knowledge in public speech.

- [x] **Step 4: Implement plan-to-action conversion**

API:

```python
decision_and_dialogue_to_action(decision, dialogue, context) -> PlayerAction
```

The result must pass existing `PlayerAction` validation.

- [x] **Step 5: Integrate only one low-risk task first**

Start with `TaskType.VOTE` because the vote action already has private audit
fields and public speech must be empty.

- [x] **Step 6: Expand to speech tasks**

Add `TaskType.SPEECH`, `SHERIFF_SPEECH`, `DEFENSE_SPEECH`, and
`WOLF_DISCUSSION` after vote tests pass.

- [x] **Step 7: Run Phase 4 tests**

Run:

```powershell
pytest tests/agents/test_planning.py tests/agents/test_player_agent.py tests/runtime/test_agent_adapter.py tests/runtime/test_vote_flow.py -q
```

Expected: pass.

---

## Phase 5: Bounded Simulator

**Purpose:** Predict likely next events from possible worlds without replacing
the rule engine.

**Files:**
- Create: `werewolf_agent/cognition/simulator.py`
- Create: `tests/cognition/test_simulator.py`
- Modify: `werewolf_agent/agents/schemas.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `werewolf_agent/runtime/context.py`

- [x] **Step 1: Write failing prediction-shape test**

Run:

```powershell
pytest tests/cognition/test_simulator.py::test_simulator_returns_bounded_prediction_cards -q
```

Expected: fail.

- [x] **Step 2: Implement simulator types**

Implement `FutureEventPrediction` and `SimulationResult`.

- [x] **Step 3: Implement first horizon**

Start with `next_day_vote_pressure`. Inputs: possible worlds, relation graph,
alive players, current day.

- [x] **Step 4: Add night-kill and claim-pressure horizons**

Keep these heuristic and bounded. Do not call LLM in the first implementation.

- [x] **Step 5: Add prompt renderer**

Render top 2 predictions with explicit "prediction, not fact" warning.

- [x] **Step 6: Run Phase 5 tests**

Run:

```powershell
pytest tests/cognition/test_simulator.py tests/agents/test_prompt_builder.py tests/runtime/test_context.py -q
```

Expected: pass.

---

## Phase 6: Persona Policy Priors

**Purpose:** Let persona tune thresholds and tone without changing truth.

**Files:**
- Create: `werewolf_agent/persona_runtime/policy.py`
- Create: `tests/persona_runtime/test_policy.py`
- Modify: `werewolf_agent/persona_runtime/router.py`
- Modify: `werewolf_agent/agents/planning.py`
- Modify: `werewolf_agent/runtime/context.py`

- [x] **Step 1: Write failing persona-prior tests**

Test that aggressive/risk-tolerant personas lower vote confidence threshold,
but do not change possible-world probabilities.

- [x] **Step 2: Implement `PersonaPolicyPrior`**

Inputs: sanitized persona snapshot and role/task.

Outputs:

- `vote_confidence_threshold_delta`;
- `claim_risk_threshold_delta`;
- `speech_directness`;
- `deception_allowed` only for compatible roles/tasks.

- [x] **Step 3: Integrate into planning**

DecisionPlan can use priors for thresholds and tie-breaking. DialoguePlan can
use priors for tone.

- [x] **Step 4: Run Phase 6 tests**

Run:

```powershell
pytest tests/persona_runtime/test_policy.py tests/agents/test_planning.py tests/agents/test_prompt_builder.py -q
```

Expected: pass.

---

## Phase 7: RAG, Reflection, And Profile Integration Cleanup

**Purpose:** Make long-term knowledge influence plans without becoming evidence.

**Files:**
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `werewolf_agent/agents/planning.py`
- Modify: `werewolf_agent/memory/reflection.py`
- Modify: `werewolf_agent/rag/prompt_renderer.py`
- Test: `tests/rag/test_prompt_renderer.py`
- Test: `tests/memory/test_reflection_v2.py`
- Test: `tests/agents/test_prompt_builder.py`

- [x] **Step 1: Write failing reference/evidence separation test**

Test that a DecisionPlan may cite RAG/reflection under `reference_refs`, but
not under `evidence_refs`.

- [x] **Step 2: Add reference ref validation**

Planning validation rejects historical references in current evidence fields.

- [x] **Step 3: Ensure prompt priority under budget**

Prompt trimmer keeps current facts/world summaries before RAG/reflection/profile
sections.

- [x] **Step 4: Run Phase 7 tests**

Run:

```powershell
pytest tests/rag/test_prompt_renderer.py tests/memory/test_reflection_v2.py tests/agents/test_prompt_builder.py -q
```

Expected: pass.

---

## Phase 8: Evaluation, API, And UI Observability

**Purpose:** Make world-model quality measurable and inspectable.

**Files:**
- Modify: `werewolf_agent/evaluation/metrics.py`
- Modify: `werewolf_agent/evaluation/reports.py`
- Modify: `werewolf_agent/api/views.py`
- Modify: `werewolf_agent/api/routes/games.py`
- Modify: `werewolf_agent/ui/static/dashboard.html`
- Modify: `werewolf_agent/ui/static/dashboard.css`
- Modify: `werewolf_agent/ui/static/dashboard.js`
- Test: `tests/evaluation/test_evaluation.py`
- Test: `tests/api/test_views.py`
- Test: `tests/ui/test_dashboard_enhanced.py`

- [x] **Step 1: Write failing metric tests**

Metrics:

- belief calibration;
- possible-world top-k hit rate after game end;
- simulator prediction hit rate;
- dialogue leakage rate;
- decision legality rate.

- [x] **Step 2: Implement metric collection**

Use post-game ground truth only in evaluation paths.

- [x] **Step 3: Add moderator-only API views**

Expose audit data only to moderator/spectator-authorized views.

- [x] **Step 4: Add optional dashboard audit panel**

Delivered scope:

- one moderator/debugger-only world-model audit panel;
- belief summary;
- possible-world cards;
- simulator prediction cards;
- decision/dialogue summary.

The earlier multi-panel target remains a future UI expansion, not part of the
completed Phase 8 implementation.

- [x] **Step 5: Run Phase 8 tests**

Run:

```powershell
pytest tests/evaluation/test_evaluation.py tests/api/test_views.py tests/ui/test_dashboard_enhanced.py -q
```

Expected: pass.

---

## Final Verification

After all phases:

```powershell
pytest tests/cognition tests/memory tests/agents tests/runtime tests/rag tests/persona_runtime tests/evaluation tests/api tests/ui -q
```

Expected:

- all selected suites pass;
- no live-player prompt leaks hidden truth;
- world-model sections remain bounded;
- existing RAG V2 and Reflection V2 tests still pass;
- deterministic rule-engine tests are unchanged.

Then run at least one mock full-game integration:

```powershell
pytest tests/integration/test_live_game_flow.py tests/integration/test_e2e_info_leak.py -q
```

Expected:

- game completes;
- no information leak assertions fail;
- action schema failures do not increase unexpectedly.

## Rollback Strategy

Each phase must be behind a safe fallback:

- Phase 1 can fall back to recompute belief in `runtime.context`.
- Phase 3 can omit possible worlds if generation fails.
- Phase 4 can fall back to direct `PlayerAction` generation.
- Phase 5 can omit simulator cards.
- Phase 6 can use neutral persona priors.
- Phase 8 UI/API additions must be read-only.

Failures in optional world-model layers must degrade to the existing agent
behavior, not break the game loop.
