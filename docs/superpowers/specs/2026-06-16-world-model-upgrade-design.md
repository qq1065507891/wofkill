# Werewolf World Model Upgrade Design

Date: 2026-06-16
Status: Phase 8 implemented; review hardening completed
Owner: Codex development session

## Problem

The project already has a working Werewolf multi-agent runtime: event sourcing,
structured prompts, RAG, reflection memory, persona routing, skills, visibility
guards, and rule-engine orchestration. The next goal is not to add another
chat prompt layer. The next goal is to make each player maintain a coherent,
auditable model of the hidden game state and use that model to plan actions and
public dialogue.

The target architecture is a phased world-model upgrade:

1. persistent in-game cognition;
2. evidence and relation graph upgrades;
3. possible worlds;
4. decision/dialogue separation;
5. forward simulation;
6. persona policy priors;
7. memory/RAG/reflection integration;
8. observability, evaluation, and UI support.

## Current Implementation Baseline

The current codebase already provides these foundations:

- `werewolf_agent.cognition.world_state`: converts `GameEvent` objects into
  structured facts.
- `werewolf_agent.cognition.belief`: maintains per-viewer role probabilities,
  faction lean, and trust.
- `werewolf_agent.memory.cognition_matrix`: defines the persisted form of
  per-viewer short-term cognition.
- `werewolf_agent.memory.relation_graph`: stores typed vote, claim, attack, and
  defense relations.
- `werewolf_agent.memory.reflection`: stores quality-gated long-term learning
  cards.
- `werewolf_agent.rag`: stores prompt-safe transferable tactical knowledge.
- `werewolf_agent.agents.schemas` and `werewolf_agent.agents.action_contract`:
  enforce structured output.
- `werewolf_agent.agents.prompt_builder`: renders current facts, memory, RAG,
  persona, strategy directives, and output contracts.
- `werewolf_agent.runtime.graph` and `werewolf_agent.runtime.game_runner`:
  orchestrate the rule-safe game lifecycle.

The main gap is lifecycle coherence. Belief is currently often rebuilt from
events during context construction, while `MemoryStore.cognition_matrices`
exists as a persistable structure but is not the single source of truth for
in-game cognition. Possible worlds and forward predictions do not yet exist as
first-class structures.

## Goals

1. Make every player maintain a persistent, visibility-safe in-game cognition
   state.
2. Keep current-game facts, hidden-role hypotheses, tactical memories, and
   public dialogue in separate layers.
3. Add possible-world reasoning constrained by ruleset role counts and each
   viewer's visibility.
4. Add a bounded simulator that predicts likely next events without replacing
   the deterministic rule engine.
5. Split private decision planning from public dialogue generation.
6. Use persona as a strategy prior, not as a source of hidden truth or rule
   overrides.
7. Make RAG and reflection influence decisions only as reference priors below
   current-game evidence.
8. Preserve prompt safety, role visibility, and structured-output contracts.
9. Provide tests and metrics for correctness, leak safety, and actual decision
   impact.

## Non-Goals

1. Do not train model weights or introduce reinforcement learning.
2. Do not let LLM output decide rule outcomes.
3. Do not expose hidden roles through world hypotheses, RAG, reflection, or
   persona.
4. Do not store full transcripts as the primary memory representation.
5. Do not make possible worlds unbounded. The live prompt receives only compact
   top-k summaries.
6. Do not replace the existing RAG V2 or Reflection Memory V2 designs.
7. Do not require external vector or embedding services for the world-model
   core.

## Layer Precedence

Live agent reasoning must obey this order:

1. rule engine, legal actions, information boundaries, and output contract;
2. current-game visible facts and role-visible private facts;
3. persistent belief/cognition state derived from visible facts;
4. possible worlds derived from belief plus ruleset constraints;
5. bounded simulator predictions derived from possible worlds;
6. strategy directives and role-critical tactical guards;
7. skills, RAG, reflection, profile, and persona policy priors;
8. dialogue style.

Lower layers may suggest or re-rank a choice. They must not override legal
actions, current facts, visibility, or role-critical hard rules.

## Phase 1: Persistent In-Game Cognition

### Purpose

Make `CognitionMatrix` the durable per-player cognition state for the current
game, not just a post-game or restored-memory artifact.

### New Component

`werewolf_agent.runtime.cognition_state.CognitionStateManager`

Responsibilities:

- initialize one `CognitionMatrix` per player at game start;
- track processed event count to avoid duplicate evidence;
- build `StructuredWorldState` from new events;
- apply `VisibilityPolicy` per viewer;
- update `BeliefState` through `BeliefUpdater`;
- sync `BeliefState` into `CognitionMatrix`;
- attach evidence references for meaningful deltas;
- expose prompt-ready belief summaries.

### Data Model

`CognitionUpdateRecord`

```json
{
  "viewer_id": "p01",
  "event_start": 12,
  "event_end": 15,
  "day": 2,
  "phase": "day",
  "deltas": [
    {
      "target_id": "p05",
      "before": {"werewolf": 0.22, "seer": 0.18},
      "after": {"werewolf": 0.41, "seer": 0.12},
      "trust_delta": -0.08,
      "evidence_refs": ["event:13:speech", "event:14:vote"]
    }
  ]
}
```

The update record is audit-only. Live prompts receive compact summaries:

```json
{
  "my_suspects": [
    {
      "player": "p05",
      "faction_lean": "wolf_lean",
      "trust": 0.31,
      "top_role_guess": "werewolf",
      "top_role_prob": 0.41,
      "evidence": ["D2 speech contradiction", "vote against confirmed good"]
    }
  ],
  "my_trusted": []
}
```

### Integration

- `GameRunner` owns a per-game `CognitionStateManager` when memory is enabled.
- `runtime.context.build_agent_context()` asks the manager for the current
  prompt belief summary. If no manager exists, it keeps the current recompute
  fallback.
- `MemoryStore` remains the serialization boundary.
- `PersistentMemoryCoordinator` can save snapshots without a schema migration
  because `MemoryStore` already serializes `cognition_matrices`.

### Safety Rules

- Viewer-specific matrices must never be shared across players.
- A villager matrix must never include wolf teammate facts.
- A seer matrix may include only that seer's own checks.
- A witch matrix may include only current role-visible potion facts.
- Historical reflection and RAG cannot be written into current-game evidence.

## Phase 2: Evidence And Relation Graph Upgrade

### Purpose

Make the relation graph strong enough to support belief and possible-world
reasoning.

### Current State

`RelationGraph` already stores typed relation events, but edges are light:
predicate, source, target, day, value, metadata. Speech attack/defend extraction
is heuristic.

### New Data Model

Extend relation event metadata without breaking the dataclass API:

```json
{
  "predicate": "spoke_against",
  "source": "p02",
  "target": "p05",
  "day": 2,
  "value": "public pressure",
  "metadata": {
    "event_ref": "event:19:speech",
    "weight": 0.65,
    "confidence": 0.72,
    "stance": "attack",
    "basis": "speech_logic",
    "visibility": "public",
    "decay_group": "day_2_discussion"
  }
}
```

### New Helpers

`werewolf_agent.memory.relation_scoring`

- `score_relation_event(fact, context) -> float`
- `relation_strength(source, target, predicates, day_window) -> float`
- `player_pressure_summary(player_id) -> dict`
- `coalition_edges(day_window) -> list`

### Integration

- `MemoryStore.import_world_state()` imports scored relation events.
- `ReviewGenerator` uses scored relations when analyzing deception.
- `BeliefUpdater` can consume relation summaries instead of re-parsing speech.
- Possible Worlds uses relation graph evidence as one scoring channel.

### Safety Rules

- Relation graph stores public/publicly-visible claims and role-visible facts
  according to viewer. It must not merge all-player private views into one live
  player prompt.
- Ground truth is allowed only in post-game review paths.

## Phase 3: Possible Worlds

### Purpose

Represent hidden-role inference as top-k constrained hypotheses rather than
independent role probabilities.

### New Component

`werewolf_agent.cognition.worlds`

Core types:

```python
@dataclass(frozen=True)
class PossibleWorld:
    world_id: str
    probability: float
    roles: dict[str, str]
    score_breakdown: dict[str, float]
    supporting_evidence: list[str]
    contradictions: list[str]

@dataclass(frozen=True)
class PossibleWorldSet:
    viewer_id: str
    generated_at_event_index: int
    worlds: list[PossibleWorld]
    marginal_role_probs: dict[str, dict[str, float]]
```

### Engine

`PossibleWorldsEngine.generate(...)`

Inputs:

- viewer id and viewer role;
- visible facts;
- current `CognitionMatrix`;
- relation graph view;
- ruleset role distribution;
- alive/dead state;
- known role-visible facts.

Outputs:

- top-k worlds, default `k=3`;
- marginal probabilities derived from worlds;
- contradiction list for impossible or low-probability assignments.

### Scoring

Score each candidate world through additive channels:

- role-count validity;
- self role and role-visible facts;
- public claims;
- seer-check claims;
- vote and attack/support graph consistency;
- death and night-result consistency;
- contradiction penalties;
- persona-independent behavioral priors.

Do not use RAG or reflection as evidence. They can influence strategy after
worlds are generated, not the truth score of the current game.

### Prompt Shape

Live prompt receives compact, uncertainty-preserving text:

```json
{
  "type": "possible_worlds",
  "top_worlds": [
    {
      "label": "World A",
      "probability": 0.46,
      "key_assignments": {"p03": "seer", "p05": "werewolf"},
      "why": ["p03 claim aligns with D1 badge flow", "p05 vote pattern conflicts"],
      "watch_for": ["p03 next check report", "p05 defense target"]
    }
  ],
  "warning": "These are hypotheses from visible evidence, not ground truth."
}
```

### Safety Rules

- Never render a hidden role as fact.
- For non-wolf players, never fix wolf teammates unless public evidence implies
  a hypothesis.
- For wolves, teammates may be fixed as known private facts, but public
  dialogue must not reveal that knowledge.
- For hybrid, master id is known only if role-visible; master faction remains
  hidden unless rules/runtime has already made it role-visible.

## Phase 4: DecisionPlan And DialoguePlan

### Purpose

Separate private choice from public expression.

### Current State

`PlayerAction` already distinguishes action types, but many calls still ask the
LLM to produce action, target, reason, confidence, and speech together.

### New Component

`werewolf_agent.agents.planning`

Core types:

```python
class DecisionPlan(BaseModel):
    action_type: ActionType
    target_id: str | None = None
    confidence: float
    private_goal: str
    evidence_refs: list[str] = []
    selected_world_ids: list[str] = []
    risk_flags: list[str] = []

class DialoguePlan(BaseModel):
    public_intent: str
    public_target_id: str | None = None
    talking_points: list[str]
    conceal: list[str] = []
    tone: str = ""
```

### Flow

For decision-heavy tasks:

```text
AgentContext
  -> DecisionPlan generation or deterministic planner
  -> validation against legal actions and worlds
  -> DialoguePlan generation
  -> PlayerAction
```

For simple speech-only tasks:

```text
AgentContext
  -> DialoguePlan
  -> SpeechPlayerAction
```

### Compatibility

The external provider still receives one tool call when possible. Internally,
the agent can perform two calls only for tasks where separation materially
improves behavior, initially `vote`, `speech`, `sheriff_speech`, and
`wolf_discussion`.

### Safety Rules

- `DecisionPlan.private_goal` and `conceal` are moderator-audit fields only.
- `DialoguePlan` must not include wolf teammate ids, night kill targets, hidden
  checks, or private simulator assumptions unless role and task allow them.
- `PlayerAction.reason` for public actions must not include private intent.

## Phase 5: Forward Simulator

### Purpose

Predict likely next events under each possible world to support planning.

### New Component

`werewolf_agent.cognition.simulator`

Core types:

```python
@dataclass(frozen=True)
class FutureEventPrediction:
    event_type: str
    probability: float
    affected_players: list[str]
    rationale: str
    world_ids: list[str]

@dataclass(frozen=True)
class SimulationResult:
    viewer_id: str
    horizon: str
    predictions: list[FutureEventPrediction]
```

### Scope

Simulator V1 predicts only:

- next vote pressure target;
- likely night kill target;
- likely claim/counterclaim pressure;
- risk that current planned speech increases suspicion.

It does not simulate full games or replace the runtime graph.

### Method

1. Start from top-k possible worlds.
2. Apply rule-engine legal-state constraints.
3. Use relation graph and current vote pressure to estimate next public moves.
4. Optionally ask an LLM to explain or rank bounded candidate predictions.
5. Render only compact prediction cards.

### Prompt Shape

```json
{
  "type": "simulation",
  "horizon": "next_day_vote",
  "predictions": [
    {
      "event": "p05 receives major pressure",
      "probability": 0.52,
      "rationale": "two current worlds require resolving p05/p03 conflict"
    }
  ],
  "warning": "Prediction, not fact."
}
```

## Phase 6: Persona Policy Priors

### Purpose

Make persona affect strategy thresholds, not truth.

### Current State

Persona profiles already carry `personality`, `speech_style`,
`risk_tolerance`, `deception_skill`, and `task_styles`. Prompt rendering already
sanitizes persona snapshots.

### New Component

`werewolf_agent.persona_runtime.policy`

`PersonaPolicyPrior`:

```json
{
  "risk_tolerance": 0.82,
  "aggression": 0.74,
  "deception_skill": 0.78,
  "leadership": 0.66,
  "vote_confidence_threshold_delta": -0.08,
  "claim_risk_threshold_delta": 0.10,
  "speech_directness": "high"
}
```

### Integration

- DecisionPlan uses priors to break ties and set confidence thresholds.
- DialoguePlan uses priors for tone, brevity, and pressure style.
- Possible Worlds never uses persona to change truth likelihood directly.

### Safety Rules

- Good roles must not receive wolf-only deception tactics.
- Persona cannot override mandatory role reporting.
- Persona cannot suppress legal-action requirements.

## Phase 7: Memory, RAG, And Reflection Integration

### Purpose

Keep long-term knowledge useful without confusing it with current evidence.

### Rules

- Current-game facts and possible worlds are evidence.
- RAG entries are tactical reference.
- Reflection entries are personal historical lessons.
- Profile entries are long-term behavioral summaries.

### Integration

- DecisionPlan can cite RAG/reflection as `reference_refs`, never
  `evidence_refs`.
- Possible Worlds ignores RAG/reflection.
- Simulator may use RAG only to generate candidate tactics, not to assert
  current hidden roles.
- Prompt builder keeps RAG/reflection below world-model sections.

## Phase 8: Observability, Evaluation, And UI

### New Metrics

- belief calibration after game end;
- possible-world top-k hit rate;
- average rank of true world approximation;
- decision-plan legality rate;
- dialogue leakage rate;
- simulator prediction hit rate;
- prompt token cost by section;
- win rate, game length, illegal action rate, parse failure rate.

### New Audit Views

- moderator-only world-model audit endpoint;
- audit records derived from real `action_trace_audit` payloads, not only
  synthetic `world_model_audit` events;
- one dashboard audit panel with belief summary, possible-world cards,
  simulator predictions, and decision/dialogue summary;
- post-game metrics derived from review records and real event-log traces.

### UI/API Scope

API should expose moderator-only world-model traces. Live player views must not
receive hidden-role hypotheses for other players unless they are the same
prompt-safe summaries already shown to that player.

Public dialogue fields are validated before conversion to `PlayerAction`.
If `DialoguePlan.public_intent` or `talking_points` copy `conceal` content or
contain explicit private markers such as wolf teammates or night-kill targets,
the plan is rejected and the normal retry/fallback path handles the failure.

## Migration Strategy

1. No database migration for Phase 1. Reuse `MemoryStore` snapshots.
2. Add optional fields to prompt context with safe defaults.
3. Keep recompute fallback for contexts created outside `GameRunner`.
4. Add new modules behind feature flags or config toggles.
5. Make tests prove old flows continue working when world model is disabled.
6. Enable Phase 1 by default only after targeted runtime and integration tests
   pass.

## Testing Strategy

Each phase follows red-green-refactor.

Phase 1 tests:

- cognition state initializes one matrix per player;
- visible events update only the correct viewer matrices;
- matrices are not left uniform after meaningful evidence;
- no private info leaks across roles;
- game-end snapshots include updated matrices.

Phase 2 tests:

- relation events carry weights and event refs;
- vote/claim/attack/defense summaries are deterministic;
- post-game review uses relation weights;
- graph import is idempotent per processed event range.

Phase 3 tests:

- generated worlds respect role counts;
- viewer-visible hard facts are fixed;
- hidden truth is not rendered as fact;
- top-k world marginals normalize;
- contradictory worlds are penalized or rejected.

Phase 4 tests:

- vote tasks produce valid DecisionPlan before PlayerAction;
- public DialoguePlan omits private goal and hidden facts;
- wolf dialogue can conceal teammate knowledge without leaking it;
- fallback path still returns legal PlayerAction.

Phase 5 tests:

- simulator returns bounded predictions;
- predictions cite worlds but do not expose hidden truth;
- invalid future event types are rejected;
- simulator can be disabled without changing action contracts.

Phase 6 tests:

- persona priors alter thresholds/tone only;
- good roles do not receive wolf deception tactics;
- role-critical directives beat persona.

Phase 7 tests:

- RAG/reflection refs are marked as reference, not evidence;
- possible worlds ignore historical memory;
- prompt budget drops historical layers before current-game world-model
  summaries.

Phase 8 tests:

- API moderator views expose audit data;
- player views remain visibility-safe;
- metrics are populated in evaluation reports.

## Rollout Order

1. Phase 1: persistent in-game cognition.
2. Phase 2: evidence and relation graph scoring.
3. Phase 3: possible worlds.
4. Phase 4: DecisionPlan/DialoguePlan split.
5. Phase 5: bounded simulator.
6. Phase 6: persona policy priors.
7. Phase 7: memory/RAG/reflection integration cleanup.
8. Phase 8: observability, evaluation, and UI.

This order is deliberate. Possible Worlds needs stable cognition and evidence.
Simulator needs Possible Worlds. Decision/Dialog separation becomes much more
valuable once the private decision can cite worlds and predictions. Persona,
RAG, and reflection should tune strategy only after the evidence stack is
stable.
