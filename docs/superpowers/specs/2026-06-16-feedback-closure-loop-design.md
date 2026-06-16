# Feedback Closure Loop Design

## Goal

Upgrade the current Werewolf agent evaluation stack from local reports into a
closed feedback system that observes live module exposure, runs same-seed
ablation, produces reviewable improvements, gates approved changes with
regression tests, and exposes the whole flow through optional LangSmith tracing
and dashboard/API views.

## Non-Goals

- Do not auto-promote self-play output into live RAG or reflection memory.
- Do not inject post-game hidden truth into live player prompts.
- Do not make LangSmith required for runtime, tests, or local development.
- Do not change RuleEngine outcomes or game rules while adding evaluation
  infrastructure.
- Do not replace the existing context-level ablation harness; extend around it.

## Current State

Already implemented:

- RAG V2 tactical schema and live prompt-safe rendering.
- Reflection V2 synthesis, quality gate, approved-only live query path.
- Persistent in-game cognition, possible worlds, bounded simulator,
  decision/dialogue planning, persona policy priors, and world-model audit.
- Feedback trace schemas, trace builder, attribution metrics, RAG retrieval
  eval, reflection effectiveness eval, world-rank metrics, diagnostics,
  improvement candidate generation, offline ablation, live-context ablation,
  feedback reports, and optional LangSmith report exporter.

Remaining gaps:

- Runtime RAG/reflection/skill/persona exposure audits are not consistently
  emitted as first-class events that join to `EvaluationTrace`.
- Full same-seed live-game ablation is not implemented; current live ablation is
  context-level and decision-only.
- Improvement candidates remain review-only objects; there is no approval,
  materialization, regression gate, or rollback metadata.
- LangSmith export is report-level; it does not trace the full decision flow.
- Dashboard/API views do not yet expose ablation comparisons, candidate review,
  regression outcomes, or trace links as a workflow.

## Core Design

The closure loop uses one invariant identity across runtime, evaluation, and
approval:

```text
decision_key = {game_id}:{player_id}:{phase}:D{day}:N{night}:{task_type}:{action_index}
```

`decision_key` matches the existing `EvaluationTraceBuilder.make_trace_id()`.
Runtime code must attach it to both `action_trace_audit` and module exposure
events. If a call site cannot know final `action_index` before action
completion, it may emit a temporary `decision_ref` containing the same fields
except index, and the final action audit resolves it. The preferred path is to
assign `action_index` centrally before building an `AgentContext`.

The loop has five layers:

1. **Observation**: runtime emits prompt-visible module exposure audit events.
2. **Evaluation**: trace builder joins action traces with module exposure.
3. **Experimentation**: same-seed batches compare baseline with module-off runs.
4. **Improvement**: diagnostics generate candidates; approved candidates are
   materialized into RAG or reflection stores.
5. **Governance**: regression gate, report, LangSmith trace, and dashboard make
   the loop auditable.

## Data Contracts

### Decision Identity

Create a shared helper in `werewolf_agent.evaluation.trace_identity`:

```python
@dataclass(frozen=True)
class DecisionIdentity:
    game_id: str
    player_id: str
    phase: str
    day_number: int
    night_number: int
    task_type: str
    action_index: int

    def trace_id(self) -> str:
        return make_trace_id(
            game_id=self.game_id,
            player_id=self.player_id,
            phase=self.phase,
            day_number=self.day_number,
            night_number=self.night_number,
            task_type=self.task_type,
            action_index=self.action_index,
        )
```

`EvaluationTraceBuilder.make_trace_id()` should delegate to this helper so
runtime and evaluation cannot drift.

### Module Exposure Event

Runtime module exposure events use event types:

- `rag_exposure_audit`
- `reflection_exposure_audit`
- `skill_exposure_audit`
- `persona_exposure_audit`

Common payload fields:

```json
{
  "trace_id": "g1:p01:vote:D2:N1:vote:4",
  "player_id": "p01",
  "phase": "vote",
  "day_number": 2,
  "night_number": 1,
  "task_type": "vote",
  "visibility": "moderator_only",
  "module": "rag",
  "items": []
}
```

RAG item fields:

```json
{
  "entry_id": "seer_duel_anchor",
  "rank": 1,
  "relevance_score": 0.82,
  "prompt_visible": true,
  "title": "对跳预言家主线",
  "situation_signature": "seer duel speech",
  "retrieval_reason": "role_phase_match"
}
```

Reflection item fields:

```json
{
  "entry_id": "reflection:v2:abc",
  "rank": 1,
  "quality_score": 0.86,
  "prompt_visible": true,
  "lesson_key": "overtrust_claim",
  "quality_status": "approved"
}
```

Skill item fields:

```json
{
  "skill_name": "vote_analysis",
  "rank": 1,
  "prompt_visible": true,
  "summary_hash": "sha256:<prompt-visible-advice-hash>",
  "advice_type": "tactical"
}
```

Persona item fields:

```json
{
  "profile_id": "aggressive_bluffer",
  "prompt_visible": true,
  "policy_keys": ["vote_confidence_threshold_delta"],
  "sanitized": true
}
```

Forbidden in exposure events:

- hidden role truth not visible to that player;
- raw LLM private reasoning;
- raw full prompts;
- unredacted API responses;
- post-game outcome labels.

## Phase 1: Runtime Exposure Audit

Add a runtime audit collector that lives beside context construction:

```python
class ModuleExposureAuditCollector:
    def record_rag(self, identity: DecisionIdentity, hits: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def record_reflection(self, identity: DecisionIdentity, cards: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def record_skill(self, identity: DecisionIdentity, analyses: dict[str, str]) -> None:
        raise NotImplementedError

    def record_persona(self, identity: DecisionIdentity, snapshot: dict[str, Any]) -> None:
        raise NotImplementedError

    def flush_events(self) -> list[GameEvent]:
        raise NotImplementedError
```

`build_agent_context()` already has access to RAG hints, reflection hints,
skill analyses, strategy directives, and persona snapshots. The collector
should record only the compact prompt-visible forms that are actually placed on
`AgentContext`.

Integration rule:

- The same node that emits `action_trace_audit` must also append pending module
  exposure audit events for that decision.
- If a context is built but no LLM/action call happens, exposure events are
  discarded.
- Tests must assert that the trace builder can build supported RAG/reflection
  exposure metrics from real runtime events with no side-channel argument.

## Phase 2: Same-Seed Full-Game Ablation

Create a full-game ablation runner that executes paired game batches:

```python
@dataclass(frozen=True)
class FullGameAblationConfig:
    batch_id: str
    seed_set: list[int]
    removed_modules: list[str]
    player_count: int
    ruleset_id: str

class FullGameAblationRunner:
    def run(self, config: FullGameAblationConfig) -> FullGameAblationReport:
        """Run paired baseline/ablated batches and return baseline-minus-ablated deltas."""
        raise NotImplementedError
```

The baseline and ablated runs must use:

- same seed;
- same ruleset snapshot;
- same role assignment;
- same deterministic fallbacks where no real model is configured.

Module toggles should be applied at context construction:

- `rag`: no `rag_hints`, no RAG exposure events;
- `reflection`: no `reflection_memory_hints`, no `error_pattern_hint`;
- `possible_worlds`: no `possible_worlds`;
- `simulator`: no `simulation_predictions`;
- `skills`: no `skill_analyses`, `skill_analysis_hints`,
  `skill_tactical_advice`;
- `persona`: no `persona_snapshot` and no persona policy priors.

Report paired deltas as:

```text
baseline metric - ablated metric
```

Metrics:

- win-rate by faction;
- vote target quality;
- illegal action rate;
- hidden-info leakage rate;
- world true-rank rate;
- simulator prediction hit rate;
- RAG/reflection harmful-transfer rate;
- cost/latency if model usage records exist.

If a metric cannot be computed from available traces, mark it unsupported with a
reason rather than returning zero.

## Phase 3: Candidate Approval And Regression Gate

Add persistent review state for generated candidates:

```python
class CandidateStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MATERIALIZED = "materialized"
    ROLLED_BACK = "rolled_back"
```

Create `werewolf_agent.evaluation.candidate_store` with an in-memory
implementation for tests and repository-backed implementation for runtime.

Approval flow:

```text
candidate generated
-> pending store
-> approve(candidate_id, reviewer, notes)
-> materialize into RAG or Reflection draft
-> run same-seed regression gate
-> promote if gate passes
-> rollback/quarantine if gate fails
```

Materialization rules:

- RAG candidates become `RAGEntry(schema_version=2)` with
  `review_status=approved` only after approval and gate pass.
- Reflection candidates become `ReflectionEntryV2` with
  `quality_status=approved` only after approval and gate pass.
- Rejected or failed candidates never enter live query results.
- Every materialized entry stores source diagnosis IDs and regression report
  refs in audit metadata.

Regression gate:

```python
class RegressionGate:
    def evaluate(
        self,
        candidate: ImprovementCandidate,
        regression_report: FullGameAblationReport,
    ) -> GateResult:
        """Return pass/fail plus blocking metric reasons."""
        raise NotImplementedError
```

Default gate blocks promotion on:

- higher hidden-info leakage;
- lower vote target quality beyond tolerance;
- lower good/wolf win-rate beyond configured tolerance for the target faction;
- increased illegal actions;
- prompt-safety validation failure.

## Phase 4: LangSmith Full Decision Trace

Add a local-first tracing abstraction:

```python
class DecisionTraceSink(Protocol):
    def start_decision(self, identity: DecisionIdentity, metadata: dict[str, Any]) -> None:
        raise NotImplementedError

    def record_stage(
        self,
        identity: DecisionIdentity,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def finish_decision(self, identity: DecisionIdentity, outcome: dict[str, Any]) -> None:
        raise NotImplementedError
```

Implement:

- `NoopDecisionTraceSink`;
- `InMemoryDecisionTraceSink` for tests;
- `LangSmithDecisionTraceSink` with lazy import and redaction.

Stages:

- `context.build.start`
- `rag.retrieve`
- `reflection.retrieve`
- `worlds.generate`
- `simulator.predict`
- `skills.apply`
- `persona.policy`
- `prompt.render`
- `llm.call`
- `parse.result`
- `action.audit`
- `evaluation.trace`

Redaction:

- hidden roles and factions are removed unless viewer is moderator trace;
- prompt text is off by default; only prompt section hashes and sizes are
  recorded unless debug export is explicitly enabled;
- raw model output can be disabled by config.

LangSmith must remain optional. Tests should monkeypatch an injected fake
client; no network calls.

## Phase 5: Dashboard And API Workflow

Add API endpoints:

- `GET /api/evaluation/feedback/report/{batch_id}`
- `GET /api/evaluation/ablation/{batch_id}`
- `GET /api/evaluation/candidates`
- `POST /api/evaluation/candidates/{id}/approve`
- `POST /api/evaluation/candidates/{id}/reject`
- `GET /api/evaluation/regressions/{candidate_id}`

Dashboard panels:

- module exposure coverage;
- ablation paired delta table;
- failed trace clusters;
- candidate review queue;
- regression gate result;
- world-model audit link;
- optional LangSmith trace link.

The first UI implementation can be compact and utilitarian. It should reuse the
existing dashboard style and avoid introducing a separate frontend framework.

## Safety Invariants

1. Live prompts never receive post-game outcome labels.
2. Candidate generation never mutates live stores.
3. Candidate approval never bypasses prompt-safety validation.
4. Failed regression gate never promotes an entry.
5. Same-seed ablation never changes RuleEngine truth.
6. LangSmith exporter is optional and redacted by default.
7. Unsupported metrics report explicit reasons.
8. `trace_id` generation has one source of truth.
9. RAG/reflection exposure events contain only prompt-visible snippets.
10. `review_only` and `rejected` entries are never returned by live query.

## Testing Strategy

Unit tests:

- trace identity formatting;
- exposure event payload builders;
- trace builder joins runtime exposure events;
- module toggle behavior in full-game config;
- regression gate pass/fail cases;
- candidate store state transitions;
- materializer prompt-safety rejection;
- LangSmith sink lazy import and redaction;
- API route authorization/view-mode behavior.

Integration tests:

- one deterministic baseline/ablated same-seed pair with fake agents;
- candidate approval followed by regression gate and materialization;
- feedback report includes ablation and candidate gate artifacts;
- dashboard API returns compact JSON without hidden truth.

Regression commands should use:

```powershell
python -m pytest tests/evaluation tests/runtime tests/rag tests/memory tests/api tests/ui -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

## Rollout Plan

1. Implement trace identity and runtime exposure audits.
2. Implement full-game same-seed ablation runner.
3. Implement candidate store, approval, materializer, and regression gate.
4. Implement decision trace sinks and optional LangSmith runtime integration.
5. Implement API/dashboard workflow.
6. Run full feedback closure regression and update progress docs.

Each phase must be independently committed and must keep existing tests passing.
