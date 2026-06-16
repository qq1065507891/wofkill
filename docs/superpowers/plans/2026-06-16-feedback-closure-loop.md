# Feedback Closure Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a closed evaluation feedback loop that joins live module exposure to decision traces, runs reproducible same-seed full-game ablations, gates reviewable improvement candidates, emits optional redacted LangSmith decision traces, and exposes moderator/debugger workflow APIs.

**Architecture:** Use one public identity (`trace_id`) from runtime through evaluation, with exposure audit events emitted only when a real action audit is emitted. Keep causal claims reproducible by limiting full-game ablation to deterministic fake agents or strict replay; fresh live-model runs report unsupported causal metrics. Candidate generation remains review-only until explicit approval, draft materialization, regression gate pass, and promotion.

**Tech Stack:** Python dataclasses/Pydantic, existing GameEvent/GameResult schemas, FastAPI routers, pytest, in-memory fakes for model/replay/store tests, optional lazy LangSmith client injection.

---

## Scope And Invariants

This plan implements the approved spec at:

`docs/superpowers/specs/2026-06-16-feedback-closure-loop-design.md`

Hard invariants:

- `trace_id` is the only public join identity. Do not introduce `decision_key` or `decision_ref`.
- `trace_id` format is `{game_id}:{player_id}:{phase}:D{day}:N{night}:{task_type}:{action_index}`.
- `action_index` is monotonic per game over emitted `action_trace_audit` records and allocated before `AgentContext` construction.
- Runtime event log is the primary exposure path; `EvaluationTraceBuilder.build(..., exposure_audits=...)` remains only a secondary import/test side-channel.
- Exposure events must not contain hidden role truth, raw prompts, raw private reasoning, unredacted API responses, or post-game outcome labels.
- Full-game same-seed causal metrics are supported only for deterministic fake/replay. Fresh `live_model` mode reports causal metrics as unsupported unless replay capture is supplied.
- Candidate regression compares baseline current system vs candidate-enabled draft namespace. It is not module-off ablation.
- LangSmith is optional, lazy, and redacted by default.
- API workflow uses existing `ViewMode.MODERATOR_FULL` and `CallerRole.MODERATOR` / `CallerRole.DEBUGGER`; no new evaluator role.

Use pytest with an explicit temp directory:

```powershell
python -m pytest <targets> -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Do not use real network, real LLM calls, or production DB writes in tests.

---

## File Map

### New Files

- `werewolf_agent/evaluation/trace_identity.py`
  - Single source of truth for `DecisionIdentity`, `make_trace_id`, and action-index allocation helpers.
- `werewolf_agent/runtime/exposure_audit.py`
  - Builds sanitized `rag_exposure_audit`, `reflection_exposure_audit`, `skill_exposure_audit`, and `persona_exposure_audit` events.
- `werewolf_agent/evaluation/full_game_ablation.py`
  - Full-game ablation config, deterministic fake/replay runner contracts, paired report schemas, unsupported metric handling.
- `werewolf_agent/evaluation/replay.py`
  - Small replay artifact loader/matcher used by full-game ablation tests and future replay capture.
- `werewolf_agent/evaluation/candidate_store.py`
  - Candidate statuses, in-memory store, repository-backed facade, transition validation, reviewer metadata.
- `werewolf_agent/evaluation/candidate_materializer.py`
  - Draft-only RAG/reflection materialization, prompt-safety validation, promotion/rollback metadata.
- `werewolf_agent/evaluation/regression_gate.py`
  - Candidate regression config/report schemas and normalized gate pass/fail logic.
- `werewolf_agent/evaluation/artifact_store.py`
  - Minimal in-memory and repository-backed stores for feedback reports, ablation reports, regression reports, and trace-link metadata used by API/dashboard workflow.
- `werewolf_agent/evaluation/decision_trace.py`
  - `DecisionTraceSink` protocol, noop/in-memory sinks, redactor, optional `LangSmithDecisionTraceSink`.
- `werewolf_agent/api/routes/evaluation.py`
  - Feedback report, ablation, candidates, approval/rejection, regression endpoints.
- `tests/evaluation/test_trace_identity.py`
- `tests/runtime/test_exposure_audit.py`
- `tests/evaluation/test_runtime_exposure_trace_join.py`
- `tests/evaluation/test_full_game_ablation.py`
- `tests/evaluation/test_candidate_store.py`
- `tests/evaluation/test_candidate_materializer.py`
- `tests/evaluation/test_regression_gate.py`
- `tests/evaluation/test_evaluation_artifact_store.py`
- `tests/evaluation/test_decision_trace_sink.py`
- `tests/api/test_evaluation_workflow_api.py`
- `tests/ui/test_feedback_dashboard.py`

### Modified Files

- `werewolf_agent/evaluation/trace_builder.py`
  - Delegate trace id formatting to `trace_identity`; scan runtime event log for exposure audit events before side-channel audits; support skill/persona exposures.
- `werewolf_agent/evaluation/feedback_schemas.py`
  - Add candidate status/regression fields only if needed by reports; keep existing dataclasses backward compatible.
- `werewolf_agent/evaluation/feedback_report.py`
  - Include full-game ablation reports and candidate gate artifacts with public/private redaction.
- `werewolf_agent/evaluation/ablation.py`
  - Reuse `apply_ablation_toggles` from full-game runner; keep existing context-level behavior and sign convention.
- `werewolf_agent/runtime/context.py`
  - Accept optional `decision_identity`, exposure collector, ablation toggles, and decision trace sink; record only prompt-visible context artifacts actually returned.
- `werewolf_agent/runtime/agent_adapter.py`
  - Allocate identity before context construction and pass identity/collector/sink to `build_agent_context` at each adapter entry.
- `werewolf_agent/runtime/nodes/_shared.py`
  - Extend `RuntimeState` with action-index state if needed; attach `trace_id` to `action_trace_audit`; append flushed exposure events beside action audit events.
- `werewolf_agent/runtime/nodes/day.py`
  - Use central action identity/allocation for vote traces and include paired exposure events.
- `werewolf_agent/runtime/nodes/night.py`
  - Use central action identity/allocation for night action traces and include paired exposure events.
- `werewolf_agent/runtime/nodes/skills.py`
  - Use central action identity/allocation for role-skill traces and include paired exposure events.
- `werewolf_agent/runtime/nodes/sheriff.py`
  - Use central action identity/allocation for sheriff traces and include paired exposure events.
- `werewolf_agent/runtime/nodes/sheriff_pk.py`
  - Use central action identity/allocation for PK traces and include paired exposure events.
- `werewolf_agent/runtime/world_model_audit.py`
  - Preserve existing audit extraction; ensure trace id passthrough if present.
- `werewolf_agent/api/app.py`
  - Mount the new evaluation router and expose in-memory workflow stores in `app.state`.
- `werewolf_agent/api/views.py`
  - Add small redaction helpers if route-level response builders need shared behavior.
- `werewolf_agent/ui/static/dashboard.html`
  - Add compact feedback-loop dashboard panels for module exposure coverage, ablation deltas, failed trace clusters, candidate queue, regression result, world-model audit link, and optional LangSmith link.
- `werewolf_agent/ui/static/dashboard.js`
  - Load `/api/evaluation/...` workflow data for moderator/debugger view and render redacted public fallback.
- `werewolf_agent/ui/static/dashboard.css`
  - Add restrained table/status styling for feedback workflow panels.
- `werewolf_agent/rag/schemas.py`
  - Only if materialization needs metadata fields not currently present; prefer existing extension/audit metadata fields.
- `werewolf_agent/memory/schemas.py`
  - Only if materialization needs metadata fields not currently present; prefer existing extension/audit metadata fields.
- `docs/implementation_plan.md`
  - Add progress entry for feedback closure loop phases.
- `docs/design/werewolf-agent-v1-design.md`
  - Update architecture status and feedback closure loop module description.

### Existing Tests To Preserve

- `tests/evaluation/test_feedback_trace_builder.py`
- `tests/evaluation/test_live_ablation_harness.py`
- `tests/evaluation/test_feedback_report.py`
- `tests/evaluation/test_langsmith_exporter.py`
- `tests/api/test_evaluation.py`
- `tests/runtime/test_context.py`
- `tests/runtime/test_vote_flow.py`
- `tests/runtime/test_graph_lifecycle.py`

---

### Task 1: Shared Trace Identity

**Files:**
- Create: `werewolf_agent/evaluation/trace_identity.py`
- Modify: `werewolf_agent/evaluation/trace_builder.py`
- Test: `tests/evaluation/test_trace_identity.py`
- Test: `tests/evaluation/test_feedback_trace_builder.py`

- [ ] **Step 1: Write failing identity tests**

Add tests that assert exact formatting, dataclass serialization behavior, and monotonic per-game allocation:

```python
from werewolf_agent.evaluation.trace_identity import (
    ActionIndexAllocator,
    DecisionIdentity,
    make_trace_id,
)


def test_make_trace_id_uses_approved_public_format() -> None:
    assert make_trace_id(
        game_id="g1",
        player_id="p01",
        phase="vote",
        day_number=2,
        night_number=1,
        task_type="vote",
        action_index=4,
    ) == "g1:p01:vote:D2:N1:vote:4"


def test_decision_identity_returns_trace_id() -> None:
    identity = DecisionIdentity("g1", "p01", "vote", 2, 1, "vote", 4)
    assert identity.trace_id() == "g1:p01:vote:D2:N1:vote:4"


def test_action_index_allocator_is_monotonic_per_game() -> None:
    allocator = ActionIndexAllocator()
    assert allocator.next("g1") == 0
    assert allocator.next("g1") == 1
    assert allocator.next("g2") == 0
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/evaluation/test_trace_identity.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because `trace_identity.py` does not exist.

- [ ] **Step 3: Implement `trace_identity.py`**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass, field


def make_trace_id(
    *,
    game_id: str,
    player_id: str,
    phase: str,
    day_number: int,
    night_number: int,
    task_type: str,
    action_index: int,
) -> str:
    return (
        f"{game_id}:{player_id}:{phase}:"
        f"D{day_number}:N{night_number}:{task_type}:{action_index}"
    )


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


@dataclass
class ActionIndexAllocator:
    _next_by_game: dict[str, int] = field(default_factory=dict)

    def next(self, game_id: str) -> int:
        index = self._next_by_game.get(game_id, 0)
        self._next_by_game[game_id] = index + 1
        return index
```

- [ ] **Step 4: Delegate existing `make_trace_id`**

In `werewolf_agent/evaluation/trace_builder.py`, import `make_trace_id` from `trace_identity` and remove the local implementation body, preserving the public import:

```python
from werewolf_agent.evaluation.trace_identity import make_trace_id
```

If any callers import `EvaluationTraceBuilder.make_trace_id()` incorrectly, do not add a class method; keep the module-level function as the compatibility surface.

- [ ] **Step 5: Run identity and trace-builder tests**

Run:

```powershell
python -m pytest tests/evaluation/test_trace_identity.py tests/evaluation/test_feedback_trace_builder.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add werewolf_agent/evaluation/trace_identity.py werewolf_agent/evaluation/trace_builder.py tests/evaluation/test_trace_identity.py
git commit -m "feat: centralize evaluation trace identity"
```

---

### Task 2: Runtime Exposure Audit Events

**Files:**
- Create: `werewolf_agent/runtime/exposure_audit.py`
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/runtime/nodes/_shared.py`
- Modify: `werewolf_agent/runtime/nodes/day.py`
- Modify: `werewolf_agent/runtime/nodes/night.py`
- Modify: `werewolf_agent/runtime/nodes/skills.py`
- Modify: `werewolf_agent/runtime/nodes/sheriff.py`
- Modify: `werewolf_agent/runtime/nodes/sheriff_pk.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `tests/runtime/test_exposure_audit.py`
- Test: `tests/runtime/test_vote_flow.py`
- Test: `tests/runtime/test_graph_lifecycle.py`

- [ ] **Step 1: Write failing collector payload tests**

Test that the collector emits sanitized events for all four module types:

```python
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector


def _identity() -> DecisionIdentity:
    return DecisionIdentity("g1", "p01", "vote", 2, 1, "vote", 4)


def test_collector_builds_rag_reflection_skill_persona_events() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_rag(_identity(), [{"entry_id": "rag1", "rank": 1, "title": "safe"}])
    collector.record_reflection(_identity(), [{"entry_id": "ref1", "rank": 1}])
    collector.record_skill(_identity(), {"vote_analysis": "push p02"})
    collector.record_persona(_identity(), {"profile_id": "aggressive", "effective_params": {"risk": 0.8}})

    events = collector.flush_events()

    assert [event.type for event in events] == [
        "rag_exposure_audit",
        "reflection_exposure_audit",
        "skill_exposure_audit",
        "persona_exposure_audit",
    ]
    assert all(event.payload["trace_id"] == "g1:p01:vote:D2:N1:vote:4" for event in events)
    assert all(event.payload["visibility"] == "moderator_only" for event in events)
```

Add a negative test that forbidden keys are removed:

```python
def test_collector_strips_forbidden_private_fields() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_rag(_identity(), [{"entry_id": "x", "target_role": "werewolf"}])
    payload = collector.flush_events()[0].payload
    assert "target_role" not in str(payload)
    assert "werewolf" not in str(payload)
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/runtime/test_exposure_audit.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because collector does not exist.

- [ ] **Step 3: Implement collector**

In `werewolf_agent/runtime/exposure_audit.py`:

- Build `GameEvent` objects with common identity fields.
- Sanitize dict/list payload recursively.
- RAG `hits` keep only `entry_id`, `rank`, `relevance_score`, `prompt_visible`, `title`, `situation_signature`, `retrieval_reason`.
- Reflection `cards` keep only `entry_id`, `rank`, `quality_score`, `prompt_visible`, `lesson_key`, `quality_status`.
- Skill `analyses` convert `dict[str, str]` or `list[dict]` into compact rows with `skill_name`, `rank`, `prompt_visible`, `summary_hash`, `advice_type`.
- Persona `snapshot` keeps `profile_id`, `prompt_visible`, `policy_keys`, `sanitized`; never copy full `effective_params`.

- [ ] **Step 4: Extend `build_agent_context` signature**

In `werewolf_agent/runtime/context.py`, add optional keyword-only args:

```python
decision_identity: DecisionIdentity | None = None,
exposure_collector: ModuleExposureAuditCollector | None = None,
ablation_toggles: Any | None = None,
decision_trace_sink: DecisionTraceSink | None = None,
```

After the final `AgentContext` is built and RAG has been injected, record exposure from the returned context only:

```python
final_context = _inject_seed_rag_hints(...)
if decision_identity is not None and exposure_collector is not None:
    exposure_collector.record_rag(decision_identity, final_context.rag_hints)
    exposure_collector.record_reflection(decision_identity, final_context.reflection_memory_hints)
    exposure_collector.record_skill(decision_identity, final_context.skill_analyses)
    exposure_collector.record_persona(decision_identity, final_context.persona_snapshot)
return final_context
```

Do not record empty modules if the corresponding prompt-visible field is empty.

- [ ] **Step 5: Add action identity allocation helper**

In `werewolf_agent/runtime/nodes/_shared.py`, add a small runtime helper:

```python
def _allocate_decision_identity(
    state: RuntimeState,
    *,
    player_id: str,
    phase: str,
    task_type: str,
    day_number: int,
    night_number: int,
) -> DecisionIdentity:
    gs = state["game_state"]
    next_index_by_game = state.setdefault("action_index_by_game", {})
    action_index = int(next_index_by_game.get(gs.game_id, 0))
    next_index_by_game[gs.game_id] = action_index + 1
    return DecisionIdentity(
        game_id=gs.game_id,
        player_id=player_id,
        phase=phase,
        day_number=day_number,
        night_number=night_number,
        task_type=task_type,
        action_index=action_index,
    )
```

Also extend `RuntimeState` with:

```python
action_index_by_game: dict[str, int]
pending_exposure_events_by_trace: dict[str, list[GameEvent]]
```

- [ ] **Step 6: Attach trace id and exposure events beside action audit**

Extend `_action_trace_event(...)` to accept:

```python
decision_identity: DecisionIdentity | None = None
```

When present, write `trace_id` and use identity fields in payload. Add helper:

```python
def _action_audit_events(
    *,
    state: RuntimeState,
    player_id: str,
    phase: str,
    action_trace: dict[str, Any],
    decision_identity: DecisionIdentity | None,
    exposure_collector: ModuleExposureAuditCollector | None,
    day_number: int = 0,
    night_number: int = 0,
) -> list[GameEvent]:
    event = _action_trace_event(
        player_id=player_id,
        phase=phase,
        action_trace=action_trace,
        day_number=day_number,
        night_number=night_number,
        decision_identity=decision_identity,
    )
    exposure_events = exposure_collector.flush_events() if exposure_collector else []
    return [*exposure_events, event]
```

Flush only after the action trace exists. If an adapter builds context but returns no action trace, discard the collector.

- [ ] **Step 7: Thread identity through adapter calls**

In `werewolf_agent/runtime/agent_adapter.py`, at each `build_agent_context(...)` call:

- accept optional `decision_identity` and `exposure_collector` parameters in adapter function signatures used by runtime nodes;
- if an adapter is called outside graph nodes/tests without identity, keep backward-compatible behavior with no exposure audit;
- pass `decision_identity`, `exposure_collector`, and optional `decision_trace_sink` into `build_agent_context`.

Do not allocate inside `build_agent_context`; allocation belongs to runtime dispatch.

- [ ] **Step 8: Update node call sites**

For each node that emits `_action_trace_event`, allocate identity before calling the agent adapter/context builder. Pass the same identity to:

- adapter/context construction;
- exposure collector;
- `_action_trace_event`.

Keep emitted event order as exposure events first, then `action_trace_audit`, so trace builder can scan either direction without relying on order.

- [ ] **Step 9: Run runtime tests**

Run:

```powershell
python -m pytest tests/runtime/test_exposure_audit.py tests/runtime/test_vote_flow.py tests/runtime/test_graph_lifecycle.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add werewolf_agent/runtime/exposure_audit.py werewolf_agent/runtime/context.py werewolf_agent/runtime/nodes/_shared.py werewolf_agent/runtime/nodes/day.py werewolf_agent/runtime/nodes/night.py werewolf_agent/runtime/nodes/skills.py werewolf_agent/runtime/nodes/sheriff.py werewolf_agent/runtime/nodes/sheriff_pk.py werewolf_agent/runtime/agent_adapter.py tests/runtime/test_exposure_audit.py tests/runtime/test_vote_flow.py tests/runtime/test_graph_lifecycle.py
git commit -m "feat: emit runtime module exposure audits"
```

---

### Task 3: Trace Builder Runtime Exposure Join

**Files:**
- Modify: `werewolf_agent/evaluation/trace_builder.py`
- Test: `tests/evaluation/test_runtime_exposure_trace_join.py`
- Test: `tests/evaluation/test_feedback_trace_builder.py`
- Test: `tests/evaluation/test_feedback_metrics.py`

- [ ] **Step 1: Write failing event-log join tests**

Add a result with `rag_exposure_audit`, `reflection_exposure_audit`, `skill_exposure_audit`, `persona_exposure_audit`, and `action_trace_audit` all carrying the same `trace_id`. Call:

```python
traces = EvaluationTraceBuilder().build(result)
```

Assert there is no `missing_source` exposure and modules include all four runtime exposure modules.

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/evaluation/test_runtime_exposure_trace_join.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because builder only uses the side-channel.

- [ ] **Step 3: Scan event log for exposure events**

In `EvaluationTraceBuilder.build(...)`:

```python
runtime_exposure_audits = self._collect_exposure_events(result.event_log)
side_channel_audits = exposure_audits or []
exposure_by_trace = self._group_exposure_audits(runtime_exposure_audits + side_channel_audits)
exposure_sources_provided = bool(runtime_exposure_audits or side_channel_audits)
```

When an `action_trace_audit` payload already has `trace_id`, use it directly. Only compute fallback trace id when runtime did not provide one.

- [ ] **Step 4: Support skill/persona exposure conversion**

Extend `_group_exposure_audits`:

```python
elif audit_type == "skill_exposure_audit":
    grouped[trace_id].extend(_skill_exposures(audit))
elif audit_type == "persona_exposure_audit":
    grouped[trace_id].extend(_persona_exposures(audit))
```

Map skill analyses to `ModuleExposure(module="skills", item_id=skill_name, ...)`.
Map persona snapshot to `ModuleExposure(module="persona", item_id=profile_id, ...)`.

- [ ] **Step 5: Keep side-channel compatibility**

Existing tests that pass `exposure_audits=[...]` must continue to pass. Add one test where event log has no exposure events but side-channel does.

- [ ] **Step 6: Run evaluation tests**

Run:

```powershell
python -m pytest tests/evaluation/test_runtime_exposure_trace_join.py tests/evaluation/test_feedback_trace_builder.py tests/evaluation/test_feedback_metrics.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/evaluation/trace_builder.py tests/evaluation/test_runtime_exposure_trace_join.py tests/evaluation/test_feedback_trace_builder.py
git commit -m "feat: join evaluation traces from runtime exposure events"
```

---

### Task 4: Full-Game Same-Seed Ablation

**Files:**
- Create: `werewolf_agent/evaluation/full_game_ablation.py`
- Create: `werewolf_agent/evaluation/replay.py`
- Modify: `werewolf_agent/evaluation/ablation.py`
- Modify: `werewolf_agent/evaluation/feedback_report.py`
- Test: `tests/evaluation/test_full_game_ablation.py`
- Test: `tests/evaluation/test_live_ablation_harness.py`

- [ ] **Step 1: Write failing config/report tests**

Cover:

- `FullGameAblationConfig` includes all approved namespace and replay fields.
- `agent_mode="live_model"` with no replay capture marks causal metrics unsupported.
- deterministic fake paired report uses baseline minus ablated deltas.
- missing replay output marks affected metrics unsupported.

Example:

```python
def test_live_model_without_replay_reports_unsupported_causal_metrics() -> None:
    runner = FullGameAblationRunner(game_runner_factory=None)
    report = runner.run(FullGameAblationConfig(
        batch_id="b1",
        seed_set=[1],
        removed_modules=["rag"],
        player_count=12,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        ruleset_snapshot={"id": "rules"},
        agent_mode="live_model",
        model_config_snapshot={"provider": "test"},
        baseline_storage_namespace="baseline",
        ablated_storage_namespace="ablated",
        replay_policy="unsupported_live_model",
    ))
    assert report.unsupported_metrics["live_win_rate_delta"] == "fresh_live_model_without_replay"
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/evaluation/test_full_game_ablation.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement schemas and unsupported mode**

In `full_game_ablation.py`, implement:

- `FullGameAblationConfig`
- `FullGameMetricDelta`
- `FullGameAblationPair`
- `FullGameAblationReport`
- `FullGameAblationRunner`

For unsupported live model:

```python
return FullGameAblationReport(
    batch_id=config.batch_id,
    mode="full_game",
    agent_mode=config.agent_mode,
    removed_modules=list(config.removed_modules),
    pair_count=0,
    metric_deltas={},
    unsupported_metrics={
        "live_win_rate_delta": "fresh_live_model_without_replay",
        "causal_decision_delta": "fresh_live_model_without_replay",
    },
    pairs=[],
)
```

- [ ] **Step 4: Implement deterministic fake runner seam**

Accept injected `game_runner_factory` so tests can use fake results:

```python
class FullGameAblationRunner:
    def __init__(self, game_runner_factory: Callable[..., GameResult] | None = None) -> None:
        self._game_runner_factory = game_runner_factory
```

The fake factory must receive seed, namespace, removed modules, ruleset snapshot, and mode. Do not call real LLMs.

- [ ] **Step 5: Implement replay matcher**

In `replay.py`, implement `ReplayArtifact`, `ReplayRecord`, and `ReplayMatcher`:

- `trace_id` mode requires exact trace id.
- `event_order` mode requires exact action count/order.
- missing output returns unsupported reason, never fresh live fallback.

- [ ] **Step 6: Apply module toggles at context seam**

Use existing `apply_ablation_toggles` for context-level field removal. Full-game runner should pass removed modules into runtime config or injected fake runner; do not duplicate field removal logic.

- [ ] **Step 7: Extend feedback report serialization**

Update `feedback_report.py` to serialize full-game reports separately from existing offline/context ablation reports:

```json
"full_game_ablations": [...]
```

Keep the existing `"ablations"` key backward compatible for `AblationReport`.

- [ ] **Step 8: Run ablation tests**

Run:

```powershell
python -m pytest tests/evaluation/test_full_game_ablation.py tests/evaluation/test_live_ablation_harness.py tests/evaluation/test_feedback_report.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add werewolf_agent/evaluation/full_game_ablation.py werewolf_agent/evaluation/replay.py werewolf_agent/evaluation/ablation.py werewolf_agent/evaluation/feedback_report.py tests/evaluation/test_full_game_ablation.py tests/evaluation/test_live_ablation_harness.py tests/evaluation/test_feedback_report.py
git commit -m "feat: add reproducible full game ablation runner"
```

---

### Task 5: Candidate Store, Materialization, And Regression Gate

**Files:**
- Create: `werewolf_agent/evaluation/candidate_store.py`
- Create: `werewolf_agent/evaluation/candidate_materializer.py`
- Create: `werewolf_agent/evaluation/regression_gate.py`
- Modify: `werewolf_agent/evaluation/feedback_schemas.py`
- Modify: `werewolf_agent/evaluation/candidates.py`
- Modify: `werewolf_agent/evaluation/feedback_report.py`
- Test: `tests/evaluation/test_candidate_store.py`
- Test: `tests/evaluation/test_candidate_materializer.py`
- Test: `tests/evaluation/test_regression_gate.py`
- Test: `tests/evaluation/test_improvement_candidates.py`

Status: implemented in commit pending for this working session. The final
implementation keeps candidate generation review-only, stores explicit reviewer
metadata for approval/rejection/materialization/rollback transitions, writes
RAG/reflection candidates only to adapter-owned draft namespaces, gates draft
promotion with baseline-vs-candidate regression checks, and exposes
`candidate_workflow` in feedback reports without private audit evidence in the
public view.

- [x] **Step 1: Write failing store transition tests**

Cover allowed and forbidden transitions:

```python
def test_candidate_store_approval_records_reviewer_metadata() -> None:
    store = InMemoryCandidateStore()
    store.add(candidate)
    record = store.approve(candidate.candidate_id, reviewer="mod1", notes="safe")
    assert record.status is CandidateStatus.APPROVED
    assert record.reviewer == "mod1"
    assert record.notes == "safe"
```

Also assert rejected candidates cannot be approved/materialized without reset.

- [x] **Step 2: Run failing store tests**

Run:

```powershell
python -m pytest tests/evaluation/test_candidate_store.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because module does not exist.

- [x] **Step 3: Implement store**

In `candidate_store.py`, implement:

- `CandidateStatus`
- `CandidateRecord`
- `CandidateStore` protocol
- `InMemoryCandidateStore`
- `RepositoryCandidateStore` facade that calls repository methods when present and otherwise raises clear `NotImplementedError`

`CandidateRecord` must store reviewer, notes, and timestamp for approve/reject writes. Use an injectable clock in tests so timestamps are deterministic.

Statuses:

```python
PENDING -> APPROVED
PENDING -> REJECTED
APPROVED -> MATERIALIZED
MATERIALIZED -> ROLLED_BACK
```

Do not allow direct `PENDING -> MATERIALIZED`.

- [x] **Step 4: Write materializer tests**

Cover:

- RAG candidate materializes as draft-only `review_only`.
- Reflection candidate materializes as draft-only `review_only`.
- prompt-unsafe candidate is rejected and not written.
- failed gate rollback quarantines draft and never promotes.
- passed gate promotes to approved in live-eligible namespace and stores source diagnosis/regression metadata.

- [x] **Step 5: Implement materializer**

In `candidate_materializer.py`, implement:

- `CandidateMaterializationResult`
- `DraftNamespace`
- `CandidateMaterializer.materialize_draft(candidate, namespace)`
- `CandidateMaterializer.promote(candidate_id, regression_report_ref)`
- `CandidateMaterializer.rollback(candidate_id, reason)`

Use adapters around existing RAG/reflection stores. If the exact store APIs differ, implement small local helper methods with tests using fakes:

```python
class DraftRAGStoreAdapter:
    def save_review_only(self, entry: Any, namespace: str) -> str: ...
    def promote(self, draft_id: str, live_namespace: str, metadata: dict[str, Any]) -> str: ...
```

Do not query draft namespaces from live runtime.

- [x] **Step 6: Write regression gate tests**

Cover pass/fail for:

- leakage increase blocks;
- illegal action increase blocks;
- vote quality drop beyond tolerance blocks;
- target faction win-rate drop beyond tolerance blocks;
- harmful transfer increase blocks;
- normalized UI values do not require sign inference.

- [x] **Step 7: Implement regression gate**

In `regression_gate.py`, implement:

- `CandidateRegressionConfig`
- `CandidateMetricDelta`
- `CandidateRegressionReport`
- `GateCheck`
- `GateResult`
- `RegressionGate`

Use sign conventions:

- improvement metric delta = `candidate - baseline`
- harm metric delta = `baseline - candidate`
- `GateCheck.normalized_value` is always "higher is better for passing" or explicitly stores comparator, threshold, and passed boolean.

- [x] **Step 8: Integrate report serialization**

In `feedback_report.py`, include candidate records and gate results when present:

```json
"candidate_workflow": {
  "pending": 0,
  "approved": 1,
  "materialized": 1,
  "rolled_back": 0,
  "gate_results": [...]
}
```

Public view must not include hidden trace evidence or raw diagnosis evidence.

- [x] **Step 9: Run candidate tests**

Run:

```powershell
python -m pytest tests/evaluation/test_candidate_store.py tests/evaluation/test_candidate_materializer.py tests/evaluation/test_regression_gate.py tests/evaluation/test_improvement_candidates.py tests/evaluation/test_feedback_report.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add werewolf_agent/evaluation/candidate_store.py werewolf_agent/evaluation/candidate_materializer.py werewolf_agent/evaluation/regression_gate.py werewolf_agent/evaluation/feedback_schemas.py werewolf_agent/evaluation/candidates.py werewolf_agent/evaluation/feedback_report.py tests/evaluation/test_candidate_store.py tests/evaluation/test_candidate_materializer.py tests/evaluation/test_regression_gate.py tests/evaluation/test_improvement_candidates.py tests/evaluation/test_feedback_report.py
git commit -m "feat: add candidate approval and regression gate"
```

---

### Task 6: Decision Trace Sink And Optional LangSmith Runtime Trace

**Files:**
- Create: `werewolf_agent/evaluation/decision_trace.py`
- Modify: `werewolf_agent/evaluation/langsmith_exporter.py`
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/runtime/nodes/_shared.py`
- Test: `tests/evaluation/test_decision_trace_sink.py`
- Test: `tests/evaluation/test_langsmith_exporter.py`

- [ ] **Step 1: Write failing sink/redaction tests**

Cover:

- `NoopDecisionTraceSink` no-ops.
- `InMemoryDecisionTraceSink` records stage order.
- `LangSmithDecisionTraceSink` does not import `langsmith` at module import time.
- fake injected client receives redacted payload.
- raw prompt/output are omitted by default.
- debug raw output flag still redacts hidden truth keys.

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/evaluation/test_decision_trace_sink.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement sink protocol and in-memory sink**

In `decision_trace.py`:

```python
class DecisionTraceSink(Protocol):
    def start_decision(self, identity: DecisionIdentity, metadata: dict[str, Any]) -> None: ...
    def record_stage(self, identity: DecisionIdentity, stage: str, payload: dict[str, Any]) -> None: ...
    def finish_decision(self, identity: DecisionIdentity, outcome: dict[str, Any]) -> None: ...
```

Implement `NoopDecisionTraceSink`, `InMemoryDecisionTraceSink`, `TraceRedactor`, and `LangSmithDecisionTraceSink`.

- [ ] **Step 4: Add stage recording at existing seams**

Record stages without changing prompt behavior:

- `context.build.start` before context assembly.
- `rag.retrieve` after RAG injection returns prompt lines.
- `reflection.retrieve` after reflection hints are attached.
- `worlds.generate` after possible worlds dict is built.
- `simulator.predict` after simulation predictions dict is built.
- `skills.apply` after skill analyses/directives are built.
- `persona.policy` after persona snapshot is attached.
- `prompt.render`, `llm.call`, `parse.result` inside agent adapter/player call seam if accessible without large refactor.
- `action.audit` when `_action_trace_event` is emitted.
- `evaluation.trace` in trace builder only when sink is explicitly passed; do not add hidden global state.

If a stage is not reachable without destabilizing `PlayerAgent`, leave a TODO-free no-op seam and document the exact missing hook in the test name. Do not fake stages that were not observed.

- [ ] **Step 5: Preserve report-level exporter**

Keep `LangSmithFeedbackExporter` behavior in `langsmith_exporter.py`; share redactor helpers if useful, but do not merge report exporter with runtime decision sink.

- [ ] **Step 6: Run tracing tests**

Run:

```powershell
python -m pytest tests/evaluation/test_decision_trace_sink.py tests/evaluation/test_langsmith_exporter.py tests/runtime/test_context.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/evaluation/decision_trace.py werewolf_agent/evaluation/langsmith_exporter.py werewolf_agent/runtime/context.py werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/nodes/_shared.py tests/evaluation/test_decision_trace_sink.py tests/evaluation/test_langsmith_exporter.py tests/runtime/test_context.py
git commit -m "feat: add redacted decision trace sinks"
```

---

### Task 7: Evaluation Artifact Stores

**Files:**
- Create: `werewolf_agent/evaluation/artifact_store.py`
- Test: `tests/evaluation/test_evaluation_artifact_store.py`

- [ ] **Step 1: Write failing artifact store tests**

Cover storing and retrieving:

- feedback reports by `batch_id`;
- ablation reports by `batch_id`;
- regression reports by `candidate_id`;
- optional trace links by `trace_id`;
- trace links listed by `batch_id` so dashboards can discover links for failed trace clusters without knowing every trace id in advance;
- missing artifacts return `None` or empty list consistently.

Example:

```python
def test_in_memory_artifact_store_round_trips_reports() -> None:
    store = InMemoryEvaluationArtifactStore()
    store.save_feedback_report("b1", {"batch_id": "b1", "trace_count": 2})
    store.save_ablation_report("b1", {"batch_id": "b1", "removed_modules": ["rag"]})
    store.save_regression_report("c1", {"candidate_id": "c1", "passed": True})
    store.save_trace_link("b1", "t1", {"trace_id": "t1", "langsmith_url": "https://example.invalid/t1"})

    assert store.get_feedback_report("b1")["trace_count"] == 2
    assert store.get_ablation_report("b1")["removed_modules"] == ["rag"]
    assert store.get_regression_report("c1")["passed"] is True
    assert store.get_trace_link("t1")["trace_id"] == "t1"
    assert store.list_trace_links("b1")[0]["trace_id"] == "t1"
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/evaluation/test_evaluation_artifact_store.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because artifact store does not exist.

- [ ] **Step 3: Implement minimal store contract**

In `artifact_store.py`, implement:

```python
class EvaluationArtifactStore(Protocol):
    def save_feedback_report(self, batch_id: str, payload: dict[str, Any]) -> None: ...
    def get_feedback_report(self, batch_id: str) -> dict[str, Any] | None: ...
    def save_ablation_report(self, batch_id: str, payload: dict[str, Any]) -> None: ...
    def get_ablation_report(self, batch_id: str) -> dict[str, Any] | None: ...
    def save_regression_report(self, candidate_id: str, payload: dict[str, Any]) -> None: ...
    def get_regression_report(self, candidate_id: str) -> dict[str, Any] | None: ...
    def save_trace_link(self, batch_id: str, trace_id: str, payload: dict[str, Any]) -> None: ...
    def get_trace_link(self, trace_id: str) -> dict[str, Any] | None: ...
    def list_trace_links(self, batch_id: str) -> list[dict[str, Any]]: ...
```

Implement:

- `InMemoryEvaluationArtifactStore`
- `RepositoryEvaluationArtifactStore` that delegates to repository methods when present and raises clear `NotImplementedError` when a write method is absent.

All returned payloads should be shallow copies to avoid accidental mutation.

- [ ] **Step 4: Run artifact store tests**

Run:

```powershell
python -m pytest tests/evaluation/test_evaluation_artifact_store.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/evaluation/artifact_store.py tests/evaluation/test_evaluation_artifact_store.py
git commit -m "feat: add evaluation artifact store"
```

---

### Task 8: Evaluation Workflow API

**Files:**
- Create: `werewolf_agent/api/routes/evaluation.py`
- Modify: `werewolf_agent/api/app.py`
- Modify: `werewolf_agent/api/schemas.py`
- Modify: `werewolf_agent/api/views.py`
- Modify: `werewolf_agent/evaluation/artifact_store.py`
- Test: `tests/api/test_evaluation_workflow_api.py`
- Test: `tests/api/test_auth.py`
- Test: `tests/api/test_evaluation.py`

- [ ] **Step 1: Write failing API auth tests**

Cover:

- `GET /api/evaluation/feedback/report/{batch_id}` returns public redacted aggregate for public/spectator if allowed by route design, but no hidden trace refs.
- ablation/candidate/regression endpoints require moderator/debugger.
- approve/reject require reviewer identity and notes/timestamp.
- elevated query-param auth uses existing `authorized_callers`.
- no `CallerRole.EVALUATOR` or new view mode is introduced.

If the app has no `/api` prefix currently, mount these paths exactly as `/api/evaluation/...` by giving the new router `prefix="/api/evaluation"`.

- [ ] **Step 2: Run failing API tests**

Run:

```powershell
python -m pytest tests/api/test_evaluation_workflow_api.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because route does not exist.

- [ ] **Step 3: Implement route module**

In `werewolf_agent/api/routes/evaluation.py`, implement:

```python
def create_evaluation_router(
    *,
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    candidate_store: CandidateStore,
    artifact_store: EvaluationArtifactStore,
) -> APIRouter:
    router = APIRouter(prefix="/api/evaluation")
```

Endpoint list:

- `GET /feedback/report/{batch_id}`
- `GET /ablation/{batch_id}`
- `GET /candidates`
- `POST /candidates/{id}/approve`
- `POST /candidates/{id}/reject`
- `GET /regressions/{candidate_id}`
- `GET /trace-links/{batch_id}`

Use helper:

```python
def _require_evaluation_admin(
    *,
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    caller_id: str,
    caller_role: CallerRole,
    session_token: str = "",
) -> CallerRole:
    resolved_role = _resolve_evaluation_caller_role(
        authorized_callers=authorized_callers,
        auth=auth,
        caller_id=caller_id,
        requested_role=caller_role,
        session_token=session_token,
    )
    if resolved_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        raise HTTPException(403, "evaluation workflow requires moderator or debugger role")
    return resolved_role
```

- [ ] **Step 4: Mount router in app**

In `api/app.py`, create in-memory stores:

```python
candidate_store = InMemoryCandidateStore()
evaluation_artifact_store = InMemoryEvaluationArtifactStore()
```

Mount:

```python
from werewolf_agent.api.routes.evaluation import create_evaluation_router
app.include_router(create_evaluation_router(
    authorized_callers=authorized_callers,
    auth=auth,
    candidate_store=candidate_store,
    artifact_store=evaluation_artifact_store,
))
```

Expose on `app.state` for tests.

- [ ] **Step 5: Add compact response schemas only where useful**

Prefer dict responses if existing API style allows. If adding Pydantic schemas, keep them in `api/schemas.py` and use existing enum types.

- [ ] **Step 6: Redact public data**

Public report payload must omit:

- hidden roles/factions;
- outcome labels;
- diagnosis evidence;
- raw trace payloads;
- LangSmith URLs;
- player IDs in prompt-safe candidate audit evidence where public unsafe.

Moderator/debugger payload may include private audit data, but still no raw prompts/model outputs by default.

- [ ] **Step 7: Run API tests**

Run:

```powershell
python -m pytest tests/api/test_evaluation_workflow_api.py tests/api/test_auth.py tests/api/test_evaluation.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add werewolf_agent/api/routes/evaluation.py werewolf_agent/api/app.py werewolf_agent/api/schemas.py werewolf_agent/api/views.py werewolf_agent/evaluation/artifact_store.py tests/api/test_evaluation_workflow_api.py tests/api/test_auth.py tests/api/test_evaluation.py
git commit -m "feat: expose evaluation feedback workflow api"
```

---

### Task 9: Feedback Dashboard Panels

**Files:**
- Modify: `werewolf_agent/ui/static/dashboard.html`
- Modify: `werewolf_agent/ui/static/dashboard.js`
- Modify: `werewolf_agent/ui/static/dashboard.css`
- Test: `tests/ui/test_feedback_dashboard.py`
- Test: `tests/ui/test_dashboard.py`
- Test: `tests/ui/test_dashboard_enhanced.py`

- [ ] **Step 1: Write failing dashboard static tests**

Add tests that read static files and assert:

- dashboard contains a feedback closure section or panels with stable IDs:
  - `feedbackExposureCoverage`
  - `feedbackAblationDeltas`
  - `feedbackFailureClusters`
  - `feedbackCandidateQueue`
  - `feedbackRegressionGate`
  - `feedbackTraceLinks`
- `dashboard.js` calls `/api/evaluation/feedback/report/`, `/api/evaluation/ablation/`, `/api/evaluation/candidates`, and `/api/evaluation/regressions/`.
- `dashboard.js` calls `/games/{game_id}/world-model-audit` or renders a stable world-model audit link from feedback data.
- `dashboard.js` calls `/api/evaluation/trace-links/` for optional sanitized trace links.
- public rendering path does not expose `target_role`, `target_faction`, `ground_truth`, or raw LangSmith URLs.

- [ ] **Step 2: Run failing UI tests**

Run:

```powershell
python -m pytest tests/ui/test_feedback_dashboard.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because panels are not present.

- [ ] **Step 3: Add dashboard markup**

In `dashboard.html`, add a moderator/debugger-only panel block near the existing world-model/RAG panels:

```html
<div class="panel mod-section" id="feedback-loop-panel">
  <div class="panel-header">Feedback Closure Loop</div>
  <div class="panel-body feedback-grid">
    <div id="feedbackExposureCoverage" class="feedback-card"></div>
    <div id="feedbackAblationDeltas" class="feedback-card"></div>
    <div id="feedbackFailureClusters" class="feedback-card"></div>
    <div id="feedbackCandidateQueue" class="feedback-card"></div>
    <div id="feedbackRegressionGate" class="feedback-card"></div>
    <div id="feedbackTraceLinks" class="feedback-card"></div>
  </div>
</div>
```

Keep it compact and consistent with existing dashboard styling.

- [ ] **Step 4: Add dashboard loader**

In `dashboard.js`, extend `loadEnhancedPanels(p)` with `loadFeedbackWorkflow(p)`.

`loadFeedbackWorkflow` should:

- use a deterministic batch id source, initially `currentGame`;
- fetch feedback report and candidates;
- fetch ablation/regression only when IDs are present or current game is available;
- render a world-model audit link for the current game when moderator/debugger view is active;
- fetch sanitized trace links through `/api/evaluation/trace-links/{batchId}` and show only labels/trace ids unless the API explicitly returns a moderator-safe URL;
- render unsupported metrics as explicit statuses, not zero values;
- never render LangSmith URLs or raw trace payloads unless current view is moderator/debugger and API returned sanitized links.

- [ ] **Step 5: Add CSS**

In `dashboard.css`, add:

```css
.feedback-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.feedback-card { min-height: 120px; padding: 10px; border: 1px solid rgba(127, 153, 199, 0.28); border-radius: 8px; background: rgba(30, 48, 85, 0.9); }
.feedback-status.pass { color: var(--green); }
.feedback-status.fail { color: var(--red); }
.feedback-status.unsupported { color: var(--muted); }
```

Maintain responsive single-column layout on narrow screens.

- [ ] **Step 6: Run UI tests**

Run:

```powershell
python -m pytest tests/ui/test_feedback_dashboard.py tests/ui/test_dashboard.py tests/ui/test_dashboard_enhanced.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/ui/static/dashboard.html werewolf_agent/ui/static/dashboard.js werewolf_agent/ui/static/dashboard.css tests/ui/test_feedback_dashboard.py tests/ui/test_dashboard.py tests/ui/test_dashboard_enhanced.py
git commit -m "feat: add feedback workflow dashboard panels"
```

---

### Task 10: End-To-End Feedback Report Integration

**Files:**
- Modify: `werewolf_agent/evaluation/runner.py`
- Modify: `werewolf_agent/evaluation/reports.py`
- Modify: `werewolf_agent/evaluation/feedback_report.py`
- Modify: `tests/integration/test_evaluation_live_game.py`
- Test: `tests/integration/test_evaluation_live_game.py`
- Test: `tests/evaluation/test_feedback_report.py`

- [ ] **Step 1: Write failing integration test**

Use deterministic fake data, not a real model:

- Build `GameResult` with runtime exposure events and action traces.
- Build `EvaluationTrace` without side-channel exposure.
- Build diagnostics/candidates.
- Run candidate gate with fake pass report.
- Build feedback report.
- Assert report contains module exposure coverage, ablation unsupported reasons, candidate workflow summary, and no private leakage in public JSON.

- [ ] **Step 2: Run failing integration target**

Run:

```powershell
python -m pytest tests/integration/test_evaluation_live_game.py::test_feedback_closure_loop_uses_runtime_exposures_without_side_channel -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL until integration glue exists.

- [ ] **Step 3: Add runner glue**

In existing evaluation runner/report modules, add helpers that compose:

```text
GameResult -> EvaluationTraceBuilder -> metrics -> diagnostics -> candidates -> FeedbackReport
```

Do not auto-approve or materialize candidates from this helper.

- [ ] **Step 4: Run integration tests**

Run:

```powershell
python -m pytest tests/integration/test_evaluation_live_game.py tests/evaluation/test_feedback_report.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/evaluation/runner.py werewolf_agent/evaluation/reports.py werewolf_agent/evaluation/feedback_report.py tests/integration/test_evaluation_live_game.py tests/evaluation/test_feedback_report.py
git commit -m "feat: integrate feedback closure report pipeline"
```

---

### Task 11: Documentation And Progress Update

**Files:**
- Modify: `docs/implementation_plan.md`
- Modify: `docs/design/werewolf-agent-v1-design.md`
- Modify: `docs/superpowers/specs/2026-06-16-feedback-closure-loop-design.md` only if implementation discovers a contract correction that must be reflected.
- Test: documentation review via grep/read commands.

- [ ] **Step 1: Update progress doc**

In `docs/implementation_plan.md`, add a dated progress section:

```markdown
## 2026-06-16 Feedback Closure Loop

- Runtime exposure audits: implemented with `trace_id` join identity.
- Full-game ablation: deterministic fake/replay supported; fresh live-model causal metrics explicitly unsupported without replay.
- Candidate workflow: pending -> approved/rejected -> draft materialized -> regression gated -> promoted/rolled back.
- LangSmith: optional redacted decision trace sink; no network dependency in tests.
- API: moderator/debugger workflow under `/api/evaluation`.
```

- [ ] **Step 2: Update architecture doc**

In `docs/design/werewolf-agent-v1-design.md`, add or update the feedback/evaluation section with:

- observation layer emits runtime module exposure audit events;
- evaluation layer joins by `trace_id`;
- experimentation layer uses deterministic fake/replay same-seed ablation;
- improvement layer requires approval and regression gate;
- governance layer provides API/dashboard and optional LangSmith.

- [ ] **Step 3: Verify no forbidden identity names leaked**

Run:

```powershell
rg -n "decision_key|decision_ref|CallerRole\.EVALUATOR|ViewMode\.EVALUATOR" werewolf_agent tests docs
```

Expected: no matches except historical docs explicitly marked obsolete. Prefer no matches.

- [ ] **Step 4: Verify docs mention unsupported live causal metrics**

Run:

```powershell
rg -n "fresh_live_model_without_replay|unsupported_live_model|strict_replay" docs werewolf_agent tests
```

Expected: matches in spec, plan, implementation, and tests.

- [ ] **Step 5: Commit**

```powershell
git add docs/implementation_plan.md docs/design/werewolf-agent-v1-design.md docs/superpowers/specs/2026-06-16-feedback-closure-loop-design.md
git commit -m "docs: update feedback closure loop progress"
```

---

### Task 12: Full Regression And Final Review

**Files:**
- No planned implementation files.
- May modify tests/docs only if regression exposes a real issue.

- [ ] **Step 1: Run focused feedback closure regression**

Run:

```powershell
python -m pytest tests/evaluation tests/runtime tests/rag tests/memory tests/api tests/ui -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS. If `tests/ui` does not exist, rerun without it and document that it is absent.

- [ ] **Step 2: Run broad existing regression slices**

Run:

```powershell
python -m pytest tests/agents tests/cognition tests/integration -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS or documented pre-existing failures. Do not mark completion with unexplained failures.

- [ ] **Step 3: Check git status**

Run:

```powershell
git status --short
```

Expected: clean except intentional uncommitted files if the user requested no commit. For this user, commit all finished implementation changes.

- [ ] **Step 4: Request code review**

Use `requesting-code-review` before final integration. Reviewer should focus on:

- trace identity drift;
- runtime exposure privacy;
- candidate promotion safety;
- full-game ablation causal claims;
- API auth/redaction.

- [ ] **Step 5: Fix review findings**

Use `receiving-code-review`; verify every accepted finding with targeted tests.

- [ ] **Step 6: Final commit if needed**

```powershell
git add <changed-files>
git commit -m "fix: harden feedback closure loop review findings"
```

- [ ] **Step 7: Final status**

Report:

- commits made;
- tests run;
- unsupported metrics intentionally reported;
- any remaining risk or skipped tests.

---

## Execution Notes

- Prefer subagent-driven execution after this plan is approved:
  - Task 1-3 can be one runtime/evaluation worker.
  - Task 4 can be one ablation worker.
  - Task 5 can be one candidate/governance worker.
  - Task 6 can be one tracing worker.
  - Task 7 can be one artifact-store worker.
  - Task 8 can be one API worker.
  - Task 9 can be one dashboard worker.
  - Main agent integrates and runs final regression.
- Workers are not alone in the codebase. They must not revert user edits or other workers' edits.
- Each worker must write tests first, run the expected failing test, implement minimally, rerun, and report changed files.
- Keep commits small and phase-aligned. If a task spans many files, commit after the tested sub-slice is green.
- Avoid real model/network/DB calls. Use fake clients, fake stores, deterministic fake game runners, and in-memory repositories.
