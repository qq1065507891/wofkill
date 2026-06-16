# Evaluation Feedback Loop Design

Date: 2026-06-16
Status: Proposed design
Owner: Codex development session

## Problem

The project now has working implementations for possible worlds, bounded
simulation, RAG V2 tactical cards, reflection memory V2, action trace audits,
and evaluation metrics. These modules are useful, but they are not yet connected
as a learning loop.

The current system can answer questions such as:

- What did the agent see?
- Which RAG cards were injected?
- Which possible worlds were shown?
- Did a world-model top-k card match the true role assignment?
- Did the final action satisfy the legal action contract?

It cannot yet reliably answer the higher-value questions:

- Did RAG improve or harm the decision?
- Did reflection memory change later behavior in the intended direction?
- Did the possible-world scorer rank the true hidden state well, or only hit
  top-k by chance?
- Did simulator predictions help the final action or distract from evidence?
- Which failed decisions should create new RAG, reflection, or scoring updates?
- After an update, does the same seed set improve without regressing safety?

Without this loop, the system can accumulate more context without proving that
the context makes agents smarter.

## Current Implementation Baseline

The current codebase already provides the required raw material:

- `werewolf_agent.cognition.worlds.PossibleWorldsEngine` generates bounded
  top-k hidden-role hypotheses from role counts and belief summaries.
- `werewolf_agent.cognition.simulator.BoundedSimulator` generates compact
  next-turn predictions from possible worlds.
- `werewolf_agent.runtime.world_model_audit` extracts sanitized
  `world_model_audit` payloads from `action_trace_audit` events.
- `werewolf_agent.rag.schemas` defines RAG V2 tactical frames.
- `werewolf_agent.rag.knowledge_service` retrieves live RAG hints under
  metadata and visibility boundaries.
- `werewolf_agent.rag.prompt_renderer` emits prompt-safe live RAG cards.
- `werewolf_agent.memory.reflection.ReflectionMemory` stores and retrieves
  approved V2 reflection cards.
- `werewolf_agent.evaluation.metrics.MetricsAggregator` computes faction,
  quality, safety, cost, and world-model metrics.
- `werewolf_agent.evaluation.runner.BatchRunner` can produce deterministic
  batch results for reproducible metric tests.

The main gap is not missing instrumentation. The main gap is feedback control:
failed traces are not converted into diagnoses, diagnoses are not converted
into reviewable improvement candidates, and candidates are not verified against
same-seed regression batches.

There is one important baseline constraint: not every exposure source is already
persisted in the event log. `world_model_audit` is available through sanitized
`action_trace_audit` payloads, while RAG and reflection exposure data may still
exist only in live context, service memory, or prompt construction paths. The
first implementation phase must therefore define the exposure-source contract
before computing attribution metrics.

## Goals

1. Add a feedback loop that turns game traces into actionable diagnostics.
2. Measure the impact of RAG, reflection, possible worlds, and simulator
   predictions separately.
3. Support same-seed ablation runs for module on/off comparisons.
4. Add module-specific evaluation metrics beyond aggregate win rate.
5. Generate human-reviewable improvement candidates for RAG, reflection memory,
   world-model scoring, and simulator heuristics.
6. Keep all ground-truth analysis strictly post-game or moderator-only.
7. Make accepted updates prove themselves through regression batches before
   being treated as improvements.
8. Keep LangSmith or other external tracing tools optional adapters, not core
   dependencies.

## Non-Goals

1. Do not train model weights or introduce reinforcement learning.
2. Do not let evaluation code modify live game state or rule outcomes.
3. Do not inject post-game ground truth into future live prompts except through
   approved, prompt-safe RAG or reflection artifacts.
4. Do not automatically promote generated candidates into live RAG or reflection
   memory without review gates.
5. Do not require LangSmith, OpenTelemetry, or any external tracing service for
   local correctness.
6. Do not replace existing RAG V2, Reflection Memory V2, or World Model designs.

## Design Principles

Evaluation must be causal enough to guide engineering decisions. A metric that
only says "win rate moved" is too coarse. The loop must preserve the chain:

```text
module exposure
-> model decision
-> public/private action
-> game outcome or local correctness check
-> diagnosis
-> reviewable candidate
-> same-seed regression
```

The system should prefer deterministic evidence first. LLM-based analysis can
summarize or cluster failures, but primary labels such as illegal action,
wrong target, leaked hidden information, missed true world, or harmful RAG hit
should be computed from structured traces whenever possible.

## Architecture

The feedback loop adds seven components.

1. `EvaluationTraceBuilder`
   - Reads `GameResult`, `event_log`, `action_trace_audit`,
     persisted module exposure audits, and `world_model_audit`.
   - Emits normalized per-decision trace records.
   - Keeps source event IDs and visibility class for every trace field.

2. `ModuleAttributionEngine`
   - Connects module exposures to downstream decisions.
   - Tracks whether RAG cards, reflection cards, possible worlds, simulator
     predictions, skills, and persona priors were present and whether the
     decision cited or aligned with them.

3. `AblationRunner`
   - Runs the same seed set under controlled module toggles.
   - Produces paired comparisons rather than unrelated aggregate batches.
   - Starts with deterministic batch support and later supports full live-agent
     batch runs.

4. `DiagnosticsEngine`
   - Converts failed traces into typed failure diagnoses.
   - Uses deterministic rules first and optional LLM clustering second.

5. `ImprovementCandidateGenerator`
   - Emits reviewable candidates, not direct mutations.
   - Candidate targets include RAG entries, reflection entries, world-model
     scoring rules, simulator heuristics, prompt guards, and evaluation golden
     cases.

6. `RegressionBatchBuilder`
   - Builds focused same-seed regression batches from failure cases.
   - Stores expected metric deltas and safety invariants.

7. `FeedbackReport`
   - Produces a compact report for CLI, dashboard, or API views.
   - Shows metrics, failure clusters, candidates, and regression deltas.

## Core Data Model

### Decision Trace Identity

Every module exposure must be joinable to one decision point. The feedback loop
uses a stable `trace_id` derived from the decision context:

```text
{game_id}:{player_id}:{phase}:D{day_number}:N{night_number}:{task_type}:{action_index}
```

Rules:

1. `action_index` is the zero-based index of the player decision among
   `action_trace_audit` events for the same `game_id`.
2. If a caller has a stronger runtime action ID, it may be used as
   `action_index`, but the generated trace ID must stay stable across replay.
3. RAG, reflection, possible-world, simulator, skill, and persona exposure
   records must either carry this `trace_id` directly or carry enough source
   refs to deterministically join to it.
4. Trace identity is audit-only. It must not be rendered in live player prompts.

### Exposure Source Contract

The trace builder should accept exposure data from two sources:

1. persisted event-log records;
2. explicit side-channel audit sources passed to the evaluator.

This is required because current code does not guarantee that all module
exposures are emitted as events.

Required contracts:

- World model: read from `world_model_audit` nested in `action_trace_audit`.
- RAG: either persist `rag_exposure_audit` records in `event_log`, or pass
  `RAGKnowledgeService.last_audit()` / `RAGInjector.audit_log()` snapshots into
  `EvaluationTraceBuilder`.
- Reflection: add `reflection_exposure_audit` records or include reflection
  exposure IDs inside `action_trace_audit`.
- Skills/persona: use `action_trace_audit.parsed_action` and strategy directive
  refs where available; otherwise mark attribution as unsupported.

`rag_exposure_audit` minimum shape:

```json
{
  "type": "rag_exposure_audit",
  "trace_id": "g1:p03:speech:D1:N0:speech:4",
  "player_id": "p03",
  "phase": "speech",
  "hits": [
    {
      "entry_id": "seed_foundation_counterclaim_analysis",
      "rank": 1,
      "relevance_score": 0.82,
      "prompt_visible": true,
      "query_signature": "role=seer phase=speech task=speech",
      "visibility": "player_perspective"
    }
  ]
}
```

`reflection_exposure_audit` minimum shape:

```json
{
  "type": "reflection_exposure_audit",
  "trace_id": "g1:p03:vote:D2:N1:vote:9",
  "player_id": "p03",
  "phase": "vote",
  "cards": [
    {
      "entry_id": "reflection_g0_p03",
      "rank": 1,
      "quality_score": 0.91,
      "prompt_visible": true,
      "query_signature": "player=p03 role=seer phase=vote",
      "visibility": "player_perspective"
    }
  ]
}
```

If exposure records are absent, the metric must report `unsupported`, not zero.
Zero means the module was evaluated and did not expose anything. Unsupported
means the current trace source cannot answer the question.

### Evaluation Trace

An evaluation trace is one decision point with enough context to reason about
module impact.

```python
@dataclass
class EvaluationTrace:
    trace_id: str
    game_id: str
    player_id: str
    role: str
    faction: str
    phase: str
    day_number: int = 0
    night_number: int = 0
    task_type: str = ""
    legal_actions: list[str] = field(default_factory=list)
    legal_targets: list[str] = field(default_factory=list)
    module_exposures: list[ModuleExposure] = field(default_factory=list)
    decision: DecisionSnapshot | None = None
    outcome: DecisionOutcome | None = None
    source_refs: list[str] = field(default_factory=list)
```

### Module Exposure

`ModuleExposure` records what a module placed into the decision context.

```python
@dataclass
class ModuleExposure:
    module: str  # rag, reflection, possible_worlds, simulator, skill, persona
    item_id: str
    rank: int = 0
    score: float = 0.0
    prompt_visible: bool = False
    cited_by_decision: bool = False
    aligned_with_decision: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

Examples:

- RAG hit: `module="rag"`, `item_id=entry_id`, `rank`, `relevance_score`.
- Reflection card: `module="reflection"`, `item_id=entry_id`,
  `quality_score`.
- Possible world: `module="possible_worlds"`, `item_id=world_id`,
  `probability`, `true_world_rank`.
- Simulator prediction: `module="simulator"`, `item_id=event_type`,
  `probability`, `future_hit`.

### Decision Outcome

```python
@dataclass
class DecisionOutcome:
    legal: bool | None = None
    target_role: str = ""
    target_faction: str = ""
    target_alive: bool | None = None
    vote_hit_wolf: bool | None = None
    protected_good: bool | None = None
    leaked_hidden_info: bool = False
    contradicted_visible_evidence: bool = False
    local_quality_score: float = 0.0
    outcome_refs: list[str] = field(default_factory=list)
```

Outcome labels are post-game evaluation data. They must not flow into live
player prompts.

### Failure Diagnosis

```python
@dataclass
class FailureDiagnosis:
    diagnosis_id: str
    trace_id: str
    category: str
    severity: str
    primary_module: str = ""
    supporting_modules: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    explanation: str = ""
```

Initial categories:

- `illegal_action`
- `hidden_info_leak`
- `wrong_vote_target`
- `missed_confirmed_evidence`
- `rag_harmful_transfer`
- `rag_low_relevance`
- `reflection_stale_or_irrelevant`
- `reflection_harmful_transfer`
- `possible_world_true_rank_low`
- `possible_world_overconfident`
- `simulator_false_positive`
- `decision_dialogue_mismatch`
- `prompt_contract_failure`

### Improvement Candidate

```python
@dataclass
class ImprovementCandidate:
    candidate_id: str
    source_diagnosis_ids: list[str]
    target_module: str
    operation: str  # create, update, downrank, disable, add_test
    priority: str
    prompt_safe_payload: dict[str, Any]
    audit_evidence: dict[str, Any] = field(default_factory=dict)
    moderator_notes: str = ""
    safety_notes: list[str] = field(default_factory=list)
    regression_seed_set: list[int] = field(default_factory=list)
    review_status: str = "pending"
```

Candidates are never applied automatically to live memory or RAG. A human or
explicit maintenance command must approve them.

Payload rules:

- `prompt_safe_payload` is the only portion that can become RAG or reflection
  prompt-visible content after review.
- `audit_evidence` may contain post-game labels, true roles, target factions,
  true-world rank, trace IDs, and source refs. It remains moderator-only.
- `moderator_notes` can explain why the candidate exists, but must not be copied
  into live prompts.
- Candidate validation must reject prompt-safe payloads containing current-game
  player IDs, hidden role truth, true-world rank, target faction labels, or
  concrete current-game facts.

## Module-Specific Evaluation

### World Model

Current `possible_world_topk_hit_rate` should remain, but it is not enough.

Add:

- `true_world_average_rank`: average rank of the closest world assignment.
- `true_world_mrr`: reciprocal rank of the closest world assignment.
- `world_score_entropy`: whether top worlds are over-concentrated.
- `covered_candidate_count`: how many candidate worlds were considered.
- `topk_probability_scope`: whether probabilities are top-k normalized or
  estimated against all candidates.
- `overconfidence_error_rate`: high-probability world cards that miss true
  assignments.

The world scorer should later accept relation evidence:

- vote edges;
- support and attack edges;
- claim and counterclaim facts;
- contradiction alerts;
- pressure summaries;
- confirmed deaths and revealed roles.

The evaluation loop should flag a candidate when the true world is consistently
ranked low in cases where relation evidence was available.

True-world rank is computed only inside post-game evaluation. The evaluator may
compare sanitized `key_assignments` from possible-world cards against
`GameResult.player_roles`, but it must not write full ground-truth role maps
back into live traces or prompt-visible fields.

Rank scope:

- `top_k_only`: only sanitized top-k cards are available.
- `full_candidate_set`: a post-game evaluator side-channel has access to full
  candidate scores.
- `unsupported`: no comparable assignments are available.

### Simulator

Current simulator metrics should distinguish event recall from event precision.

Add:

- `simulator_prediction_precision`
- `simulator_prediction_recall`
- `simulator_false_positive_rate`
- `simulator_action_alignment_rate`

Predictions should be evaluated by event class, not only as a global hit rate:

- vote pressure;
- night kill pressure;
- claim or counterclaim;
- power-role death risk;
- wolf push target;
- good-side split risk.

### RAG

RAG needs both offline retrieval quality and live decision attribution.

Add a golden query set:

```yaml
- query_id: golden_seer_sheriff_counterclaim_001
  role: seer
  phase: sheriff_speech
  task_type: speech
  situation: "role=seer game_phase=day task=speech actions=sheriff_claim"
  expected_entry_ids:
    - seed_foundation_counterclaim_analysis
    - seed_rule_seer_badge_flow_01
  forbidden_entry_ids:
    - seed_godview_review_01
```

Metrics:

- `rag_recall_at_1`
- `rag_recall_at_3`
- `rag_mrr`
- `rag_ndcg_at_3`
- `rag_live_use_rate`
- `rag_harmful_transfer_rate`
- `rag_coverage_by_role_phase`
- `rag_filter_violation_count`

Each live RAG hit should have a retrieval explanation:

- rule score;
- vector score;
- merged score;
- matched tags;
- quality grade;
- role and phase match class;
- filter decisions.

The explanation stays audit-only. The live prompt continues to receive only the
prompt-safe tactical card.

If the runtime cannot yet persist RAG exposure, Phase 1 must implement either
event-log persistence or an evaluator side-channel before `rag_live_use_rate` and
`rag_harmful_transfer_rate` are considered supported.

### Reflection Memory

Reflection memory needs impact metrics, not only quality gates.

Add:

- `reflection_injection_rate`
- `reflection_citation_rate`
- `reflection_alignment_rate`
- `reflection_beneficial_rate`
- `reflection_harmful_rate`
- `reflection_stale_rate`
- `reflection_repeated_no_effect_count`

`query_live()` should eventually rank by structured relevance:

```text
quality_score
+ same player
+ same role
+ same faction
+ phase overlap
+ situation pattern overlap
+ trigger signal overlap
+ recentness
- repeated no-effect penalty
- harmful attribution penalty
```

A reflection card that is repeatedly injected and associated with worse local
decisions should produce a candidate to downrank it to `review_only`.

Reflection exposure must be audited at prompt-construction time. The audit must
record `entry_id`, rank, `quality_score`, query signature, and whether the card
was prompt-visible. Without this audit, reflection impact metrics are
unsupported.

## Ablation Design

Ablation should compare paired runs with identical seeds.

Initial module toggles:

```python
@dataclass(frozen=True)
class EvaluationToggles:
    rag_enabled: bool = True
    reflection_enabled: bool = True
    possible_worlds_enabled: bool = True
    simulator_enabled: bool = True
    skill_advice_enabled: bool = True
    persona_prior_enabled: bool = True
```

Minimum experiment matrix:

1. baseline: all enabled;
2. no RAG;
3. no reflection;
4. no possible worlds;
5. no simulator;
6. no RAG and no reflection;
7. world model only: possible worlds plus simulator enabled, RAG and reflection
   disabled.

Comparison must use paired deltas:

```text
same seed, same ruleset, same role assignment
baseline metric - ablated metric
```

Global win rate remains useful but secondary. Primary ablation signals are local
quality metrics:

- vote target quality;
- illegal action rate;
- leakage rate;
- decision-dialogue consistency;
- world true-rank;
- RAG or reflection harmful-transfer rate.

There are two ablation modes:

1. Offline trace ablation
   - Replays existing traces and recomputes metrics while hiding selected
     module exposure records.
   - Valid for trace completeness, attribution sensitivity, diagnostics, and
     report generation.
   - Does not prove that a live agent would make a different decision.

2. Live-agent ablation harness
   - Runs the full prompt/runtime pipeline with module toggles applied before
     prompt construction.
   - Required for causal claims about RAG, reflection, possible worlds,
     simulator, skills, or persona improving live decisions.
   - More expensive and should run after offline trace tests are stable.

The existing deterministic `BatchRunner` is useful for schema, metric, and report
tests, but it does not exercise live prompt construction. Any metric requiring
real RAG or reflection influence must be marked `unsupported` under deterministic
mock batches unless explicit exposure fixtures are supplied.

## Feedback Candidate Policy

The loop emits candidates under strict gates.

### RAG Candidates

Allowed operations:

- create a new seed candidate from repeated high-confidence failure clusters;
- update `applicability` or `counter_signals`;
- downrank low-quality or harmful entries;
- add golden query tests;
- add forbidden query tests.

Not allowed:

- auto-approve self-play candidates into live RAG;
- store current-game hidden roles in prompt-visible fields;
- promote god-view analysis into player-facing tactical cards.

### Reflection Candidates

Allowed operations:

- downrank approved cards that repeatedly correlate with harmful decisions;
- mark stale cards as `review_only`;
- suggest a replacement prompt card after post-game review;
- add tests for retrieval relevance.

Not allowed:

- transfer concrete historical player IDs;
- transfer historical hidden role truth into live prompt facts;
- auto-delete operator-owned memory without explicit cleanup command.

### World Model Candidates

Allowed operations:

- add relation-graph features to world scoring;
- tune scoring weights;
- add true-rank regression tests;
- expand simulator event classes.

Not allowed:

- use ground truth before game end;
- make possible worlds override legal actions;
- expose full hidden role assignment to non-moderator views.

## Storage And API

Phase 1 can store feedback artifacts inside existing evaluation JSON outputs.
No database migration is required at first.

Recommended persisted objects:

- `evaluation_traces`
- `failure_diagnoses`
- `improvement_candidates`
- `ablation_runs`
- `regression_batches`

Later, these can be added to repository storage if the dashboard needs durable
querying across many batches.

API and UI should remain read-only initially:

- show module impact summary;
- show failure clusters;
- show candidate queue;
- show before/after regression deltas.

Mutation endpoints for candidate approval should come later and require explicit
operator authorization.

## LangSmith Position

LangSmith is useful for visualizing long LLM workflows, prompt/response traces,
latency, cost, and per-step metadata. It is not required for the core feedback
loop.

Recommended policy:

1. Build the local trace schema first.
2. Make every trace serializable without external services.
3. Add an optional LangSmith exporter that maps local traces to LangSmith runs.
4. Never make tests depend on LangSmith network availability.
5. Never send hidden-role or private audit data to external tracing unless the
   operator explicitly enables it and accepts the privacy boundary.

This keeps observability useful without turning it into a production dependency.

## Implementation Phases

### Phase 1: Trace Normalization

Create focused schemas and a trace builder that converts current `GameResult`
objects into normalized evaluation traces.

This phase must also add the exposure-source contract:

- stable `trace_id` derivation;
- optional side-channel audit inputs;
- `rag_exposure_audit` shape;
- `reflection_exposure_audit` shape;
- unsupported metric markers when a source is absent.

Expected files:

- `werewolf_agent/evaluation/feedback_schemas.py`
- `werewolf_agent/evaluation/trace_builder.py`
- `tests/evaluation/test_feedback_trace_builder.py`

### Phase 2: Module Attribution Metrics

Extend metrics with module exposure and attribution rates.

Expected files:

- `werewolf_agent/evaluation/feedback_metrics.py`
- `werewolf_agent/evaluation/schemas.py`
- `werewolf_agent/evaluation/metrics.py`
- `tests/evaluation/test_feedback_metrics.py`

### Phase 3: RAG Golden Retrieval Evaluation

Add an offline golden query evaluator for the existing RAG store.

Expected files:

- `config/rag_eval/golden_queries.yaml`
- `werewolf_agent/rag/evaluation.py`
- `tests/rag/test_retrieval_eval.py`

### Phase 4: Reflection Effectiveness Evaluation

Add reflection attribution metrics and candidates for stale or harmful cards.

Expected files:

- `werewolf_agent/memory/reflection_effectiveness.py`
- `tests/memory/test_reflection_effectiveness.py`

### Phase 5: World-Model Rank Metrics

Add true-world rank and probability-scope metrics.

Expected files:

- `werewolf_agent/evaluation/world_model_eval.py`
- `werewolf_agent/evaluation/schemas.py`
- `werewolf_agent/evaluation/metrics.py`
- `tests/evaluation/test_world_model_rank_metrics.py`

### Phase 6: Diagnostics And Candidates

Add deterministic diagnosis rules and reviewable improvement candidates.

Expected files:

- `werewolf_agent/evaluation/diagnostics.py`
- `werewolf_agent/evaluation/candidates.py`
- `tests/evaluation/test_feedback_diagnostics.py`
- `tests/evaluation/test_improvement_candidates.py`

### Phase 7: Ablation Runner

Add same-seed module toggles and paired comparison reports in two steps.

Step 7A: offline trace ablation.

- Operates on `EvaluationTrace` records.
- Tests attribution and diagnostics without rerunning live agents.
- Marks causal decision-impact metrics as unsupported.

Step 7B: live-agent ablation harness.

- Applies toggles before context construction.
- Runs the full runtime pipeline.
- Supports causal comparisons for RAG, reflection, possible worlds, simulator,
  skill advice, and persona priors.

Expected files:

- `werewolf_agent/evaluation/ablation.py`
- `werewolf_agent/evaluation/runner.py`
- `tests/evaluation/test_ablation_runner.py`

### Phase 8: Reports And Optional Export

Add a compact report object and optional LangSmith export adapter.

Expected files:

- `werewolf_agent/evaluation/feedback_report.py`
- `werewolf_agent/evaluation/langsmith_exporter.py`
- `tests/evaluation/test_feedback_report.py`
- `tests/evaluation/test_langsmith_exporter.py`

The exporter must be optional and skipped when LangSmith dependencies or
environment variables are absent.

## Testing Strategy

Tests should be deterministic and local.

Required coverage:

1. Trace builder extracts RAG, reflection, possible-world, simulator, and
   decision data from event logs.
2. Trace builder strips or labels private data correctly.
3. RAG golden query evaluator computes recall, MRR, and forbidden-hit counts.
4. Reflection effectiveness identifies cited, aligned, stale, and harmful cards.
5. World-model metrics compute true-world rank and overconfidence errors.
6. Diagnostics classify common failure cases.
7. Candidate generator emits reviewable candidates without mutating live stores.
8. Ablation runner compares paired same-seed results.
9. Optional LangSmith exporter can serialize traces without network calls.
10. Candidate prompt payload rejects hidden truth, current-game player IDs, and
    target faction labels.
11. Feedback candidate generation never calls `store_v2()`, `save_rag_entries()`,
    or other live persistence methods without an explicit approval path.
12. Live prompt renderers exclude outcome labels, true roles, true-world rank,
    and post-game diagnoses.

Recommended command groups:

```powershell
python -m pytest tests/evaluation/test_feedback_trace_builder.py tests/evaluation/test_feedback_metrics.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/rag/test_retrieval_eval.py tests/memory/test_reflection_effectiveness.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/evaluation/test_world_model_rank_metrics.py tests/evaluation/test_feedback_diagnostics.py tests/evaluation/test_ablation_runner.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

## Success Criteria

The upgrade is successful when:

1. Every evaluated decision can show which modules influenced its context.
2. RAG and reflection have measurable live-use and harmful-transfer rates.
3. Possible worlds report true-world rank, not only top-k hit rate.
4. Simulator predictions report precision and recall by event class.
5. Failure diagnoses produce reviewable candidates with source trace IDs.
6. Accepted candidates can be validated by same-seed regression batches.
7. No feedback artifact leaks hidden truth into live player prompts.
8. The loop runs locally without LangSmith or network dependencies.
9. Metrics that cannot be computed from available traces report `unsupported`
   instead of silently reporting `0.0`.

## Risks

1. Attribution can be overstated. Mitigation: mark alignment and citation
   separately, and require ablation for stronger causal claims.
2. Candidate generation can create noisy work. Mitigation: require repeated
   failure clusters and severity thresholds.
3. Metrics can encourage overfitting to deterministic tests. Mitigation: keep
   regression seed sets separate from development fixtures.
4. Trace payloads can grow large. Mitigation: store source refs and compact
   module exposure records, not full prompts or full transcripts.
5. External tracing can leak sensitive data. Mitigation: make LangSmith export
   opt-in and scrub private fields by default.
6. Unsupported metrics can be mistaken for poor performance. Mitigation: carry
   explicit support status with every attribution metric.

## Open Decisions

1. How broad the first live-agent ablation harness should be.
   - Recommendation: keep Phase 7A offline trace ablation local and cheap, then
     make Phase 7B start with a small same-seed live-agent smoke batch before
     expanding to larger regression suites.
2. Whether feedback candidates should be stored in memory/RAG tables or separate
   evaluation artifacts.
   - Recommendation: keep candidates separate until approved.
3. Whether RAG retrieval explanations should be exposed in the dashboard.
   - Recommendation: expose to moderator/debugger views only.
4. Whether LangSmith should be added immediately.
   - Recommendation: defer until local trace schema and tests are stable.
