# Evaluation Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, testable feedback loop that normalizes decision traces, attributes RAG/reflection/world-model/simulator exposure, diagnoses failures, and produces reviewable improvement candidates without leaking post-game truth into live prompts.

**Architecture:** Start with a small evaluation trace core: dataclass schemas plus a trace builder that consumes `GameResult` event logs and optional side-channel exposure audits. Later phases add metrics, RAG golden retrieval eval, reflection effectiveness, world-rank metrics, diagnostics, ablation, and reporting on top of the same trace contract.

**Tech Stack:** Python dataclasses, existing `werewolf_agent.evaluation.schemas.GameResult`, pytest, existing event-log dictionaries, no network dependencies.

---

## File Structure

- Create `werewolf_agent/evaluation/feedback_schemas.py`
  - Owns feedback-loop dataclasses and constants.
  - Must stay storage-agnostic and import-light.
- Create `werewolf_agent/evaluation/trace_builder.py`
  - Converts `GameResult` plus optional exposure audit records into `EvaluationTrace` objects.
  - Handles stable `trace_id`, `unsupported` source status, RAG/reflect/world-model exposure parsing, and post-game outcome labels.
- Create `tests/evaluation/test_feedback_trace_builder.py`
  - Covers Phase 1 red/green behavior.
- Later create `werewolf_agent/evaluation/feedback_metrics.py`
  - Computes module attribution metrics from traces.
- Later create `werewolf_agent/rag/evaluation.py`
  - Evaluates golden retrieval queries.
- Later create `config/rag_eval/golden_queries.yaml`
  - Holds offline RAG golden queries.
- Later create `werewolf_agent/memory/reflection_effectiveness.py`
  - Computes reflection-specific usefulness/staleness signals.
- Later create `werewolf_agent/evaluation/world_model_eval.py`
  - Computes true-world rank and overconfidence metrics.
- Later create `werewolf_agent/evaluation/diagnostics.py`
  - Classifies trace failures into typed diagnoses.
- Later create `werewolf_agent/evaluation/candidates.py`
  - Produces reviewable improvement candidates with prompt-safe/audit-only separation.
- Later create `werewolf_agent/evaluation/ablation.py`
  - Runs offline trace ablation and later delegates live-agent ablation.
- Later create `werewolf_agent/evaluation/feedback_report.py`
  - Serializes feedback reports for CLI/API/dashboard.
- Later create `werewolf_agent/evaluation/langsmith_exporter.py`
  - Optional exporter. Must not be required by tests.

## Task 1: Phase 1 Feedback Trace Schemas

**Files:**
- Create: `werewolf_agent/evaluation/feedback_schemas.py`
- Test: `tests/evaluation/test_feedback_trace_builder.py`

- [ ] **Step 1: Write failing schema tests**

Add tests for:

```python
from werewolf_agent.evaluation.feedback_schemas import (
    MetricSupport,
    ModuleExposure,
    ImprovementCandidate,
    validate_candidate_prompt_safe,
)


def test_module_exposure_defaults_to_unsupported_when_source_missing():
    exposure = ModuleExposure(module="rag", item_id="seed_1")
    assert exposure.support == MetricSupport.SUPPORTED
    assert exposure.prompt_visible is False


def test_candidate_prompt_payload_rejects_hidden_truth():
    candidate = ImprovementCandidate(
        candidate_id="c1",
        source_diagnosis_ids=["d1"],
        target_module="rag",
        operation="create",
        priority="high",
        prompt_safe_payload={"recommended_use": "根据 p03 的真实身份调整"},
        audit_evidence={"target_role": "werewolf"},
    )
    assert validate_candidate_prompt_safe(candidate) is False
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/evaluation/test_feedback_trace_builder.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because `werewolf_agent.evaluation.feedback_schemas` does not exist.

- [ ] **Step 3: Implement minimal schemas**

Implement:

- `MetricSupport` enum: `supported`, `unsupported`
- `ModuleExposure`
- `DecisionSnapshot`
- `DecisionOutcome`
- `EvaluationTrace`
- `FailureDiagnosis`
- `ImprovementCandidate`
- `validate_candidate_prompt_safe(candidate)`

Prompt-safe rejection should catch:

- `p\d+`, `player_\d+`, `agent_\d+`
- `真实身份`, `底牌`, `target_role`, `target_faction`, `true_world_rank`
- role truth phrases in prompt payload values.

- [ ] **Step 4: Run schema tests to verify GREEN**

Run the same pytest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/evaluation/feedback_schemas.py tests/evaluation/test_feedback_trace_builder.py
git commit -m "feat: add feedback trace schemas"
```

## Task 2: Phase 1 Trace Identity And Action Trace Extraction

**Files:**
- Modify: `werewolf_agent/evaluation/trace_builder.py`
- Modify: `tests/evaluation/test_feedback_trace_builder.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

- stable `trace_id` derived from `game_id`, `player_id`, phase, day/night,
  task type, and action index;
- extraction from `action_trace_audit` event logs;
- legal action and final action copied into `DecisionSnapshot`.

- [ ] **Step 2: Run tests to verify RED**

Expected: FAIL because `trace_builder.py` does not exist.

- [ ] **Step 3: Implement `EvaluationTraceBuilder`**

Create:

```python
class EvaluationTraceBuilder:
    def build(self, result: GameResult, exposure_audits: list[dict[str, Any]] | None = None) -> list[EvaluationTrace]:
        ...
```

Implement only `action_trace_audit` parsing first.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/evaluation/test_feedback_trace_builder.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/evaluation/trace_builder.py tests/evaluation/test_feedback_trace_builder.py
git commit -m "feat: build feedback traces from action audits"
```

## Task 3: Phase 1 RAG And Reflection Exposure Join

**Files:**
- Modify: `werewolf_agent/evaluation/trace_builder.py`
- Modify: `tests/evaluation/test_feedback_trace_builder.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

- `rag_exposure_audit` records join to matching `trace_id`;
- `reflection_exposure_audit` records join to matching `trace_id`;
- missing RAG/reflection exposure source creates an unsupported source marker,
  not a zero-valued supported metric.

- [ ] **Step 2: Run tests to verify RED**

Expected: FAIL because exposure join is not implemented.

- [ ] **Step 3: Implement exposure join**

Implement:

- `_exposures_from_rag_audit()`
- `_exposures_from_reflection_audit()`
- unsupported source markers on traces when no source is available.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/evaluation/test_feedback_trace_builder.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/evaluation/trace_builder.py tests/evaluation/test_feedback_trace_builder.py
git commit -m "feat: join feedback module exposure audits"
```

## Task 4: Phase 1 World Model And Outcome Labels

**Files:**
- Modify: `werewolf_agent/evaluation/trace_builder.py`
- Modify: `tests/evaluation/test_feedback_trace_builder.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

- possible-world cards from nested `world_model_audit` become module exposures;
- simulator predictions become module exposures;
- post-game `GameResult.player_roles` produces `DecisionOutcome.target_role`
  and `target_faction`;
- outcome labels are stored only in `DecisionOutcome`, not in prompt-safe
  exposure metadata.

- [ ] **Step 2: Run tests to verify RED**

Expected: FAIL because world-model exposure/outcome labels are missing.

- [ ] **Step 3: Implement world-model exposure and outcome extraction**

Use only sanitized assignments from audit payload and post-game role map from
`GameResult`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/evaluation/test_feedback_trace_builder.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/evaluation/trace_builder.py tests/evaluation/test_feedback_trace_builder.py
git commit -m "feat: attach world model feedback traces"
```

## Task 5: Phase 2 Module Attribution Metrics

**Files:**
- Create: `werewolf_agent/evaluation/feedback_metrics.py`
- Deferred: `werewolf_agent/evaluation/schemas.py`
- Deferred: `werewolf_agent/evaluation/metrics.py`
- Test: `tests/evaluation/test_feedback_metrics.py`

- [x] Add metrics for exposure count, supported/unsupported source counts,
  citation rate, alignment rate, and harmful-transfer placeholder inputs.
- [x] Keep new metrics separate from existing world-model metrics until stable.
- [x] Add tests with synthetic traces.
- [x] Commit with `feat: add feedback attribution metrics`.

Implementation note:
- Prompt-visible, citation, alignment, and harmful-transfer rates use supported
  exposures as the denominator so unsupported source markers expose missing
  instrumentation without lowering quality rates.

## Task 6: Phase 3 RAG Golden Retrieval Eval

**Files:**
- Create: `config/rag_eval/golden_queries.yaml`
- Create: `werewolf_agent/rag/evaluation.py`
- Test: `tests/rag/test_retrieval_eval.py`

- [x] Add golden query schema.
- [x] Compute recall@1, recall@3, MRR, nDCG@3, forbidden hit count.
- [x] Test with fake retriever and seed entries.
- [x] Commit with `feat: add rag retrieval evaluation`.

Implementation note:
- Golden queries are evaluation-only YAML fixtures. They validate that
  referenced expected/forbidden entry IDs exist in the RAG seed corpus.

## Task 7: Phase 4 Reflection Effectiveness

**Files:**
- Create: `werewolf_agent/memory/reflection_effectiveness.py`
- Test: `tests/memory/test_reflection_effectiveness.py`

- [x] Compute reflection injection, citation, alignment, stale, and harmful flags
  from `EvaluationTrace`.
- [x] Add candidate hints for repeated no-effect or harmful cards.
- [x] Commit with `feat: evaluate reflection effectiveness`.

Implementation note:
- Candidate hints are audit outputs only. This phase does not mutate
  `ReflectionMemory` or persistence stores.

## Task 8: Phase 5 World Model Rank Metrics

**Files:**
- Create: `werewolf_agent/evaluation/world_model_eval.py`
- Modify: `werewolf_agent/evaluation/schemas.py`
- Modify: `werewolf_agent/evaluation/metrics.py`
- Test: `tests/evaluation/test_world_model_rank_metrics.py`

- [x] Compute top-k-only true-world rank from sanitized assignments and
  `GameResult.player_roles`.
- [x] Add support status when comparable assignments are missing.
- [x] Commit with `feat: add world model rank metrics`.

Implementation note:
- Unsupported world-rank samples are counted separately and excluded from
  true-world rank rate denominators.

## Task 9: Phase 6 Diagnostics And Candidates

**Files:**
- Create: `werewolf_agent/evaluation/diagnostics.py`
- Create: `werewolf_agent/evaluation/candidates.py`
- Test: `tests/evaluation/test_feedback_diagnostics.py`
- Test: `tests/evaluation/test_improvement_candidates.py`

- [x] Classify illegal action, hidden info leak, wrong target, low true-world
  rank, RAG harmful transfer, reflection harmful transfer, simulator false
  positive.
- [x] Generate `ImprovementCandidate` objects without mutating RAG or memory
  stores.
- [x] Add negative tests that patch store methods and assert they are not called.
- [x] Commit with `feat: diagnose feedback failures`.

Implementation note:
- Candidate generation returns review-only objects and filters prompt-unsafe
  payloads; it does not write RAG entries, reflection cards, or persistence rows.

## Task 10: Phase 7 Ablation

**Files:**
- Create: `werewolf_agent/evaluation/ablation.py`
- Modify: `werewolf_agent/evaluation/runner.py`
- Test: `tests/evaluation/test_ablation_runner.py`

- [x] Add offline trace ablation first.
- [x] Mark causal live-agent metrics unsupported in offline mode.
- [x] Add live-agent harness only after trace tests are stable.
- [x] Commit with `feat: add feedback ablation runner`.

Implementation note:
- Offline trace ablation removes selected module exposures from immutable
  `EvaluationTrace` copies and compares baseline vs ablated attribution
  summaries. Live win-rate deltas and causal decision deltas are explicitly
  reported as unsupported in offline mode; no live-agent harness is introduced
  in this phase.
- Phase 7B adds context-level live-context ablation using injected runners. It
  compares paired decision deltas only and still marks full live win-rate causal
  metrics as unsupported until same-seed full-game batches are implemented.

## Task 11: Phase 8 Reports And Optional LangSmith Export

**Files:**
- Create: `werewolf_agent/evaluation/feedback_report.py`
- Create: `werewolf_agent/evaluation/langsmith_exporter.py`
- Test: `tests/evaluation/test_feedback_report.py`
- Test: `tests/evaluation/test_langsmith_exporter.py`

- [x] Add report JSON serialization.
- [x] Add optional LangSmith exporter that does not import or require LangSmith
  unless called.
- [x] Add tests that run without network or LangSmith dependencies.
- [x] Commit with `feat: report feedback loop results`.

Implementation note:
- Feedback reports serialize compact aggregate artifacts only: trace counts,
  source refs, module attribution metrics, diagnoses, improvement candidates,
  and ablation summaries. Raw traces and hidden post-game outcome labels are not
  emitted by default.
- The LangSmith exporter has no import-time dependency on LangSmith. Payload
  construction is local-only; actual export lazily imports LangSmith unless an
  injected client is supplied.

## Verification Commands

Run progressively:

```powershell
python -m pytest tests/evaluation/test_feedback_trace_builder.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/evaluation/test_feedback_trace_builder.py tests/evaluation/test_feedback_metrics.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/rag/test_retrieval_eval.py tests/memory/test_reflection_effectiveness.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/evaluation/test_world_model_rank_metrics.py tests/evaluation/test_feedback_diagnostics.py tests/evaluation/test_ablation_runner.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/evaluation/test_live_ablation_harness.py tests/evaluation/test_ablation_runner.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/evaluation/test_feedback_report.py tests/evaluation/test_langsmith_exporter.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```
