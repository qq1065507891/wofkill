# Attribution Engine Design

Date: 2026-06-19
Status: Design (awaiting implementation plan)
Owner: Claude development session

## Problem

The evaluation feedback loop has consumers for module-decision attribution that
have no producers. Concretely:

1. `ModuleExposure.cited_by_decision` and `aligned_with_decision`
   (`feedback_schemas.py:24-25`) default to `False` and are never populated.
   `EvaluationTraceBuilder._rag_exposures` / `_reflection_exposures` /
   `_world_model_exposures` (`trace_builder.py:166-299`) set only
   `module`/`item_id`/`rank`/`score`/`prompt_visible`/`metadata`.
2. `_is_harmful_transfer` (`feedback_metrics.py:84`) reads
   `exposure.metadata["harmful_transfer"]`, but no code ever writes that key, so
   it always returns `False`. `ModuleAttributionSummary.harmful_count` /
   `harmful_rate` are therefore always zero.
3. `regression_gate.py:152-161` checks `harmful_transfer_rate`, and
   `regression_gate.py:131-140` checks `judge_consistency_rate`, but no producer
   emits those metric keys. The `monitoring-closure-fix` plan added
   `required_metrics` fail-closed and a `vote_quality` producer, and revived
   `DecisionOutcome.legal` / `leaked_hidden_info` so `diagnostics` can fire
   `illegal_action` / `hidden_info_leak`. It explicitly deferred the
   `judge_consistency_rate` and `harmful_transfer_rate` producers to this plan.
4. `judge_speech_consistency(context, action)` (`llm_judge.py:74`) runs three
   deterministic checks, but the `public_fact_reference` dimension needs
   `context.public_facts`, and `action_trace_audit` payloads carry no context
   snapshot — so the judge cannot run end-to-end on a trace today.

The practical consequence: the regression gate's `harmful_transfer` and
`judge_consistency_rate` checks are still no-ops on any real ablation pair, and
the feedback loop cannot tell whether a RAG card / reflection card / possible
world / simulator prediction was merely shown, actually cited, aligned with the
decision, or caused a bad outcome.

## Goals

1. Build a post-game `AttributionEngine` that annotates every module exposure on
   an `EvaluationTrace` with `cited_by_decision`, `aligned_with_decision`, and a
   `harmful_transfer` metadata flag — for the four cognition modules: `rag`,
   `reflection`, `possible_worlds`, `simulator`.
2. Produce a `judge_consistency_rate` by running `judge_speech_consistency`
   end-to-end per trace, with the `public_facts` context rebuilt post-game from
   the event log.
3. Produce a `harmful_transfer_rate` aggregated across the four cognition
   modules, feeding the existing gate consumer with no consumer-side change.
4. Wire both producers into `FullGameAblationRunner` so ablation pairs carry
   the metrics into the regression gate.
5. Keep the change out of the runtime hot path: zero runtime edits, zero
   `action_trace_audit` payload growth, zero schema/DB/rule-engine change.
6. Provide deterministic, unit-testable attribution logic.

## Non-Goals

1. Do not make causal claims. Attribution is correlation (spec
   `evaluation-feedback-loop-design.md:842`: "Attribution can be overstated.
   Mitigation: mark alignment and citation separately, and require ablation for
   stronger causal claims."). Strong causal claims stay the job of the
   live-agent ablation harness.
2. Do not cover `skill` or `persona` attribution in this plan. They are
   behavioural priors (shape style/tone) whose "cited by decision" semantics
   are ill-defined; forcing attribution would produce noise. They remain
   exposure-audited but un-attributed.
3. Do not re-rank reflection cards by `beneficial`/`harmful` signal in this
   plan. The `cited ∧ aligned ∧ correct` (`beneficial`) byproduct is computed
   and stored on the exposure but not fed into `reflection.query_live` ranking
   (spec line 514 lists that as a future downrank signal). Tracked, not done.
4. Do not change `ActionContract`, output parsers, or example prompts to add an
   explicit `cited_modules` field. Citation is inferred from text, not declared
   by the LLM.
5. Do not change the `reflections` DB schema, RAG schema, or any rule-engine
   behaviour.
6. Do not run attribution or the judge on the runtime decision path.

## Design Decisions (locked in brainstorming)

1. **`cited_by_decision` = text match (Jaccard).** `cited` is `True` when
   `Jaccard(tokenize(decision.reason + speech), tokenize(exposure_representative_text)) >= 0.15`.
   Reuses the reflection module's existing `_jaccard` / `_tokenize_situation`.
   The 0.15 floor is tuned for short texts (avoid false negatives); player IDs
   like `p03` are tokens and are captured, so `possible_worlds`/`simulator`
   need no special handling for citation.

2. **Scope = four cognition modules:** `rag`, `reflection`, `possible_worlds`,
   `simulator`. Excludes `skill` / `persona` (Non-Goal 2).

3. **`aligned_with_decision` = per-module direction rule**, independent of
   outcome. `harmful_transfer` = `cited ∧ aligned ∧ outcome_is_bad`. The three
   fields are separate signals (spec "mark alignment and citation separately"),
   so `cited ∧ ¬aligned` (shown but ignored) is NOT harmful (no impact), and
   `cited ∧ aligned ∧ correct` is `beneficial` (a stored byproduct).

4. **Timing = post-game.** `AttributionEngine.annotate(traces, result)` runs
   after `EvaluationTraceBuilder.build`. The judge's `public_facts` context is
   rebuilt post-game by filtering `result.event_log` to the trace's
   `day_number`/`phase` and re-running `build_world_state` — no runtime audit
   payload change (spec "store source refs and compact module exposure records,
   not full prompts").

5. **`cited` text match = unified Jaccard** (not per-module keyword extraction).
   Per-module differences are carried by `aligned`; `cited` stays a single
   uniform text-similarity signal.

## Architecture

A new post-game pass sits between trace assembly and metric aggregation:

```text
GameResult
  │
  ▼
EvaluationTraceBuilder.build(result)          # unchanged: assembles traces + exposures
  │                                           #   (cited/aligned stay False, no harmful flag)
  ▼
AttributionEngine.annotate(traces, result)    # NEW: pure post-game pass
  │   for each trace:
  │     rebuild public_facts (filter event_log by trace day/phase → build_world_state)
  │     for each exposure in trace.module_exposures (rag/reflection/possible_worlds/simulator):
  │       exposure.cited_by_decision = _cited(trace.decision, exposure)
  │       exposure.aligned_with_decision = _aligned(trace.decision, exposure, trace.faction)
  │       if _is_harmful(exposure, trace.outcome):
  │           exposure.metadata["harmful_transfer"] = True
  │     trace.outcome.local_quality_score = judge_speech_consistency(context, action).consistency_score
  │
  ▼
FullGameAblationRunner                         # produces harmful_transfer_rate + judge_consistency_rate
  │   metrics = _game_metrics(result)          # existing: win_rate / vote_quality / illegal
  │   metrics["harmful_transfer_rate"] = harmful_rate(traces)
  │   metrics["judge_consistency_rate"] = mean_consistency(traces)
  │
  ▼
RegressionGate                                 # existing consumers now receive real signals
```

`EvaluationTraceBuilder` is unchanged. Attribution is a separate responsibility
in a separate module, so the trace assembler stays focused on joining audit
events to exposures, and the attribution engine stays focused on
decision-vs-exposure semantics.

## Components

### `AttributionEngine` (`evaluation/attribution.py`, new)

```python
class AttributionEngine:
    def annotate(self, traces: list[EvaluationTrace], result: GameResult) -> list[EvaluationTrace]:
        """Post-game: fill cited/aligned/harmful on each cognition exposure,
        and set outcome.local_quality_score from the consistency judge."""
```

It mutates exposure fields in place (exposures are `frozen=True` dataclasses —
see Implementation Note A for the rebuild approach) and returns the traces.

### `cited` — `_cited(decision, exposure) -> bool`

```python
decision_text = decision.reason + " " + (parsed_action.speech or "")
exposure_text = _exposure_representative_text(exposure)
return _jaccard(_tokenize(decision_text), _tokenize(exposure_text)) >= _CITED_THRESHOLD  # 0.15
```

`_exposure_representative_text(exposure)` per module:

| module | representative text |
|---|---|
| `rag` | `metadata["title"] + " " + metadata["lesson"] + " " + metadata["recommended_action"]` (V2 tactical frame fields; fall back to `metadata["title"]` alone if absent) |
| `reflection` | `metadata["theme"] + " " + metadata["lesson"] + " " + metadata["recommended_action"]` (same fallback) |
| `possible_worlds` | `" ".join(f"{pid}={role}" for pid, role in metadata["key_assignments"].items())` (player IDs are tokens) |
| `simulator` | `metadata["event"] + " " + " ".join(metadata["affected_players"])` |

`_tokenize` reuses the reflection `_tokenize_situation` regex
(`[a-z0-9_]+|[一-鿿]`). `_jaccard` is imported from
`werewolf_agent.memory.reflection` (already shared) or duplicated as a small
helper in `attribution.py` to avoid a memory→evaluation dependency (see
Implementation Note B).

### `aligned` — `_aligned(decision, exposure, faction) -> bool`

Per-module direction rule. `target_id` is `decision.target_id`; for speech
traces without a target, fall back to player IDs extracted from
`decision.reason`.

| module | aligned when |
|---|---|
| `possible_worlds` | `target_id` (or any reason-mentioned player) is in the world's wolf-role assignments (`key_assignments[pid] in {"werewolf","wolf"}`) |
| `simulator` | `target_id` (or reason-mentioned player) is in `metadata["affected_players"]` |
| `rag` / `reflection` | `decision.reason` contains an action verb from the `_looks_actionable` token set (`"先","不要","避免","必须","优先","核验","比较","列"`) that also appears in the exposure's `recommended_action` — i.e. the decision adopted the recommended action verb |

For non-vote/non-speech actions (`wolf_kill`, `use_poison`, `hunter_shot`,
`sheriff_vote`), use `target_id` with the `possible_worlds`/`simulator` rules
above; `rag`/`reflection` keep the reason-verb rule (most of those actions have
a `reason`). No target and no verb match → `aligned = False`.

### `harmful` — `_is_harmful(exposure, outcome) -> bool`

```python
return (
    exposure.cited_by_decision
    and exposure.aligned_with_decision
    and _outcome_is_bad(outcome)
)
```

`_outcome_is_bad(outcome)` reuses the signals revived/added by
`monitoring-closure-fix` plus the existing `diagnostics._is_wrong_good_target`:

```python
def _outcome_is_bad(outcome: DecisionOutcome) -> bool:
    if outcome is None:
        return False
    if outcome.legal is False:
        return True
    if outcome.leaked_hidden_info:
        return True
    return False  # vote_hit_wolf / wrong_target handled by the caller via faction (see below)
```

The `vote_hit_wolf` / `wrong_target` signals need the trace's `faction` and
`decision.action_type`, so they are evaluated in `_is_harmful`'s caller with
access to the trace, not inside `_outcome_is_bad`:

```python
def _trace_outcome_is_bad(trace: EvaluationTrace) -> bool:
    outcome = trace.outcome
    if _outcome_is_bad(outcome):
        return True
    if trace.faction == "good" and trace.decision and trace.decision.action_type == "vote":
        if outcome is not None and outcome.vote_hit_wolf is False:
            return True
    # wrong_target: good-faction harmful action hit a good-faction player
    if trace.decision and trace.decision.action_type in {"vote","use_poison","hunter_shot","sheriff_vote"}:
        if trace.faction == "good" and getattr(outcome, "target_faction", "") == "good":
            return True
    return False
```

When harmful, set `exposure.metadata["harmful_transfer"] = True` so the existing
`feedback_metrics._is_harmful_transfer`, `diagnostics._is_harmful`, and the
gate's `harmful_transfer_rate` consumer all activate with **zero consumer-side
change**.

### `judge_consistency` producer

Per trace with non-empty speech:

```python
visible_facts = _rebuild_visible_facts(result.event_log, trace)   # filter by day_number/phase → build_world_state → VisibilityPolicy.filter_visible_facts
context = {
    "role": trace.role,
    "faction": trace.faction,
    "public_claim": _derive_public_claim(trace),   # from player_roles/public speech claims if available
    "public_facts": visible_facts,
    "visible_facts": visible_facts,
}
action = {"speech": parsed.speech, "reason": trace.decision.reason}
judgment = judge_speech_consistency(context, action)
trace.outcome.local_quality_score = judgment.consistency_score
```

`_rebuild_visible_facts` filters `result.event_log` to events with
`day_number <= trace.day_number` (and matching phase scope), runs
`build_world_state` (`cognition.world_state`) to get facts, then
`VisibilityPolicy.filter_visible_facts(world_state, trace.player_id, trace.role)`
to keep only what this player could see. This is deterministic given the event
log and reuses the single `build_world_state` source the runtime itself uses.

`judge_consistency_rate` = mean of `judgment.consistency_score` over traces
whose `decision.reason + speech` is non-empty (empty-action traces are skipped,
not counted as 0).

### Metric producers

In `FullGameAblationRunner` (after each `_run_game`):

```python
metrics = _game_metrics(result)                  # existing
traces = EvaluationTraceBuilder().build(result)
AttributionEngine().annotate(traces, result)
metrics["harmful_transfer_rate"] = _harmful_rate(traces)
metrics["judge_consistency_rate"] = _mean_consistency(traces)
```

```python
def _harmful_rate(traces: list[EvaluationTrace]) -> float:
    supported = 0
    harmful = 0
    for t in traces:
        for e in t.module_exposures:
            if e.module not in {"rag","reflection","possible_worlds","simulator"}:
                continue
            if e.support != MetricSupport.SUPPORTED:
                continue
            supported += 1
            if e.metadata.get("harmful_transfer") is True:
                harmful += 1
    return harmful / supported if supported else 0.0

def _mean_consistency(traces: list[EvaluationTrace]) -> float | None:
    scores = [t.outcome.local_quality_score for t in traces
              if t.outcome and t.decision and (t.decision.reason or _speech(t).strip())
              and t.outcome.local_quality_score > 0]
    return sum(scores) / len(scores) if scores else None
```

Replay path (`_result_from_replay_record`, sparse `GameResult` with empty
`event_log`/`action_records`): cannot build traces → both keys omitted. Callers
running replay ablations must not set
`required_metrics=("harmful_transfer_rate","judge_consistency_rate")` without
enriching the replay `GameResult` (the `monitoring-closure-fix`
`required_metrics` fail-closed then governs).

## Data Flow

```text
FullGameAblationRunner.run
  └─ per seed: baseline = _run_game(...); ablated = _run_game(...)
        └─ _run_game returns full GameResult (event_log + action_records + player_roles/factions)
  └─ per pair:
        baseline_metrics = _enriched_metrics(baseline)   # _game_metrics + traces + annotate + 2 new keys
        ablated_metrics  = _enriched_metrics(ablated)
        pair = FullGameAblationPair(..., baseline_metrics, ablated_metrics)
  └─ _metric_deltas(pairs) → FullGameAblationReport.metric_deltas now includes
     harmful_transfer_rate + judge_consistency_rate deltas
  └─ RegressionGate.evaluate(config, baseline_metrics, candidate_metrics, ...)
        → harmful_transfer (lower-is-better) + judge_consistency_rate (higher-is-better)
          checks now receive real values; required_metrics fail-closed triggers
          if a producer is silently absent
```

## Safety / Boundaries

1. **No runtime change.** `AttributionEngine` is post-game only. No
   `action_trace_audit` payload growth, no decision-path latency, no
   `moderator_full` leak (it reads `GameResult`, which is a post-game artifact
   already allowed to hold ground truth for evaluation — same boundary as
   `world_model_eval`).
2. **No live-prompt leak.** Attribution annotates `EvaluationTrace` objects that
   never flow back into a live player `AgentContext`. The harmful flag is on
   the exposure metadata of an evaluation trace, not on a prompt field.
3. **Determinism.** `_rebuild_visible_facts` is a pure function of
   `(event_log, trace.day_number, trace.phase, trace.player_id, trace.role)`.
   Same input → same facts → same judge score. Reuses the single
   `build_world_state` source so it cannot drift from the runtime's own fact
   extraction unless the runtime changes that function (single source).
4. **Attribution is correlation, not causation** (spec line 842). The design
   never claims a card *caused* a bad decision; it flags correlation
   (cited ∧ aligned ∧ bad). Strong causal claims require the live-agent
   ablation harness (out of scope).
5. **`harmful = cited ∧ aligned ∧ bad` is the strictest reasonable definition**
   — minimises false positives. `cited ∧ ¬aligned` (shown, ignored) is not
   harmful because it had no behavioural effect. This is documented as an
   explicit trade-off: some real harmful transfers where the LLM half-followed
   a card may be missed, but the flagged set is high-precision.
6. **Judge context rebuild stays in-memory post-game** — `public_facts` are
   never persisted into an audit payload (spec "not full prompts"), only
   recomputed when the engine runs.

## Testing

- `tests/evaluation/test_attribution.py` (new):
  - `_cited`: Jaccard above/below 0.15 threshold for each of the four modules'
    representative text; player-ID token capture for possible_worlds/simulator.
  - `_aligned`: each module's direction rule (possible_worlds wolf-assignment
    match/mismatch; simulator affected_players match; rag/reflection
    action-verb adoption).
  - `_is_harmful` / `_trace_outcome_is_bad`: the cited∧aligned∧bad four-quadrant
    matrix; each bad-outcome signal (`legal=False`, `leaked=True`,
    good-voter `vote_hit_wolf=False`, wrong_target).
  - `judge` producer: rebuilt `public_facts` yield a non-empty
    `PublicEvidenceIndex`; `judge_consistency_rate` mean over traces.
  - `annotate`: end-to-end on a fixture `GameResult` → exposures carry
    cited/aligned/harmful; `local_quality_score` populated.
- `tests/evaluation/test_full_game_ablation.py` (extend):
  - `_enriched_metrics` emits `harmful_transfer_rate` + `judge_consistency_rate`
    on the deterministic-fallback path; replay path omits them.
- `tests/evaluation/test_regression_gate.py` (extend):
  - End-to-end: an ablation pair carrying the two metrics, with
    `required_metrics=("judge_consistency_rate","harmful_transfer_rate")`,
    passes when metrics are present and fails-closed when absent — proving the
    `monitoring-closure-fix` gate now has a real producer behind it.

## Open Risks

1. **`EvaluationTrace` / `ModuleExposure` are `@dataclass(frozen=True)`.**
   Annotating cited/aligned/harmful requires rebuilding the exposure (and its
   parent trace) with updated fields, not mutating in place. Implementation
   must use `dataclasses.replace` chains (see Implementation Note A). This is
   mechanical but verbose; a small `_replace_exposure` / `_replace_trace` helper
   keeps it readable.
2. **`_jaccard` / `_tokenize_situation` live in `memory.reflection`.** Importing
   them into `evaluation.attribution` creates an `evaluation → memory`
   dependency. Preferred: move the two pure helpers into a shared
   `evaluation/text_similarity.py` (or `memory/text_helpers.py`) and have both
   `reflection` and `attribution` import from there. Fallback: duplicate the
   ~10-line helpers in `attribution.py` (DRY violation but no cycle). Decision
   deferred to the implementation plan (see Open Decision 1).
3. **`public_claim` derivation for the judge.** `judge_speech_consistency`'s
   `identity_consistency` reads `context.public_claim`. The trace does not
   carry an explicit public_claim; it is derived from the player's public
   speech/role-claim events in `event_log` up to this trace's day/phase. If
   none, `public_claim=""` (the judge's identity dimension then only fires on
   wolf self-identification). Acceptable; documented.
4. **Jaccard 0.15 threshold tuning.** The floor is a starting point; short
   texts can swing between 0.1 and 0.2. The threshold is a module constant
   (`_CITED_THRESHOLD`) so it can be tuned without API change; the test suite
   pins behaviour at 0.15.
5. **`beneficial` byproduct unused.** `cited ∧ aligned ∧ correct` is computable
   but not consumed this plan. Storing it on exposure metadata
   (`metadata["beneficial"]=True`) is cheap and unlocks reflection downrank
   later, but adds a field; decision deferred (Open Decision 2).

## Open Decisions

1. **Shared text-similarity helper location.** Move `_jaccard` /
   `_tokenize_situation` to a shared module (recommended, breaks the
   evaluation→memory edge) vs duplicate in `attribution.py`. Recommend: move to
   `evaluation/text_similarity.py` and have `memory.reflection` re-import, since
   `evaluation` is the broader consumer and `memory.reflection` already has an
   `evaluation`-adjacent role.
2. **Store `beneficial` on exposure metadata.** Cheap byproduct, unlocks
   future reflection downrank. Recommend: yes, store
   `metadata["beneficial"]=True` for `cited ∧ aligned ∧ ¬bad`, since the
   attribution pass already computes all three.

## Out of Scope / Future

- `skill` / `persona` attribution (behavioural priors; ill-defined citation
  semantics).
- Feeding `beneficial` / `harmful` into `reflection.query_live` re-ranking
  (spec line 514 downrank).
- Live-agent ablation harness causal deltas (spec Phase 7B).
- Per-module harmful rates broken out in the gate (currently the gate consumes
  the aggregate `harmful_transfer_rate` across the four modules; per-module
  breakdown stays in `feedback_report.module_metrics` for review).

## Self-Review

- **Placeholders:** none. Each component has concrete pseudocode or a table.
- **Internal consistency:** `harmful = cited ∧ aligned ∧ bad` is stated
  identically in Goals, Design Decisions, Components, and Safety. The four
  modules are listed identically in Goals, Non-Goals (excluded skill/persona),
  Components tables, and `_harmful_rate`.
- **Scope:** one post-game engine + two metric producers + gate wiring. Single
  implementation plan sized.
- **Ambiguity:** `aligned` for rag/reflection is pinned to the
  `_looks_actionable` verb set (reused, not invented); the frozen-dataclass
  rebuild approach is named (`replace`); the `_jaccard` location is an explicit
  Open Decision rather than left vague.
