# Monitoring Closure Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two P0 break-points in the evaluation feedback loop so the regression gate actually guards decision-quality regressions and the failure diagnostics actually surface illegal actions and information leaks.

**Architecture:** Four independent, backward-compatible fixes inside `werewolf_agent/evaluation/`. (1) Add a `required_metrics` fail-closed knob to `RegressionGate` so a missing producer is visible instead of silently skipped. (2) Make `FullGameAblationRunner._game_metrics` emit `vote_quality` (good-faction stance accuracy) from `action_records`. (3) Extract the legality / dialogue-leak helpers out of `metrics.py` into a new `decision_helpers.py` to break the `metrics -> trace_builder` import cycle, then have `EvaluationTraceBuilder` populate `DecisionOutcome.legal` / `leaked_hidden_info` so `diagnose_trace_failures` can fire `illegal_action` / `hidden_info_leak`. (4) Remove `"werewolf"` from `FeedbackReport._PRIVATE_AUDIT_TOKENS` so the public-facing report no longer scrubs the public `werewolf_win_rate` aggregate.

**Tech Stack:** Python 3.11, pytest, dataclasses, conda env `wofkill`, PowerShell. Tests run with `--basetemp` to dodge the local pytest-xdist temp-dir permission issue noted in `PROGRESS.md`.

---

## File Structure

- **Create** `werewolf_agent/evaluation/decision_helpers.py` — shared decision-legality + dialogue-leak predicates (extracted from `metrics.py` to break the import cycle). Single responsibility: pure functions over an `action_trace` dict.
- **Modify** `werewolf_agent/evaluation/metrics.py` — delete the migrated `_TARGET_REQUIRED_ACTIONS` / `_decision_is_legal_from_trace` / `_dialogue_leaked_from_trace` definitions; re-export them as private aliases from `decision_helpers` so all in-file call sites stay unchanged.
- **Modify** `werewolf_agent/evaluation/regression_gate.py` — add `required_metrics` to `CandidateRegressionConfig`; emit a failed `GateCheck` for any required metric absent from both sides.
- **Modify** `werewolf_agent/evaluation/full_game_ablation.py` — `_game_metrics` emits `vote_quality` when good-faction vote records exist.
- **Modify** `werewolf_agent/evaluation/trace_builder.py` — `_decision_outcome` takes the `action_trace` dict and fills `legal` / `leaked_hidden_info` via `decision_helpers`.
- **Modify** `werewolf_agent/evaluation/feedback_report.py` — drop `"werewolf"` from `_PRIVATE_AUDIT_TOKENS`.
- **Test** `tests/evaluation/test_regression_gate.py`, `test_full_game_ablation.py`, `test_feedback_trace_builder.py`, `test_feedback_diagnostics.py`, `test_feedback_report.py`.

**Design boundaries respected:** No change to game rules, role abilities, or the rule engine. No change to the `reflections` DB schema. Diagnostics still never feed hidden truth back into a live player context — they read `action_trace_audit` events (already visibility-safe) and ground-truth `player_roles` / `player_factions` only inside `DecisionOutcome`, which is a post-game evaluation artifact, never injected into a prompt.

---

## Task 1: Regression gate fail-closed on required metrics

**Why:** `RegressionGate.evaluate` silently `return`s when a checked metric is absent from both `baseline_metrics` and `candidate_metrics` (`regression_gate.py:200-201`, `:232-233`). The gate checks `vote_quality` / `judge_consistency_rate` / `harmful_transfer_rate`, but no producer emits those keys into the metrics dict that the candidate pipeline feeds the gate, so those checks are no-ops in practice. A `required_metrics` knob makes a missing producer fail loudly instead.

**Files:**
- Modify: `werewolf_agent/evaluation/regression_gate.py:9-18` (config), `:86-184` (evaluate)
- Test: `tests/evaluation/test_regression_gate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_regression_gate.py`:

```python
from werewolf_agent.evaluation.regression_gate import (
    CandidateRegressionConfig,
    RegressionGate,
)


def test_gate_fails_when_required_metric_is_missing():
    config = CandidateRegressionConfig(
        candidate_id="c1",
        required_metrics=("vote_quality",),
    )
    report = RegressionGate().evaluate(
        config,
        baseline_metrics={"good_win_rate": 0.5},
        candidate_metrics={"good_win_rate": 0.5},
        prompt_safe=True,
    )
    assert report.passed is False
    assert any(
        "vote_quality" in reason and "missing" in reason
        for reason in report.blocked_reasons
    )


def test_gate_passes_when_required_metric_is_present():
    config = CandidateRegressionConfig(
        candidate_id="c1",
        required_metrics=("vote_quality",),
    )
    report = RegressionGate().evaluate(
        config,
        baseline_metrics={"vote_quality": 0.8, "good_win_rate": 0.5},
        candidate_metrics={"vote_quality": 0.8, "good_win_rate": 0.5},
        prompt_safe=True,
    )
    assert report.passed is True


def test_gate_required_metrics_default_empty_is_backward_compatible():
    # No required_metrics set → historical behaviour unchanged.
    config = CandidateRegressionConfig(candidate_id="c1")
    report = RegressionGate().evaluate(
        config,
        baseline_metrics={"good_win_rate": 0.5},
        candidate_metrics={"good_win_rate": 0.5},
        prompt_safe=True,
    )
    assert report.passed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate wofkill; python -m pytest tests/evaluation/test_regression_gate.py::test_gate_fails_when_required_metric_is_missing tests/evaluation/test_regression_gate.py::test_gate_passes_when_required_metric_is_present tests/evaluation/test_regression_gate.py::test_gate_required_metrics_default_empty_is_backward_compatible -q --basetemp=.pytest_tmp\mon`
Expected: the first test FAILS (gate currently passes because `vote_quality` is absent → check is skipped, no blocked reason). The other two PASS already.

- [ ] **Step 3: Add the config field**

In `werewolf_agent/evaluation/regression_gate.py`, add the field to `CandidateRegressionConfig` (after `harmful_transfer_increase_tolerance`):

```python
    harmful_transfer_increase_tolerance: float = 0.0
    # Metrics that MUST be present in baseline or candidate; if a producer
    # forgets to emit one, the gate fails closed instead of silently skipping
    # the check. Default empty preserves historical behaviour.
    required_metrics: tuple[str, ...] = ()
```

- [ ] **Step 4: Enforce required metrics inside evaluate**

In `RegressionGate.evaluate`, insert the required-metric loop AFTER the `prompt_safety` check is appended and BEFORE `blocked_reasons` is computed (currently `regression_gate.py:163-176`):

```python
        if not prompt_safe:
            checks.append(GateCheck(
                name="prompt_safety",
                passed=False,
                reason="prompt_safety_failed",
            ))
        else:
            checks.append(GateCheck(name="prompt_safety", passed=True))

        for required in config.required_metrics:
            if required in baseline_metrics or required in candidate_metrics:
                continue
            checks.append(GateCheck(
                name=f"required_{required}",
                passed=False,
                reason=f"required_metric_missing:{required}",
                metric=required,
            ))

        blocked_reasons = [
            check.reason
            for check in checks
            if not check.passed and check.reason
        ]
```

- [ ] **Step 5: Run the full gate test file to verify pass + no regressions**

Run: `python -m pytest tests/evaluation/test_regression_gate.py -q --basetemp=.pytest_tmp\mon`
Expected: PASS (existing tests still green; new tests green).

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/evaluation/regression_gate.py tests/evaluation/test_regression_gate.py
git commit -m "fix: regression gate fails closed on missing required metrics"
```

---

## Task 2: Full-game ablation emits vote_quality

**Why:** `_game_metrics` (`full_game_ablation.py:242-249`) only emits `good_win_rate` / `werewolf_win_rate` / `illegal_action_count`. The gate consumes `vote_quality` but nobody produces it, so even after Task 1 a caller who sets `required_metrics=("vote_quality",)` would always fail on the deterministic-fallback path. `vote_quality` here is good-faction stance accuracy — the fraction of good-faction votes that hit a werewolf — which is a direct decision-quality signal derivable purely from `GameResult.action_records` + `player_factions` (same computation pattern as `metrics.py:323-331`).

**Files:**
- Modify: `werewolf_agent/evaluation/full_game_ablation.py:242-249`
- Test: `tests/evaluation/test_full_game_ablation.py`

- [ ] **Step 1: Confirm the ActionRecord constructor signature**

Run: `python -m pytest tests/evaluation/test_full_game_ablation.py -q --basetemp=.pytest_tmp\mon` (baseline green), then read `werewolf_agent/evaluation/schemas.py` around the `ActionRecord` / `ActionVerdict` definitions to confirm the exact field names used by existing tests in `test_full_game_ablation.py`. Mirror those field names in the test below.

- [ ] **Step 2: Write the failing test**

Append to `tests/evaluation/test_full_game_ablation.py`. Use the same `GameResult` / `ActionRecord` import pattern this file already uses (copy the import line from the top of the file):

```python
def test_game_metrics_includes_vote_quality_from_action_records():
    from werewolf_agent.evaluation.full_game_ablation import _game_metrics
    # p01 (good) votes p03 (werewolf) -> correct
    # p02 (good) votes p01 (good)     -> wrong
    # => good-faction stance accuracy = 1/2 = 0.5
    result = _make_result_with_votes(
        player_factions={"p01": "good", "p02": "good", "p03": "werewolf"},
        votes=[("p01", "p03"), ("p02", "p01")],
        winning_faction="good",
    )
    metrics = _game_metrics(result)
    assert metrics["vote_quality"] == 0.5


def test_game_metrics_omits_vote_quality_when_no_good_votes():
    from werewolf_agent.evaluation.full_game_ablation import _game_metrics
    result = _make_result_with_votes(
        player_factions={"p01": "werewolf", "p02": "werewolf"},
        votes=[("p01", "p02")],
        winning_faction="werewolf",
    )
    metrics = _game_metrics(result)
    assert "vote_quality" not in metrics
```

Add the helper `_make_result_with_votes` in the same test file (above the two tests), constructing a `GameResult` with `action_records=[ActionRecord(player_id=v, action_type="vote", target_id=t, verdict=<OK enum>, phase="day_vote", day_number=1) for v,t in votes]`, empty `event_log`, and the given `player_factions` / `winning_faction`. Reuse the `ActionRecord` / `ActionVerdict` imports already at the top of `test_full_game_ablation.py`; if they are absent, add `from werewolf_agent.evaluation.schemas import GameResult, ActionRecord, ActionVerdict` and use the verdict enum value the existing tests use.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation/test_full_game_ablation.py::test_game_metrics_includes_vote_quality_from_action_records tests/evaluation/test_full_game_ablation.py::test_game_metrics_omits_vote_quality_when_no_good_votes -q --basetemp=.pytest_tmp\mon`
Expected: FAIL — `_game_metrics` does not return a `vote_quality` key.

- [ ] **Step 4: Implement vote_quality extraction**

In `werewolf_agent/evaluation/full_game_ablation.py`, replace `_game_metrics` (lines 242-249) with:

```python
def _vote_quality_from_result(result: GameResult) -> float | None:
    """Good-faction stance accuracy: good votes hitting a werewolf / good votes.

    Returns None when there are no good-faction votes (e.g. the replay path,
    whose GameResult has empty action_records) so the key is omitted rather
    than reported as a misleading 0.0.
    """
    correct = 0
    total = 0
    for record in result.action_records:
        if record.action_type != "vote" or not record.target_id:
            continue
        if result.player_factions.get(record.player_id) == "good":
            total += 1
            if result.player_factions.get(record.target_id) == "werewolf":
                correct += 1
    if total == 0:
        return None
    return round(correct / total, 6)


def _game_metrics(result: GameResult) -> dict[str, float]:
    metrics: dict[str, float] = {
        "good_win_rate": 1.0 if result.winning_faction == "good" else 0.0,
        "werewolf_win_rate": 1.0 if result.winning_faction == "werewolf" else 0.0,
        "illegal_action_count": float(
            sum(1 for event in result.event_log if _event_type(event) == "illegal_action")
        ),
    }
    vote_quality = _vote_quality_from_result(result)
    if vote_quality is not None:
        metrics["vote_quality"] = vote_quality
    return metrics
```

- [ ] **Step 5: Run the ablation test file to verify pass + no regressions**

Run: `python -m pytest tests/evaluation/test_full_game_ablation.py -q --basetemp=.pytest_tmp\mon`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/evaluation/full_game_ablation.py tests/evaluation/test_full_game_ablation.py
git commit -m "feat: full-game ablation emits vote_quality from action_records"
```

---

## Task 3: Populate DecisionOutcome.legal / leaked_hidden_info (revive diagnostics)

**Why:** `EvaluationTraceBuilder._decision_outcome` (`trace_builder.py:328-343`) only fills `target_role` / `target_faction` / `vote_hit_wolf`. It never sets `legal` or `leaked_hidden_info`, so `diagnose_trace_failures` (`diagnostics.py:59`, `:67`) — the `illegal_action` and `hidden_info_leak` categories — can never fire. Those two are the most important failure modes the feedback loop is supposed to surface, and right now they are dead code. The legality and leak logic already exists in `metrics.py` (`_decision_is_legal_from_trace`, `_dialogue_leaked_from_trace`), but `metrics` imports `trace_builder` (`metrics.py:39`), so we must extract those helpers into a cycle-free module first.

**Files:**
- Create: `werewolf_agent/evaluation/decision_helpers.py`
- Modify: `werewolf_agent/evaluation/metrics.py:39-61` and `:1047-1110`
- Modify: `werewolf_agent/evaluation/trace_builder.py:8-16` (imports), `:107-123` (call site), `:328-343` (function)
- Test: `tests/evaluation/test_feedback_trace_builder.py`, `tests/evaluation/test_feedback_diagnostics.py`

- [ ] **Step 1: Create decision_helpers.py with the migrated helpers**

Create `werewolf_agent/evaluation/decision_helpers.py`:

```python
"""Decision-legality and dialogue-leak helpers.

Extracted from ``metrics.py`` so both ``metrics`` and ``trace_builder`` can
share them without the ``metrics -> trace_builder`` import cycle. These are
pure functions over an ``action_trace`` audit dict; they read no game state.
"""

from __future__ import annotations

from typing import Any

# Action types that must carry a non-empty target_id to be legal.
TARGET_REQUIRED_ACTIONS = {
    "vote",
    "wolf_kill",
    "use_poison",
    "check_alignment",
    "choose_master",
    "hunter_shot",
    "badge_transfer",
    "sheriff_vote",
}


def decision_is_legal_from_trace(trace: dict[str, Any]) -> bool | None:
    """Return False when the action violates legal_actions/legal_targets.

    Returns None when legality cannot be decided (no action_type or no
    legal_* lists recorded), so callers can distinguish "unknown" from
    "illegal".
    """
    parsed = trace.get("parsed_action")
    parsed = parsed if isinstance(parsed, dict) else {}
    decision = parsed.get("decision_plan")
    decision = decision if isinstance(decision, dict) else {}
    action_type = str(
        trace.get("final_action_type")
        or parsed.get("action_type")
        or decision.get("action_type")
        or ""
    )
    if not action_type:
        return None
    legal_actions = trace.get("legal_actions")
    if isinstance(legal_actions, list) and legal_actions and action_type not in legal_actions:
        return False
    target_id = parsed.get("target_id") or decision.get("target_id")
    if action_type in TARGET_REQUIRED_ACTIONS and not target_id:
        return False
    legal_targets = trace.get("legal_targets")
    if (
        target_id
        and isinstance(legal_targets, list)
        and legal_targets
        and target_id not in legal_targets
    ):
        return False
    return True


def dialogue_leaked_from_trace(trace: dict[str, Any]) -> bool | None:
    """Return True when the public dialogue text leaks a concealed secret.

    Returns None when there is no dialogue_plan to inspect.
    """
    parsed = trace.get("parsed_action")
    if not isinstance(parsed, dict):
        return None
    dialogue = parsed.get("dialogue_plan")
    if not isinstance(dialogue, dict):
        return None
    public_parts = [
        dialogue.get("public_intent"),
        *(dialogue.get("talking_points") or []),
        parsed.get("reason"),
        parsed.get("speech"),
        parsed.get("speech_text"),
    ]
    public_text = "\n".join(str(part or "") for part in public_parts).lower()
    if not public_text:
        return False
    for secret in dialogue.get("conceal") or []:
        secret_text = str(secret or "").strip().lower()
        if len(secret_text) >= 4 and secret_text in public_text:
            return True
    return any(
        marker in public_text
        for marker in (
            "wolf teammate",
            "my teammate",
            "night kill",
            "private goal",
            "狼队友",
            "我的队友",
            "夜刀",
            "真实身份",
        )
    )
```

- [ ] **Step 2: Repoint metrics.py at the new helpers**

In `werewolf_agent/evaluation/metrics.py`:

Delete the local `_TARGET_REQUIRED_ACTIONS` block (currently lines 53-61) and the two function definitions `_decision_is_legal_from_trace` and `_dialogue_leaked_from_trace` (currently lines 1047-1110). Add this import block next to the existing `trace_builder` import (around line 39-40):

```python
from werewolf_agent.evaluation.decision_helpers import (
    TARGET_REQUIRED_ACTIONS as _TARGET_REQUIRED_ACTIONS,
    decision_is_legal_from_trace as _decision_is_legal_from_trace,
    dialogue_leaked_from_trace as _dialogue_leaked_from_trace,
)
```

This keeps every existing in-file call site (e.g. `_decision_is_legal_from_trace(...)`, `_TARGET_REQUIRED_ACTIONS`) working unchanged via the private aliases.

- [ ] **Step 3: Verify metrics still imports and tests still pass**

Run: `python -m pytest tests/evaluation/test_remaining_metrics.py -q --basetemp=.pytest_tmp\mon`
Expected: PASS (proves the extraction did not change behaviour).

- [ ] **Step 4: Write the failing trace-builder tests**

Append to `tests/evaluation/test_feedback_trace_builder.py`. Mirror the existing `action_trace_audit` event construction used elsewhere in that file (copy the helper that builds a `GameResult` with an `action_trace_audit` event from an existing test, then specialise the `action_trace` dict):

```python
def test_outcome_flags_illegal_action_type():
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    action_trace = {
        "final_action_type": "vote",
        "legal_actions": ["speech"],  # vote not in legal_actions -> illegal
        "legal_targets": ["p03"],
        "parsed_action": {"target_id": "p03"},
    }
    result = _build_result_with_action_trace(action_trace, player_id="p01")
    traces = EvaluationTraceBuilder().build(result)
    assert traces, "expected at least one trace"
    assert traces[0].outcome.legal is False


def test_outcome_flags_dialogue_leak():
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    action_trace = {
        "final_action_type": "speech",
        "parsed_action": {
            "speech": "我怀疑狼队友会刀我",  # contains the 狼队友 leak marker
            "dialogue_plan": {
                "public_intent": "我怀疑狼队友会刀我",
                "conceal": ["夜刀目标"],
            },
        },
    }
    result = _build_result_with_action_trace(action_trace, player_id="p01")
    traces = EvaluationTraceBuilder().build(result)
    assert traces[0].outcome.leaked_hidden_info is True


def test_outcome_legal_unknown_without_action_type():
    from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder

    action_trace = {"parsed_action": {}}  # no action_type
    result = _build_result_with_action_trace(action_trace, player_id="p01")
    traces = EvaluationTraceBuilder().build(result)
    assert traces[0].outcome.legal is None
    assert traces[0].outcome.leaked_hidden_info is False
```

Add the `_build_result_with_action_trace(action_trace, *, player_id)` helper in the same test file: it constructs a `GameResult` (reuse the file's existing GameResult import) whose `event_log` is a single `{"type": "action_trace_audit", "payload": {"player_id": player_id, "action_trace": action_trace}}` event, with `player_roles` / `player_factions` empty or minimal. Copy the GameResult construction shape from an existing passing test in this file.

- [ ] **Step 5: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation/test_feedback_trace_builder.py::test_outcome_flags_illegal_action_type tests/evaluation/test_feedback_trace_builder.py::test_outcome_flags_dialogue_leak tests/evaluation/test_feedback_trace_builder.py::test_outcome_legal_unknown_without_action_type -q --basetemp=.pytest_tmp\mon`
Expected: the first two FAIL (`outcome.legal` / `leaked_hidden_info` are never set), the third passes trivially.

- [ ] **Step 6: Wire decision_helpers into trace_builder**

In `werewolf_agent/evaluation/trace_builder.py` add to the imports (top of file, alongside the other `werewolf_agent.evaluation` imports):

```python
from werewolf_agent.evaluation.decision_helpers import (
    decision_is_legal_from_trace,
    dialogue_leaked_from_trace,
)
```

Change the call site in `_trace_from_action_event` (currently `trace_builder.py:121`) to pass the `action_trace` dict:

```python
            outcome=_decision_outcome(result, decision, action_trace),
```

Replace `_decision_outcome` (currently `trace_builder.py:328-343`) with:

```python
def _decision_outcome(
    result: GameResult,
    decision: DecisionSnapshot,
    action_trace: dict[str, Any],
) -> DecisionOutcome:
    target_id = decision.target_id or ""
    target_role = result.player_roles.get(target_id, "")
    target_faction = result.player_factions.get(target_id, "")
    vote_hit_wolf = None
    if decision.action_type == "vote" and target_faction:
        vote_hit_wolf = target_faction == "werewolf"
    legal: bool | None = None
    leaked: bool = False
    if isinstance(action_trace, dict):
        legal = decision_is_legal_from_trace(action_trace)
        leak_decision = dialogue_leaked_from_trace(action_trace)
        leaked = leak_decision is True
    return DecisionOutcome(
        legal=legal,
        target_role=target_role,
        target_faction=target_faction,
        vote_hit_wolf=vote_hit_wolf,
        leaked_hidden_info=leaked,
        outcome_refs=[f"player_roles:{target_id}"] if target_id and target_role else [],
    )
```

- [ ] **Step 7: Run trace-builder tests to verify pass**

Run: `python -m pytest tests/evaluation/test_feedback_trace_builder.py -q --basetemp=.pytest_tmp\mon`
Expected: PASS.

- [ ] **Step 8: Write the diagnostics revival test**

Append to `tests/evaluation/test_feedback_diagnostics.py`:

```python
def test_illegal_outcome_produces_illegal_action_diagnosis():
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures
    from werewolf_agent.evaluation.feedback_schemas import (
        DecisionOutcome,
        DecisionSnapshot,
        EvaluationTrace,
    )

    trace = EvaluationTrace(
        trace_id="t1",
        game_id="g1",
        player_id="p01",
        role="villager",
        faction="good",
        phase="day_vote",
        decision=DecisionSnapshot(action_type="vote", target_id="p02"),
        outcome=DecisionOutcome(legal=False, target_faction="good"),
    )
    diagnoses = diagnose_trace_failures([trace])
    categories = {d.category for d in diagnoses}
    assert "illegal_action" in categories


def test_leaked_outcome_produces_hidden_info_leak_diagnosis():
    from werewolf_agent.evaluation.diagnostics import diagnose_trace_failures
    from werewolf_agent.evaluation.feedback_schemas import (
        DecisionOutcome,
        DecisionSnapshot,
        EvaluationTrace,
    )

    trace = EvaluationTrace(
        trace_id="t2",
        game_id="g1",
        player_id="p01",
        role="villager",
        faction="good",
        phase="speech",
        decision=DecisionSnapshot(action_type="speech"),
        outcome=DecisionOutcome(leaked_hidden_info=True),
    )
    diagnoses = diagnose_trace_failures([trace])
    categories = {d.category for d in diagnoses}
    assert "hidden_info_leak" in categories
```

- [ ] **Step 9: Run diagnostics tests to verify pass**

Run: `python -m pytest tests/evaluation/test_feedback_diagnostics.py -q --basetemp=.pytest_tmp\mon`
Expected: PASS (these tests pass even before Task 3 because they construct the trace directly; they lock in the revived behaviour end-to-end and guard against future regressions in `diagnostics.py`).

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/evaluation/decision_helpers.py werewolf_agent/evaluation/metrics.py werewolf_agent/evaluation/trace_builder.py tests/evaluation/test_feedback_trace_builder.py tests/evaluation/test_feedback_diagnostics.py
git commit -m "fix: populate decision outcome legality/leak to revive failure diagnostics"
```

---

## Task 4: Stop scrubbing the public werewolf_win_rate aggregate

**Why:** `FeedbackReport._PRIVATE_AUDIT_TOKENS` (`feedback_report.py:36-44`) contains `"werewolf"`. `werewolf` is a public faction word (the winning side), not a hidden identity, so the public-view scrubber (`to_json_dict(include_private_audit=False)`) deletes any metric key/value containing it — including the legitimate public `werewolf_win_rate` aggregate. Hidden player↔role mappings are already covered by the `_PLAYER_ID_RE` filter and the `_PRIVATE_AUDIT_KEYS` set (`actual_role`, `target_role`, etc.); the bare token `werewolf` adds only false positives.

**Files:**
- Modify: `werewolf_agent/evaluation/feedback_report.py:36-44`
- Test: `tests/evaluation/test_feedback_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_feedback_report.py`. Reuse the existing `FeedbackReport` / `ModuleAttributionSummary` imports at the top of the file:

```python
def test_public_view_keeps_werewolf_win_rate():
    import json
    from werewolf_agent.evaluation.feedback_report import FeedbackReport
    from werewolf_agent.evaluation.feedback_metrics import ModuleAttributionSummary

    report = FeedbackReport(
        report_id="r1",
        batch_id="b1",
        trace_count=0,
        module_metrics={"vote": ModuleAttributionSummary(exposure_count=0)},
    )
    # Inject a public aggregate that contains the "werewolf" token.
    data = report.to_json_dict(include_private_audit=False)
    # Simulate a module metric value carrying the public token.
    data["module_metrics"]["werewolf_aggregate"] = {"werewolf_win_rate": 0.6}
    serialized = json.dumps(data, ensure_ascii=False)
    assert "werewolf_win_rate" in serialized


def test_public_view_still_scrubs_private_actual_role():
    import json
    from werewolf_agent.evaluation.feedback_report import FeedbackReport

    report = FeedbackReport(
        report_id="r2",
        batch_id="b1",
        trace_count=0,
        module_metrics={},
    )
    data = report.to_json_dict(include_private_audit=False)
    data["candidates"] = [{"audit_evidence": {"actual_role": "werewolf"}}]
    serialized = json.dumps(data, ensure_ascii=False)
    assert "actual_role" not in serialized
```

Note: `test_public_view_keeps_werewolf_win_rate` constructs the dict directly to target the scrubber's token-matching path precisely; if `test_feedback_report.py` already has a helper that round-trips a `FeedbackReport` through `to_json_dict`, prefer extending that helper instead.

- [ ] **Step 2: Run tests to verify the first fails**

Run: `python -m pytest tests/evaluation/test_feedback_report.py::test_public_view_keeps_werewolf_win_rate tests/evaluation/test_feedback_report.py::test_public_view_still_scrubs_private_actual_role -q --basetemp=.pytest_tmp\mon`
Expected: the first may already pass depending on the exact scrubber call site (the token set only bites strings passed through `_is_public_safe_text`); the second must PASS. If the first passes, locate the scrubber call site that drops `werewolf_win_rate` by grepping `_PRIVATE_AUDIT_TOKENS` usage and adjust the test to exercise that exact path so it fails before the fix. (The grep is part of this step: run `findstr /N "_PRIVATE_AUDIT_TOKENS" werewolf_agent\evaluation\feedback_report.py`.)

- [ ] **Step 3: Remove the token**

In `werewolf_agent/evaluation/feedback_report.py`, edit `_PRIVATE_AUDIT_TOKENS` (lines 36-44) to drop the `"werewolf"` line:

```python
_PRIVATE_AUDIT_TOKENS = {
    "actual_role",
    "ground_truth",
    "key_assignments",
    "target_faction",
    "target_role",
    "true_world_rank",
}
```

`_PRIVATE_AUDIT_KEYS` (lines 28-35) is unchanged — it never contained `werewolf`.

- [ ] **Step 4: Run the feedback-report test file to verify pass + no regressions**

Run: `python -m pytest tests/evaluation/test_feedback_report.py -q --basetemp=.pytest_tmp\mon`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/feedback_report.py tests/evaluation/test_feedback_report.py
git commit -m "fix: stop scrubbing public werewolf_win_rate in feedback report"
```

---

## Task 5: Whole-module regression + PROGRESS update

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run the full evaluation suite**

Run: `python -m pytest tests/evaluation -q --basetemp=.pytest_tmp\mon`
Expected: all PASS.

- [ ] **Step 2: Run the cognition + runtime suites that touch DecisionOutcome / traces**

Run: `python -m pytest tests/cognition tests/runtime -q --basetemp=.pytest_tmp\mon`
Expected: all PASS (sanity check that the metrics/trace_builder extraction did not leak).

- [ ] **Step 3: Compile-check the package**

Run: `python -m compileall -q werewolf_agent`
Expected: no output (no stale imports).

- [ ] **Step 4: Update PROGRESS.md**

Add a new section at the top of `PROGRESS.md` (following the existing entry style) recording: the four fixes, the files changed, the verification commands above, and the open risks (judge_consistency_rate / harmful_transfer_rate producers are still absent — tracked as the next plan `judge-and-harmful-producers`). Note explicitly that `required_metrics` defaults to empty so historical gate callers are unaffected, and that the diagnostics revival is backward compatible because `DecisionOutcome.legal` / `leaked_hidden_info` default to `None` / `False`.

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: log monitoring closure fix in PROGRESS"
```

---

## Open Risks / Out of Scope

- `judge_consistency_rate` and `harmful_transfer_rate` still have no producer that feeds the gate. After this plan, a caller who wants those guarded must set `required_metrics=("judge_consistency_rate", "harmful_transfer_rate")` AND wire a producer (running `judge_speech_consistency` over traces for the former; aggregating `feedback_metrics` harmful flags for the latter). That wiring needs the traces pipeline and is deferred to a separate plan `judge-and-harmful-producers`.
- The replay path of `FullGameAblationRunner` (`_result_from_replay_record`) builds a sparse `GameResult` with empty `action_records`, so `vote_quality` is omitted there by design (Task 2 returns `None`). Callers running replay ablations must not set `required_metrics=("vote_quality",)` without also enriching the replay `GameResult`.
- Balance-audit design-intent deltas (high wolf win-rate, seer D1 exile) are intentionally NOT touched — per `memory/feedback-balance-audit-design-scope.md` those reflect real-Werewolf bias, not bugs.

## Self-Review

- **Spec coverage:** The two P0 break-points (gate silent-skip, dead illegal/leak diagnostics) are covered by Tasks 1-3. The `werewolf` token false-positive is Task 4. Decision-quality producer coverage starts with Task 2 (`vote_quality`); the remaining producers are explicitly out of scope and tracked above.
- **Placeholder scan:** No TBD / TODO. Every code step shows the actual code. Two steps (Task 2 Step 1 and Task 4 Step 2) include a read/grep to confirm exact local field names / call sites because those depend on test-file fixtures the implementer must mirror — this is verification, not a placeholder.
- **Type consistency:** `decision_is_legal_from_trace` / `dialogue_leaked_from_trace` are defined in Task 3 Step 1 and consumed in Task 3 Step 6 with matching signatures. `_decision_outcome` gains an `action_trace: dict[str, Any]` parameter used at the single call site in Step 6. `required_metrics: tuple[str, ...]` is added in Task 1 Step 3 and read in Task 1 Step 4. `_vote_quality_from_result` returns `float | None` and the `None` branch is asserted in Task 2 Step 2.
