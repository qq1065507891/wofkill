# Semantic Public Evidence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the over-strict semantic claim-retention blocker with a V2 public-evidence safety gate that allows supported revisions and always exposes safe, specific rejection reasons.

**Architecture:** Keep claim classification and public support decisions in `balance_public_claims.py`; add a structured validation result in `semantic_repair_audit.py`; propagate stable reason codes through `RetryInfo`, retry prompts, logs, and terminal traces. Version semantic audit rows so acceptance reporting can dual-read V1 history while applying V2 success and release thresholds only to complete V2 batches.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, Ruff, existing game audit/acceptance pipeline.

---

## File responsibility map

- `werewolf_agent/agents/semantic_repair_audit.py`: V2 audit fields, stable rejection codes, safe explanations, compatibility wrapper.
- `werewolf_agent/agents/trace_schemas.py`: retry reason-code schema.
- `werewolf_agent/agents/player_action_flow.py`: validation ordering, targeted retries, safe logging.
- `werewolf_agent/evaluation/acceptance_terminal_semantic_metrics.py`: V1/V2 dual-read metrics.
- `scripts/evaluate_audit_closure_thresholds.py`: V2-aligned release thresholds.
- Agent, runtime, evaluation, and script test files: contract and regression evidence.

### Task 1: Define the V2 validator contract

**Files:**
- Modify: `tests/agents/test_semantic_repair_invariants.py`
- Modify: `tests/agents/test_player_retry.py`
- Modify: `werewolf_agent/agents/semantic_repair_audit.py`

- [ ] **Step 1: Write failing tests for allowed supported revisions**

Add independent tests proving all three cases are accepted:

```python
result = validate_semantic_repair(context, source, final)
assert result.accepted is True
assert result.reason_codes == ()
assert result.audit["semantic_gate_version"] == 2
assert result.audit["unsupported_public_claim_count"] == 0
```

Cases: final adds a claim already supported by `public_claim_ledger`; final drops one verified source claim; final changes a legal, not-yet-executed speech target. Replace old tests that expected dropped claims or target changes to fail.

- [ ] **Step 2: Write failing tests for all rejection reasons**

Cover `unsupported_public_claim`, `speaker_attribution_changed`, and `negation_changed`. Add a multi-failure case where one retained claim has both its non-empty speaker and its negation changed, plus a separate unsupported claim; require every applicable code in stable, duplicate-free ordering:

```python
assert result.reason_codes == (
    "unsupported_public_claim",
    "speaker_attribution_changed",
    "negation_changed",
)
assert result.audit["rejection_reason_codes"] == list(result.reason_codes)
```

Parameterize the fixed-message helpers and assert each single reason code produces its corresponding safe Chinese explanation (`公开证据`, `说话人归属`, or `否定关系`) and correction hint. For a subset of reasons, assert the text mentions only that subset, contains no speech sentinel/private role truth, and never contains the obsolete blanket guidance `不得新增任何事实` or `保留全部论点`.

- [ ] **Step 3: Verify RED**

Run:

```powershell
$env:LANGCHAIN_TRACING_V2='false'
$env:LANGSMITH_TRACING='false'
$env:TMP=(Resolve-Path '.tmp').Path
$env:TEMP=$env:TMP
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-t1-red -o addopts='' tests/agents/test_semantic_repair_invariants.py tests/agents/test_player_retry.py -q
```

Expected: FAIL because the structured validator and V2 fields do not exist and old behavior rejects supported changes.

- [ ] **Step 4: Implement the minimal validator**

Add to `semantic_repair_audit.py`:

```python
@dataclass(frozen=True)
class SemanticRepairValidationResult:
    accepted: bool
    reason_codes: tuple[str, ...]
    audit: dict[str, Any]
```

Compute unsupported final claims as `final_claims - final_verified_claims`. Reject only unsupported claims, changed speaker attribution, and changed negation. Keep target, introduced-claim, and retained-claim fields as observations. Add `semantic_gate_version=2`, `unsupported_public_claim_count`, and `rejection_reason_codes` to every new audit.

Add fixed-message helpers:

```python
semantic_repair_rejection_message(reason_codes)
semantic_repair_correction_hint(reason_codes)
```

They must never interpolate speech text, role truth, or provider errors. Keep `semantic_repair_retains_verified_claim()` as a boolean wrapper over the new validator. Synchronize the Chinese module description and set `修改日期: 2026-07-19`.

- [ ] **Step 5: Verify GREEN and commit**

Repeat Step 3 with basetemp `pytest-semantic-t1-green`; expect exit 0. Then:

```powershell
git add -- werewolf_agent/agents/semantic_repair_audit.py tests/agents/test_semantic_repair_invariants.py tests/agents/test_player_retry.py
git commit -m "fix: narrow semantic repair evidence gate"
```

### Task 2: Propagate safe reason codes through retry traces

**Files:**
- Modify: `tests/agents/test_schemas.py`
- Modify: `tests/agents/test_task_terminal_fallbacks.py`
- Modify: `werewolf_agent/agents/trace_schemas.py`
- Modify: `werewolf_agent/agents/trace_builder.py`

- [ ] **Step 1: Write failing schema and privacy tests**

Require `RetryInfo().reason_codes == []`. Build a terminal trace from:

```python
RetryInfo(
    error_code="semantic_claim_retention",
    error_message="private rejected speech text",
    reason_codes=["unsupported_public_claim"],
)
```

Use unique sentinels in `raw_text`, parsed speech, and `error_message`. Assert `trace.retry["reason_codes"]` survives, while all three private sentinels are absent from serialized terminal JSON. The deterministic `final_action` must remain available.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-t2-red -o addopts='' tests/agents/test_schemas.py tests/agents/test_task_terminal_fallbacks.py -q
```

Expected: FAIL because `RetryInfo.reason_codes` does not exist and terminal traces retain rejected `raw_text`/`parsed_action`.

- [ ] **Step 3: Implement the compatibility field**

Add to `RetryInfo`:

```python
reason_codes: list[str] = Field(default_factory=list)
```

Do not add a second `ActionTrace` field. In `trace_builder.py`, when building a terminal fallback whose retry has non-empty semantic `reason_codes`, serialize `raw_text=""` and `parsed_action=None`; keep `final_action`, the V2 aggregate audit, stable failure codes, and `retry.reason_codes`. Other failure types retain their existing audit behavior. Update both module descriptions/dates as required.

- [ ] **Step 4: Verify GREEN and commit**

Repeat Step 2 with basetemp `pytest-semantic-t2-green`; expect exit 0. Then commit the four explicit files with message `feat: trace semantic rejection reasons`.

### Task 3: Reorder runtime validation and expose reasons

**Files:**
- Modify: `tests/agents/test_player_agent.py`
- Modify: `werewolf_agent/agents/player_action_flow.py`

- [ ] **Step 1: Write the allowed-revision integration test**

Replace `test_semantic_repair_retries_until_verified_claim_is_retained` with `test_semantic_repair_allows_supported_revision`. Attempt 1 fails speech quality; attempt 2 drops the old claim or adds a public-ledger-supported claim. Assert only two provider calls, the second speech is accepted, and the V2 audit has no rejection reasons.

- [ ] **Step 2: Write the ordering and reason test**

Add `test_semantic_repair_runs_before_speech_quality`. Make attempt 2 contain an unsupported public claim and also configure the speech-quality mock to fail if called. Assert:

```python
assert retry.error_code == "semantic_claim_retention"
assert retry.reason_codes == ["unsupported_public_claim"]
```

Assert speech quality was called only for attempt 1, proving the repair gate runs first on subsequent candidates.

- [ ] **Step 3: Write reason propagation tests**

Add `test_semantic_repair_reason_codes_reach_prompt_and_success_trace`. Exercise a semantic rejection followed by a successful repair. Capture the next fully rendered prompt and the returned successful trace; assert the same reason codes are present in retry metadata and `action.trace.retry["reason_codes"]`. Assert the prompt contains the safe Chinese explanation and correction hint for each actual failure, omits explanations for reasons that did not occur, excludes rejected-speech/private-role sentinels, and excludes `不得新增任何事实` and `保留全部论点`. This locks both prompt rendering and the top-of-loop `RetryInfo` reconstruction contract instead of testing only immediate rejection state.

- [ ] **Step 4: Write terminal log/trace privacy tests**

Add `test_semantic_repair_terminal_log_trace_privacy`. With `caplog`, exhaust retries and assert:

```python
assert "last_reason_codes=unsupported_public_claim" in caplog.text
assert rejected_speech not in caplog.text
assert action.trace.retry["reason_codes"] == ["unsupported_public_claim"]
assert action.trace.retry.get("error_message") is None
```

Also require an intermediate rejection log containing agent/task/attempt/reason codes only.

- [ ] **Step 5: Verify RED**

```powershell
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-t3-red -o addopts='' tests/agents/test_player_agent.py -k "semantic_repair_allows_supported_revision or semantic_repair_runs_before_speech_quality or semantic_repair_reason_codes_reach_prompt_and_success_trace or semantic_repair_terminal_log_trace_privacy" -q
```

Expected: FAIL because current ordering runs speech quality first, allowed revisions are rejected, and logs omit reasons.

- [ ] **Step 6: Implement the minimal flow change**

After action legality, call `validate_semantic_repair()` before `_speech_quality_error()` whenever a repair source exists. On failure, create `RetryInfo` with all reason codes and helper-generated safe message/hint, log stable codes, reject the attempt, and continue. When reconstructing `RetryInfo` at the top of the next loop, copy `reason_codes` along with the existing error fields so both the next prompt and any success/terminal trace retain them. Reuse `validation.audit` for successful trace emission. Keep the first speech-quality failure as the only source initializer. Add `last_reason_codes` to the exhausted-retries warning, using `none` when empty. Do not change provider fallback, vote quality, legal-action checks, or deterministic terminal fallback.

Update the module description only if needed and set its modification date.

- [ ] **Step 7: Verify GREEN and commit**

Repeat Step 5 with basetemp `pytest-semantic-t3-green`; expect exit 0. Commit the two files with message `fix: report semantic gate rejection reasons`.

### Task 4: Align V2 acceptance metrics and thresholds

**Files:**
- Modify: `tests/evaluation/test_semantic_acceptance_invariants.py`
- Modify: `tests/evaluation/test_report_acceptance_metrics.py`
- Modify: `tests/scripts/test_evaluate_audit_closure_thresholds.py`
- Modify: `werewolf_agent/evaluation/acceptance_terminal_semantic_metrics.py`
- Modify: `scripts/evaluate_audit_closure_thresholds.py`

- [ ] **Step 1: Write failing V2 success tests**

Create a V2 accepted row with `target_preserved=False`, `introduced_claim_count=1`, partial verified-claim retention, preserved speaker/negation, and zero unsupported claims. Assert it counts as semantic success while the three observation rates still report the changes.

- [ ] **Step 2: Write V1/V2 support tests**

Cover:

- V1-only: old success calculation remains; V2 safety support is false and rate is `None`.
- V2-only valid: safety support true and rate `1.0`.
- V2 unsafe: support true and rate below `1.0`.
- V1/V2 mixed: support false and rate `None`.
- malformed/missing V2 unsupported count: fail closed.

- [ ] **Step 3: Write threshold replacement tests**

Assert `_THRESHOLDS` removes target preservation, no-new-claim, and verified-claim-retention hard gates, and adds:

```python
(
    "semantic_repair_public_evidence_safety_rate",
    "==",
    1.0,
    "semantic_repair_public_evidence_safety_metrics_supported",
)
```

Prove missing/unsupported V2 safety evidence makes `overall_pass=False`.

- [ ] **Step 4: Verify RED**

```powershell
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-t4-red -o addopts='' tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py tests/scripts/test_evaluate_audit_closure_thresholds.py -q
```

Expected: FAIL because current metrics apply V1 retention semantics to all rows and thresholds remain obsolete.

- [ ] **Step 5: Implement dual-read metrics and thresholds**

In `acceptance_terminal_semantic_metrics.py`, preserve the current calculation for rows without a version. For V2, success requires `success=True`, speaker and negation preservation, and `unsupported_public_claim_count=0`; target/no-new/retention remain observational. Compute safety rate as safe V2 rows divided by V2 rows only when the projection is supported, the semantic set is non-empty, every row is V2, and every unsupported count is a non-negative integer. Otherwise return support false and rate `None`.

Replace the three obsolete script thresholds with the V2 safety threshold. Update both Python module descriptions/dates as required by repository instructions.

- [ ] **Step 6: Verify GREEN and commit**

Repeat Step 4 with basetemp `pytest-semantic-t4-green`; expect exit 0. Commit the five explicit files with message `fix: align semantic gate acceptance metrics`.

### Task 5: Lock and repair public evidence snapshot parity

**Files:**
- Modify: `tests/runtime/test_public_ledger.py`
- Modify: `werewolf_agent/evaluation/balance_public_claims.py`

- [ ] **Step 1: Add the parity regression**

Build a `GameState` with a public p05 role claim, a moderator-only p06 role truth/claim, and another public sheriff/PK/last-words speech. Assert:

```python
ledger = build_public_claim_text_ledger(game_state)
history = public_speech_history(game_state.events)
assert [(item["speaker"], item["text"]) for item in ledger] == history
```

Validate and sanitize the same candidate against both snapshots; require identical support/unsupported results and absence of p06 private content.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-t5 -o addopts='' tests/runtime/test_public_ledger.py -q
```

Expected: FAIL because `public_speech_history()` filters event types but currently ignores both typed and legacy payload visibility.

- [ ] **Step 3: Implement visibility-equivalent history extraction**

In `balance_public_claims.py`, make `public_speech_history()` accept only events whose effective visibility is public. For `GameEvent`, use `event_visibility(event) is EventVisibility.PUBLIC`; for legacy mappings, resolve the explicit top-level visibility first and then `payload["visibility"]`, defaulting to public only when neither is present. Keep the speech event-type set exactly aligned with `build_public_claim_text_ledger()`. Do not widen visibility or import role truth. Update the module description/date.

- [ ] **Step 4: Verify GREEN and commit**

Repeat Step 2 with basetemp `pytest-semantic-t5-green`; expect exit 0. Commit the two explicit files with message `fix: align public evidence snapshots`.

### Task 6: Full verification and final handoff

**Files:** Verify only; preserve the user's existing untracked artifacts.

- [ ] **Step 1: Run the complete focused suite**

```powershell
$env:LANGCHAIN_TRACING_V2='false'
$env:LANGSMITH_TRACING='false'
$env:TMP=(Resolve-Path '.tmp').Path
$env:TEMP=$env:TMP
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-focused -o addopts='' tests/agents/test_semantic_repair_invariants.py tests/agents/test_player_retry.py tests/agents/test_schemas.py tests/agents/test_task_terminal_fallbacks.py tests/agents/test_player_agent.py tests/runtime/test_public_ledger.py tests/runtime/test_agent_action_audit.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py tests/scripts/test_evaluate_audit_closure_thresholds.py -q
```

Expected: exit 0 with a final pass count. A timeout or progress reaching 100% without the final summary is unverified.

Also run the direct projection and SRP contract suites affected by the acceptance-metric module:

```powershell
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-projection -o addopts='' tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_balance_audit_srp_split.py -q
```

- [ ] **Step 2: Run broader affected regressions**

```powershell
python -m pytest -n 0 --basetemp .tmp\pytest-semantic-regression -o addopts='' tests/agents tests/runtime/test_context.py tests/runtime/test_day_discussion.py tests/runtime/test_sheriff_flow.py tests/runtime/test_sheriff_node_split.py tests/evaluation/test_final_acceptance_integrity.py tests/scripts/test_run_real_game.py -q
```

Split the batch if it approaches the execution timeout; report only completed summaries.

- [ ] **Step 3: Run static and diff checks**

```powershell
python -m ruff check werewolf_agent/agents/semantic_repair_audit.py werewolf_agent/agents/trace_schemas.py werewolf_agent/agents/trace_builder.py werewolf_agent/agents/player_action_flow.py werewolf_agent/evaluation/balance_public_claims.py werewolf_agent/evaluation/acceptance_terminal_semantic_metrics.py scripts/evaluate_audit_closure_thresholds.py tests/agents/test_semantic_repair_invariants.py tests/agents/test_player_retry.py tests/agents/test_schemas.py tests/agents/test_task_terminal_fallbacks.py tests/agents/test_player_agent.py tests/runtime/test_public_ledger.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py tests/scripts/test_evaluate_audit_closure_thresholds.py
git diff --check
git status --short
```

Expected: Ruff and diff check exit 0. Status contains no unintended tracked files and still preserves the pre-existing untracked `.claude/worktrees/`, `artifacts/`, and audit draft.

- [ ] **Step 4: Inspect history and commit only verified corrections**

```powershell
git log -10 --oneline --decorate
git status --short
```

Confirm no credentials, raw rejected speech, private role truth, or unrelated cleanup entered commits. If verification required a focused correction, commit only that correction and its test; do not create an empty commit.
