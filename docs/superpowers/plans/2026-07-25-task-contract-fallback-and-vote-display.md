# Task Contract, Fallback Observability, and Vote Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate public speech, internal discussion summaries, and reflections into explicit task contracts; make terminal failures safely diagnosable; make truncated metrics honest; and display integer vote units as real 1/1.5 vote values without changing rule outcomes.

**Architecture:** Keep provider routing, provider retries, structured-mode fallback, and attempt identity in the shared execution layer. Move parsing, validation, repair, and terminal fallback into cohesive task-specific modules. Preserve existing event fields as internal-unit compatibility aliases, add versioned canonical unit/display fields, and migrate all display consumers through one decoder.

**Tech Stack:** Python 3.11+, Pydantic v2, immutable runtime models/events, pytest, YAML rulesets, vanilla JavaScript dashboard.

**Spec:** `docs/superpowers/specs/2026-07-25-task-contract-fallback-and-vote-display-design.md`

---

## File Structure

### New files

- `werewolf_agent/runtime/vote_display.py` — exact integer/half-integer vote conversion, V1/V2 payload decoding, and display formatting.
- `werewolf_agent/agents/discussion_summary.py` — `DiscussionSummary` Schema, legacy coercion, accessor, and deterministic text projection.
- `werewolf_agent/agents/speech_repair_requirements.py` — immutable cumulative public-speech repair constraints.
- `werewolf_agent/agents/safe_failure_diagnostics.py` — allowlisted Schema/quality failure projection without rejected text.
- `werewolf_agent/agents/reflection_contract.py` — reflection-specific generation, parsing, retry, and terminal-result contract.
- `tests/runtime/test_vote_display.py`
- `tests/agents/test_discussion_summary.py`
- `tests/agents/test_speech_repair_requirements.py`
- `tests/agents/test_safe_failure_diagnostics.py`
- `tests/agents/test_reflection_contract.py`
- `tests/integration/test_task_contract_acceptance.py`

### Existing files with focused changes

- `werewolf_agent/runtime/nodes/day_vote.py` — emit V2 vote payloads and display values.
- `werewolf_agent/runtime/nodes/judge_broadcast_helpers.py` — give judge prompts/display real vote values.
- `werewolf_agent/runtime/context_public_summary.py` — decode vote payloads through the shared helper.
- `werewolf_agent/runtime/directives/wolf.py` — read canonical vote units for ranking.
- `werewolf_agent/agents/prompt_schemas.py` — add `DISCUSSION_SUMMARY`.
- `werewolf_agent/agents/prompt_output.py` and `werewolf_agent/agents/player.py` — route the new task without treating it as public speech.
- `werewolf_agent/runtime/nodes/runtime_state.py` — correct `discussion_positions` type.
- `werewolf_agent/runtime/nodes/summary.py` — use summary and reflection contracts; report exact counts.
- `werewolf_agent/runtime/context.py` — consume summary through the compatibility accessor.
- `werewolf_agent/agents/player_action_flow.py` — orchestrate cumulative repair requirements and safe diagnostics.
- `werewolf_agent/agents/semantic_repair_audit.py` — return complete stable rejection sets.
- `werewolf_agent/runtime/reflection_transaction.py` — preserve exact contract failure boundaries and count invariants.
- `werewolf_agent/runtime/agent_reflection_support.py` — compatibility facade over the reflection contract.
- `werewolf_agent/evaluation/acceptance_reflection_metrics.py` — validate old and new reflection counts.
- `werewolf_agent/evaluation/acceptance_audit.py` and `scripts/run_real_game.py` — make event-derived metrics unsupported on truncated projections.
- `werewolf_agent/ui/static/dashboard.js`, `scripts/print_game_audit.py`, and
  `scripts/run_real_game_reports.py` — render decoded vote-display values and metric support states.
- Existing focused tests under `tests/runtime/`, `tests/agents/`, `tests/evaluation/`, and `tests/scripts/`.

Do not split unrelated logic or retrofit headers in untouched files. New Python files must use the repository’s Chinese module-docstring/header convention.

---

### Task 1: Versioned Vote Unit and Display Conversion

**Files:**
- Create: `werewolf_agent/runtime/vote_display.py`
- Create: `tests/runtime/test_vote_display.py`
- Modify: `werewolf_agent/runtime/nodes/day_vote.py:166-218`
- Modify: `werewolf_agent/runtime/nodes/day_vote.py:222-316`
- Test: `tests/runtime/test_vote_flow.py`
- Test: `tests/runtime/test_judge_flow.py`

- [ ] **Step 1: Write failing conversion and decoder tests**

Cover exact values, V1 compatibility, V2 invariants, and fail-closed conflicts:

```python
def test_vote_units_to_display_preserves_half_votes() -> None:
    assert vote_units_to_display(2, base_vote_weight=2) == Decimal("1")
    assert vote_units_to_display(3, base_vote_weight=2) == Decimal("1.5")
    assert vote_units_to_display(21, base_vote_weight=2) == Decimal("10.5")


def test_decode_v2_rejects_alias_unit_conflict() -> None:
    payload = {
        "vote_weight_format_version": 2,
        "base_vote_weight": 2,
        "weighted_tally": {"p02": 20},
        "weighted_tally_units": {"p02": 21},
        "weighted_tally_display": {"p02": 10.5},
    }
    with pytest.raises(VotePayloadError):
        decode_vote_resolved_payload(payload)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\vote-display-red -o addopts='' tests/runtime/test_vote_display.py -q
```

Expected: collection/import failure because `werewolf_agent.runtime.vote_display` does not exist.

- [ ] **Step 3: Implement the minimal exact conversion API**

Use `Decimal` only at the presentation boundary. Rule comparisons continue to use integers.

```python
def vote_units_to_display(units: int, *, base_vote_weight: int) -> Decimal:
    if type(units) is not int or type(base_vote_weight) is not int:
        raise TypeError("vote units and base_vote_weight must be integers")
    if units < 0 or base_vote_weight <= 0:
        raise ValueError("vote units must be non-negative and base_vote_weight positive")
    return Decimal(units) / Decimal(base_vote_weight)


def format_vote_count(value: Decimal) -> str:
    return format(value.quantize(Decimal("1")) if value == value.to_integral()
                  else value.normalize(), "f")
```

Add separate decoders for `judge_broadcast/vote_tally` and `vote_resolved`. They must enforce the spec’s V1/V2 precedence and never guess a missing historical base weight.

- [ ] **Step 4: Run conversion tests and verify they pass**

Run the Step 2 command.

Expected: all tests in `test_vote_display.py` pass.

- [ ] **Step 5: Write failing runtime-event tests**

Update/add assertions that V2 events preserve compatibility aliases while adding canonical fields:

```python
assert payload["vote_weight_format_version"] == 2
assert payload["base_vote_weight"] == 2
assert payload["weighted_tally"] == {"p03": 3, "p02": 2}
assert payload["weighted_tally_units"] == payload["weighted_tally"]
assert payload["weighted_tally_display"] == {"p03": 1.5, "p02": 1}
assert payload["vote_weight_units"] == {"p01": 3, "p02": 2}
assert payload["vote_weights_display"] == {"p01": 1.5, "p02": 1}
```

For the `judge_broadcast` event with `judge_method=vote_tally`, assert the complete
V2 contract:

```python
assert payload["vote_weight_format_version"] == 2
assert payload["base_vote_weight"] == 2
assert payload["tally"] == payload["tally_units"]
assert payload["sheriff_weight"] == payload["sheriff_weight_units"] == 3
assert payload["tally_display"] == {"p03": 1.5, "p02": 1}
assert payload["sheriff_weight_display"] == 1.5
assert "警长1.5票" in payload["message"]
assert "警长3票" not in payload["message"]
```

- [ ] **Step 6: Run focused runtime tests and verify the new assertions fail**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\vote-events-red -o addopts='' tests/runtime/test_vote_flow.py tests/runtime/test_judge_flow.py -q
```

Expected: failures for missing version/display fields and the old `警长3票` label.

- [ ] **Step 7: Emit V2 fields and format all runtime vote text**

In `_broadcast_vote_details()` and `resolve_vote()`:

- keep `tally`, `sheriff_weight`, `weighted_tally`, and `vote_weights` in internal units;
- add mandatory version/base fields;
- add canonical `*_units`;
- add `*_display` values;
- use the exact singular field name `vote_weight_units` in `vote_resolved`;
- format debug and judge messages from display values.

Do not change `RuleEngine.resolve_vote()`, `vote_weight()`, or tie resolution.

- [ ] **Step 8: Run vote runtime and rule tests**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\vote-events-green -o addopts='' tests/runtime/test_vote_display.py tests/runtime/test_vote_flow.py tests/runtime/test_judge_flow.py tests/rules/test_rule_engine_v1.py -q
```

Expected: all selected tests pass; existing exile/tie results remain unchanged.

- [ ] **Step 9: Commit Task 1**

```powershell
git add werewolf_agent/runtime/vote_display.py werewolf_agent/runtime/nodes/day_vote.py tests/runtime/test_vote_display.py tests/runtime/test_vote_flow.py tests/runtime/test_judge_flow.py
git commit -m "fix: version vote units and display values"
```

---

### Task 2: Migrate Vote Consumers Without Double Conversion

**Files:**
- Modify: `werewolf_agent/runtime/nodes/judge_broadcast_helpers.py`
- Modify: `werewolf_agent/runtime/context_public_summary.py`
- Modify: `werewolf_agent/runtime/directives/wolf.py`
- Modify: `werewolf_agent/agents/judge_dynamic_broadcasts.py`
- Modify: `werewolf_agent/ui/static/dashboard.js`
- Modify: `scripts/print_game_audit.py`
- Modify: `scripts/run_real_game_reports.py`
- Test: `tests/runtime/test_context_public_summary.py`
- Test: `tests/runtime/test_strategy_directives.py`
- Test: `tests/runtime/test_judge_flow.py`
- Test: `tests/ui/test_dashboard_enhanced.py`
- Test: `tests/test_game_audit.py`
- Test: `tests/scripts/test_run_real_game.py`

- [ ] **Step 1: Write failing consumer tests for V1 and V2**

Add paired fixtures:

- V1 payload with internal-unit aliases only;
- V2 payload with aliases, units, and display fields;
- V2 malformed payload with conflicting alias/units.

Assert public summaries, dashboard vote rows, CLI audit, and real-game reports show
`1.5` rather than `3`; wolf ranking uses unit values without dividing; and malformed
V2 is ignored/fail-closed. Add a dashboard helper contract:

```javascript
function voteDisplayTally(data, rulesetBaseVoteWeight) {
  if (data.vote_weight_format_version === 2) {
    return data.weighted_tally_display || data.tally_display || {};
  }
  if (Number.isInteger(rulesetBaseVoteWeight) && rulesetBaseVoteWeight > 0) {
    const units = data.weighted_tally || data.tally || {};
    return Object.fromEntries(
      Object.entries(units).map(([playerId, value]) => [
        playerId,
        value / rulesetBaseVoteWeight,
      ]),
    );
  }
  return null;
}
```

The dashboard must show a clear unsupported/legacy label when a V1 payload cannot
be decoded with a known ruleset base; it must not print raw internal units as votes.
Paired tests must pass `rulesetBaseVoteWeight=2` for the decodable V1 fixture and
omit it for the unsupported V1 fixture.

- [ ] **Step 2: Run the consumer tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\vote-consumers-red -o addopts='' tests/runtime/test_context_public_summary.py tests/runtime/test_strategy_directives.py tests/runtime/test_judge_flow.py tests/ui/test_dashboard_enhanced.py tests/test_game_audit.py tests/scripts/test_run_real_game.py -q
```

Expected: at least the V2 display and malformed-payload assertions fail.

- [ ] **Step 3: Replace direct payload reads with the shared decoder**

Examples:

```python
decoded = decode_vote_resolved_payload(payload, ruleset_base_vote_weight=base)
weighted_units = decoded.weighted_tally_units
weighted_display = decoded.weighted_tally_display
```

- ranking/strategy uses `weighted_units`;
- human-readable context and judge prompts use `weighted_display`;
- dashboard rows use V2 display maps and never raw aliases;
- `print_game_audit.py` and `run_real_game_reports.py` use the decoder and render
  unknown historical bases as unsupported;
- no consumer divides the display result again.

- [ ] **Step 4: Run focused consumer tests**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Search for remaining direct reads**

```powershell
rg -n "weighted_tally|vote_weights|sheriff_weight|judge_method.*vote_tally" werewolf_agent scripts -g "*.py" -g "*.js"
```

Expected: remaining direct reads are event construction, compatibility tests, or
explicitly documented unit consumers. If the search finds another production
presentation consumer, add its path and its focused test to this task before
implementation, then include both in the explicit commit command below.

- [ ] **Step 6: Commit Task 2**

```powershell
git add werewolf_agent/runtime/nodes/judge_broadcast_helpers.py werewolf_agent/runtime/context_public_summary.py werewolf_agent/runtime/directives/wolf.py werewolf_agent/agents/judge_dynamic_broadcasts.py werewolf_agent/ui/static/dashboard.js scripts/print_game_audit.py scripts/run_real_game_reports.py tests/runtime/test_context_public_summary.py tests/runtime/test_strategy_directives.py tests/runtime/test_judge_flow.py tests/ui/test_dashboard_enhanced.py tests/test_game_audit.py tests/scripts/test_run_real_game.py
git commit -m "fix: consume versioned vote display payloads"
```

---

### Task 3: Separate Discussion Summary From Public Speech

**Files:**
- Create: `werewolf_agent/agents/discussion_summary.py`
- Create: `tests/agents/test_discussion_summary.py`
- Modify: `werewolf_agent/agents/prompt_schemas.py:23-43`
- Modify: `werewolf_agent/agents/prompt_output.py:47-55`
- Modify: `werewolf_agent/agents/player.py:98-105`
- Modify: `werewolf_agent/runtime/nodes/runtime_state.py:72`
- Modify: `werewolf_agent/runtime/nodes/summary.py:65-135`
- Modify: `werewolf_agent/runtime/context.py:142,219-223`
- Modify: `config/models.yaml`
- Test: `tests/runtime/test_context.py`
- Test: `tests/runtime/test_summary_visibility.py`

- [ ] **Step 1: Write failing Schema and migration tests**

```python
def test_legacy_summary_string_upgrades_to_v2() -> None:
    summary = discussion_summary_for_player(
        {"discussion_positions": {"p01": "我怀疑p03"}},
        "p01",
    )
    assert summary.summary == "我怀疑p03"
    assert summary.suspected_players == []


def test_v2_version_schema_conflict_fails_closed() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {"p01": "legacy string"},
    }
    assert discussion_summary_for_player(state, "p01") is None
```

Also cover:

- unversioned mappings that fully validate as V2 and are upgraded in memory;
- unknown explicit versions;
- mixed valid/invalid player entries;
- the upgraded state writes `discussion_positions_version=2`;
- text projection is deterministic and contains no private fields.

- [ ] **Step 2: Run new tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\discussion-summary-red -o addopts='' tests/agents/test_discussion_summary.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement `DiscussionSummary` and compatibility accessors**

Use a strict Pydantic model:

```python
class DiscussionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    suspected_players: list[str] = Field(default_factory=list)
    trusted_players: list[str] = Field(default_factory=list)
    vote_target: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
```

Implement exact read precedence from the spec. Do not silently coerce a conflicting explicit V2 payload.

- [ ] **Step 4: Run Schema tests and verify they pass**

Run the Step 2 command.

- [ ] **Step 5: Write failing runtime tests for task separation**

Assert:

- `summarize_positions()` dispatches `TaskType.DISCUSSION_SUMMARY`;
- `_speech_quality_error()` is not invoked;
- no public speech event is emitted;
- model failure returns a deterministic V2 summary;
- deterministic fallback audit records
  `task=discussion_summary`, `outcome=deterministic_fallback`, and a safe
  `failure_code`;
- the prompt receives the legacy-compatible text projection;
- day speech and vote context consumers call `discussion_summary_for_player()`
  rather than reading the state mapping directly;
- runtime state declares the actual mapping shape.

- [ ] **Step 6: Run runtime tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\discussion-runtime-red -o addopts='' tests/runtime/test_context.py tests/runtime/test_summary_visibility.py -q
```

Expected: failures because the node still uses `TaskType.SPEECH` and strings.

- [ ] **Step 7: Add the task type and migrate the runtime node**

- add `DISCUSSION_SUMMARY = "discussion_summary"`;
- do not add it to public `_SPEECH_INTENT_TASKS`;
- give it only the narrow `DiscussionSummary` JSON Schema route; it must never
  request or parse the generic `PlayerAction` contract;
- add explicit model-profile task mappings where needed, inheriting the profile default otherwise;
- write `discussion_positions_version=2`;
- update downstream context access through the accessor.

After migration, run:

```powershell
rg -n "discussion_positions.*get|\\[.*discussion_positions|discussion_positions\\[" werewolf_agent -g "*.py"
```

Expected: direct indexing/`.get(player_id)` exists only inside
`discussion_summary.py` compatibility code. All prompt/vote/speech consumers use
the accessor.

- [ ] **Step 8: Run summary, prompt, and context regressions**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\discussion-runtime-green -o addopts='' tests/agents/test_discussion_summary.py tests/agents/test_prompt_mode_isolation.py tests/runtime/test_context.py tests/runtime/test_summary_visibility.py tests/runtime/test_vote_quality.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 3**

```powershell
git add werewolf_agent/agents/discussion_summary.py werewolf_agent/agents/prompt_schemas.py werewolf_agent/agents/prompt_output.py werewolf_agent/agents/player.py werewolf_agent/runtime/nodes/runtime_state.py werewolf_agent/runtime/nodes/summary.py werewolf_agent/runtime/context.py config/models.yaml tests/agents/test_discussion_summary.py tests/agents/test_prompt_mode_isolation.py tests/runtime/test_context.py tests/runtime/test_summary_visibility.py tests/runtime/test_vote_quality.py
git commit -m "refactor: separate discussion summary task contract"
```

---

### Task 4: Make Public Speech Repair Constraints Cumulative

**Files:**
- Create: `werewolf_agent/agents/speech_repair_requirements.py`
- Create: `tests/agents/test_speech_repair_requirements.py`
- Modify: `werewolf_agent/agents/semantic_repair_audit.py:219-239`
- Modify: `werewolf_agent/agents/player_action_flow.py:610-673`
- Modify: `werewolf_agent/agents/player_retry_hints.py`
- Test: `tests/agents/test_player_agent.py`
- Test: `tests/agents/test_player_retry.py`
- Test: `tests/evaluation/test_semantic_acceptance_invariants.py`

- [ ] **Step 1: Write failing cumulative-requirement tests**

```python
def test_requirements_accumulate_without_forgetting_prior_safety_constraints() -> None:
    requirements = SpeechRepairRequirements.empty()
    requirements = requirements.merge(
        missing_requirements=("evidence_basis",),
        required_target="p11",
    )
    requirements = requirements.merge(
        forbidden_claim_keys=("role_claim:p11:seer",),
        preserve_negations=True,
    )
    assert requirements.required_target == "p11"
    assert requirements.missing_requirements == ("evidence_basis",)
    assert requirements.forbidden_claim_keys == ("role_claim:p11:seer",)
    assert requirements.preserve_negations is True
```

Cover deterministic ordering, immutable collections, conflicting required targets, and allowlisted evidence refs.

- [ ] **Step 2: Run the new tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\speech-requirements-red -o addopts='' tests/agents/test_speech_repair_requirements.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement the immutable requirements model**

Use a frozen dataclass or frozen Pydantic model. Reject contradictory required targets instead of silently replacing them. Expose a compact, structured prompt projection rather than concatenated free-form history.

- [ ] **Step 4: Run the new tests**

Expected: all requirement-model tests pass.

- [ ] **Step 5: Write failing end-to-end repair tests**

Model the observed D3 chain:

1. attempt 1 lacks evidence basis;
2. attempt 2 introduces `unsupported_public_claim`;
3. attempt 3 must retain the target, remove the unsupported claim, and satisfy quality.

Assert all applicable stable reason codes are available together and no earlier requirement disappears.

Add terminal fallback assertions:

```python
assert action.trace.generated_by == "terminal_fallback"
assert action.trace.fallback_kind == "ordinary_speech"
assert action.target_id == verified_source_target
assert unsupported_role_claim not in action.speech
assert private_night_claim not in action.speech
assert action.speech_uses_only_public_facts is True
```

Use the existing semantic audit fields rather than adding the illustrative
`speech_uses_only_public_facts` attribute if no such public model field exists;
the test must still prove the property through the sanitizer/public-evidence
validator.

- [ ] **Step 6: Run repair tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\speech-repair-red -o addopts='' tests/agents/test_player_agent.py tests/agents/test_player_retry.py tests/evaluation/test_semantic_acceptance_invariants.py -q
```

Expected: the convergence and terminal fallback contract tests fail against the
sequential one-error repair.

- [ ] **Step 7: Integrate requirements into `player_action_flow`**

- quality and semantic validation return structured violations;
- merge violations into one requirements object;
- retry hints serialize only the structured safe projection;
- repair state never resets prior safety requirements;
- success requires both semantic and quality gates in the same attempt;
- max retry count remains unchanged.

- [ ] **Step 8: Run focused repair regressions**

Run the Step 6 command.

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 4**

```powershell
git add werewolf_agent/agents/speech_repair_requirements.py werewolf_agent/agents/semantic_repair_audit.py werewolf_agent/agents/player_action_flow.py werewolf_agent/agents/player_retry_hints.py tests/agents/test_speech_repair_requirements.py tests/agents/test_player_agent.py tests/agents/test_player_retry.py tests/evaluation/test_semantic_acceptance_invariants.py
git commit -m "fix: converge public speech repair constraints"
```

---

### Task 5: Add Safe Field-Level Failure Diagnostics

**Files:**
- Create: `werewolf_agent/agents/safe_failure_diagnostics.py`
- Create: `tests/agents/test_safe_failure_diagnostics.py`
- Modify: `werewolf_agent/agents/player_action_flow.py`
- Modify: `werewolf_agent/agents/trace_schemas.py`
- Modify: `werewolf_agent/agents/trace_builder.py`
- Modify: `werewolf_agent/model_gateway/generation_attempt_context.py`
- Test: `tests/agents/test_player_agent.py`
- Test: `tests/model_gateway/test_generation_attempt_context.py`

- [ ] **Step 1: Write failing allowlist/redaction tests**

Pass representative Pydantic errors and hostile raw inputs. Assert output contains only:

- field paths;
- stable error types;
- missing quality keys;
- normalized public claim keys;
- task contract/mode/attempt metadata.

Assert rejected text, correction hints, provider body, role secrets, URLs with credentials, and arbitrary validation input never appear.

- [ ] **Step 2: Run diagnostics tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\safe-diagnostics-red -o addopts='' tests/agents/test_safe_failure_diagnostics.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement an allowlist-only diagnostic model**

Never sanitize by subtracting a denylist from arbitrary input. Construct diagnostics from known-safe enumerations:

```python
class SafeFailureDiagnostic(BaseModel):
    decision_id: str
    opaque_request_id: str
    task_contract: str
    attempt: int
    structured_output_mode: str
    failure_stage: str
    reason_codes: tuple[str, ...]
    field_paths: tuple[str, ...] = ()
    error_types: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    unsupported_claim_keys: tuple[str, ...] = ()
    mode_downgraded: bool = False
    terminal_fallback: bool = False
```

`decision_id` and `opaque_request_id` must come from the existing identity
objects and remain opaque; the diagnostic builder must not derive them from raw
prompt/model content. `unsupported_claim_keys` accepts only normalized,
allowlisted semantic audit keys, never source sentences.

- [ ] **Step 4: Run the new tests**

Expected: all diagnostic unit tests pass.

- [ ] **Step 5: Write failing trace integration tests**

Assert schema-validation retries persist safe paths/types while raw rejected speech and correction hints remain empty/redacted, including terminal fallback.

- [ ] **Step 6: Integrate diagnostics into attempt and action traces**

Keep existing compatibility fields. Add the safe diagnostic as an optional structured field. Do not restore fields removed by commits `a5372c5`, `0231705`, or `2497069`.

- [ ] **Step 7: Run trace and privacy regressions**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\safe-diagnostics-green -o addopts='' tests/agents/test_safe_failure_diagnostics.py tests/agents/test_player_agent.py tests/agents/test_trace_builder.py tests/model_gateway/test_generation_attempt_context.py tests/runtime/test_exposure_audit.py -q
```

Expected: all selected tests pass and rejected text remains absent.

- [ ] **Step 8: Commit Task 5**

```powershell
git add werewolf_agent/agents/safe_failure_diagnostics.py werewolf_agent/agents/player_action_flow.py werewolf_agent/agents/trace_schemas.py werewolf_agent/agents/trace_builder.py werewolf_agent/model_gateway/generation_attempt_context.py tests/agents/test_safe_failure_diagnostics.py tests/agents/test_player_agent.py tests/agents/test_trace_builder.py tests/model_gateway/test_generation_attempt_context.py tests/runtime/test_exposure_audit.py
git commit -m "feat: record safe field-level failure diagnostics"
```

---

### Task 6: Introduce a Reflection-Specific Generation Contract

**Files:**
- Create: `werewolf_agent/agents/reflection_contract.py`
- Create: `tests/agents/test_reflection_contract.py`
- Modify: `werewolf_agent/runtime/agent_reflection_support.py:64-159`
- Modify: `werewolf_agent/runtime/nodes/summary.py:276-390`
- Modify: `werewolf_agent/runtime/reflection_transaction.py`
- Modify: `werewolf_agent/evaluation/acceptance_reflection_metrics.py`
- Test: `tests/runtime/test_agent_reflection_support.py`
- Test: `tests/runtime/test_reflection_transaction.py`
- Test: `tests/runtime/test_reflection_security_contract.py`
- Test: `tests/evaluation/test_report_acceptance_metrics.py`

- [ ] **Step 1: Write failing reflection-contract parser tests**

Reuse the existing `ReflectionDraft`/claim/lesson domain types from `werewolf_agent.memory.reflection_synthesis`; do not duplicate them.

Cover:

- valid reflection draft;
- missing nested field;
- wrong field type;
- forbidden extra field;
- three failed attempts producing one `not_generated` result;
- safe field paths/types in diagnostics;
- no raw draft in terminal output.

- [ ] **Step 2: Run new tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\reflection-contract-red -o addopts='' tests/agents/test_reflection_contract.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement the narrow reflection generation entry point**

The entry point should return a typed result, not a `PlayerAction`:

```python
@dataclass(frozen=True)
class ReflectionContractResult:
    draft: ReflectionDraft | None
    terminal_failure_code: str | None
    terminal_failure_stage: str | None
    diagnostics: tuple[SafeFailureDiagnostic, ...]
```

Reuse the shared model router and structured-mode retry machinery. Do not run public speech quality or public speech fallback.

- [ ] **Step 4: Run contract unit tests**

Expected: all reflection-contract tests pass.

- [ ] **Step 5: Write failing runtime compatibility tests**

Assert a one-player failure maps exactly to:

```python
assert entry["verification"]["status"] == "not_generated"
assert entry["transaction_state"] == "not_requested"
assert entry["failure_stage"] == "schema"
assert entry["failure_code"] == "schema_validation"
assert entry["entry_id"] is None
```

Assert `reflection_complete` retains existing fields and adds:

```python
assert payload["generated_count"] == 11
assert payload["not_generated_count"] == 1
assert payload["persisted_entry_count"] == 0
assert payload["status"] == "partial"
assert payload["persistence_complete"] is False
```

Add restored transaction tests for old payloads and new partial payloads.
Also cover:

- `no_valid_entries` restoration with every entry explicitly failed;
- `persistence_failed` restoration with verified entries not persisted;
- empty `entry_id`/claim/lesson identifiers rejected where the stage requires
  them;
- all count invariants for complete, partial, no-valid, and persistence-failed
  payloads.

- [ ] **Step 6: Run runtime/evaluation tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\reflection-runtime-red -o addopts='' tests/runtime/test_agent_reflection_support.py tests/runtime/test_reflection_transaction.py tests/runtime/test_reflection_security_contract.py tests/evaluation/test_report_acceptance_metrics.py -q
```

Expected: failures for missing count fields and the old `PlayerAction` generation path.

- [ ] **Step 7: Migrate `_agent_reflection()` and event construction**

- call the reflection contract directly;
- keep `reflection_verification` as the adapter return surface;
- map terminal failure to the existing transaction stage/failure fields;
- preserve event type and legacy fields;
- add count invariants;
- log generation counts as `[复盘生成] 处理N位：有效X，未生成Y`;
- leave persistence counts to the runner’s actual persistence audit.

- [ ] **Step 8: Run reflection regressions**

Run the Step 6 command plus:

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\reflection-memory-green -o addopts='' tests/memory/test_reflection_synthesis.py tests/runtime/test_reflection_security_contract.py tests/evaluation/test_acceptance_game_projection.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 6**

```powershell
git add werewolf_agent/agents/reflection_contract.py werewolf_agent/runtime/agent_reflection_support.py werewolf_agent/runtime/nodes/summary.py werewolf_agent/runtime/reflection_transaction.py werewolf_agent/evaluation/acceptance_reflection_metrics.py tests/agents/test_reflection_contract.py tests/runtime/test_agent_reflection_support.py tests/runtime/test_reflection_transaction.py tests/runtime/test_reflection_security_contract.py tests/evaluation/test_report_acceptance_metrics.py
git commit -m "refactor: add reflection-specific output contract"
```

---

### Task 7: Mark Truncated Event-Derived Metrics Unsupported

**Files:**
- Modify: `scripts/run_real_game.py:160-300`
- Modify: `scripts/run_real_game.py:870-1035`
- Modify: `scripts/run_real_game.py:1155-1170`
- Modify: `werewolf_agent/evaluation/acceptance_audit.py`
- Modify: `werewolf_agent/evaluation/game_projection.py`
- Test: `tests/scripts/test_run_real_game.py`
- Test: `tests/evaluation/test_acceptance_game_projection.py`
- Test: `tests/evaluation/test_semantic_acceptance_invariants.py`
- Test: `tests/evaluation/test_report_acceptance_metrics.py`

- [ ] **Step 1: Write failing truncated-projection metric tests**

For `events=[]` plus `_acceptance_projection_unsupported_reason=json_item_limit_exceeded`, assert:

```python
assert quality["fallback_rate"] is None
assert quality["fallback_count"] is None
assert quality["fallback_metrics_supported"] is False
assert quality["fallback_metrics_unsupported_reason"] == "json_item_limit_exceeded"
```

Add a control case where complete events contain no fallback and the supported rate is exactly `0.0`.
Add deterministic cases for:

- a partial export marker with some surviving events;
- stdout diagnostics containing terminal fallback while structured events are
  truncated;
- every event-derived fallback count/rate returned by
  `compute_game_quality_score`, including speech, reflection, wolf plan,
  decision execution, semantic repair, possible-world, and power-role metrics;
- report serialization and console rendering of `None`/unsupported values.

The stdout count remains a separately named diagnostic and must never be copied
into the official structured denominator.

- [ ] **Step 2: Run focused tests and verify failure**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\truncated-metrics-red -o addopts='' tests/scripts/test_run_real_game.py tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_semantic_acceptance_invariants.py -q
```

Expected: truncated case currently reports zero and fails.

- [ ] **Step 3: Add one support-state helper**

Centralize the decision:

```python
def event_metric_support(source: Mapping[str, Any]) -> MetricSupport:
    if source.get("_acceptance_projection_supported") is False:
        return MetricSupport(False, source.get(
            "_acceptance_projection_unsupported_reason", "projection_unsupported"
        ))
    events = source.get("events")
    if not isinstance(events, list) or not events:
        return MetricSupport(False, "missing_events")
    return MetricSupport(True, None)
```

Every event-derived metric must consult this before calculating a count/rate.
Add a table-driven test that enumerates the exported event-derived metric keys
and asserts each unsupported count/rate is `None` with its matching support flag
false. This prevents newly added metrics from silently defaulting to zero.
Add a separate unmarked `events=[]` fixture and assert
`fallback_metrics_unsupported_reason == "missing_events"`; this must not depend
on a pre-existing projection-support marker.

- [ ] **Step 4: Make logging nullable-safe**

Replace `%.3f` assumptions. Example terminal log:

```text
fallback_rate=unsupported(json_item_limit_exceeded)
```

Low-quality classification must only compare rates when support is true.

- [ ] **Step 5: Run metric and report regressions**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\truncated-metrics-green -o addopts='' tests/scripts/test_run_real_game.py tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_game_balance_batch.py -q
```

Expected: all selected tests pass; supported true-zero remains `0.0`.

- [ ] **Step 6: Commit Task 7**

```powershell
git add scripts/run_real_game.py werewolf_agent/evaluation/acceptance_audit.py werewolf_agent/evaluation/game_projection.py tests/scripts/test_run_real_game.py tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_game_balance_batch.py
git commit -m "fix: mark truncated fallback metrics unsupported"
```

---

### Task 8: Cross-Batch Regression and Fresh-Game Acceptance

**Files:**
- Create: `tests/integration/test_task_contract_acceptance.py`
- Modify only the exact production/test files implicated if this integration
  test exposes a defect.
- Verify: all files changed in Tasks 1–7.

- [ ] **Step 1: Write deterministic cross-contract acceptance tests**

Build one controlled runtime fixture with deterministic agents/providers that
exercises all required branches without external network access:

1. ordinary and sheriff votes create a V2 1/1.5 display payload;
2. a discussion summary Schema failure uses deterministic summary fallback and
   is not labeled speech;
3. a public speech fails quality, then unsupported claim repair, then reaches a
   public-facts-only terminal fallback;
4. a reflection fails Schema three times and produces one explicit failed
   transaction;
5. the completed event set reports real fallback values;
6. the same game projected through `json_item_limit_exceeded` reports
   unsupported/null event-derived metrics.

Do not rely on random roles or live providers for these branches.

- [ ] **Step 2: Run the deterministic acceptance test**

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\task-contract-acceptance -o addopts='' tests/integration/test_task_contract_acceptance.py -q
```

Expected: all deterministic cross-contract scenarios pass.

- [ ] **Step 3: Run focused cross-domain regression batches**

Run in separate auditable batches:

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\final-agent -o addopts='' tests/agents/test_player_agent.py tests/agents/test_player_retry.py tests/agents/test_prompt_mode_isolation.py tests/agents/test_discussion_summary.py tests/agents/test_speech_repair_requirements.py tests/agents/test_safe_failure_diagnostics.py tests/agents/test_reflection_contract.py -q
```

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\final-runtime -o addopts='' tests/runtime/test_vote_display.py tests/runtime/test_vote_flow.py tests/runtime/test_judge_flow.py tests/runtime/test_context.py tests/runtime/test_summary_visibility.py tests/runtime/test_reflection_transaction.py tests/runtime/test_reflection_security_contract.py -q
```

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp\final-evaluation -o addopts='' tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_semantic_acceptance_invariants.py tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_game_balance_batch.py tests/scripts/test_run_real_game.py -q
```

Expected: each command completes with a final all-passed summary. A command that times out before its summary remains unverified.

- [ ] **Step 4: Run static and repository hygiene checks**

```powershell
python -m compileall -q werewolf_agent scripts tests
git diff --check
```

Run targeted Ruff checks on changed Python files using the repository’s configured Ruff command. Expected: no new errors in changed files.

- [ ] **Step 5: Review generated/untracked artifacts before cleanup**

```powershell
git status --short
```

Do not delete pre-existing user artifacts. Remove only newly generated, workspace-contained test/cache artifacts after resolving and verifying their absolute paths.

- [ ] **Step 6: Run a fresh real game**

Use a new fixed seed and explicit artifact directory. Do not reuse
`g_3334463270` as post-fix evidence.

```powershell
New-Item -ItemType Directory -Force -Path artifacts\task_contract_acceptance\seed-725001 | Out-Null
$env:LANGCHAIN_TRACING_V2='false'
$env:LANGSMITH_TRACING='false'
python scripts/run_real_game.py --seed 725001 --game-id task_contract_725001 --max-steps 500 --delay -1 --output-dir artifacts\task_contract_acceptance\seed-725001 2>&1 | Tee-Object -FilePath artifacts\task_contract_acceptance\seed-725001\runner.stdout.log
```

Expected artifacts:

- `artifacts\task_contract_acceptance\seed-725001\runner.stdout.log`
- `artifacts\task_contract_acceptance\seed-725001\game_task_contract_725001.json`

The command must exit successfully and print a terminal `GAME_COMPLETE` summary.
If it times out or lacks a terminal summary, record it as unverified.

Capture:

- game ID and elapsed time;
- every provider retry;
- every terminal fallback by task contract;
- discussion-summary deterministic fallbacks;
- reflection generated/not-generated/persisted counts;
- vote messages and V2 payloads;
- metric support flags.

- [ ] **Step 7: Verify production acceptance**

Confirm from the fresh game:

- ordinary votes display as `1`, sheriff votes as `1.5`;
- rule exile/tie outcome agrees with internal units;
- no discussion summary warning is labeled `TaskType.SPEECH`;
- speech failures expose safe structured requirements without rejected text;
- reflection failure, if any, includes safe field paths/types and exact counts;
- truncated exports show unsupported/null metrics, not zero;
- stdout and structured report do not contradict each other.

- [ ] **Step 8: Record branch-level verification evidence**

Create or update:

```text
docs/audits/2026-07-25-task-contract-fallback-vote-display-verification.md
```

Record exact test pass counts, commit IDs, fresh game ID, artifact paths, every
observed fallback/retry, and whether each production criterion was observed or
only deterministically verified.

- [ ] **Step 9: Commit deterministic acceptance and verification record**

Stage the deterministic integration test and verification document, plus only
any exact integration corrections shown by `git status --short`:

```powershell
git add tests/integration/test_task_contract_acceptance.py docs/audits/2026-07-25-task-contract-fallback-vote-display-verification.md
git commit -m "test: verify task contracts and vote display"
```

If integration corrections exist, commit those exact files separately before
the verification-record commit; never use a placeholder, wildcard, or broad
directory staging command.

---

## Completion Gate

Do not claim completion until:

- every task commit exists or is explicitly documented as a no-op;
- all focused regression commands completed with final summaries;
- `compileall`, targeted Ruff, and `git diff --check` are clean for changed files;
- a fresh game was inspected;
- internal vote units and displayed vote values are both verified;
- metric support states are honest;
- no rejected private/model text was reintroduced into traces.
