# Game Balance Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the latest live-game werewolf-side skew by fixing sheriff-vote contract pollution, seer credibility closure, weak wolf-plan kill gating, good-side friendly-fire guidance, and balance guardrails.

**Architecture:** Keep RuleEngine and V1 rules unchanged. Make surgical changes in prompt contracts, runtime strategy gates, cognition observability, and pure saved-log evaluation metrics so each failure mode is independently testable.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, LangGraph runtime nodes, existing `PlayerAgent`, `ActionContract`, cognition, directive, and evaluation modules.

---

## File Structure

- Modify: `werewolf_agent/evaluation/balance_audit.py`
  - Add saved-log metrics for sheriff control, sheriff-vote fallback, hunter
    friendly fire, weak planned kills, and power-role fallback.
- Modify: `tests/evaluation/test_game_balance_batch.py`
  - Add metric and warning tests using small saved-style game dicts.
- Modify: `werewolf_agent/agents/action_contract.py`
  - Ensure sheriff-vote contracts do not include exile-vote audit fields.
- Modify: `werewolf_agent/agents/prompt_builder.py`
  - Render vote-audit field guidance only for exile voting; add targeted
    sheriff-vote retry hints; keep seer credibility section tested.
- Modify: `werewolf_agent/agents/player.py`
  - Preserve parse/retry behavior, but expose targeted sheriff-vote schema
    correction when extra vote-audit fields caused failure.
- Modify: `tests/agents/test_prompt_builder.py`
  - Add prompt contract tests for sheriff-vote field isolation.
- Modify: `tests/agents/test_player_agent.py`
  - Add parser/retry regression tests using recent sheriff-vote raw text.
- Modify: `werewolf_agent/runtime/context.py`
  - Keep seer credibility summary generation and expose enough data for audit.
- Modify: `werewolf_agent/runtime/nodes/day.py`
  - Emit moderator-only `seer_credibility_audit` when public seer lines exist
    during day discussion / vote context construction.
- Modify: `tests/runtime/test_context.py`
  - Add runtime context tests for non-empty seer credibility.
- Modify: `tests/runtime/test_day_discussion.py`
  - Add audit-event test for seer credibility events.
- Modify: `werewolf_agent/runtime/nodes/_shared.py`
  - Harden `_planned_wolf_kill` so weak plans cannot automatically kill without
    matching evidence and legal live target.
- Modify: `werewolf_agent/runtime/wolf_strategy.py`
  - Preserve evidence-quality computation and clarify weak-plan output.
- Modify: `tests/runtime/test_wolf_strategy.py`
  - Add weak-plan kill gating tests.
- Modify: `werewolf_agent/runtime/directives/hunter.py`
  - Strengthen multi-evidence and no-shot guidance.
- Modify: `werewolf_agent/runtime/directives/idiot.py`
  - Strengthen pre-reveal anti-herd guidance.
- Modify: `werewolf_agent/runtime/directives/witch.py`
  - Keep public-source poison rule and add late-game pressure wording only when
    evidence candidates exist.
- Modify: `tests/runtime/test_strategy_directives.py`
  - Add focused directive tests for hunter / idiot / witch repair behavior.

---

### Task 1: Balance Audit Guardrails

**Files:**
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Modify: `tests/evaluation/test_game_balance_batch.py`

- [ ] **Step 1: Write failing metric tests**

Add tests that create saved-style game dicts with:

```python
def test_balance_audit_flags_sheriff_werewolf_and_d1_seer_exile():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "werewolf"},
            "p02": {"role": "seer"},
        },
        "events": [
            {"type": "sheriff_elected", "payload": {"sheriff_id": "p01"}},
            {"type": "vote_resolved", "payload": {"day_number": 1, "exiled": "p02", "votes": [
                {"voter": "p01", "target": "p02", "reason": "x"},
            ]}},
        ],
        "deaths": [{"player_id": "p02", "reason": "exile", "resolution_batch": "day_1_vote"}],
    }

    audit = compute_balance_audit([game])

    assert audit["sheriff_werewolf_rate"] == 1.0
    assert audit["d1_seer_exile_rate"] == 1.0
```

Add companion tests for:

- `sheriff_vote_fallback_rate`
- `hunter_friendly_fire_rate`
- `weak_plan_kill_rate`
- warnings: `sheriff_werewolf_rate_high`, `sheriff_vote_fallback_high`,
  `hunter_friendly_fire_high`, `weak_plan_kill_high`

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m pytest tests/evaluation/test_game_balance_batch.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL because the new keys do not exist.

- [ ] **Step 3: Implement pure saved-log metrics**

In `compute_balance_audit`, add helper functions:

```python
def _sheriff_werewolf_rate(games: list[dict[str, Any]]) -> float: ...
def _sheriff_vote_fallback_rate(games: list[dict[str, Any]]) -> float: ...
def _hunter_friendly_fire_rate(games: list[dict[str, Any]]) -> float: ...
def _weak_plan_kill_rate(games: list[dict[str, Any]]) -> float: ...
def _power_role_fallback_rate(games: list[dict[str, Any]]) -> float: ...
```

Implementation notes:

- Build `roles = {player_id: role}` from `game["players"]`.
- Sheriff role comes from `sheriff_elected.payload.sheriff_id`.
- Sheriff-vote fallback comes from action traces whose task/phase is
  `sheriff_vote` and whose trace has `fallback_reason`, `parse_error`, or
  `structured_failure_reason`.
- Hunter friendly fire counts `death.reason == "hunter_shot"` whose target role
  is not `werewolf`.
- Weak plan kill rate counts `wolf_kill_selected` events sourced from
  `wolf_team_plan` where same-night plan quality is `weak` or `none`.

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests/evaluation/test_game_balance_batch.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/evaluation/balance_audit.py tests/evaluation/test_game_balance_batch.py
git commit -m "test(eval): add balance guardrails for recent live-game skew"
```

---

### Task 2: Sheriff Vote Contract Isolation

**Files:**
- Modify: `werewolf_agent/agents/action_contract.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `tests/agents/test_prompt_builder.py`
- Modify: `tests/agents/test_player_agent.py`

- [ ] **Step 1: Write failing ActionContract test**

Add a test that builds a sheriff-vote contract and asserts vote-audit fields are
absent:

```python
def test_sheriff_vote_contract_omits_exile_vote_audit_fields():
    from werewolf_agent.agents.action_contract import ActionContract
    from werewolf_agent.agents.schemas import ActionType, OutputMode, TaskType

    contract = ActionContract.build(
        output_mode=OutputMode.FULL_ACTION,
        task_type=TaskType.SHERIFF_VOTE,
        legal_actions=[ActionType.SHERIFF_VOTE],
        legal_targets=["p01", "p02"],
    )

    props = contract.json_schema["properties"]
    for field in (
        "seer_stance",
        "vote_basis",
        "standing_with_seer",
        "suspect_reason",
        "not_voting_reason",
        "private_reason",
    ):
        assert field not in props
        assert field not in contract.required_fields
```

- [ ] **Step 2: Write failing prompt test**

Build a `PlayerPromptBuilder` context with `task_type=TaskType.SHERIFF_VOTE`
and assert the rendered prompt does not include:

- `seer_stance`
- `vote_basis`
- `private_reason`
- `投票时必须先在心里完成判断`

It should still include legal targets and `sheriff_vote`.

- [ ] **Step 3: Write parser/retry regression test**

Use a recent raw sheriff-vote shape with extra fields:

```python
raw = {
    "action_type": "sheriff_vote",
    "target_id": "p04",
    "speech": "",
    "reason": "p04验人动机不自洽",
    "confidence": 0.6,
    "seer_stance": "undecided",
    "vote_basis": "speech_logic",
    "private_reason": "extra",
}
```

Assert the first validation error is classified as sheriff-vote extra-field
pollution and the retry hint says to remove exile-vote audit fields.

- [ ] **Step 4: Run failing tests**

Run:

```powershell
python -m pytest tests/agents/test_prompt_builder.py tests/agents/test_player_agent.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL on new assertions.

- [ ] **Step 5: Implement contract split**

In `ActionContract.build` and helper schema functions:

- Only `TaskType.VOTE` gets `vote_audit_properties()`.
- `TaskType.SHERIFF_VOTE` remains the base five-field action schema.
- TARGET_CHOICE mode should also add vote-audit fields only for `TaskType.VOTE`.

- [ ] **Step 6: Implement prompt gating**

In `PlayerPromptBuilder._build_phase_context`:

- Replace `is_vote_context = ctx.task_type == TaskType.VOTE` with explicit
  exile-vote naming such as `is_exile_vote_context`.
- Add a separate small branch for `TaskType.SHERIFF_VOTE`:

```python
lines.append("警长投票只选警长候选人，不是放逐投票。")
lines.append("输出只包含 action_type、target_id、speech、reason、confidence。")
```

Do not render vote-audit field guidance for sheriff voting.

- [ ] **Step 7: Implement targeted retry hint**

In `PlayerPromptBuilder._build_retry_hint` or the retry data construction path
in `PlayerAgent`, detect schema errors that mention `sheriff_vote.` plus any
vote-audit field. Append:

```text
警长投票不是放逐投票，请删除 seer_stance、vote_basis、standing_with_seer、
suspect_reason、not_voting_reason、private_reason，只保留 action_type、target_id、
speech、reason、confidence。
```

- [ ] **Step 8: Verify**

Run:

```powershell
python -m pytest tests/agents/test_prompt_builder.py tests/agents/test_player_agent.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add werewolf_agent/agents/action_contract.py werewolf_agent/agents/prompt_builder.py werewolf_agent/agents/player.py tests/agents/test_prompt_builder.py tests/agents/test_player_agent.py
git commit -m "fix(agents): isolate sheriff vote contract from exile vote audit fields"
```

---

### Task 3: Seer Credibility Observability And Prompt Closure

**Files:**
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/runtime/nodes/day.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `tests/runtime/test_context.py`
- Modify: `tests/runtime/test_day_discussion.py`
- Modify: `tests/agents/test_seer_credibility_section.py`

- [ ] **Step 1: Write failing context test**

Create a game state with two public seer claimants:

```python
events = [
    GameEvent(type="speech", payload={"speaker": "p01", "text": "我是预言家，N1验p03狼人"}),
    GameEvent(type="speech", payload={"speaker": "p02", "text": "我是预言家，N1验p04好人"}),
]
```

Build agent context for a live player and assert:

```python
assert ctx.seer_credibility["seer_lines"]
assert {line["claimant"] for line in ctx.seer_credibility["seer_lines"]} == {"p01", "p02"}
```

- [ ] **Step 2: Write failing audit-event test**

Run the relevant day-discussion or day-vote node with public seer claims and
assert a moderator-only event exists:

```python
events = [e for e in result["game_state"].events if e.type == "seer_credibility_audit"]
assert events
assert events[-1].payload["visibility"] == "moderator_only"
assert "hidden_role" not in json.dumps(events[-1].payload)
```

- [ ] **Step 3: Run failing tests**

Run:

```powershell
python -m pytest tests/runtime/test_context.py tests/runtime/test_day_discussion.py tests/agents/test_seer_credibility_section.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL where audit event is missing or summary is empty.

- [ ] **Step 4: Ensure context computes seer credibility**

In `runtime/context.py`, keep using `SeerClaimCredibilityEngine` with visible
facts only. If the summary is empty in the test, fix visible fact extraction so
public role/check claims are passed in chronological order.

- [ ] **Step 5: Emit compact audit events**

In `runtime/nodes/day.py`, after context construction for day discussion or
vote phases, append:

```python
GameEvent(
    type="seer_credibility_audit",
    payload={
        "day_number": gs.day_number,
        "visibility": "moderator_only",
        "seer_lines": ctx.seer_credibility.get("seer_lines", []),
    },
)
```

Only emit when `seer_lines` is non-empty. Do not include actual roles.

- [ ] **Step 6: Strengthen prompt-section tests**

In `tests/agents/test_seer_credibility_section.py`, assert rendered lines
include status and evidence but cap at three lines and strip long/noisy text.

- [ ] **Step 7: Verify**

Run:

```powershell
python -m pytest tests/cognition/test_claim_credibility.py tests/runtime/test_context.py tests/runtime/test_day_discussion.py tests/agents/test_seer_credibility_section.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add werewolf_agent/runtime/context.py werewolf_agent/runtime/nodes/day.py werewolf_agent/agents/prompt_builder.py tests/runtime/test_context.py tests/runtime/test_day_discussion.py tests/agents/test_seer_credibility_section.py
git commit -m "feat(cognition): audit and render seer line credibility in live context"
```

---

### Task 4: Weak Wolf Plan Kill Gating

**Files:**
- Modify: `werewolf_agent/runtime/nodes/_shared.py`
- Modify: `werewolf_agent/runtime/wolf_strategy.py`
- Modify: `tests/runtime/test_wolf_strategy.py`
- Modify: `tests/runtime/test_wolf_flow.py`

- [ ] **Step 1: Write failing weak-plan kill test**

Add a test where plan quality is weak, primary target is alive, but
`evidence_from_discussion` has no matching target:

```python
def test_planned_wolf_kill_rejects_weak_plan_without_target_evidence():
    from werewolf_agent.runtime.nodes._shared import _planned_wolf_kill

    state = _make_state_with_plan(
        target="p05",
        evidence_quality="weak",
        evidence_from_discussion=[{"wolf_id": "p01", "target": "p06"}],
    )

    assert _planned_wolf_kill(state) is None
```

- [ ] **Step 2: Write failing stale target test**

Create a plan whose primary target is dead or no longer legal. Assert
`_planned_wolf_kill(state) is None`, even when `evidence_quality="strong"`.

- [ ] **Step 3: Run failing tests**

Run:

```powershell
python -m pytest tests/runtime/test_wolf_strategy.py tests/runtime/test_wolf_flow.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL on new cases if weak/stale plans still produce kills.

- [ ] **Step 4: Harden `_planned_wolf_kill`**

In `_planned_wolf_kill`:

- Return `None` when `evidence_quality == "none"`.
- For `weak`, require at least one evidence item whose `target` equals selected
  target.
- For `strong`, still verify target is alive and legal.
- Do not silently fall back from invalid primary to backup unless backup also
  has matching evidence and is legal.

- [ ] **Step 5: Clarify weak-plan output**

In `wolf_strategy.build_wolf_team_plan_from_discussion`, keep `weak` plans
visible for audit but ensure comments/tests state they are advisory unless
runtime validation accepts the target.

- [ ] **Step 6: Verify**

Run:

```powershell
python -m pytest tests/runtime/test_wolf_strategy.py tests/runtime/test_wolf_flow.py tests/integration/test_wolf_team_plan_e2e.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/runtime/nodes/_shared.py werewolf_agent/runtime/wolf_strategy.py tests/runtime/test_wolf_strategy.py tests/runtime/test_wolf_flow.py
git commit -m "fix(runtime): block automatic wolf kills from weak stale plans"
```

---

### Task 5: Good-Side Friendly-Fire Hardening

**Files:**
- Modify: `werewolf_agent/runtime/directives/hunter.py`
- Modify: `werewolf_agent/runtime/directives/idiot.py`
- Modify: `werewolf_agent/runtime/directives/witch.py`
- Modify: `tests/runtime/test_strategy_directives.py`

- [ ] **Step 1: Write hunter friendly-fire tests**

Add tests that ensure hunter directives say:

- at least two independent public evidence sources are required before shooting;
- if the main suspect may be witch / hunter / idiot and evidence is thin, prefer
  `no_action`;
- exile last words must not say "must shoot" without naming a target.

- [ ] **Step 2: Write idiot anti-herd test**

Before reveal, assert idiot directive contains:

- avoid becoming an easy exile;
- do not blindly follow sheriff / near-unanimous push;
- revealing survives exile but loses vote, so it is not free value.

- [ ] **Step 3: Write witch late-game poison test**

Using existing poison candidate helpers, assert:

- no evidence candidates -> directive recommends no poison / hold poison;
- evidence candidate exists in low-player-count state -> directive says poison
  can be considered and must cite public source.

- [ ] **Step 4: Run failing tests**

Run:

```powershell
python -m pytest tests/runtime/test_strategy_directives.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: FAIL on missing directive text.

- [ ] **Step 5: Implement directive text only**

Make minimal text changes in the three directive modules. Do not alter
RuleEngine skill legality.

- [ ] **Step 6: Verify**

Run:

```powershell
python -m pytest tests/runtime/test_strategy_directives.py tests/runtime/test_hunter_flow.py tests/runtime/test_idiot_reveal.py tests/runtime/test_witch_flow.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add werewolf_agent/runtime/directives/hunter.py werewolf_agent/runtime/directives/idiot.py werewolf_agent/runtime/directives/witch.py tests/runtime/test_strategy_directives.py
git commit -m "fix(strategy): reduce good-side power role friendly fire"
```

---

### Task 6: Saved-Log Regression Report And Final Verification

**Files:**
- Create: `scripts/analyze_recent_balance.py`
- Modify: `tests/scripts/test_recent_balance_analysis.py` if a scripts test
  package pattern already exists; otherwise add coverage to
  `tests/evaluation/test_game_balance_batch.py`.

- [ ] **Step 1: Write failing script test**

Add a test that calls a pure function:

```python
def test_recent_balance_report_includes_new_guardrails(tmp_path):
    from scripts.analyze_recent_balance import build_recent_balance_report

    report = build_recent_balance_report(["game_g_3224969250.json"])

    assert "sheriff_vote_fallback_rate" in report
    assert "weak_plan_kill_rate" in report
```

- [ ] **Step 2: Implement script**

Create `scripts/analyze_recent_balance.py` with:

```python
def build_recent_balance_report(paths: Sequence[str | Path]) -> dict[str, Any]:
    games = load_game_logs(paths)
    return compute_balance_audit(games)
```

Add CLI output as compact JSON. Keep it pure; no LLM calls.

- [ ] **Step 3: Verify script against recent seven games**

Run:

```powershell
python scripts/analyze_recent_balance.py game_g_850282163.json game_g_1416328333.json game_g_415824166.json game_g_1600154180.json game_g_934007012.json game_g_3819861640.json game_g_3224969250.json
```

Expected: prints metrics and warnings reproducing the current skew.

- [ ] **Step 4: Run focused suites**

Run:

```powershell
python -m pytest tests/evaluation/test_game_balance_batch.py tests/agents/test_prompt_builder.py tests/agents/test_player_agent.py tests/cognition/test_claim_credibility.py tests/runtime/test_context.py tests/runtime/test_day_discussion.py tests/runtime/test_wolf_strategy.py tests/runtime/test_wolf_flow.py tests/runtime/test_strategy_directives.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 5: Run rule and runtime smoke suites**

Run:

```powershell
python -m pytest tests/rules/test_rule_engine_v1.py tests/runtime/test_vote_flow.py tests/runtime/test_sheriff_flow.py tests/runtime/test_game_runner.py -q --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: PASS.

- [ ] **Step 6: Compile and diff check**

Run:

```powershell
python -m compileall -q werewolf_agent tests scripts
git diff --check
```

Expected: both exit 0.

- [ ] **Step 7: Optional live batch**

If model credentials and time are available, run a 10-game smoke batch followed
by a 20-game balance batch. Compare:

- wolf win rate
- D1 seer exile rate
- sheriff-vote fallback rate
- weak planned kill rate
- hunter friendly-fire rate

Expected live target:

- `sheriff_vote_fallback_rate < 0.05`
- `weak_plan_kill_rate == 0`
- wolf win rate review threshold: investigate if still above 70%

- [ ] **Step 8: Commit**

```powershell
git add scripts/analyze_recent_balance.py tests/evaluation/test_game_balance_batch.py
git commit -m "chore(eval): add recent balance regression report"
```

---

## Final Delivery Checklist

- [ ] New balance metrics reproduce the seven-game failure baseline.
- [ ] Sheriff-vote contract and prompt no longer request exile-vote audit fields.
- [ ] Seer credibility appears in prompt tests and moderator-only audit events.
- [ ] Weak/no-evidence wolf plans cannot automatically choose night kills.
- [ ] Hunter, idiot, and witch directives include evidence thresholds.
- [ ] Focused tests pass with `--basetemp E:\NLP\agent\wofkill\.pytest_tmp`.
- [ ] `find_boundary_violations` remains zero on recent saved logs.
- [ ] Live-batch evaluation is run or explicitly marked blocked by credentials/time.

