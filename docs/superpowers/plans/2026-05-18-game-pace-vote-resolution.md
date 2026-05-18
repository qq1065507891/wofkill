# Game Pace Vote Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix overly long 12-player Werewolf games by making day votes produce real eliminations, preventing stale votes, and adding deterministic anti-stall protections.

**Architecture:** Keep RuleEngine as the deterministic authority. Runtime graph nodes own per-phase lifecycle state, agent adapters enforce legal action contracts, and evaluation metrics verify that games progress at a human-like pace without information leakage.

**Tech Stack:** Python, LangGraph runtime graph, pytest, existing `GameState` / `RuleEngine` models, JSON replay outputs.

---

## Scope

This plan targets the real symptom seen in `game_g_789.json`: no `exile` deaths, repeated `exiled: p01`, and many `second_tie_no_exile` events. It does not rebalance roles or change the public 12-player pre-witch-hunter-idiot-hybrid ruleset. The fix lives in runtime state lifecycle, agent decision constraints, and evaluation guardrails.

## File Structure

- Modify: `werewolf_agent/runtime/graph.py`
  - Reset day vote state at the start of each new vote.
  - Ensure revotes route back through `day_vote` and collect fresh votes from PK candidates.
  - Route no-exile outcomes cleanly without carrying stale vote state into the next day.
- Modify: `werewolf_agent/runtime/agent_adapter.py`
  - Make day vote mandatory when `allow_abstain: false`.
  - Add task hints for vote pressure and legal target selection.
  - Return structured vote fallback metadata.
- Modify: `werewolf_agent/agents/player.py`
  - Improve fallback behavior for mandatory target actions.
  - Prefer deterministic context-aware fallback hooks where supplied.
- Modify: `werewolf_agent/engine/rule_engine.py`
  - Preserve official tie rules by default.
  - Add optional anti-stall vote resolution policy controlled by ruleset/runtime config, not hard-coded identity knowledge.
- Modify: `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`
  - Add explicit simulation-only pace policy defaults.
  - Keep base rules unchanged.
- Modify: `werewolf_agent/evaluation/metrics.py`
  - Add game pace metrics: day exile rate, stale vote reuse, consecutive no-exile days, median finish night.
- Modify: `scripts/run_real_game.py`
  - Emit full action trace logs without truncating speeches, vote reasons, or night decisions.
- Modify: `tests/runtime/test_runtime.py`
  - Cover vote lifecycle and runtime routing.
- Modify: `tests/rules/test_rule_engine_v1.py`
  - Cover anti-stall policy while preserving existing tie behavior.
- Modify: `tests/agents/test_agents.py`
  - Cover mandatory vote action constraints and fallback.
- Modify: `tests/evaluation/test_evaluation.py`
  - Cover pace metrics.
- Optional modify: `scripts/run_real_game.py`
  - Emit concise pace report after real game runs.

---

### Task 1: Prevent Stale Vote Reuse Across Days

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing test for fresh day votes**

Add a test that enters `day_vote` with old `exile_votes` from a previous day and verifies the node calls agents again instead of returning the old votes.

```python
def test_day_vote_ignores_stale_votes_when_day_changes() -> None:
    from werewolf_agent.runtime.graph import day_vote

    engine = _new_engine()
    gs = _sample_game_state(day_number=2)
    registry = _fake_registry(votes={"p02": "p05", "p03": "p05"})

    result = day_vote({
        "game_state": gs,
        "engine": engine,
        "agent_registry": registry,
        "exile_votes": {"p01": "p01"},
        "exile_vote_day": 1,
        "revote": False,
    })

    assert result["exile_vote_day"] == 2
    assert result["exile_votes"] != {"p01": "p01"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runtime/test_runtime.py::test_day_vote_ignores_stale_votes_when_day_changes -v`

Expected: FAIL because `day_vote` currently reuses `existing_votes`.

- [ ] **Step 3: Implement vote lifecycle key**

In `day_vote`, treat `exile_votes` as reusable only if `exile_vote_day == gs.day_number` and `exile_vote_revote == state.get("revote", False)`.

```python
same_vote_window = (
    state.get("exile_vote_day") == gs.day_number
    and state.get("exile_vote_revote") == state.get("revote", False)
)
existing_votes = state.get("exile_votes", {}) if same_vote_window else {}
```

Return:

```python
return {
    "exile_votes": votes,
    "exile_vote_day": gs.day_number,
    "exile_vote_revote": state.get("revote", False),
    "revote": state.get("revote", False),
}
```

- [ ] **Step 4: Run targeted runtime tests**

Run: `pytest tests/runtime/test_runtime.py::test_day_vote_ignores_stale_votes_when_day_changes -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/graph.py tests/runtime/test_runtime.py
git commit -m "fix: reset exile votes between days"
```

---

### Task 2: Route PK Revote Back Through Fresh Day Vote

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing test for revote route**

This is the critical regression: `tie_revote` must not go directly to `resolve_vote_node`. It must clear the first-round votes, mark the revote window, then route to `day_vote` so agents cast fresh votes.

```python
def test_tie_revote_routes_back_to_day_vote() -> None:
    graph = build_game_graph()

    edges = _edge_targets_for(graph, "tie_revote")

    assert "day_vote" in edges
    assert "resolve_vote_node" not in edges
```

- [ ] **Step 2: Write failing test for revote freshness**

```python
def test_tie_revote_clears_first_round_votes() -> None:
    from werewolf_agent.runtime.graph import tie_revote

    result = tie_revote({
        "exile_votes": {"p01": "p05", "p02": "p06"},
        "exile_vote_day": 3,
        "exile_vote_revote": False,
    })

    assert result["revote"] is True
    assert result["exile_votes"] == {}
    assert result["exile_vote_revote"] is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/runtime/test_runtime.py::test_tie_revote_routes_back_to_day_vote tests/runtime/test_runtime.py::test_tie_revote_clears_first_round_votes -v`

Expected: FAIL because the graph currently routes `tie_revote` directly to `resolve_vote_node`.

- [ ] **Step 4: Implement revote reset**

Change `tie_revote` to clear votes and mark the revote window:

```python
def tie_revote(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    return {
        "exile_votes": {},
        "exile_vote_day": gs.day_number,
        "exile_vote_revote": True,
        "revote": True,
}
```

- [ ] **Step 5: Fix graph edge**

Change the graph edge:

```python
graph.add_edge("tie_revote", "day_vote")
```

Do not route `tie_revote` directly to `resolve_vote_node`; otherwise revote uses an empty tally and immediately produces `second_tie_no_exile`.

- [ ] **Step 6: Run route and revote tests**

Run: `pytest tests/runtime/test_runtime.py::test_tie_revote_routes_back_to_day_vote tests/runtime/test_runtime.py::test_tie_revote_clears_first_round_votes tests/runtime/test_runtime.py::test_route_after_vote_tie -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/runtime/graph.py tests/runtime/test_runtime.py
git commit -m "fix: route tie revotes through fresh voting"
```

---

### Task 3: Make Voting Mandatory When Abstention Is Disabled

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for mandatory vote legal actions**

```python
def test_agent_day_vote_disallows_no_action_when_abstain_disabled() -> None:
    context = _capture_context_from_agent_day_vote()

    assert ActionType.VOTE in context.legal_actions
    assert ActionType.NO_ACTION not in context.legal_actions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_agents.py::test_agent_day_vote_disallows_no_action_when_abstain_disabled -v`

Expected: FAIL because `agent_day_vote` currently passes `[ActionType.VOTE, ActionType.NO_ACTION]`.

- [ ] **Step 3: Change legal actions based on ruleset**

In `agent_day_vote`, read:

```python
allow_abstain = engine.ruleset.raw["day_flow"]["vote"].get("allow_abstain", False)
legal_actions = [ActionType.VOTE]
if allow_abstain:
    legal_actions.append(ActionType.NO_ACTION)
```

Pass `legal_actions=legal_actions`.

- [ ] **Step 4: Add mandatory fallback behavior**

In `PlayerAgent._fallback_action`, keep the existing first-legal-action behavior, but ensure mandatory target actions choose a legal target. Add a test that when legal actions are `[VOTE]` and legal targets are `["p05"]`, fallback returns `VOTE p05`.

- [ ] **Step 5: Run agent tests**

Run: `pytest tests/agents/test_agents.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/runtime/agent_adapter.py werewolf_agent/agents/player.py tests/agents/test_agents.py
git commit -m "fix: enforce mandatory day votes"
```

---

### Task 4: Ensure Vote Majority Produces Real Exile Death

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing integration-style runtime test**

```python
def test_majority_vote_creates_exile_death_once() -> None:
    from werewolf_agent.runtime.graph import resolve_vote, resolve_exile

    engine = _new_engine()
    gs = _sample_game_state()
    state = {
        "game_state": gs,
        "engine": engine,
        "exile_votes": {"p01": "p05", "p02": "p05", "p03": "p05"},
        "revote": False,
    }

    state.update(resolve_vote(state))
    state.update(resolve_exile(state))

    deaths = state["game_state"].deaths
    assert [d for d in deaths if d.player_id == "p05" and d.reason == "exile"]
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `pytest tests/runtime/test_runtime.py::test_majority_vote_creates_exile_death_once -v`

Expected: PASS if runtime already resolves exile correctly in isolated flow; FAIL if `_vote_result` or state merge loses the result. If it passes, keep it as regression coverage for the real bug.

- [ ] **Step 3: Do not depend on `_vote_result` for graph routing**

`_vote_result` is not part of `RuntimeState` and may be dropped by LangGraph channels. `resolve_exile` must recover the target from the last `vote_resolved` event in `gs.events`.

Expected implementation shape:

```python
exiled_id = None
for event in reversed(gs.events):
    if event.type == "vote_resolved":
        exiled_id = event.payload.get("exiled")
        break
```

Keep `_vote_result` only as a local/unit-test convenience if needed; never rely on it across graph nodes.

- [ ] **Step 4: Add replay assertion for `game_g_789` symptom**

Add a helper test using synthetic events: if a `vote_resolved` event has `exiled != None` and `reason == "majority"`, later events/deaths must include `player_exiled` or idiot reveal for that player before the next `enter_night`.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/graph.py tests/runtime/test_runtime.py
git commit -m "test: guard majority vote exile resolution"
```

---

### Task 5: Add Simulation Anti-Stall Policy Without Changing Base Rules

**Files:**
- Modify: `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`
- Modify: `werewolf_agent/engine/rule_engine.py`
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/rules/test_rule_engine_v1.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Add config section**

Add under `day_flow.vote`:

```yaml
    simulation_pace:
      enabled: true
      max_consecutive_no_exile_days: 1
      forced_consolidation_policy: top_tied_or_top_suspects
      final_tie_breaker: sheriff_then_seeded_random
```

- [ ] **Step 2: Store PK candidates on first tie**

When `resolve_vote` returns `first_tie_pk`, the runtime must preserve the tied candidates for the PK round. Add them to the `vote_resolved` payload and runtime state:

```python
payload={
    "exiled": None,
    "reason": "first_tie_pk",
    "tied": result.tied_player_ids,
}
return {
    "game_state": gs,
    "pk_candidates": result.tied_player_ids,
}
```

If `VoteResult` does not yet expose tied candidates, add an optional `tied_player_ids: list[str]` field to the model and populate it in `RuleEngine.resolve_vote`.

- [ ] **Step 3: Write preserving-rule test**

Existing test `test_second_tie_creates_no_exile_and_enters_night` must continue to pass when anti-stall is not explicitly invoked.

Run: `pytest tests/rules/test_rule_engine_v1.py::test_second_tie_creates_no_exile_and_enters_night -v`

Expected: PASS.

- [ ] **Step 4: Write anti-stall test for tied candidates**

```python
def test_anti_stall_policy_breaks_repeated_second_tie_from_pk_candidates() -> None:
    engine = make_engine()
    state = make_state()

    result = engine.resolve_vote(
        state,
        votes={},
        revote=True,
        consecutive_no_exile_days=2,
        pk_candidates=["w1", "w2"],
        rng_seed="game-pace-test",
    )

    assert result.exiled_player_id in {"w1", "w2"}
    assert result.reason in {"anti_stall_tie_break", "anti_stall_empty_tally"}
```

- [ ] **Step 5: Extend `resolve_vote` signature carefully**

Add optional keyword parameters with defaults:

```python
def resolve_vote(
    self,
    state: GameState,
    *,
    votes: dict[str, str],
    revote: bool,
    consecutive_no_exile_days: int = 0,
    pk_candidates: list[str] | None = None,
    rng_seed: str | None = None,
) -> VoteResult:
```

Only use anti-stall when config is enabled and `consecutive_no_exile_days` exceeds the configured max.

- [ ] **Step 6: Implement deterministic tie-break**

Tie-break order:

1. If active sheriff voted for one of the tied candidates, exile that candidate.
2. If `revote=True` and tally is empty, choose from `pk_candidates`, not from all legal targets.
3. Otherwise pick stable seeded random from tied candidates.
4. Record reason `anti_stall_tie_break` or `anti_stall_empty_tally`.

- [ ] **Step 7: Add runtime test for empty revote candidate scope**

```python
def test_empty_revote_anti_stall_uses_pk_candidates_only() -> None:
    result = resolve_vote({
        "game_state": _sample_game_state(),
        "engine": _new_engine(),
        "exile_votes": {},
        "revote": True,
        "pk_candidates": ["p05", "p06"],
        "consecutive_no_exile_days": 2,
    })

    event = result["game_state"].events[-1]
    assert event.payload["exiled"] in {"p05", "p06"}
```

- [ ] **Step 8: Run rule and runtime tests**

Run: `pytest tests/rules/test_rule_engine_v1.py tests/runtime/test_runtime.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add config/rulesets/pre_witch_hunter_idiot_mixed.yaml werewolf_agent/engine/rule_engine.py werewolf_agent/runtime/graph.py tests/rules/test_rule_engine_v1.py tests/runtime/test_runtime.py
git commit -m "feat: add simulation vote anti-stall policy"
```

---

### Task 6: Track Consecutive No-Exile Days In Runtime

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing counter tests**

```python
def test_no_exile_counter_increments_on_second_tie() -> None:
    state = _state_after_vote_result(exiled=None, reason="second_tie_no_exile", no_exile_days=0)
    result = _update_no_exile_counter(state)
    assert result["consecutive_no_exile_days"] == 1


def test_no_exile_counter_resets_on_exile() -> None:
    state = _state_after_vote_result(exiled="p05", reason="majority", no_exile_days=2)
    result = _update_no_exile_counter(state)
    assert result["consecutive_no_exile_days"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/runtime/test_runtime.py::test_no_exile_counter_increments_on_second_tie tests/runtime/test_runtime.py::test_no_exile_counter_resets_on_exile -v`

Expected: FAIL because no counter exists.

- [ ] **Step 3: Add runtime counter update**

In `resolve_vote`, after `engine.resolve_vote`, return:

```python
next_no_exile_days = (
    state.get("consecutive_no_exile_days", 0) + 1
    if result.reason == "second_tie_no_exile"
    else 0
)
```

Pass the current counter into `engine.resolve_vote`.

- [ ] **Step 4: Run runtime tests**

Run: `pytest tests/runtime/test_runtime.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/graph.py tests/runtime/test_runtime.py
git commit -m "feat: track consecutive no-exile days"
```

---

### Task 7: Add Vote Pressure To Agent Context

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Optional modify: `werewolf_agent/agents/player.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Write test for vote prompt context**

```python
def test_vote_context_mentions_mandatory_exile_pressure() -> None:
    context = _build_vote_context()
    prompt = PlayerAgent(...)._build_system_prompt(context)

    assert "必须投票" in prompt
    assert "不能弃票" in prompt
    assert "连续无人出局" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_agents.py::test_vote_context_mentions_mandatory_exile_pressure -v`

Expected: FAIL until prompt/context includes these hints.

- [ ] **Step 3: Add safe public vote pressure hints**

Add only public/runtime information to context:

- Current day number.
- Consecutive no-exile day count.
- Legal targets.
- Statement that abstention is not legal when `allow_abstain: false`.

Do not include hidden identities, private intentions, or moderator-only role data.

- [ ] **Step 4: Run info-leak and agent tests**

Run: `pytest tests/agents/test_agents.py tests/integration/test_e2e_info_leak.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/agent_adapter.py werewolf_agent/agents/player.py tests/agents/test_agents.py
git commit -m "feat: add mandatory vote pressure hints"
```

---

### Task 8: Add Game Pace Evaluation Metrics

**Files:**
- Modify: `werewolf_agent/evaluation/metrics.py`
- Test: `tests/evaluation/test_evaluation.py`

- [ ] **Step 1: Write metrics tests**

```python
def test_game_pace_metrics_detect_stale_votes_and_no_exile_streak() -> None:
    result = _evaluation_result_with_events([
        vote_resolved(day=1, exiled="p01", votes={"p02": "p01"}),
        vote_resolved(day=2, exiled="p01", votes={"p02": "p01"}),
        vote_resolved(day=3, exiled=None, reason="second_tie_no_exile"),
        vote_resolved(day=4, exiled=None, reason="second_tie_no_exile"),
    ])

    metrics = compute_quality_metrics(result)

    assert metrics["stale_vote_reuse_count"] == 1
    assert metrics["max_consecutive_no_exile_days"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_evaluation.py::test_game_pace_metrics_detect_stale_votes_and_no_exile_streak -v`

Expected: FAIL because metrics do not exist.

- [ ] **Step 3: Implement pace metrics**

Add:

- `day_exile_rate`
- `stale_vote_reuse_count`
- `max_consecutive_no_exile_days`
- `finish_night_number`
- `pace_target_met`

Set `pace_target_met` true when:

- `finish_night_number <= 8`
- `max_consecutive_no_exile_days <= 1`
- no stale vote reuse
- if at least 3 days occurred, `day_exile_rate >= 0.5`

- [ ] **Step 4: Run evaluation tests**

Run: `pytest tests/evaluation/test_evaluation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/metrics.py tests/evaluation/test_evaluation.py
git commit -m "feat: add game pace metrics"
```

---

### Task 9: Add Full Runtime Action Trace And Pace Report

**Files:**
- Modify: `scripts/run_real_game.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/agents/player.py`
- Optional modify: `werewolf_agent/evaluation/reports.py`
- Test: `tests/agents/test_agents.py`
- Test: manual smoke command

- [ ] **Step 1: Add full trace output requirements**

Each player operation must be reconstructable from logs. Record:

- Agent id and task type.
- Full raw model text.
- Parsed action JSON.
- Final accepted action.
- Legal actions and legal targets.
- Retry count, parse/validation errors, and fallback reason.
- Full public speech text, without truncation.
- Full vote target and vote reason.
- Full wolf kill, witch, seer, hunter, sheriff, and hybrid decision payloads.
- Visibility label for every event.

Do not log hidden role identities into public/player-visible logs. Full trace may be moderator/private audit output only.

- [ ] **Step 2: Write failing test for action trace fields**

```python
def test_agent_action_trace_records_raw_text_and_fallback() -> None:
    action, retry = agent.act(context)

    trace = action.trace

    assert trace.raw_text
    assert trace.legal_actions
    assert trace.legal_targets
    assert trace.final_action_type == action.action_type.value
```

- [ ] **Step 3: Implement structured action trace**

Add a lightweight trace object or payload returned by `PlayerAgent.act` and propagated by adapter functions. If changing the return type is too invasive, add trace to returned adapter dicts:

```python
return {
    "vote_target": target,
    "action_trace": {
        "agent_id": voter_id,
        "task_type": "vote",
        "raw_text": retry_info.raw_text,
        "final_action": action.model_dump(),
        "legal_actions": [a.value for a in legal_actions],
        "legal_targets": legal_targets,
        "retry": retry_info.model_dump() if retry_info else None,
    },
}
```

- [ ] **Step 4: Add full event logging**

When runtime nodes append events, include full text fields in the event payload. Do not truncate in stored events. If display output needs truncation, do it only in UI rendering, never in saved JSON or audit logs.

- [ ] **Step 5: Add pace report output requirements**

After each real game, print:

- Winner.
- Finish day/night.
- Total deaths.
- Wolf kills.
- Exile deaths.
- `second_tie_no_exile` count.
- Max consecutive no-exile days.
- Stale vote reuse count.
- API calls and 429 count if available.

- [ ] **Step 6: Implement report formatting**

Keep the summary text-only and concise, but write the complete action trace to the saved game JSON. Remove truncation like `text[:80]` from audit output.

- [ ] **Step 7: Run targeted tests and short deterministic smoke test**

Run: `pytest tests/agents/test_agents.py -v`

Expected: PASS.

Run: `python scripts/run_real_game.py --seed 789 --max-steps 80`

Expected: Report includes pace metrics even if the game does not finish within 80 steps, and saved JSON contains full speech/action trace text without truncation.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_real_game.py werewolf_agent/runtime/agent_adapter.py werewolf_agent/agents/player.py werewolf_agent/evaluation/reports.py tests/agents/test_agents.py
git commit -m "feat: log full runtime action traces"
```

---

### Task 10: Full Verification

**Files:**
- No planned source changes.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest tests/runtime/test_runtime.py tests/rules/test_rule_engine_v1.py tests/agents/test_agents.py tests/evaluation/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 2: Run information leak regression**

Run:

```bash
pytest tests/integration/test_e2e_info_leak.py tests/integration/test_visibility_replay.py -v
```

Expected: PASS.

- [ ] **Step 3: Run real game validation**

Run:

```bash
python scripts/run_real_game.py --seed 789
```

Expected:

- Game finishes.
- No information leakage.
- No stale vote reuse.
- PK revote routes through `day_vote` and collects fresh votes.
- Empty PK revote anti-stall selects only from PK candidates.
- `max_consecutive_no_exile_days <= 1`.
- At least one real `exile` death or white-idiot reveal from day vote.
- Saved game JSON contains full, untruncated action text for speeches and decisions.
- Finish no later than night 8 for this seed, with target median night 4-6 across a seed batch.

- [ ] **Step 4: Run seed batch**

Run:

```bash
python scripts/run_real_game.py --seeds 101,202,303,404,505 --max-steps 220
```

Expected:

- Median finish night: 4-6.
- Night 8+ games: <= 10%.
- Day exile rate: >= 70% for games with at least 2 full days.
- Information leakage: 0.

- [ ] **Step 5: Final commit**

```bash
git status --short
git add .
git commit -m "fix: normalize live game pace"
```

---

## Risks And Guardrails

- Do not make RuleEngine reveal hidden roles to solve vote quality.
- Do not replace official tie policy globally; anti-stall must be simulation/runtime policy and testable.
- Do not route PK revote directly to vote resolution; it must re-enter `day_vote`.
- Do not use all legal targets for empty PK revote anti-stall; use stored PK candidates.
- Do not rely on `_vote_result` across graph nodes; use durable `vote_resolved` events.
- Do not let fallback always choose the first legal target long term; use it only as a schema-safety fallback.
- Keep public prompts about pressure and legal targets, not secret alignment.
- Keep full action traces in moderator/audit logs, not public player-visible context.
- Preserve existing tests for hunter shot, witch action, idiot reveal, sheriff badge, and visibility.

## Done Criteria

- `game_g_789` symptom cannot recur: no cross-day stale votes, no repeated majority against already-dead targets, and no majority vote without exile resolution.
- PK revote cannot become an automatic empty-vote `second_tie_no_exile`; it must collect fresh votes through `day_vote`.
- Anti-stall tie breaking in PK contexts is scoped to PK candidates.
- `_vote_result` is not required for cross-node exile resolution.
- Full action trace logs preserve untruncated speeches and decision text for debugging.
- Real games finish through normal day/night elimination pressure, not only wolf night kills.
- Seed batch has human-like pace distribution: median finish night 4-6, night 8+ under 10%.
- Existing no-information-leak checks still pass.
