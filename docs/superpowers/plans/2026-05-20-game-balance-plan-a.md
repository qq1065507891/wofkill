# Game Balance Plan A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce structural werewolf-side bias by fixing deterministic fallbacks, requiring evidence-backed wolf plans and good-side votes, and adding batch metrics that catch regressions.

**Architecture:** Keep `RuleEngine` victory and adjudication rules unchanged. Adjust runtime strategy/adapters so agent failures and weak evidence do not collapse into deterministic wolf-favorable defaults, then verify with focused unit tests plus a small deterministic batch audit.

**Tech Stack:** Python, pytest, LangGraph runtime nodes, existing `PlayerAgent`, `wolf_strategy`, `vote_quality`, and evaluation modules.

---

## File Structure

- Modify `werewolf_agent/agents/player.py`
  - Fix invalid action examples so prompts do not teach schema-invalid `faction_goal` values.
  - Preserve the existing schema and retry/fallback path.

- Modify `werewolf_agent/runtime/agent_adapter.py`
  - Replace vote fallback target selection with evidence-aware deterministic selection.
  - Inject full day discussion and anti-herd guidance into vote contexts.
  - Keep player visibility constraints unchanged.

- Modify `werewolf_agent/runtime/vote_quality.py`
  - Add reusable helpers for evidence-aware fallback ranking and herd-risk detection.
  - Keep existing basis extraction API compatible.

- Modify `werewolf_agent/runtime/wolf_strategy.py`
  - Require discussion evidence before a `wolf_team_plan` can name night kill targets.
  - Mark fallback plan fields with evidence quality so runtime can avoid treating static fallback as consensus.

- Modify `werewolf_agent/runtime/graph.py`
  - Stop `_build_wolf_team_plan` from auto-selecting first alive non-wolves as kill targets.
  - Make `_planned_wolf_kill` obey evidence quality.
  - Reset vote window state after resolved exile/no-exile to avoid stale vote reuse.

- Add or modify tests:
  - `tests/agents/test_agents.py`
  - `tests/runtime/test_vote_quality.py`
  - `tests/runtime/test_wolf_strategy.py`
  - `tests/runtime/test_game_runner.py`
  - Optional: `tests/evaluation/test_game_balance_batch.py`

---

### Task 1: Fix Schema Examples And Vote Fallback Bias

**Files:**
- Modify: `werewolf_agent/agents/player.py:510-527`
- Modify: `werewolf_agent/runtime/agent_adapter.py:851-920`
- Modify: `werewolf_agent/runtime/vote_quality.py`
- Test: `tests/agents/test_agents.py`
- Test: `tests/runtime/test_vote_quality.py`

- [ ] **Step 1: Add failing test for valid prompt examples**

Add a test that builds a `PlayerAgent` prompt for a wolf-kill context and asserts examples only use legal `PrivateIntent.faction_goal` enum values.

```python
def test_wolf_examples_use_valid_private_intent_goals():
    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.agents.schemas import ActionType, AgentContext, PrivateIntent, TaskType

    class DummyRouter:
        def generate(self, **kwargs):
            raise AssertionError("not called")

    agent = PlayerAgent("p01", DummyRouter())
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.NIGHT_ACTION,
        own_role="werewolf",
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=["p05"],
    )

    prompt = agent._build_system_prompt(ctx)

    assert "eliminate_villager" not in prompt
    assert "frame_villager" not in prompt
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
python -m pytest tests/agents/test_agents.py::test_wolf_examples_use_valid_private_intent_goals -q
```

Expected: FAIL because the prompt currently contains invalid enum examples.

- [ ] **Step 3: Fix the wolf examples**

Change the wolf-kill example to use `push_good_player_out` or `aggressive_push`. Change the no-kill example to use `confuse_good` or `deep_hook`.

- [ ] **Step 4: Add failing tests for vote fallback target selection**

In `tests/runtime/test_vote_quality.py`, add tests for a helper such as `choose_vote_fallback_target(gs, voter_id, legal_targets, reason_context)`:

```python
def test_vote_fallback_does_not_always_pick_first_legal_target():
    from werewolf_agent.core.models import GameEvent, GameState, PlayerState
    from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 5)
    }
    gs = GameState(
        game_id="g_vote",
        day_number=1,
        players=players,
        events=[
            GameEvent(type="speech", payload={
                "speaker": "p03",
                "day_number": 1,
                "text": "我刚才说我是预言家，但警徽流前后矛盾，逻辑不通。",
            }),
        ],
    )

    target = choose_vote_fallback_target(gs, "p01", ["p02", "p03", "p04"])

    assert target == "p03"
```

- [ ] **Step 5: Implement evidence-aware fallback helper**

Add `choose_vote_fallback_target` to `vote_quality.py`.

Implementation rules:
- Score legal targets by mentions in current-day speeches with basis words from `extract_vote_basis`.
- Exclude `voter_id`.
- If no evidence exists, use stable seeded random from `game_id`, `day_number`, `voter_id`, and legal targets.
- Never simply return `legal_targets[0]` unless the seeded choice happens to choose it.

- [ ] **Step 6: Wire vote fallback into `agent_day_vote`**

Replace:

```python
if target is None and legal_targets:
    target = legal_targets[0]
```

with:

```python
if target is None and legal_targets:
    from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target
    target = choose_vote_fallback_target(gs, voter_id, legal_targets)
```

- [ ] **Step 7: Add vote context anti-herd guidance**

In `agent_day_vote`, include:
- `build_day_discussion_summary(gs, gs.day_number)`
- `build_vote_pressure_context(gs, voter_id, pk_candidates=state.get("pk_candidates"))`
- A directive that says good-side voters should not follow a near-unanimous push without at least one concrete basis.

Do not reveal hidden roles.

- [ ] **Step 8: Verify Task 1 tests**

Run:

```powershell
python -m pytest tests/agents/test_agents.py tests/runtime/test_vote_quality.py -q
```

Expected: PASS.

---

### Task 2: Require Evidence For Wolf Team Kill Plans

**Files:**
- Modify: `werewolf_agent/runtime/wolf_strategy.py`
- Modify: `werewolf_agent/runtime/graph.py:233-300`
- Test: `tests/runtime/test_wolf_strategy.py`

- [ ] **Step 1: Add failing test for no-evidence wolf plan**

In `tests/runtime/test_wolf_strategy.py`, add:

```python
def test_wolf_plan_without_discussion_evidence_has_no_kill_target():
    from werewolf_agent.runtime.wolf_strategy import build_wolf_team_plan_from_discussion

    gs = _make_wolf_gs()
    consensus = {
        "night_kill_primary": None,
        "night_kill_backup": None,
        "evidence_from_discussion": [],
        "agreement_count": 0,
        "total_wolves": 4,
    }

    plan = build_wolf_team_plan_from_discussion(gs, consensus=consensus)

    assert plan.get("night_kill_primary") is None
    assert plan.get("night_kill_backup") is None
    assert plan.get("evidence_quality") == "none"
```

- [ ] **Step 2: Add failing test for runtime not using static fallback kill**

Test `_planned_wolf_kill` with a plan that has `evidence_quality="none"` and a target. Expected result is `None`.

```python
def test_planned_wolf_kill_ignores_low_evidence_plan():
    from werewolf_agent.runtime.graph import _planned_wolf_kill, RuntimeState

    gs = _make_wolf_gs()
    state = {
        "game_state": gs,
        "wolf_team_plan": {
            "night_kill_primary": "p05",
            "evidence_quality": "none",
        },
    }

    assert _planned_wolf_kill(state) is None
```

- [ ] **Step 3: Implement evidence quality in wolf plans**

In `build_wolf_team_plan_from_discussion`, compute:
- `"strong"` when `agreement_count > total_wolves / 2` and at least one evidence item supports the primary target.
- `"weak"` when a target exists but no strict majority exists.
- `"none"` when no discussion evidence supports a target.

- [ ] **Step 4: Stop static fallback from selecting kill targets**

Change `_build_wolf_team_plan` so it may still assign roles and public story, but does not set `night_kill_primary`, `night_kill_backup`, or `day_push_target` from seat order. Those fields should remain `None` unless prior plan has an alive target with evidence quality not `"none"`.

- [ ] **Step 5: Gate `_planned_wolf_kill` by evidence**

In `_planned_wolf_kill`, return `None` if:
- `plan.get("evidence_quality") == "none"`, or
- target has no matching entry in `evidence_from_discussion`, unless `evidence_quality == "strong"`.

Then let existing agent consensus or no-kill fallback handle the night.

- [ ] **Step 6: Verify Task 2 tests**

Run:

```powershell
python -m pytest tests/runtime/test_wolf_strategy.py -q
```

Expected: PASS.

---

### Task 3: Prevent Stale Vote Reuse Across Vote Windows

**Files:**
- Modify: `werewolf_agent/runtime/graph.py:1202-1357`
- Test: `tests/runtime/test_game_runner.py`

- [ ] **Step 1: Add failing test for new vote window after exile**

Create a unit-level test around `day_vote` or graph state progression that proves a new day does not reuse the previous `exile_votes`.

Suggested focused test:

```python
def test_day_vote_does_not_reuse_previous_day_votes_without_same_window():
    from werewolf_agent.runtime.graph import day_vote

    gs = GameState(
        game_id="g_vote_window",
        phase="day",
        day_number=2,
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        },
    )
    state = {
        "game_state": gs,
        "exile_votes": {"p01": "p02"},
        "exile_vote_day": 1,
        "exile_vote_revote": False,
        "revote": False,
    }

    result = day_vote(state)

    assert result["exile_votes"] == {}
    assert result["exile_vote_day"] == 2
```

- [ ] **Step 2: Add state reset after resolved vote path**

When `resolve_vote` returns an exile or second-tie no-exile, explicitly clear:
- `exile_votes`
- `vote_action_traces`
- `exile_vote_revote`

Keep `exile_vote_day` set to current day for audit only if needed. The next `day_vote` should only treat votes as reusable when `same_vote_window` is true.

- [ ] **Step 3: Make PK revote reuse explicit**

Ensure `tie_revote` is the only place that intentionally opens a same-day revote window. PK candidates must be carried in `pk_candidates`, not by reusing old votes.

- [ ] **Step 4: Verify Task 3 tests**

Run:

```powershell
python -m pytest tests/runtime/test_game_runner.py -q
```

Expected: PASS.

---

### Task 4: Add Batch Balance Audit Guardrails

**Files:**
- Create: `werewolf_agent/evaluation/balance_audit.py`
- Create: `tests/evaluation/test_game_balance_batch.py`
- Optional modify: `scripts/run_real_game.py`

- [ ] **Step 1: Add balance audit data model/helper tests**

Add tests that compute metrics from saved-style game dicts:

```python
def test_balance_audit_flags_high_wolf_win_rate():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        {"winning_faction": "werewolf", "events": [], "deaths": []}
        for _ in range(20)
    ]

    audit = compute_balance_audit(games)

    assert audit["wolf_win_rate"] == 1.0
    assert "wolf_win_rate_high" in audit["warnings"]
```

- [ ] **Step 2: Implement `compute_balance_audit`**

Metrics:
- `games`
- `wolf_win_rate`
- `good_win_rate`
- `fallback_action_rate`
- `schema_failure_rate`
- `seer_day1_exile_rate`
- `witch_night1_death_rate`
- `mean_vote_concentration`
- `weak_wolf_plan_kill_count`

Warnings:
- `wolf_win_rate_high` if at least 10 games and wolf win rate > 0.75.
- `schema_failure_high` if schema failures exceed 5%.
- `seer_day1_exile_high` if at least 10 games and rate > 0.35.
- `weak_wolf_plan_kills_present` if any wolf kill was made from weak/no evidence plan.

- [ ] **Step 3: Add saved-log loader helper**

Add a helper:

```python
def load_game_logs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    ...
```

Keep it pure JSON loading. Do not call LLM providers.

- [ ] **Step 4: Add optional script integration**

Either add a small CLI script later, or extend `scripts/run_real_game.py` summary to print balance-relevant metrics for a single game. Do not block core tests on real LLM calls.

- [ ] **Step 5: Verify evaluation tests**

Run:

```powershell
python -m pytest tests/evaluation/test_game_balance_batch.py -q
```

Expected: PASS.

---

### Task 5: Final Verification

**Files:**
- No new code unless earlier tasks reveal a defect.

- [ ] **Step 1: Run focused test suite**

Run:

```powershell
python -m pytest tests/agents/test_agents.py tests/runtime/test_vote_quality.py tests/runtime/test_wolf_strategy.py tests/runtime/test_game_runner.py tests/evaluation/test_game_balance_batch.py -q
```

Expected: PASS.

- [ ] **Step 2: Run rule-engine regression tests**

Run:

```powershell
python -m pytest tests/rules/test_rule_engine_v1.py -q
```

Expected: PASS. This confirms the plan did not change adjudication truth.

- [ ] **Step 3: Run non-LLM deterministic smoke**

Run:

```powershell
python -m pytest tests/runtime/test_runtime.py tests/integration/test_live_game_flow.py -q
```

Expected: PASS.

- [ ] **Step 4: Optional real-game batch**

Only if API keys and time budget are available, run 10-20 real LLM games with fixed seeds and feed saved JSON logs into `compute_balance_audit`.

Expected after this plan:
- No schema failures from prompt examples.
- Fallbacks no longer always pick first legal target.
- Weak/no-evidence wolf plans do not produce automatic night kills.
- Wolf win rate should no longer be structurally 100% in small batches, though exact rates remain model-dependent.

- [ ] **Step 5: Document results**

Update `PROGRESS.md` with:
- Changed files.
- Focused test results.
- Any real-game batch audit numbers if run.
