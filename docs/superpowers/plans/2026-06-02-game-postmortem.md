# Game g_3528592081 Post-mortem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 confirmed bugs from game g_3528592081 post-mortem (vote validation/fallback, wolf seer priority, fake seer info leak, solo-wolf plan fallback). One additional design point (sheriff election tie) requires user clarification before any code change.

**Architecture:** Two independent streams: (A) vote/fallback fixes in `runtime/vote_quality.py` + `agents/player.py` + `runtime/agent_adapter.py`; (B) wolf strategy fixes in `runtime/directives/wolf.py` + `runtime/agent_adapter.py` + `runtime/nodes/_shared.py`. Each task is self-contained and committable independently.

**Tech Stack:** Python 3.12, LangGraph, pytest, Pydantic, RuleEngine

**Reference:**
- Source game: `game_g_3528592081.json` (433 KB, finished 2026-06-02 01:59, good wins)
- Findings ledger: `PROGRESS.md` (section "Game g_3528592081 Post-mortem — 2026-06-02")

---

## Issue 0: Add sheriff election PK + revote (NEW per user 2026-06-02)

**User decision:** Sheriff election tie should follow the same pattern as exile voting tie: first tie → PK speech by tied candidates only, then revote among tied candidates; second tie → no sheriff, badge lost. This unifies the rule across voting domains.

**Files:**
- Modify: `werewolf_agent/core/models.py:30-60` (add `sheriff_pk_candidates`, `sheriff_tie_count` to `GameState`)
- Modify: `werewolf_agent/engine/sheriff.py:60-75` (`resolve_sheriff_vote` add tie_count parameter)
- Create: `werewolf_agent/runtime/nodes/sheriff_pk.py` (new module: `sheriff_pk_speech`, `sheriff_revote`)
- Modify: `werewolf_agent/runtime/nodes/sheriff.py:271-302` (`sheriff_vote` route to PK on first tie)
- Modify: `werewolf_agent/runtime/graph.py:251-258, 336-440` (add nodes + routes)
- Test: `tests/runtime/test_sheriff_flow.py` (add PK tests)

- [ ] **Step 1: Add tie_count + pk_candidates to GameState**

In `werewolf_agent/core/models.py`, in the `GameState` dataclass around line 60, add two new fields:

```python
sheriff_tie_count: int = 0
sheriff_pk_candidates: list[str] = field(default_factory=list)
```

Update `__post_init__` (line 50-57) to also defensive-copy `sheriff_pk_candidates`:

```python
object.__setattr__(self, "sheriff_pk_candidates", list(self.sheriff_pk_candidates) if self.sheriff_pk_candidates else [])
```

- [ ] **Step 2: Write failing test for first-tie → PK behavior**

In `tests/runtime/test_sheriff_flow.py`, add at end of file:

```python
class TestSheriffElectionPK:
    def test_first_tie_triggers_pk_speech(self):
        """First sheriff vote tie should route to sheriff_pk_speech with tied candidates."""
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.nodes.sheriff import sheriff_vote
        from werewolf_agent.runtime.nodes._shared import RuntimeState

        gs = GameState(
            game_id="g_test",
            players={
                "p01": PlayerState(player_id="p01", alive=True),
                "p05": PlayerState(player_id="p05", alive=True),
                "p08": PlayerState(player_id="p08", alive=True),
            },
            sheriff_candidates=["p01", "p05", "p08"],
            day_number=1,
        )
        state: RuntimeState = {
            "game_state": gs,
            "engine": None,  # resolve_sheriff_vote does not need engine for tie
            "sheriff_votes": {"v_a": "p01", "v_b": "p05", "v_c": "p08"},
            "sheriff_withdrawing": [],
        }
        result = sheriff_vote(state)
        gs_out = result["game_state"]
        # On first tie, tie_count should be 1, no sheriff, and pk_candidates set
        assert gs_out.sheriff_id is None
        assert gs_out.sheriff_tie_count == 1
        assert set(gs_out.sheriff_pk_candidates) == {"p01", "p05", "p08"}

    def test_second_tie_skips_to_no_election(self):
        """When tie_count is already 1, a second tie resolves to no sheriff."""
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.nodes.sheriff import sheriff_vote
        from werewolf_agent.runtime.nodes._shared import RuntimeState

        gs = GameState(
            game_id="g_test",
            players={
                "p01": PlayerState(player_id="p01", alive=True),
                "p05": PlayerState(player_id="p05", alive=True),
            },
            sheriff_candidates=["p01", "p05"],
            sheriff_tie_count=1,  # already tied once
            day_number=1,
        )
        state: RuntimeState = {
            "game_state": gs,
            "sheriff_votes": {"v_a": "p01", "v_b": "p05"},
            "sheriff_withdrawing": [],
        }
        result = sheriff_vote(state)
        gs_out = result["game_state"]
        assert gs_out.sheriff_id is None
        # Second tie: tie_count should be reset, no PK loop
        events = [e for e in gs_out.events if e.type == "sheriff_no_election"]
        assert len(events) == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/runtime/test_sheriff_flow.py::TestSheriffElectionPK -v`
Expected: FAIL (current `sheriff_vote` does not set `sheriff_tie_count` or `sheriff_pk_candidates`)

- [ ] **Step 4: Update `sheriff_vote` to handle first-tie → PK routing**

In `werewolf_agent/runtime/nodes/sheriff.py`, replace the no-election branch (lines 294-302) with:

```python
# No election from vote tie — route to PK on first tie
elected_id = event.payload.get("sheriff_id")
if elected_id:
    # ... existing election branch unchanged ...
    pass
else:
    tied = event.payload.get("tied", [])
    tie_count = gs.sheriff_tie_count
    if tie_count == 0 and tied:
        # First tie → enter PK with tied candidates
        gs = replace(
            gs,
            sheriff_tie_count=1,
            sheriff_pk_candidates=tied,
            events=gs.events + [GameEvent(
                type="sheriff_vote_tie_first",
                payload={"tied": tied},
            )],
        )
        gs, _ = _judge_broadcast(
            phase="sheriff_vote_tie_first",
            message=f"警下投票首次平票，{', '.join(_player_display(state, c) for c in tied)} 进入 PK 发言环节",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        return {"game_state": gs}
    # Second tie (or no candidates) → no sheriff
    gs, _ = _judge_broadcast(
        phase="sheriff_no_election",
        message="投票未选出警长，警徽流失，本局无警长",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    # Reset tie_count for next election opportunity (D2 re-election etc.)
    gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
    speech_order = choose_no_sheriff_speech_order(gs)
    return {"game_state": gs, "speech_order": speech_order}
```

- [ ] **Step 5: Create `sheriff_pk.py` with PK speech and revote nodes**

Create file `werewolf_agent/runtime/nodes/sheriff_pk.py`:

```python
"""Sheriff election PK speech and revote nodes (after first vote tie)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import agent_sheriff_election_speech
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _action_trace_event,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
    AGENT_TIMEOUTS,
)
from werewolf_agent.runtime.sheriff_policy import choose_sheriff_led_speech_order


def sheriff_pk_speech(state: RuntimeState) -> dict[str, Any]:
    """Only sheriff_pk_candidates give speeches during PK phase."""
    gs: GameState = state["game_state"]
    pk_candidates = list(gs.sheriff_pk_candidates or [])
    events: list[GameEvent] = []

    if not pk_candidates:
        # Edge case: no candidates recorded — skip to no_election
        gs, _ = _judge_broadcast(
            phase="sheriff_no_election",
            message="警徽流失，本局无警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
        return {"game_state": gs}

    pk_names = ", ".join(_player_display(state, c) for c in pk_candidates)
    gs, _ = _judge_broadcast(
        phase="sheriff_pk_speech_start",
        message=f"首次平票，{pk_names} 进入 PK 发言环节，请依次发言",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    for candidate_id in pk_candidates:
        result = _dispatch_agent(
            state,
            agent_sheriff_election_speech,
            candidate_id,
            timeout_override=AGENT_TIMEOUTS.sheriff_speech,
        )
        speech_text = result.get("speech_text", "") if result else ""
        logger.debug(f"  [警长PK发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
        events.append(GameEvent(
            type="sheriff_pk_speech",
            payload={
                "speaker": candidate_id,
                "day_number": gs.day_number,
                "text": speech_text,
            },
        ))
        if result and result.get("action_trace"):
            events.append(_action_trace_event(
                player_id=candidate_id,
                phase="sheriff_pk_speech",
                action_trace=result["action_trace"],
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))

    gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


def sheriff_revote(state: RuntimeState) -> dict[str, Any]:
    """Revote after sheriff PK — only sheriff_pk_candidates are eligible."""
    from werewolf_agent.runtime.sheriff_policy import (
        eligible_sheriff_voters,
        filter_sheriff_votes_to_eligible,
    )
    from werewolf_agent.engine.rule_engine import RuleEngine

    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    pk_candidates = list(gs.sheriff_pk_candidates or [])

    if not pk_candidates:
        gs, _ = _judge_broadcast(
            phase="sheriff_no_election",
            message="无 PK 候选人，警徽流失",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="sheriff_revote_start",
        message=f"PK 发言结束，警下玩家重新投票选出警长（仅 {', '.join(_player_display(state, c) for c in pk_candidates)} 可选）",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    # Reuse sheriff_vote logic but constrain candidates to pk_candidates
    # by passing pk_candidates as the candidates list.
    from werewolf_agent.runtime.agent_adapter import agent_sheriff_vote
    from werewolf_agent.runtime.sheriff_policy import eligible_sheriff_voters

    withdrew = list(state.get("sheriff_withdrawing", []))
    # Voters exclude all PK candidates (they cannot vote in their own PK)
    voters = [
        pid for pid in gs.players
        if gs.players[pid].alive and pid not in pk_candidates
    ]

    votes: dict[str, str] = {}
    vote_records: list[dict[str, Any]] = []
    for voter_id in voters:
        result = _dispatch_agent(state, agent_sheriff_vote, voter_id, pk_candidates)
        if result and result.get("vote_target"):
            votes[voter_id] = result["vote_target"]
            vote_records.append({"voter": voter_id, "target": result["vote_target"]})
            logger.debug(f"  [警长复投] {_player_display(state, voter_id)} 投给 {_player_display(state, result['vote_target'])}")

    if vote_records:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_vote_record",
            payload={"votes": vote_records, "candidates": pk_candidates, "revote": True},
        )])

    gs, event = engine.resolve_sheriff_vote(gs, votes=votes, candidates=pk_candidates)
    gs = replace(gs, events=gs.events + [event])

    elected_id = event.payload.get("sheriff_id")
    if elected_id:
        gs = replace(gs, sheriff_id=elected_id, sheriff_badge_state="active",
                     sheriff_tie_count=0, sheriff_pk_candidates=[])
        gs, _ = _judge_broadcast(
            phase="sheriff_elected",
            message=f"{_player_display(state, elected_id)} 当选警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        speech_order = choose_sheriff_led_speech_order(gs, elected_id)
        return {"game_state": gs, "speech_order": speech_order}

    # Revote also tied → no sheriff (second tie)
    gs, _ = _judge_broadcast(
        phase="sheriff_no_election",
        message="复投仍未选出警长，警徽流失，本局无警长",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
    from werewolf_agent.runtime.sheriff_policy import choose_no_sheriff_speech_order
    speech_order = choose_no_sheriff_speech_order(gs)
    return {"game_state": gs, "speech_order": speech_order}
```

- [ ] **Step 6: Add `sheriff_pk_speech` and `sheriff_revote` to nodes `__init__` exports**

In `werewolf_agent/runtime/nodes/__init__.py`, find the section that exports `sheriff_speech` and `sheriff_vote` (around line 60-100), and add the new exports:

```python
from werewolf_agent.runtime.nodes.sheriff_pk import (
    sheriff_pk_speech,
    sheriff_revote,
)
```

Add to the `__all__` list (if present) and to any explicit import block.

- [ ] **Step 7: Update graph.py to wire the new nodes and routes**

In `werewolf_agent/runtime/graph.py`:

7a. Add the imports (around line 90):

```python
from werewolf_agent.runtime.nodes.sheriff_pk import (
    sheriff_pk_speech,
    sheriff_revote,
)
```

7b. Add a new route function (after `route_after_sheriff_vote`, around line 258):

```python
def route_after_sheriff_pk_speech(state: RuntimeState) -> str:
    return _route_after_sheriff_phase(state, "sheriff_revote")


def route_after_sheriff_revote(state: RuntimeState) -> str:
    """After revote, go to next phase (deaths or free_discussion)."""
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    if not _deaths_already_announced(gs):
        return "announce_deaths"
    return "free_discussion"
```

7c. Update `route_after_sheriff_vote` to route to `sheriff_pk_speech` on first tie. Replace the function (lines 251-258) with:

```python
def route_after_sheriff_vote(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    # If first tie, route to PK speech
    if gs.sheriff_tie_count == 1 and gs.sheriff_pk_candidates:
        return "sheriff_pk_speech"
    if not _deaths_already_announced(gs):
        return "announce_deaths"
    return "free_discussion"
```

7d. Register the new nodes and edges. Find the section that adds `sheriff_speech` and `sheriff_vote` (around line 336-340), and after it add:

```python
graph.add_node("sheriff_pk_speech", sheriff_pk_speech)
graph.add_node("sheriff_revote", sheriff_revote)
```

Find the section that adds edges for `sheriff_vote` (around line 399-405), and add after it:

```python
graph.add_conditional_edges("sheriff_pk_speech", route_after_sheriff_pk_speech, {
    "sheriff_revote": "sheriff_revote",
    "resolve_self_destruct": "resolve_self_destruct",
    "announce_deaths": "announce_deaths",
    "free_discussion": "free_discussion",
})
graph.add_conditional_edges("sheriff_revote", route_after_sheriff_revote, {
    "resolve_self_destruct": "resolve_self_destruct",
    "announce_deaths": "announce_deaths",
    "free_discussion": "free_discussion",
})
```

- [ ] **Step 8: Update existing test for new behavior**

The existing test `test_sheriff_vote_tie_does_not_produce_sheriff` (`tests/rules/test_rule_engine_v1.py:767-777`) tests the engine-level `resolve_sheriff_vote` and still passes (the engine itself does not decide PK — the runtime does). Keep this test as-is.

Add a new test confirming the second-tie no-election behavior (already in Step 2's test class).

- [ ] **Step 9: Run sheriff flow tests**

Run: `python -m pytest tests/runtime/test_sheriff_flow.py tests/rules/test_rule_engine_v1.py -q --tb=short`
Expected: All pass

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/core/models.py werewolf_agent/engine/sheriff.py werewolf_agent/runtime/nodes/sheriff_pk.py werewolf_agent/runtime/nodes/sheriff.py werewolf_agent/runtime/nodes/__init__.py werewolf_agent/runtime/graph.py tests/runtime/test_sheriff_flow.py
git commit -m "feat(sheriff): add PK + revote for first-tie in sheriff election"
```

---

## Execution Order (REVISED)

Recommended sequence: Task 0 (sheriff PK) → Task 1 (vote割裂) → Task 2 (vote quality) → Task 3 (seer priority) → Task 4 (seer claim guardrail) → Task 5 (solo-wolf fallback). Each task is independently committable and testable.

### Task 1: Fix vote fallback target / reason consistency (Issue 6)

**Root cause:** In `werewolf_agent/agents/player.py:780-786`, `_fallback_action` picks a fallback target from `_vote_fallback_target` and embeds it in the reason string at line 812-813. In `runtime/agent_adapter.py:962-968`, `agent_day_vote` overwrites the fallback target with the LLM's intended target — but the reason string still references the fallback target. Result: vote_target = LLM's choice, reason = fallback's choice → audit shows inconsistent layers.

**Fix:** Decouple target from reason string. Use LLM's target when valid, fallback's target only as a last resort; never mix. Add `fallback_target_used: bool` to action trace for transparency.

**Files:**
- Modify: `werewolf_agent/agents/player.py:780-823` (remove target from reason template; track fallback_used)
- Modify: `werewolf_agent/agents/schemas.py` (add `fallback_target_used` to `ActionTrace`)
- Modify: `werewolf_agent/runtime/agent_adapter.py:962-968` (consistent target propagation)
- Test: `tests/agents/test_player_agent.py` (add new test for consistency)

- [ ] **Step 1: Write failing test for fallback/reason consistency**

In `tests/agents/test_player_agent.py`, add at end of file:

```python
class TestVoteFallbackConsistency:
    """Vote fallback must not produce a reason mentioning a different target."""

    def test_fallback_reason_does_not_embed_target(self):
        """When fallback fires, reason should not contain a target ID string."""
        from werewolf_agent.agents.player import _fallback_reason
        from werewolf_agent.agents.schemas import FallbackAction

        action = FallbackAction(
            action_type="vote",
            target_id="p07",
            speech="",
            reason="结构化输出失败",
            confidence=0.0,
        )
        reason = _fallback_reason(action)
        # Reason must NOT contain any pXX target reference
        import re
        assert not re.search(r"p\d{2}", reason), (
            f"fallback reason must not embed target_id, got: {reason!r}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agents/test_player_agent.py::TestVoteFallbackConsistency::test_fallback_reason_does_not_embed_target -v`
Expected: FAIL with "fallback reason must not embed target_id, got: '...p07...'"

- [ ] **Step 3: Refactor `_fallback_reason` to drop target embedding**

In `werewolf_agent/agents/player.py`, replace lines 802-823 (the `_fallback_reason` function) with:

```python
def _fallback_reason(action: FallbackAction) -> str:
    """Return a reason that does NOT embed the target_id.

    The caller is responsible for substituting the actual target into the
    log display. This prevents the audit trail from showing "chose p07" while
    the actual vote_target is a different player.
    """
    return "结构化输出失败，按当前可见线索选择默认目标"
```

- [ ] **Step 4: Update the fallback-action caller to use simple reason**

In `werewolf_agent/agents/player.py`, replace the call site that builds the `FallbackAction` (find the existing block that calls `_fallback_action` and constructs the audit `ActionTrace` with the embedded reason). The new pattern is:

```python
fallback_action = _fallback_action(...)
# reason no longer embeds target_id; caller substitutes it in the audit log
trace = ActionTrace(
    ...,
    fallback_target_used=True,
    fallback_target_id=fallback_action.target_id,
    parsed_action=fallback_action,
    reason=_fallback_reason(fallback_action),
)
```

The key invariant: `fallback_action.target_id` and `trace.reason` are independent — modifying one does not change the other.

- [ ] **Step 5: Add `fallback_target_used` field to ActionTrace**

In `werewolf_agent/agents/schemas.py`, find the `ActionTrace` model (around line 140) and add a new field:

```python
class ActionTrace(BaseModel):
    # ... existing fields ...
    fallback_target_used: bool = False
    fallback_target_id: str | None = None
```

- [ ] **Step 6: Set fallback flags in player.py when fallback fires**

In `werewolf_agent/agents/player.py`, find the code path that constructs the audit `ActionTrace` (search for `ActionTrace(`). When `_fallback_action` is invoked, set:

```python
trace.fallback_target_used = True
trace.fallback_target_id = fallback_action.target_id
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/agents/test_player_agent.py::TestVoteFallbackConsistency -v`
Expected: PASS

- [ ] **Step 8: Run full agent test suite**

Run: `python -m pytest tests/agents/ -q --tb=short`
Expected: All pass (no regressions)

- [ ] **Step 9: Commit**

```bash
git add werewolf_agent/agents/player.py werewolf_agent/agents/schemas.py tests/agents/test_player_agent.py
git commit -m "fix(vote): decouple fallback target from reason string; add fallback_target_used flag"
```

---

### Task 2: Relax vote quality validation (Issue 5)

**Root cause:** `werewolf_agent/runtime/vote_quality.py:81-116, 135-193` requires strict basis detection via regex on `reason + speech` text. When the regex fails, the vote is rejected and retried. After 3 retries (`agents/player.py:157` `max_retries=3`), the fallback fires. 6 of 6 fallback votes in g_3528592081 are due to `vote_quality` or `empty_response` errors. The correction hint at `agents/player.py:449-458` does not include valid enum values, so retries often repeat the same mistake.

**Fix:** Default `vote_basis=fallback, seer_stance=no_claim` when the regex finds no basis pattern, rather than rejecting. Augment correction hint with valid enum values.

**Files:**
- Modify: `werewolf_agent/runtime/vote_quality.py:135-193` (`validate_structured_vote_action`)
- Modify: `werewolf_agent/agents/player.py:449-458` (`_vote_quality_error` correction hint)
- Test: `tests/runtime/test_vote_quality.py` (add new test for relaxed validation)

- [ ] **Step 1: Locate `validate_structured_vote_action` and inspect the basis check**

Run: `grep -n "validate_structured_vote_action\|validate_vote_reason" werewolf_agent/runtime/vote_quality.py`
Expected output shows the function around line 135-193.

Read lines 140-200 to see the current basis check logic.

- [ ] **Step 2: Write failing test for relaxed basis detection**

In `tests/runtime/test_vote_quality.py`, add at end of file (or create if it doesn't exist):

```python
class TestValidateStructuredVoteAction:
    def test_missing_basis_defaults_to_fallback_not_error(self):
        """When basis regex finds nothing, default to fallback basis (no error)."""
        from werewolf_agent.runtime.vote_quality import validate_structured_vote_action

        action = {
            "action_type": "vote",
            "target_id": "p07",
            "speech": "我跟p07的票",
            "reason": "我没看出什么明显理由",
            "confidence": 0.5,
            "seer_stance": "undecided",
            "vote_basis": "fallback",
            "standing_with_seer": "",
            "suspect_reason": "",
            "not_voting_reason": "",
            "private_reason": "保守票",
        }
        # Should NOT raise; should return action unchanged or with default basis
        result = validate_structured_vote_action(action)
        assert result.get("vote_basis") in ("fallback", "speech_logic")
        assert result.get("seer_stance") in ("undecided", "no_claim")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/runtime/test_vote_quality.py::TestValidateStructuredVoteAction::test_missing_basis_defaults_to_fallback_not_error -v`
Expected: FAIL with current strict validation

- [ ] **Step 4: Update `validate_structured_vote_action` to default basis instead of erroring**

In `werewolf_agent/runtime/vote_quality.py`, find the section that calls `validate_vote_reason` (around line 150-180). Wrap the basis check to default rather than raise:

```python
# Old (likely):
basis_pattern = validate_vote_reason(reason, speech)
if not basis_pattern:
    raise ValueError("投票依据缺乏具体逻辑依据")

# New:
basis_pattern = validate_vote_reason(reason, speech)
if not basis_pattern:
    # Default to fallback basis rather than rejecting the vote
    action["vote_basis"] = "fallback"
    action["seer_stance"] = action.get("seer_stance") or "no_claim"
elif action.get("vote_basis") not in VALID_VOTE_BASIS_VALUES:
    # If LLM provided a value not in enum, normalize to detected basis
    action["vote_basis"] = "fallback"
```

Add a module-level constant near the top of the file (or in `agents/schemas.py`):

```python
VALID_VOTE_BASIS_VALUES = {
    "seer_check", "counterclaim", "speech_logic", "vote_history",
    "self_claim", "fallback", "no_action",
}
```

Verify the same set is used by the Pydantic enum in `werewolf_agent/agents/schemas.py` (search for `vote_basis` enum) — if the Pydantic enum differs, update it to match.

- [ ] **Step 5: Augment correction hint with valid enum values**

In `werewolf_agent/agents/player.py:449-458`, find `_vote_quality_error`. Update the hint string to include the enum values:

```python
def _vote_quality_error(...) -> str:
    # ... existing logic ...
    return (
        "投票依据缺乏具体逻辑依据。需要包含：怀疑对象、行为变化、PK发言、之前投票行为等。"
        f"有效 vote_basis 值: {sorted(VALID_VOTE_BASIS_VALUES)}。"
        f"有效 seer_stance 值: ['support', 'oppose', 'undecided', 'no_claim']。"
        "请重新生成。"
    )
```

Import `VALID_VOTE_BASIS_VALUES` from `vote_quality`.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/runtime/test_vote_quality.py::TestValidateStructuredVoteAction -v`
Expected: PASS

- [ ] **Step 7: Run full vote quality test suite**

Run: `python -m pytest tests/runtime/test_vote_quality.py tests/agents/test_player_agent.py -q --tb=short`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add werewolf_agent/runtime/vote_quality.py werewolf_agent/agents/player.py werewolf_agent/agents/schemas.py tests/runtime/test_vote_quality.py
git commit -m "fix(vote): default vote_basis to fallback when no basis detected; enrich retry hint with enum values"
```

---

### Task 3: Wire claimed-Seer priority into wolf kill selection (Issue 4)

**Root cause:** `werewolf_agent/runtime/strategy/wolf.py:13-129, 145-153` defines `has_publicly_claimed_seer()` and `evaluate_wolf_kill_target()` (which scores claimed-seer at +6), but **neither is wired into the wolf kill consensus flow as an explicit directive**. Wolves during private discussion see only a generic "优先击杀对狼队威胁最大的玩家" prompt (`runtime/agent_adapter.py:447-452`) and never get a concrete player ID.

In g_3528592081, real Seer p03 publicly claimed Day 1 and led the good team. Wolves p01/p02/p07/p08 failed to identify p03 as the real Seer across 3 nights and instead killed p09 (idiot), p06 (villager), p10 (villager).

**Fix:** After `_evaluate_wolf_kill_target()` in `runtime/agent_adapter.py:444`, detect any publicly-claimed Seer and inject `wolf_high_priority_target` into the strategy_directive for the wolf discussion and kill prompts.

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py:444-470` (`_single_wolf_vote` and prompt construction)
- Test: `tests/runtime/test_strategy_directives.py` (add test for claimed-seer injection)

- [ ] **Step 1: Find `_single_wolf_vote` and `_evaluate_wolf_kill_target` callsites**

Run: `grep -n "_single_wolf_vote\|_evaluate_wolf_kill_target\|has_publicly_claimed_seer" werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/nodes/_shared.py`
Expected: shows `_single_wolf_vote` around line 444, importing `evaluate_wolf_kill_target` from `strategy/wolf`.

- [ ] **Step 2: Write failing test for claimed-seer injection**

In `tests/runtime/test_strategy_directives.py`, add at end of file:

```python
class TestWolfSeerPriorityInjection:
    def test_claimed_seer_appears_in_wolf_kill_directive(self):
        """When a player has publicly claimed Seer, the wolf kill prompt must name them explicitly."""
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer
        from werewolf_agent.runtime.agent_adapter import _build_wolf_kill_directive

        # Build minimal gs with p03 having publicly claimed Seer (via speech event)
        gs = GameState(
            game_id="g_test",
            players={
                "p03": PlayerState(player_id="p03", role="seer", alive=True),
                "p05": PlayerState(player_id="p05", role="villager", alive=True),
            },
            events=[
                {"type": "speech", "payload": {"speaker": "p03", "text": "我是预言家，昨晚查了p05是好人"}},
            ],
        )
        # Simulate that p03 has publicly claimed
        assert has_publicly_claimed_seer(gs, "p03") is True

        directive = _build_wolf_kill_directive(gs, wolf_id="p01", plan={})
        assert "p03" in str(directive), (
            f"wolf kill directive should explicitly name claimed Seer p03, got: {directive}"
        )
```

- [ ] **Step 3: Run test to verify it fails (or skip if `_build_wolf_kill_directive` doesn't exist)**

If the function does not exist yet, this test fails at the import. Create the function shell in `runtime/agent_adapter.py`:

```python
def _build_wolf_kill_directive(gs, *, wolf_id, plan):
    """Build a directive for wolf kill selection that names the highest-priority target."""
    from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer

    parts = []
    # Find any publicly-claimed Seer
    for pid, p in gs.players.items():
        if p.alive and p.role != "werewolf" and has_publicly_claimed_seer(gs, pid):
            parts.append(
                f"高优先级击杀目标: {pid} — 该玩家已公开跳预言家，必须优先击杀"
            )
    if not parts:
        # No publicly-claimed Seer — fall back to evaluate_wolf_kill_target
        from werewolf_agent.runtime.strategy.wolf import evaluate_wolf_kill_target
        scores = evaluate_wolf_kill_target(gs)
        for pid, score in sorted(scores.items(), key=lambda x: -x[1])[:3]:
            parts.append(f"击杀候选: {pid} (评分 {score})")
    return "\n".join(parts)
```

Run: `python -m pytest tests/runtime/test_strategy_directives.py::TestWolfSeerPriorityInjection -v`
Expected: PASS (after creating the function shell)

- [ ] **Step 4: Wire `_build_wolf_kill_directive` into the wolf discussion and kill prompts**

In `werewolf_agent/runtime/agent_adapter.py`, find the prompt construction for wolf discussion (around line 543-562) and wolf kill (around line 444-452). In both, before the existing prompt, insert:

```python
from werewolf_agent.runtime.agent_adapter import _build_wolf_kill_directive
directive = _build_wolf_kill_directive(gs, wolf_id=wolf_id, plan=plan)
state_with_directive = {**state, "wolf_high_priority_target": directive}
```

Then in the prompt string, add at the top:

```python
prompt += f"\n\n## 击杀优先级\n{state_with_directive['wolf_high_priority_target']}\n"
```

- [ ] **Step 5: Run wolf strategy tests**

Run: `python -m pytest tests/runtime/test_strategy_directives.py -q --tb=short`
Expected: All pass (including the new test)

- [ ] **Step 6: Run full runtime tests**

Run: `python -m pytest tests/runtime/ -q --tb=short`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/runtime/agent_adapter.py tests/runtime/test_strategy_directives.py
git commit -m "feat(wolf): inject claimed-Seer kill priority into wolf discussion and kill prompts"
```

---

### Task 4: Add information-consistency guardrail for fake Seer (Issue 3)

**Root cause:** In g_3528592081, fake Seer p08 publicly claimed "I checked p04 and p09 last night (N1)" — but the Seer rule allows only 1 check per night. This is a **rule-violation leak**, not a vocabulary issue. The strategy at `werewolf_agent/runtime/directives/wolf.py:13-22` does not remind the agent of the "1 check per night" constraint, and there is no post-generation check on the public `speech` for impossible claims.

**Fix:** Add a `validate_seer_claim` post-check on the public `speech` field. If the speech claims multiple "I checked X" in the same night, trigger a retry. Also strengthen the fake-seer strategy prompt to remind the agent of the 1-check-per-night rule.

**Files:**
- Create: `werewolf_agent/runtime/seer_claim_validator.py` (new module)
- Modify: `werewolf_agent/runtime/directives/wolf.py:13-22` (add 1-check constraint reminder)
- Modify: `werewolf_agent/runtime/agent_adapter.py` (call validator after wolf speech generation)
- Test: `tests/runtime/test_seer_claim_validator.py` (new test file)

- [ ] **Step 1: Locate the wolf public speech dispatch site**

Run: `grep -n "agent_wolf_speech\|wolf.*speech\|action_type.*speech" werewolf_agent/runtime/agent_adapter.py | head -20`
Expected: shows the dispatch site for wolf public speeches. This is where we will inject the post-check.

Read the function body to see the existing retry pattern (look for `retry_count`, `max_retries`, or a `correction_hint` field) — the validator hook must follow the same shape.

- [ ] **Step 2: Create the validator module**

Create file `werewolf_agent/runtime/seer_claim_validator.py`:

```python
"""Validate that public seer claims respect the 1-check-per-night rule."""

from __future__ import annotations

import re
from typing import Any

# Match "我第N夜查了X" / "昨晚我验了X" / "我在第N夜验了X是好人/狼人"
# Captures the night number and the target player ID.
_CLAIM_PATTERN = re.compile(
    r"我[在第]?\s*(\d+)\s*夜[查验过]*了?\s*(?:是\s*)?(?:好人|狼人)?[，,。]?\s*(p\d{2})"
    r"|昨[晚天]\s*(?:我\s*)?[查验过]*了?\s*(p\d{2})",
    re.IGNORECASE,
)


def extract_seer_claims(speech: str) -> list[dict[str, Any]]:
    """Extract seer-style claims from a public speech.

    Returns a list of dicts with keys: night (int | None), target_id (str).
    """
    if not speech:
        return []
    claims: list[dict[str, Any]] = []
    for m in _CLAIM_PATTERN.finditer(speech):
        night_str, target_with_night, target_no_night = m.groups()
        if target_with_night:
            claims.append({
                "night": int(night_str) if night_str else None,
                "target_id": target_with_night,
            })
        elif target_no_night:
            claims.append({"night": None, "target_id": target_no_night})
    return claims


def validate_seer_claim(speech: str, day_number: int) -> str | None:
    """Return an error message if the speech violates seer claim rules.

    Rules:
    - At most 1 check claim per night.
    - Claims about future nights (night > day_number) are forbidden.
    - Claiming checks in night 0 (pre-game) is forbidden.
    """
    claims = extract_seer_claims(speech)
    if not claims:
        return None

    # Group by night
    by_night: dict[int | None, list[str]] = {}
    for c in claims:
        by_night.setdefault(c["night"], []).append(c["target_id"])

    # Check 1-check-per-night
    for night, targets in by_night.items():
        if night is not None and len(targets) > 1:
            return (
                f"公开发言中第{night}夜声称查验了 {len(targets)} 人，"
                f"违反预言家一夜只查一人的规则: {', '.join(targets)}"
            )

    # Check no future-night claims
    for night in by_night:
        if night is not None and night > day_number:
            return (
                f"公开发言中声称第{night}夜查验，但当前是D{day_number}，"
                "未来夜晚的查验结果不可能已知"
            )

    # Check night 0 claims
    if 0 in by_night:
        return "公开发言中声称第0夜查验，游戏开始前不可能有查验"

    return None
```

- [ ] **Step 2: Write tests for the validator**

Create file `tests/runtime/test_seer_claim_validator.py`:

```python
"""Tests for seer claim validation (1-check-per-night rule)."""

from werewolf_agent.runtime.seer_claim_validator import (
    extract_seer_claims,
    validate_seer_claim,
)


class TestExtractSeerClaims:
    def test_single_claim_n1(self):
        claims = extract_seer_claims("我在第1夜查了p04是好人")
        assert len(claims) == 1
        assert claims[0]["night"] == 1
        assert claims[0]["target_id"] == "p04"

    def test_two_claims_same_night(self):
        claims = extract_seer_claims("我第1夜查了p04是村民，也查了p09是村民")
        assert len(claims) == 2
        assert all(c["night"] == 1 for c in claims)


class TestValidateSeerClaim:
    def test_single_claim_passes(self):
        assert validate_seer_claim("我查了p04是好人", day_number=1) is None

    def test_two_claims_same_night_fails(self):
        err = validate_seer_claim(
            "我第1夜查了p04是村民，也查了p09是村民", day_number=1
        )
        assert err is not None
        assert "违反" in err

    def test_future_night_claim_fails(self):
        err = validate_seer_claim("我第3夜查了p05是狼人", day_number=1)
        assert err is not None
        assert "未来" in err or "不可能" in err

    def test_night_zero_claim_fails(self):
        err = validate_seer_claim("我第0夜查了p05是狼人", day_number=1)
        assert err is not None

    def test_no_claim_passes(self):
        assert validate_seer_claim("我觉得p04的发言有矛盾", day_number=1) is None
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/runtime/test_seer_claim_validator.py -v`
Expected: All pass (the module is new and self-contained)

- [ ] **Step 4: Wire validator into wolf public speech generation**

In `werewolf_agent/runtime/agent_adapter.py`, find where the wolf's public `speech` is finalized (search for `agent_wolf_speech` or similar; also look at `sheriff_speech` dispatch for wolves). Add a post-check after the LLM returns the action:

```python
from werewolf_agent.runtime.seer_claim_validator import validate_seer_claim

# After receiving action from LLM, before returning:
if action.get("action_type") == "speech" and action.get("speech"):
    err = validate_seer_claim(action["speech"], day_number=gs.day_number)
    if err:
        # Trigger a structured retry with the error as correction hint
        return {
            "error": err,
            "correction_hint": err,
            "retry": True,
        }
```

The exact retry wiring depends on the existing dispatch pattern. Match the retry style used in `_vote_quality_error` (Task 2).

- [ ] **Step 5: Strengthen the fake-seer strategy prompt**

In `werewolf_agent/runtime/directives/wolf.py:13-22`, update the `fake_seer` strategy string to add:

```python
"fake_seer": (
    "你是悍跳狼（假预言家）。白天发言策略：\n"
    "1) 跳预言家，报出你的'验人结果'。必须像真预言家一样有验人逻辑链\n"
    "2) **关键规则**：预言家一夜只能查验 1 人，绝对不能声称同一夜查验多人\n"
    "3) 如果场上有真预言家跳了，你必须对跳——质疑真预言家的验人逻辑和警徽流\n"
    "4) 报验人的标准格式：'我在第X夜验了[玩家]，结果是[好人/狼人]'\n"
    "5) 你可以报一个好人做金水来拉拢，或者报一个好人查杀来推人\n"
    "6) 不要暴露你的队友——你'验出'的好人可以是你的队友（假金水）\n"
    "7) 如果真预言家验出了狼（你的队友），你必须质疑该验人结果的可信度\n"
    "8) 不要慌张——对跳预言家是正常游戏行为，保持自信和逻辑连贯"
),
```

- [ ] **Step 6: Run seer claim tests + wolf directive tests**

Run: `python -m pytest tests/runtime/test_seer_claim_validator.py tests/runtime/test_strategy_directives.py -q --tb=short`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/runtime/seer_claim_validator.py werewolf_agent/runtime/directives/wolf.py werewolf_agent/runtime/agent_adapter.py tests/runtime/test_seer_claim_validator.py
git commit -m "feat(guardrail): validate 1-check-per-night rule for fake Seer public claims"
```

---

### Task 5: Add solo-wolf fallback target heuristic (Issue 2)

**Root cause:** In g_3528592081 N3, only p02 (wolf) was alive. The wolf plan's `night_kill_primary` was `None` (no discussion evidence) and `evidence_quality` was `"none"`, so `_planned_wolf_kill()` (`runtime/nodes/_shared.py:592-616`) returned `None`. The legacy fallback `_legacy_wolf_consensus` (line 59-169) ran the agent directly, but with no strategic context, the agent picked p10 (villager) instead of the real Seer p03.

**Fix:** In `_build_wolf_team_plan` (`runtime/nodes/_shared.py:545-580`), when there is 1 alive wolf, set a default `night_kill_primary` from `day_push_target` or the publicly-claimed Seer (using `has_publicly_claimed_seer` from `runtime/strategy/wolf.py`). This ensures solo wolves always have a strategic kill target even without team discussion.

**Files:**
- Modify: `werewolf_agent/runtime/nodes/_shared.py:545-580` (`_build_wolf_team_plan`)
- Test: `tests/runtime/test_wolf_flow.py` (add solo-wolf fallback test)

- [ ] **Step 1: Locate `_build_wolf_team_plan` and inspect**

Run: `grep -n "_build_wolf_team_plan\|has_publicly_claimed_seer" werewolf_agent/runtime/nodes/_shared.py werewolf_agent/runtime/strategy/wolf.py`
Expected: shows `_build_wolf_team_plan` at line 545-580 in `_shared.py`, and `has_publicly_claimed_seer` in `strategy/wolf.py`.

- [ ] **Step 2: Write failing test for solo-wolf default target**

In `tests/runtime/test_wolf_flow.py`, add at end of file:

```python
class TestSoloWolfFallbackTarget:
    def test_solo_wolf_default_targets_claimed_seer(self):
        """When only 1 wolf is alive and no plan, default kill target = claimed Seer."""
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.nodes._shared import _build_wolf_team_plan

        gs = GameState(
            game_id="g_test",
            players={
                "p02": PlayerState(player_id="p02", role="werewolf", alive=True),
                "p03": PlayerState(player_id="p03", role="seer", alive=True),
                "p05": PlayerState(player_id="p05", role="villager", alive=True),
            },
            night_number=3,
            day_number=2,
            events=[
                {"type": "speech", "payload": {"speaker": "p03", "text": "我是预言家"}},
            ],
        )
        plan = _build_wolf_team_plan(gs, previous_plan=None)
        # Solo wolf (1 alive) should default to claimed Seer p03
        assert plan.get("night_kill_primary") == "p03", (
            f"expected default target p03 (claimed seer), got {plan.get('night_kill_primary')}"
        )

    def test_solo_wolf_no_claimed_seer_uses_day_push(self):
        """When no claimed Seer, default to day_push_target from previous plan."""
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.nodes._shared import _build_wolf_team_plan

        gs = GameState(
            game_id="g_test",
            players={
                "p02": PlayerState(player_id="p02", role="werewolf", alive=True),
                "p05": PlayerState(player_id="p05", role="villager", alive=True),
            },
            night_number=3,
            day_number=2,
            events=[],
        )
        prev_plan = {"day_push_target": "p05", "night_kill_primary": None}
        plan = _build_wolf_team_plan(gs, previous_plan=prev_plan)
        # No claimed Seer → use day_push_target
        assert plan.get("night_kill_primary") == "p05"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/runtime/test_wolf_flow.py::TestSoloWolfFallbackTarget -v`
Expected: FAIL (current logic doesn't set default)

- [ ] **Step 4: Update `_build_wolf_team_plan` to set default target for solo wolves**

In `werewolf_agent/runtime/nodes/_shared.py:545-580`, modify the function to add solo-wolf fallback logic. Replace the function body (after the `wolves = _alive_wolves(gs)` line) with:

```python
def _build_wolf_team_plan(
    gs: GameState,
    *,
    previous_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wolves = _alive_wolves(gs)
    if not wolves:
        return {}

    assignments = list(wolves)
    roles = {
        "fake_seer": assignments[0] if len(assignments) > 0 else None,
        "pusher": assignments[1] if len(assignments) > 1 else None,
        "hooker": assignments[2] if len(assignments) > 2 else None,
        "deep_cover": assignments[3] if len(assignments) > 3 else None,
    }
    previous_plan = previous_plan or {}
    can_reuse_previous = previous_plan.get("evidence_quality") not in (None, "none")
    primary = _first_alive_target(gs, previous_plan.get("night_kill_primary")) if can_reuse_previous else None
    backup = _first_alive_target(gs, previous_plan.get("night_kill_backup")) if can_reuse_previous else None
    day_push = _first_alive_target(gs, previous_plan.get("day_push_target")) if can_reuse_previous else None

    # Solo-wolf fallback: when only 1 wolf is alive and no inherited primary,
    # default to publicly-claimed Seer (highest strategic value), then day_push.
    if len(wolves) == 1 and not primary:
        from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer
        claimed_seer_target = None
        for pid, p in gs.players.items():
            if p.alive and p.role != "werewolf" and has_publicly_claimed_seer(gs, pid):
                claimed_seer_target = pid
                break
        if claimed_seer_target:
            primary = claimed_seer_target
        elif day_push:
            primary = day_push

    return {
        "night_number": gs.night_number,
        **roles,
        "night_kill_primary": primary,
        "night_kill_backup": backup,
        "day_push_target": day_push,
        "evidence_from_discussion": previous_plan.get("evidence_from_discussion", []),
        "evidence_quality": previous_plan.get("evidence_quality", "none") if can_reuse_previous else "none",
        "public_story": "警上制造预言家对立，冲锋位打抗推目标，倒钩位保留质疑队友空间，深水位做中立复盘。",
        "hooking_intent": {
            "player_id": roles.get("hooker"),
            "policy": "可以轻踩或投票队友换取好人信任，但公开文本必须表现为独立逻辑判断。",
        },
    }
```

- [ ] **Step 5: Run solo-wolf tests to verify they pass**

Run: `python -m pytest tests/runtime/test_wolf_flow.py::TestSoloWolfFallbackTarget -v`
Expected: PASS

- [ ] **Step 6: Run full wolf flow tests**

Run: `python -m pytest tests/runtime/test_wolf_flow.py tests/runtime/test_night_flow.py -q --tb=short`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/runtime/nodes/_shared.py tests/runtime/test_wolf_flow.py
git commit -m "feat(wolf): solo-wolf defaults to claimed Seer when no plan evidence"
```

---

## Execution Order

Recommended sequence: Task 1 (vote割裂) → Task 2 (vote quality) → Task 3 (seer priority) → Task 4 (seer claim guardrail) → Task 5 (solo-wolf fallback). Each task is independently committable and testable.

Tasks 1+2 share the vote-fallback module — review both together.
Tasks 3+4 both modify wolf agent behavior — review both together.
Task 5 is solo-wolf specific — independent.

## Verification (end of plan)

After all tasks complete, run the full test suite to confirm no regressions:

```bash
python -m pytest tests/ -q --tb=short
```

Then run a fresh end-to-end game simulation (existing `tests/integration/test_live_game_flow.py`) to confirm the fixes work in practice:

```bash
python -m pytest tests/integration/test_live_game_flow.py -q --tb=short
```

If all pass, update PROGRESS.md to mark all 5 issues resolved.

## Open Risks

- **Issue 0 (sheriff PK)**: NOT addressed in this plan. Requires design doc + CLAUDE.md clarification before any code change. Flag for user.
- **Task 1 + 2 interaction**: Both modify vote_quality / fallback. If both are merged in one PR, the interaction should be tested end-to-end (vote where LLM returns target but fails quality — fallback's target vs LLM's target).
- **Task 3 + 4 + 5 all modify wolf agent**: Each is in a different file but the wolf's prompt context accumulates directives. Test that a wolf receiving all directives at once does not exceed prompt budget (rough check: 6 directive strings × ~200 chars = 1200 chars added).
- **Backward compat**: All 5 fixes are additive. No existing tests should break. If any do, the test is checking outdated behavior — discuss with user before updating.
