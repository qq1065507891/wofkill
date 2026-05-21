# Game Balance Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove runtime boundary leaks that made recent games over-favor wolves, then add a public information ledger so good-side agents can reference claims, checks, votes, last words, and badge flow without hidden information.

**Architecture:** Keep fixes surgical. Runtime graph nodes own transient flow flags, while wolf strategy owns discussion-derived targets. Public information must be derived only from public `GameEvent` payloads and exposed through `visible_world_state`; it must never inspect real roles, private night actions, wolf chat, moderator-only audits, or hidden rule-engine state.

**Tech Stack:** Python, pytest, existing `werewolf_agent.runtime` graph, strategy, visible-state, and cognition extraction modules.

---

### Task 1: Reset Day Increment Marker

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [x] **Step 1: Write the failing test**

Add `test_announce_deaths_resets_first_day_increment_marker` to verify `announce_deaths()` returns `day_number_already_incremented=False` after using the first-day pre-increment path.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\runtime\test_runtime.py::test_announce_deaths_resets_first_day_increment_marker -q --basetemp=.pytest-tmp`

Expected before fix: fails with missing `day_number_already_incremented`.

- [x] **Step 3: Write minimal implementation**

Return `day_number_already_incremented=False` from `announce_deaths()`.

- [x] **Step 4: Run test to verify it passes**

Run the same targeted pytest command.

### Task 2: Drop Stale Wolf Targets

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/runtime/wolf_strategy.py`
- Test: `tests/runtime/test_runtime.py`

- [x] **Step 1: Write the failing test**

Add `test_wolf_discussion_drops_stale_targets_without_current_discussion_evidence` to verify a new night with vague wolf discussion does not reuse old `night_kill_primary`, `night_kill_backup`, or `day_push_target`.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\runtime\test_runtime.py::test_wolf_discussion_drops_stale_targets_without_current_discussion_evidence -q --basetemp=.pytest-tmp`

Expected before fix: fails because old target `v1` is reused.

- [x] **Step 3: Write minimal implementation**

Limit graph static fallback to wolf role assignments and public story. Remove previous-plan fallback for `day_push_target` after a new discussion starts.

- [x] **Step 4: Run related tests**

Run wolf discussion and wolf strategy tests before full-suite verification.

---

## Phase 2: Public Information Ledger

**Design boundary:** The ledger is a public memory aid, not a truth oracle. It records what players publicly said or did, not whether those statements are true. For example, a seer check claim records `speaker=p03 target=p08 result=wolf claimed=true`; it must not compare that claim to p08's actual role.

**Files:**
- Create: `werewolf_agent/runtime/public_ledger.py`
- Modify: `werewolf_agent/runtime/visible_state.py`
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_public_ledger.py`
- Test: `tests/runtime/test_visible_state.py`

### Task 3: Define Public Ledger Schema And Builder

**Files:**
- Create: `werewolf_agent/runtime/public_ledger.py`
- Test: `tests/runtime/test_public_ledger.py`

- [x] **Step 1: Write failing tests for ledger extraction**

Create `tests/runtime/test_public_ledger.py` with tests that build a `GameState(events=[...])` and call `build_public_ledger(gs)`.

Required test cases:

```python
def test_public_ledger_extracts_role_claims_check_claims_and_badge_flow() -> None:
    gs = GameState(events=[
        GameEvent(type="speech", payload={
            "speaker": "p03",
            "day_number": 1,
            "text": "我是预言家，昨晚验p08查杀，警徽流p05 p07",
        }),
    ])

    ledger = build_public_ledger(gs)

    assert ledger["role_claims"] == [
        {"day": 1, "speaker": "p03", "role": "seer", "source_event": "speech"}
    ]
    assert ledger["seer_check_claims"] == [
        {"day": 1, "speaker": "p03", "target": "p08", "result": "wolf", "source_event": "speech"}
    ]
    assert ledger["badge_flow_claims"] == [
        {"day": 1, "speaker": "p03", "targets": ["p05", "p07"], "source_event": "speech"}
    ]
```

```python
def test_public_ledger_ignores_private_and_moderator_only_events() -> None:
    gs = GameState(events=[
        GameEvent(type="wolf_discussion", payload={
            "wolf_id": "p01",
            "text": "刀p08",
            "visibility": "werewolf_team_only",
        }),
        GameEvent(type="action_trace_audit", payload={
            "player_id": "p02",
            "visibility": "moderator_only",
            "action_trace": {"parsed_action": {"true_role": "seer"}},
        }),
    ])

    ledger = build_public_ledger(gs)

    assert ledger["role_claims"] == []
    assert ledger["seer_check_claims"] == []
    assert ledger["vote_records"] == []
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests\runtime\test_public_ledger.py -q --basetemp=.pytest-tmp`

Expected: FAIL because `werewolf_agent.runtime.public_ledger` does not exist.

- [x] **Step 3: Implement minimal builder**

Create `werewolf_agent/runtime/public_ledger.py` with:

```python
from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.cognition.world_state import _infer_claims_from_text

PUBLIC_EVENT_TYPES = {
    "speech",
    "sheriff_speech",
    "tie_pk_speech",
    "exile_last_words",
    "vote_resolved",
    "badge_transferred",
    "badge_torn",
    "judge_broadcast",
}

PRIVATE_VISIBILITIES = {"werewolf_team_only", "moderator_only", "seer_private", "witch_private"}


def build_public_ledger(game_state: GameState) -> dict[str, list[dict[str, Any]]]:
    ledger: dict[str, list[dict[str, Any]]] = {
        "role_claims": [],
        "seer_check_claims": [],
        "badge_flow_claims": [],
        "vote_records": [],
        "last_words": [],
        "badge_events": [],
    }
    for event in game_state.events:
        if not _is_public_event(event):
            continue
        if event.type in {"speech", "sheriff_speech", "tie_pk_speech", "exile_last_words"}:
            _add_speech_items(ledger, event)
        elif event.type == "vote_resolved":
            _add_vote_items(ledger, event, game_state)
        elif event.type in {"badge_transferred", "badge_torn"}:
            _add_badge_item(ledger, event, game_state)
    return ledger
```

Implementation notes:
- Reuse `_infer_claims_from_text()` so claim parsing stays consistent with cognition code.
- Keep output as plain dictionaries because `visible_world_state` is already JSON-like.
- Include only player ids, claimed roles/results, text snippets, day numbers, and source event type.
- Do not import `RuleEngine`, inspect `PlayerState.role`, or read private action traces.

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests\runtime\test_public_ledger.py -q --basetemp=.pytest-tmp`

Expected: PASS.

### Task 4: Add Vote Reasons And Last Words To Public Ledger

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/runtime/public_ledger.py`
- Test: `tests/runtime/test_public_ledger.py`
- Test: `tests/runtime/test_runtime.py`

- [x] **Step 1: Write failing tests for vote and last-word visibility**

Add tests that verify:
- `vote_resolved` contributes a stable `vote_records` list with each voter, target, day, and optional reason.
- `exile_last_words` contributes `last_words` with speaker, day, and text.

Example expected shape:

```python
assert ledger["vote_records"] == [
    {"day": 2, "voter": "p01", "target": "p08", "reason": "跟预言家查杀", "source_event": "vote_resolved"}
]
assert ledger["last_words"] == [
    {"day": 2, "speaker": "p08", "text": "我不是狼，重点看p03。", "source_event": "exile_last_words"}
]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests\runtime\test_public_ledger.py tests\runtime\test_runtime.py::test_day_vote_records_vote_reasons_for_public_ledger -q --basetemp=.pytest-tmp`

Expected: FAIL because `vote_resolved` currently stores only exile result, not per-voter public records.

- [x] **Step 3: Store public vote records in `vote_resolved`**

In `werewolf_agent/runtime/graph.py`, extend the `vote_resolved` payload built in `day_vote_result()`:

```python
payload: dict[str, Any] = {
    "exiled": result.exiled_player_id,
    "reason": result.reason,
    "day_number": gs.day_number,
    "votes": [
        {
            "voter": voter_id,
            "target": target_id,
            "reason": _public_vote_reason((state.get("vote_action_traces") or {}).get(voter_id)),
        }
        for voter_id, target_id in sorted((state.get("exile_votes") or {}).items())
    ],
}
```

Add a small helper near `_action_trace_event()`:

```python
def _public_vote_reason(action_trace: dict[str, Any] | None) -> str:
    if not action_trace:
        return ""
    parsed = action_trace.get("parsed_action") or {}
    reason = parsed.get("reason") or action_trace.get("reason") or ""
    return str(reason)[:200]
```

Do not copy private intent, true role, chain-of-thought, or full action trace into the public payload.

- [x] **Step 4: Implement vote and last-word extraction**

In `public_ledger.py`, parse:
- `vote_resolved.payload["votes"]` into `ledger["vote_records"]`
- `exile_last_words` speech events into `ledger["last_words"]`

- [x] **Step 5: Run tests**

Run: `python -m pytest tests\runtime\test_public_ledger.py tests\runtime\test_runtime.py::test_day_vote_records_vote_reasons_for_public_ledger -q --basetemp=.pytest-tmp`

Expected: PASS.

### Task 5: Expose Ledger In Visible Player State

**Files:**
- Modify: `werewolf_agent/runtime/visible_state.py`
- Test: `tests/runtime/test_visible_state.py`

- [x] **Step 1: Write failing visible-state tests**

Add tests verifying:
- `build_visible_player_state(gs)["public_ledger"]` exists.
- A villager and a werewolf receive the same public ledger through `build_agent_context()`.
- The ledger excludes wolf chat, seer private checks, witch private info, and moderator-only action traces.

Example:

```python
def test_visible_state_includes_public_ledger() -> None:
    gs = GameState(events=[
        GameEvent(type="speech", payload={
            "speaker": "p03",
            "day_number": 1,
            "text": "我是预言家，验p08查杀",
        }),
    ])

    visible = build_visible_player_state(gs)

    assert visible["public_ledger"]["role_claims"][0]["speaker"] == "p03"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests\runtime\test_visible_state.py -q --basetemp=.pytest-tmp`

Expected: FAIL because `public_ledger` is not yet exposed.

- [x] **Step 3: Add ledger to `build_visible_player_state()`**

Modify `werewolf_agent/runtime/visible_state.py`:

```python
from werewolf_agent.runtime.public_ledger import build_public_ledger


def build_visible_player_state(game_state: GameState) -> dict[str, Any]:
    return {
        ...
        "public_ledger": build_public_ledger(game_state),
    }
```

- [x] **Step 4: Run visible-state tests**

Run: `python -m pytest tests\runtime\test_visible_state.py -q --basetemp=.pytest-tmp`

Expected: PASS.

### Task 6: Prevent Public Ledger Information Leaks

**Files:**
- Test: `tests/runtime/test_public_ledger.py`
- Test: `tests/integration/test_e2e_info_leak.py`

- [x] **Step 1: Add explicit leak-regression tests**

Add tests covering:
- A real seer `seer_check` event with private visibility is not included unless the seer publicly claimed it in speech.
- `wolf_team_plan` and `wolf_discussion` never appear in public ledger.
- `action_trace_audit.parsed_action.private_intent.true_role` never appears in public ledger values.

Example:

```python
def test_public_ledger_does_not_expose_real_seer_check() -> None:
    gs = GameState(events=[
        GameEvent(type="seer_check", payload={
            "seer_id": "p03",
            "target_id": "p08",
            "alignment": "werewolf",
            "visibility": "seer_private",
        }),
    ])

    ledger = build_public_ledger(gs)

    assert ledger["seer_check_claims"] == []
```

- [x] **Step 2: Run leak tests**

Run: `python -m pytest tests\runtime\test_public_ledger.py tests\integration\test_e2e_info_leak.py -q --basetemp=.pytest-tmp`

Expected: PASS after implementation.

### Task 7: Final Verification

**Files:**
- No new files.

- [x] **Step 1: Run targeted runtime tests**

Run: `python -m pytest tests\runtime\test_public_ledger.py tests\runtime\test_visible_state.py tests\runtime\test_runtime.py -q --basetemp=.pytest-tmp`

Expected: PASS.

- [x] **Step 2: Run full suite**

Run: `python -m pytest -q --basetemp=.pytest-tmp`

Expected: PASS with the existing skipped tests unchanged.

- [x] **Step 3: Manual review checklist**

Verify in diff:
- `public_ledger.py` reads only `GameState.events`.
- No code path reads `PlayerState.role` to decide public claims.
- No `action_trace_audit`, `wolf_discussion`, `wolf_team_plan`, `seer_check`, or `witch_decision_audit` payload appears in public ledger unless converted through public speech text.
- `visible_world_state["public_ledger"]` is identical for all players before role-specific private fields are added.
