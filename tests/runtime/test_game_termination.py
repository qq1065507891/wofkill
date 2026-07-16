# -*- coding: utf-8 -*-
"""
验证游戏终态转换、中止事件与安全应急产物。

作者: Project contributors
创建日期: 2026-07-16
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.game_termination import (
    abort_game,
    finish_game,
    validate_aborted_game,
    write_emergency_abort,
)


def test_finish_game_requires_winner_and_makes_terminal_state_immutable() -> None:
    with pytest.raises(ValueError, match="winner"):
        finish_game(GameState(game_id="g-no-winner", phase="finished"))

    finished = finish_game(GameState(
        game_id="g-finished", phase="finished", winning_faction="good",
    ))

    assert finished.status == "finished"
    assert finished.termination_reason is None
    assert finish_game(finished) is finished
    with pytest.raises(RuntimeError, match="terminal"):
        abort_game(finished, reason="step_limit", last_node="vote", step=9)


def test_abort_game_records_one_moderator_only_v2_event_with_context() -> None:
    now = datetime(2026, 7, 16, 8, 30, tzinfo=timezone.utc)
    state = GameState(
        game_id="g-abort", phase="night", day_number=2, night_number=3,
        events=[GameEvent(type="enter_night")],
    )

    aborted = abort_game(
        state,
        reason="unrecoverable_runtime_error",
        last_node="wolf_action",
        step=17,
        exception=RuntimeError("private prompt must not leak"),
        now=now,
    )

    assert aborted.status == "aborted"
    assert aborted.termination_reason == "unrecoverable_runtime_error"
    assert aborted.winning_faction is None
    events = [event for event in aborted.events if event.type == "game_aborted"]
    assert len(events) == 1
    event = events[0]
    assert event.visibility is EventVisibility.MODERATOR_ONLY
    assert event.schema_version == "2"
    assert event.game_id == "g-abort"
    assert event.occurred_at == now
    assert event.payload == {
        "termination_reason": "unrecoverable_runtime_error",
        "last_node": "wolf_action",
        "phase": "night",
        "step": 17,
        "exception_type": "RuntimeError",
    }
    assert abort_game(
        aborted, reason="step_limit", last_node="other", step=99,
    ) is aborted
    assert len([event for event in aborted.events if event.type == "game_aborted"]) == 1
    with pytest.raises(RuntimeError, match="terminal"):
        finish_game(aborted)


def test_legacy_aborted_state_is_constructible_but_runtime_validation_fails_closed() -> None:
    legacy = GameState(
        game_id="g-legacy-abort",
        status="aborted",
        termination_reason="legacy_runtime_error",
    )

    with pytest.raises(ValueError, match="exactly one.*game_aborted"):
        validate_aborted_game(legacy)
    with pytest.raises(ValueError, match="exactly one.*game_aborted"):
        abort_game(
            legacy,
            reason="legacy_runtime_error",
            last_node="unknown",
            step=0,
        )


def test_repository_boundary_rejects_direct_malformed_aborted_write() -> None:
    from werewolf_agent.storage.memory_store import InMemoryGameRepository

    malformed = GameState(
        game_id="g-direct-abort",
        status="aborted",
        termination_reason="legacy_runtime_error",
    )

    with pytest.raises(ValueError, match="exactly one.*game_aborted"):
        InMemoryGameRepository().save_game(malformed)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda state, event: replace(state, events=[]),
            "exactly one.*game_aborted",
        ),
        (
            lambda state, event: replace(
                state,
                events=[replace(event, payload={
                    key: value for key, value in event.payload.items() if key != "step"
                })],
            ),
            "missing payload fields.*step",
        ),
        (
            lambda state, event: replace(
                state,
                events=[replace(event, visibility=EventVisibility.PUBLIC)],
            ),
            "moderator_only",
        ),
        (
            lambda state, event: replace(
                state,
                events=[GameEvent(
                    type="game_aborted",
                    payload=event.payload,
                    visibility=EventVisibility.MODERATOR_ONLY,
                )],
            ),
            "complete V2",
        ),
        (
            lambda state, event: replace(state, events=[event, event]),
            "exactly one.*game_aborted",
        ),
        (
            lambda state, event: replace(
                state,
                events=[replace(event, payload={
                    **event.payload, "termination_reason": "conflict",
                })],
            ),
            "termination_reason.*match",
        ),
        (
            lambda state, event: replace(
                state,
                events=[replace(event, payload={**event.payload, "phase": "day"})],
            ),
            "phase.*match",
        ),
    ],
)
def test_aborted_runtime_validation_rejects_incomplete_or_conflicting_event(
    mutate, expected,
) -> None:
    valid = abort_game(
        GameState(game_id="g-invalid-event", phase="night"),
        reason="step_limit",
        last_node="wolf_action",
        step=12,
    )
    event = next(item for item in valid.events if item.type == "game_aborted")
    invalid = mutate(valid, event)

    with pytest.raises(ValueError, match=expected):
        validate_aborted_game(invalid)
    with pytest.raises(ValueError, match=expected):
        abort_game(
            invalid,
            reason="step_limit",
            last_node="wolf_action",
            step=12,
        )


def test_valid_aborted_state_is_idempotent_without_second_event() -> None:
    aborted = abort_game(
        GameState(game_id="g-idempotent", phase="day"),
        reason="step_limit",
        last_node="vote",
        step=50,
    )

    validate_aborted_game(aborted)
    repeated = abort_game(
        aborted,
        reason="unrecoverable_runtime_error",
        last_node="other",
        step=99,
    )

    assert repeated is aborted
    assert len([event for event in repeated.events if event.type == "game_aborted"]) == 1


@pytest.mark.parametrize(
    "field_name, invalid_value, expected",
    [
        ("termination_reason", 7, "termination_reason.*non-blank string"),
        ("termination_reason", "   ", "termination_reason.*non-blank string"),
        ("phase", 7, "phase.*non-blank string"),
        ("phase", "   ", "phase.*non-blank string"),
    ],
)
def test_aborted_validation_rejects_non_string_or_blank_state_fields(
    field_name, invalid_value, expected,
) -> None:
    valid = abort_game(
        GameState(game_id="g-state-fields", phase="night"),
        reason="step_limit", last_node="node", step=1,
    )
    invalid = replace(valid, **{field_name: invalid_value})

    with pytest.raises(ValueError, match=expected):
        validate_aborted_game(invalid)


def test_aborted_event_must_be_final_and_have_unique_v2_identity() -> None:
    valid = abort_game(
        GameState(game_id="g-final-event", phase="night"),
        reason="step_limit", last_node="node", step=1,
    )
    aborted_event = valid.events[-1]
    trailing = replace(aborted_event, type="diagnostic")

    with pytest.raises(ValueError, match="final event"):
        validate_aborted_game(replace(valid, events=[aborted_event, trailing]))

    duplicate_identity = replace(aborted_event, type="diagnostic")
    with pytest.raises(ValueError, match="unique V2 identity"):
        validate_aborted_game(
            replace(valid, events=[duplicate_identity, aborted_event])
        )


def test_game_state_rejects_invalid_explicit_terminal_contracts() -> None:
    with pytest.raises(ValueError, match="finished.*winner"):
        GameState(game_id="g-invalid-finish", status="finished")
    with pytest.raises(ValueError, match="aborted.*termination_reason"):
        GameState(game_id="g-invalid-abort", status="aborted")


def test_emergency_abort_artifact_is_atomic_and_strictly_allowlisted(tmp_path) -> None:
    state = abort_game(
        GameState(game_id="safe-game", phase="day", day_number=4, night_number=3),
        reason="step_limit",
        last_node="vote",
        step=50,
        exception=RuntimeError("SECRET_PROMPT"),
        now=datetime(2026, 7, 16, 9, 0, tzinfo=timezone.utc),
    )

    path = write_emergency_abort(state, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "emergency_abort_safe-game.json"
    assert set(payload) == {
        "game_id", "status", "termination_reason", "last_node", "phase",
        "day_number", "night_number", "step", "exception_type", "occurred_at",
    }
    assert payload["status"] == "aborted"
    assert payload["last_node"] == "vote"
    assert payload["exception_type"] == "RuntimeError"
    assert "SECRET_PROMPT" not in path.read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.tmp")) == []


def test_emergency_abort_rejects_path_escape_even_if_state_is_forged(tmp_path) -> None:
    state = abort_game(
        GameState(game_id="safe", phase="night"),
        reason="step_limit", last_node=None, step=1,
    )
    forged = replace(state, game_id="../escape")

    with pytest.raises(ValueError, match="game_id"):
        write_emergency_abort(forged, tmp_path)
