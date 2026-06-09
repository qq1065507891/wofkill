"""Tests for M3-2 day filter on public history helpers.

P1 fix: ``collect_public_vote_history(gs)`` and
``collect_death_order(gs)`` used to return the full game
history.  By day 5 the LLM was looking at day-1 vote
patterns alongside the current day, diluting focus on the
information that matters *now*.  Both helpers now accept an
optional ``current_day`` filter; ``None`` (default) preserves
the pre-fix behavior so back-compat callers keep working.
"""

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.runtime.directives._shared import (
    collect_death_order,
    collect_public_vote_history,
)


def _gs_with_votes(votes_per_day: dict[int, str]) -> GameState:
    """Build a GameState with one vote_resolved event per day."""
    events: list[GameEvent] = []
    for day, exiled in votes_per_day.items():
        events.append(
            GameEvent(
                type="vote_resolved",
                payload={"day_number": day, "exiled": exiled, "votes": []},
            )
        )
    return GameState(
        game_id="t",
        phase="day",
        day_number=max(votes_per_day.keys()) if votes_per_day else 1,
        players={
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager")
            for i in range(1, 5)
        },
        events=events,
    )


def test_collect_vote_history_filters_to_current_day() -> None:
    """M3-2: helper must accept current_day param, filter to that day."""
    gs = _gs_with_votes({1: "p01", 2: "p02", 3: "p03", 4: "p04"})
    # Without filter: all 4 days
    full = collect_public_vote_history(gs)
    assert full.count("D") == 4, f"expected 4 day markers, got: {full!r}"
    # With filter=2: only days 1-2
    recent = collect_public_vote_history(gs, current_day=2)
    assert "p01" in recent and "p02" in recent, (
        f"days 1-2 should be included; got: {recent!r}"
    )
    assert "p03" not in recent and "p04" not in recent, (
        f"day 3+ votes should be filtered; got: {recent!r}"
    )


def test_collect_vote_history_default_no_filter() -> None:
    """M3-2: default current_day=None preserves back-compat behavior."""
    gs = _gs_with_votes({1: "p01", 2: "p02", 3: "p03"})
    full = collect_public_vote_history(gs)
    # All 3 days included
    for marker in ["p01", "p02", "p03"]:
        assert marker in full, f"{marker} should be in unfiltered output; got: {full!r}"


def test_collect_death_order_filters_to_current_day() -> None:
    """M3-2: collect_death_order also accepts current_day filter."""
    gs = GameState(
        game_id="t",
        phase="day",
        day_number=3,
        players={
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager")
            for i in range(1, 5)
        },
        deaths=[
            Death(
                player_id="p01",
                reason="exile",
                timing="day_vote",
                resolution_batch="day_1",
                source_player_id=None,
                can_leave_last_words=True,
                triggered_skills=[],
            ),
            Death(
                player_id="p02",
                reason="exile",
                timing="day_vote",
                resolution_batch="day_3",
                source_player_id=None,
                can_leave_last_words=True,
                triggered_skills=[],
            ),
            Death(
                player_id="p03",
                reason="wolf_kill",
                timing="night",
                resolution_batch="night_4",
                source_player_id=None,
                can_leave_last_words=False,
                triggered_skills=[],
            ),
        ],
    )
    # Filter to day 2: only p01 (day 1 exile) is public; night deaths hidden
    recent = collect_death_order(gs, current_day=2)
    assert "p01" in recent, f"day 1 exile should be included; got: {recent!r}"
    assert "p02" not in recent, (
        f"day 3 > 2 should be filtered; got: {recent!r}"
    )
    # p03 (night death) is unrelated to the day filter contract — the
    # existing helper still emits all player_ids regardless of
    # public/private labelling, so the M3-2 fix doesn't touch that
    # behavior.  Sanity-check it appears in the unfiltered output.
    full = collect_death_order(gs)
    assert "p01" in full and "p02" in full and "p03" in full, (
        f"unfiltered should include all deaths; got: {full!r}"
    )


def test_collect_death_order_keeps_malformed_batch() -> None:
    """M3-2: malformed resolution_batch is retained, not dropped.

    A logger.warning is emitted (engine regression signal).  See
    review I-1: silently dropping would mask engine bugs.
    """
    import logging
    from unittest.mock import patch

    gs = GameState(
        game_id="t",
        phase="day",
        day_number=5,
        players={
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager")
            for i in range(1, 5)
        },
        deaths=[
            Death(
                player_id="p01",
                reason="exile",
                timing="day_vote",
                resolution_batch="day_BAD",
                source_player_id=None,
                can_leave_last_words=True,
                triggered_skills=[],
            ),
        ],
    )
    # Capture warnings from the _shared module's logger
    with patch.object(
        logging.getLogger("werewolf_agent.runtime.directives._shared"),
        "warning",
    ) as mock_warn:
        out = collect_death_order(gs, current_day=5)
    assert "p01" in out, (
        f"Malformed batch should be kept (per never-silently-drop rule); got: {out!r}"
    )
    assert mock_warn.called, (
        "engine regression signal: malformed resolution_batch should emit a warning"
    )
    # Even with current_day=1, malformed day_BAD batch should be kept defensively.
    out2 = collect_death_order(gs, current_day=1)
    assert "p01" in out2, (
        f"Even with current_day=1, malformed day_BAD batch should be kept; got: {out2!r}"
    )


def test_collect_death_order_night_batch_does_not_warn() -> None:
    """M3-2: ``night_N`` is structurally valid -- the warning is
    reserved for true engine regressions (e.g. ``day_BAD``).  See
    review I-1 refinement: avoid noisy logs for the normal
    night-batch case.
    """
    import logging
    from unittest.mock import patch

    gs = GameState(
        game_id="t",
        phase="day",
        day_number=5,
        players={
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager")
            for i in range(1, 5)
        },
        deaths=[
            Death(
                player_id="p03",
                reason="wolf_kill",
                timing="night",
                resolution_batch="night_4",
                source_player_id=None,
                can_leave_last_words=False,
                triggered_skills=[],
            ),
        ],
    )
    with patch.object(
        logging.getLogger("werewolf_agent.runtime.directives._shared"),
        "warning",
    ) as mock_warn:
        collect_death_order(gs, current_day=2)
    assert not mock_warn.called, (
        "night_N is a structurally valid batch; the warning is "
        "reserved for true engine regressions (day_BAD, day_1_extra)"
    )
