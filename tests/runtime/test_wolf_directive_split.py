"""Tests for M3-3 wolf day/night directive split.

Issue M3-3: ``build_wolf_directive`` historically built BOTH day-only keys
(wolf_day_push_target) AND night-only keys (wolf_no_kill_conditions).
After the fix, ``build_wolf_day_directive`` and ``build_wolf_night_directive``
own their respective halves; ``build_wolf_directive`` is a back-compat
shim returning the merged dict.
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.directives.wolf import (
    build_wolf_directive,
    build_wolf_day_directive,
    build_wolf_night_directive,
)


def _gs() -> GameState:
    """Build a minimal 12-player GameState with 4 wolves."""
    return GameState(
        game_id="t",
        phase="day",
        day_number=3,
        night_number=3,
        players={
            f"p{i:02d}": PlayerState(
                id=f"p{i:02d}",
                role="werewolf" if i <= 4 else "villager",
                alive=True,
            )
            for i in range(1, 13)
        },
    )


def test_wolf_day_directive_has_push_target_no_no_kill() -> None:
    """M3-3: day directive must contain a day-push key, NOT no_kill.

    ``wolf_no_kill_conditions`` is a NIGHT-only decision aid and must
    not leak into the day-speech prompt.
    """
    # day_push_target must be in plan AND target must be alive for the
    # day directive to emit ``wolf_day_push_target``.
    d = build_wolf_day_directive(
        _gs(), "p01",
        {"fake_seer": "p02", "day_push_target": "p07"},
    )
    # Day-push: any of these keys is acceptable
    day_push_keys = [k for k in d if "push" in k.lower() or "day_push" in k]
    assert day_push_keys, (
        f"Day directive missing push target; got keys: {list(d.keys())}"
    )
    # Night-only keys must NOT appear in day
    assert "wolf_no_kill_conditions" not in d, (
        f"Day directive leaked no_kill_conditions (night-only); "
        f"got: {list(d.keys())}"
    )


def test_wolf_night_directive_has_no_kill_no_push() -> None:
    """M3-3: night directive must contain no_kill, NOT day-push.

    ``wolf_day_push_target`` is a DAY-only decision aid; ``wolf_no_kill_conditions``
    belongs to the night-action prompt so the LLM can see empty-knife
    as a legal option.
    """
    d = build_wolf_night_directive(_gs(), "p01", None)
    assert "wolf_no_kill_conditions" in d, (
        f"Night directive missing no_kill_conditions; got: {list(d.keys())}"
    )
    # Day-push must NOT appear in night
    day_push_keys = [k for k in d if "push" in k.lower() or "day_push" in k]
    assert not day_push_keys, (
        f"Night directive leaked day-push target; got: {day_push_keys}"
    )


def test_wolf_directive_shim_merges_both() -> None:
    """M3-3: ``build_wolf_directive`` is a back-compat shim returning merged dict.

    Old callers (tests + context.py) pass ``wolf_team_plan`` without a
    task_type; the shim must still return BOTH halves so they don't
    silently lose information.
    """
    d = build_wolf_directive(
        _gs(), "p01",
        {"fake_seer": "p02", "day_push_target": "p07"},
    )
    assert "wolf_no_kill_conditions" in d, (
        "Shim should merge night-side (no_kill) into back-compat dict"
    )
    day_push_keys = [k for k in d if "push" in k.lower() or "day_push" in k]
    assert day_push_keys, (
        f"Shim should merge day-side (push target); got: {list(d.keys())}"
    )
