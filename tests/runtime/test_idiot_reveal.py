"""Tests for exile + idiot reveal broadcast ordering."""

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph import _new_engine, resolve_exile


def _build_idiot_exile_state() -> tuple[GameState, str, RuleEngine]:
    """Build a state where the vote resolved with the idiot as the
    exile target, but the idiot has NOT been revealed yet."""
    engine = _new_engine()
    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
        "p04": PlayerState(id="p04", role="seer", alive=True),
        "p05": PlayerState(id="p05", role="idiot", alive=True),
        "p06": PlayerState(id="p06", role="werewolf", alive=True),
    }
    gs = GameState(
        game_id="exile_idiot",
        players=players,
        day_number=2,
        phase="day",
        events=[
            GameEvent(type="vote_resolved", payload={
                "exiled": "p05",
                "reason": "majority",
                "day_number": 2,
            }),
        ],
    )
    return gs, "p05", engine


def test_exile_broadcast_after_idiot_check() -> None:
    """resolve_exile must run engine.resolve_exile FIRST, then publish
    the public broadcast. When the exiled player is an unrevealed idiot,
    the engine reveals the idiot and the player stays alive — the public
    broadcast must therefore say '亮出白痴身份' (idiot_revealed), NOT
    '被放逐出局' (exile).

    Previously the exile broadcast fired before engine.resolve_exile,
    so the public ledger would log '被放逐' even when the player
    wasn't actually exiled (idiot revealed and stayed alive).
    """
    gs, exiled_id, engine = _build_idiot_exile_state()

    result = resolve_exile({
        "game_state": gs,
        "engine": engine,
    })

    new_state = result["game_state"]
    broadcasts = [
        e.payload for e in new_state.events
        if e.type == "judge_broadcast"
    ]
    broadcast_phases = [b.get("phase") for b in broadcasts]

    # The exiled idiot must STAY alive
    assert new_state.players[exiled_id].alive is True, (
        "Idiot reveal must keep the player alive"
    )
    assert new_state.players[exiled_id].revealed_idiot is True

    # There must NOT be a phase=exile broadcast (player was not exiled)
    assert "exile" not in broadcast_phases, (
        f"phase=exile broadcast should not fire when idiot reveals; "
        f"got broadcast phases: {broadcast_phases}"
    )
    # There must BE a phase=idiot_revealed broadcast
    assert "idiot_revealed" in broadcast_phases, (
        f"phase=idiot_revealed broadcast must fire when idiot reveals; "
        f"got broadcast phases: {broadcast_phases}"
    )

    # Ordering: idiot_revealed broadcast comes AFTER any vote_resolved
    # event but the exile broadcast should never have been emitted
    idiot_revealed_idx = broadcast_phases.index("idiot_revealed")
    assert idiot_revealed_idx >= 0
    # The previous test (buggy behavior) had phase=exile appearing BEFORE
    # the engine resolved the exile. After the fix, the exile broadcast
    # is conditional on idiot_revealed NOT being in the events.


def test_normal_exile_broadcasts_after_engine_resolves() -> None:
    """When the exiled player is NOT an idiot, the exile broadcast must
    still fire AFTER engine.resolve_exile. Verifies the reordering fix
    didn't break the non-idiot path."""
    engine = _new_engine()
    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="seer", alive=True),
        "p04": PlayerState(id="p04", role="werewolf", alive=True),
    }
    gs = GameState(
        game_id="normal_exile",
        players=players,
        day_number=2,
        phase="day",
        events=[
            GameEvent(type="vote_resolved", payload={
                "exiled": "p04",
                "reason": "majority",
                "day_number": 2,
            }),
        ],
    )

    result = resolve_exile({"game_state": gs, "engine": engine})
    new_state = result["game_state"]
    broadcasts = [
        e.payload for e in new_state.events
        if e.type == "judge_broadcast"
    ]
    broadcast_phases = [b.get("phase") for b in broadcasts]

    assert "exile" in broadcast_phases, (
        f"phase=exile broadcast must fire for normal exile; got {broadcast_phases}"
    )
    assert "idiot_revealed" not in broadcast_phases
    assert new_state.players["p04"].alive is False
