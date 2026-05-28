"""Shared player-visible runtime state builders."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.public_ledger import build_public_ledger
from werewolf_agent.runtime.timeline import (
    TIMELINE_ORDER_NOTE,
    build_timeline_facts,
    current_phase_label,
    phase_label,
)


def build_visible_player_state(game_state: GameState) -> dict[str, Any]:
    """Build public fields common to all player contexts."""
    deaths = list(game_state.deaths)
    # Only reveal deaths after the judge has publicly announced them.
    # During the sheriff election on day 1, deaths are already recorded in
    # game_state.deaths but have NOT been announced yet — players must not
    # see them prematurely (e.g. in their election speeches).
    death_announced = any(
        e.type == "judge_broadcast" and e.payload.get("phase") == "death_announce"
        for e in game_state.events
    )
    if not death_announced:
        # Keep only deaths that were announced in a prior day (exile, hunter shot).
        # Night deaths are never visible until the first death_announce broadcast.
        deaths = [d for d in deaths if d.timing != "night"]

    return {
        "phase": game_state.phase,
        "day": game_state.day_number,
        "night": game_state.night_number,
        "phase_label": current_phase_label(
            game_state.phase,
            day_number=game_state.day_number,
            night_number=game_state.night_number,
        ),
        "timeline_note": TIMELINE_ORDER_NOTE,
        "timeline_facts": build_timeline_facts(
            game_state.phase,
            day_number=game_state.day_number,
            night_number=game_state.night_number,
        ),
        "alive_players": [
            pid for pid, player in game_state.players.items() if player.alive
        ],
        "dead_players": [
            {"id": death.player_id, "reason": death.reason if death.reason in ("exile", "hunter_shot") else "night"}
            for death in deaths
        ],
        "sheriff_id": game_state.sheriff_id,
        "badge_state": game_state.sheriff_badge_state,
        "public_ledger": _compact_public_ledger(build_public_ledger(game_state)),
    }


def _compact_public_ledger(
    ledger: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {key: value for key, value in ledger.items() if value}


def build_public_summary(game_state: GameState) -> str:
    """Build a compact phase summary for contexts without event replay."""
    parts: list[str] = [TIMELINE_ORDER_NOTE]
    if game_state.day_number > 0:
        parts.append(phase_label("day", game_state.day_number))
    if game_state.night_number > 0:
        parts.append(phase_label("night", game_state.night_number))
    alive = sum(1 for player in game_state.players.values() if player.alive)
    parts.append(f"存活 {alive} 人")
    if game_state.sheriff_id:
        parts.append(f"警长: {game_state.sheriff_id}")
    return "。".join(parts) + "。"
