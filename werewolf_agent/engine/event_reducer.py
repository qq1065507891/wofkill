"""Event-sourced state reducer: replay GameEvents into GameState mutations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState


def _apply_idiot_reveal(raw: dict[str, Any], state: GameState, player_id: str) -> GameState:
    """Apply idiot reveal state transitions."""
    player = state.players[player_id]
    if player.role != "idiot" or player.revealed_idiot:
        return state
    after = raw["roles"]["idiot"]["abilities"]["state_after_reveal"]
    updated = replace(
        player,
        alive=after["alive"],
        revealed_idiot=after["revealed_idiot"],
        vote_enabled=not after["vote_disabled"],
        badge_eligible=not after["badge_ineligible"],
        exile_immune=after["exile_immune"],
    )
    return replace(state, players={**state.players, player_id: updated})


class EventReducer:
    """Reduces GameEvents into GameState mutations for deterministic replay."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def reduce_event(self, state: GameState, event: GameEvent) -> GameState:
        etype = event.type
        payload = event.payload

        if etype == "player_died":
            pid = payload["player_id"]
            player = state.players[pid]
            if player.alive:
                updated = replace(player, alive=False)
                new_players = {**state.players, pid: updated}
                death = Death(
                    player_id=pid,
                    reason=payload.get("reason", "unknown"),
                    timing=payload.get("timing", "unknown"),
                    resolution_batch=payload.get("resolution_batch", "unknown"),
                    source_player_id=payload.get("source_player_id"),
                    can_leave_last_words=payload.get("can_leave_last_words"),
                    triggered_skills=list(payload.get("triggered_skills", [])),
                )
                return replace(
                    state,
                    players=new_players,
                    deaths=state.deaths + [death],
                    events=state.events + [event],
                )
            return replace(state, events=state.events + [event])

        if etype == "idiot_revealed":
            pid = payload["player_id"]
            return replace(
                _apply_idiot_reveal(self._raw, state, pid),
                events=state.events + [event],
            )

        if etype == "werewolf_self_destructed":
            pid = payload["player_id"]
            player = state.players[pid]
            if player.alive:
                updated = replace(player, alive=False)
                new_players = {**state.players, pid: updated}
                death = Death(
                    player_id=pid,
                    reason="self_destruct",
                    timing="day_discussion",
                    resolution_batch=f"day_{payload.get('day_number', '?')}_self_destruct",
                    can_leave_last_words=payload.get("can_leave_last_words"),
                    triggered_skills=list(payload.get("triggered_skills", [])),
                )
                return replace(
                    state,
                    players=new_players,
                    deaths=state.deaths + [death],
                    events=state.events + [event],
                )
            return replace(state, events=state.events + [event])

        if etype == "hybrid_master_chosen":
            return replace(
                state,
                hybrid_master_id=payload["master_id"],
                hybrid_master_faction=self._faction_for_player(state, payload["master_id"]),
                events=state.events + [event],
            )

        if etype == "sheriff_elected":
            return replace(
                state,
                sheriff_id=payload["sheriff_id"],
                sheriff_badge_state="active",
                events=state.events + [event],
            )

        if etype == "player_exiled":
            pid = payload["player_id"]
            player = state.players[pid]
            if player.alive and not player.exile_immune:
                if player.role == "idiot" and not player.revealed_idiot:
                    return replace(
                        _apply_idiot_reveal(self._raw, state, pid),
                        events=state.events + [event],
                    )
                updated = replace(player, alive=False)
                new_players = {**state.players, pid: updated}
                death = Death(
                    player_id=pid,
                    reason="exile",
                    timing="day_vote",
                    resolution_batch=payload.get("resolution_batch", "day_vote"),
                    can_leave_last_words=payload.get("can_leave_last_words"),
                    triggered_skills=list(payload.get("triggered_skills", [])),
                )
                return replace(
                    state,
                    players=new_players,
                    deaths=state.deaths + [death],
                    events=state.events + [event],
                )
            return replace(state, events=state.events + [event])

        if etype in {"victory", "victory_checked"}:
            winner = payload.get("winner") or payload.get("winning_faction")
            if winner is None:
                return replace(state, events=state.events + [event])
            hybrid_result = payload.get("hybrid_result")
            if hybrid_result is None:
                hybrid_result = self._hybrid_result(state, winner)
            return replace(
                state,
                winning_faction=winner,
                hybrid_result=hybrid_result,
                phase="finished",
                events=state.events + [event],
            )

        # Badge decisions
        if etype == "badge_torn":
            return replace(
                state,
                sheriff_id=None,
                sheriff_badge_state="torn",
                events=state.events + [event],
            )
        if etype == "badge_transferred":
            return replace(
                state,
                sheriff_id=payload["new_sheriff_id"],
                sheriff_badge_state="active",
                events=state.events + [event],
            )

        # Witch potion tracking
        if etype == "witch_antidote_used":
            return replace(
                state, antidote_used=True, events=state.events + [event],
            )
        if etype == "witch_poison_used":
            return replace(
                state, poison_used=True, events=state.events + [event],
            )

        # Game started
        if etype == "game_started":
            new_players = {}
            for pid, pdata in payload.get("players", {}).items():
                if isinstance(pdata, dict):
                    new_players[pid] = PlayerState(**pdata)
                else:
                    new_players[pid] = pdata
            return replace(
                state,
                players=new_players or state.players,
                phase="night",
                events=state.events + [event],
            )

        # Pause / resume
        if etype == "game_paused":
            return replace(state, paused=True, events=state.events + [event])
        if etype == "game_resumed":
            return replace(state, paused=False, events=state.events + [event])

        # Default: just append event
        return replace(state, events=state.events + [event])

    def reduce_events(self, state: GameState, events: list[GameEvent]) -> GameState:
        for event in events:
            state = self.reduce_event(state, event)
        return state

    def _faction_for_player(self, state: GameState, player_id: str) -> str:
        role = state.players[player_id].role
        return self._raw["roles"][role]["faction"]

    def _hybrid_result(self, state: GameState, winning_faction: str) -> str | None:
        if state.hybrid_master_faction is None:
            return None
        return "win" if state.hybrid_master_faction == winning_faction else "lose"
