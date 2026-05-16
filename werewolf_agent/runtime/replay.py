"""Replay: reconstruct GameState from initial seed + ruleset + event log."""

from __future__ import annotations

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine


def replay_from_events(
    engine: RuleEngine,
    initial_state: GameState,
    events: list[GameEvent],
) -> GameState:
    return engine.reduce_events(initial_state, events)


def extract_event_log(state: GameState) -> list[GameEvent]:
    return list(state.events)
