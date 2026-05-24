"""Central timeout contract for real-game agent calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTimeouts:
    wolf_discussion_per_player: float = 180.0
    wolf_discussion_total: float = 600.0
    wolf_consensus: float = 180.0
    seer: float = 180.0
    witch: float = 180.0
    day_speech: float = 240.0
    day_vote: float = 180.0
    hunter_shot: float = 120.0


AGENT_TIMEOUTS = AgentTimeouts()
