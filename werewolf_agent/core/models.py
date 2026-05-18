from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlayerState:
    id: str
    role: str
    alive: bool = True
    revealed_idiot: bool = False
    vote_enabled: bool = True
    badge_eligible: bool = True
    exile_immune: bool = False


@dataclass(frozen=True)
class Death:
    player_id: str
    reason: str
    timing: str
    resolution_batch: str
    source_player_id: str | None = None
    can_leave_last_words: bool | None = None
    triggered_skills: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GameState:
    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    players: dict[str, PlayerState] = field(default_factory=dict)
    game_id: str = ""
    phase: str = "setup"
    day_number: int = 0
    night_number: int = 0
    hybrid_master_id: str | None = None
    hybrid_master_faction: str | None = None
    sheriff_id: str | None = None
    sheriff_badge_state: str = "none"
    sheriff_candidates: list[str] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    private_intents: dict[str, dict[str, Any]] = field(default_factory=dict)
    antidote_used: bool = False
    poison_used: bool = False
    deaths: list[Death] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)
    winning_faction: str | None = None
    hybrid_result: str | None = None
    paused: bool = False


@dataclass(frozen=True)
class Action:
    type: str
    target_id: str | None = None


@dataclass(frozen=True)
class RuleResult:
    accepted: bool
    error_code: str | None = None


@dataclass(frozen=True)
class AlignmentResult:
    alignment: str
    role: str | None = None


@dataclass(frozen=True)
class VictoryResult:
    winner: str | None
    reason: str | None = None


@dataclass(frozen=True)
class BadgeDecisionOptions:
    can_transfer: bool
    can_tear: bool
    transfer_targets: list[str]


@dataclass(frozen=True)
class VoteResult:
    exiled_player_id: str | None
    next_phase: str
    reason: str
    tied_player_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GameEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisibleContext:
    view_mode: str
    visible_sections: set[str]
