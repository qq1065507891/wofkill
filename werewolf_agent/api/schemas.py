"""API schemas: request/response models with view mode classification.

Design doc §12.1: three view modes — public, player_view, moderator_full.
During live play, player agents must never receive moderator_full.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# View modes
# ---------------------------------------------------------------------------

class ViewMode(str, Enum):
    PUBLIC = "public"
    PLAYER_VIEW = "player_view"
    MODERATOR_FULL = "moderator_full"


class CallerRole(str, Enum):
    PLAYER_AGENT = "player_agent"
    SPECTATOR = "spectator"
    MODERATOR = "moderator"
    DEBUGGER = "debugger"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditEvent(BaseModel):
    timestamp: str = ""
    caller_id: str = ""
    caller_role: CallerRole
    requested_view: ViewMode
    game_id: str = ""
    endpoint: str = ""
    granted: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# API requests
# ---------------------------------------------------------------------------

class CreateGameRequest(BaseModel):
    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    player_count: int = 12
    seed: int | None = None


class GameActionRequest(BaseModel):
    caller_id: str = ""
    caller_role: CallerRole = CallerRole.MODERATOR


class PrivateStateRequest(BaseModel):
    caller_id: str = ""
    caller_role: CallerRole = CallerRole.PLAYER_AGENT
    player_id: str = ""
    view_mode: ViewMode = ViewMode.PLAYER_VIEW


class TimelineRequest(BaseModel):
    caller_id: str = ""
    caller_role: CallerRole = CallerRole.SPECTATOR
    view_mode: ViewMode = ViewMode.PUBLIC
    day_filter: int | None = None


class ReplayRequest(BaseModel):
    caller_id: str = ""
    caller_role: CallerRole = CallerRole.MODERATOR
    view_mode: ViewMode = ViewMode.MODERATOR_FULL


class EvaluationRequest(BaseModel):
    caller_id: str = ""
    caller_role: CallerRole = CallerRole.MODERATOR
    view_mode: ViewMode = ViewMode.MODERATOR_FULL


class CognitiveDiffRequest(BaseModel):
    caller_id: str = ""
    caller_role: CallerRole = CallerRole.DEBUGGER
    player_id: str = ""
    view_mode: ViewMode = ViewMode.MODERATOR_FULL


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------

class PlayerPublicInfo(BaseModel):
    player_id: str
    alive: bool = True
    vote_enabled: bool = True
    is_sheriff: bool = False
    revealed_role: str | None = None  # Only for revealed idiot / dead


class PublicStateResponse(BaseModel):
    game_id: str
    ruleset_id: str
    day_number: int
    night_number: int
    phase: str
    sheriff_id: str | None = None
    badge_state: str = "none"
    players: list[PlayerPublicInfo] = Field(default_factory=list)
    alive_count: int = 0
    winning_faction: str | None = None


class PlayerPrivateInfo(BaseModel):
    player_id: str
    role: str
    alive: bool = True
    vote_enabled: bool = True
    badge_eligible: bool = True
    # Role-specific fields (only for own view)
    antidote_available: bool | None = None
    poison_available: bool | None = None
    current_wolf_kill_target_id: str | None = None
    master_id: str | None = None
    wolf_teammates: list[str] | None = None


class PrivateStateResponse(BaseModel):
    viewer_id: str
    view_mode: ViewMode
    player_info: PlayerPrivateInfo
    source_annotation: str = ""


class TimelineEvent(BaseModel):
    event_type: str
    day: int = 0
    phase: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    is_public: bool = True


class TimelineResponse(BaseModel):
    game_id: str
    view_mode: ViewMode
    events: list[TimelineEvent] = Field(default_factory=list)
    total_count: int = 0


class ReplaySnapshot(BaseModel):
    day: int
    phase: str
    public_state: dict[str, Any] = Field(default_factory=dict)
    player_views: dict[str, dict[str, Any]] = Field(default_factory=dict)
    moderator_full: dict[str, Any] | None = None


class ReplayResponse(BaseModel):
    game_id: str
    view_mode: ViewMode
    snapshots: list[ReplaySnapshot] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    game_id: str
    faction_win_rate: dict[str, float] = Field(default_factory=dict)
    player_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    info_leak_count: int = 0
    audit_events: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    game_id: str
    view_mode: ViewMode
    metrics: EvaluationMetrics | None = None
    source_annotation: str = ""


class CognitiveDiffEntry(BaseModel):
    target_player: str
    actual_role: str | None = None  # Only in moderator_full
    actual_faction: str | None = None  # Only in moderator_full
    guessed_role: str = "unknown"
    guessed_confidence: float = 0.0
    faction_read: str = "unknown"
    trust: float = 0.5
    key_evidence: list[str] = Field(default_factory=list)
    belief_changes: list[dict[str, Any]] = Field(default_factory=list)


class CognitiveDiffResponse(BaseModel):
    game_id: str
    viewer_id: str
    view_mode: ViewMode
    entries: list[CognitiveDiffEntry] = Field(default_factory=list)
    source_annotation: str = ""


class GameInfo(BaseModel):
    game_id: str
    ruleset_id: str
    status: str = "created"
    player_count: int = 12


class GameCreateResponse(BaseModel):
    game: GameInfo
    message: str = ""


class GameActionResponse(BaseModel):
    game_id: str
    action: str
    success: bool
    message: str = ""


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    audit_logged: bool = False
