"""API view functions: build responses filtered by view mode.

Each view function takes a GameState and view mode, and returns
only the data that the caller is authorized to see.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.api.schemas import (
    CognitiveDiffEntry,
    CognitiveDiffResponse,
    EvaluationMetrics,
    EvaluationResponse,
    PlayerPrivateInfo,
    PlayerPublicInfo,
    PrivateStateResponse,
    PublicStateResponse,
    ReplayResponse,
    ReplaySnapshot,
    TimelineEvent,
    TimelineResponse,
    ViewMode,
)
from werewolf_agent.core.models import GameState


# ---------------------------------------------------------------------------
# Public state view
# ---------------------------------------------------------------------------

def build_public_state(game_state: GameState) -> PublicStateResponse:
    players = []
    for pid, p in game_state.players.items():
        revealed = None
        if p.revealed_idiot:
            revealed = "idiot"
        players.append(PlayerPublicInfo(
            player_id=pid,
            alive=p.alive,
            vote_enabled=p.vote_enabled,
            is_sheriff=(pid == game_state.sheriff_id),
            revealed_role=revealed,
        ))
    return PublicStateResponse(
        game_id=game_state.game_id,
        ruleset_id=game_state.ruleset_id,
        day_number=game_state.day_number,
        night_number=game_state.night_number,
        phase=game_state.phase,
        sheriff_id=game_state.sheriff_id,
        badge_state=game_state.sheriff_badge_state,
        players=players,
        alive_count=sum(1 for p in game_state.players.values() if p.alive),
        winning_faction=game_state.winning_faction,
    )


# ---------------------------------------------------------------------------
# Private state view
# ---------------------------------------------------------------------------

def build_private_state(
    game_state: GameState,
    player_id: str,
    view_mode: ViewMode,
) -> PrivateStateResponse:
    p = game_state.players.get(player_id)
    if p is None:
        return PrivateStateResponse(
            viewer_id=player_id,
            view_mode=view_mode,
            player_info=PlayerPrivateInfo(player_id=player_id, role="unknown"),
        )

    info = PlayerPrivateInfo(
        player_id=player_id,
        role=p.role,
        alive=p.alive,
        vote_enabled=p.vote_enabled,
        badge_eligible=p.badge_eligible,
    )

    # Role-specific private fields (only own view)
    if p.role == "witch":
        info.antidote_available = not game_state.antidote_used
        info.poison_available = not game_state.poison_used
        info.current_wolf_kill_target_id = _current_wolf_kill_target(game_state)
    if p.role == "hybrid" and game_state.hybrid_master_id:
        info.master_id = game_state.hybrid_master_id
    if p.role == "werewolf" and view_mode == ViewMode.PLAYER_VIEW:
        info.wolf_teammates = [
            wid for wid, wp in game_state.players.items()
            if wp.role == "werewolf" and wid != player_id
        ]

    annotation = ""
    if view_mode == ViewMode.MODERATOR_FULL:
        annotation = "来源: moderator_full | 包含所有私有信息"

    return PrivateStateResponse(
        viewer_id=player_id,
        view_mode=view_mode,
        player_info=info,
        source_annotation=annotation,
    )


# ---------------------------------------------------------------------------
# Timeline view
# ---------------------------------------------------------------------------

def build_timeline(
    game_state: GameState,
    view_mode: ViewMode,
    day_filter: int | None = None,
    viewer_id: str | None = None,
) -> TimelineResponse:
    events = []
    viewer = game_state.players.get(viewer_id or "")
    for event in game_state.events:
        # Public events: always included
        if event.type in _PUBLIC_EVENT_TYPES:
            te = _build_public_event(event)
            if day_filter is not None and te.day != day_filter:
                continue
            events.append(te)
        elif view_mode == ViewMode.MODERATOR_FULL:
            te = _build_event(event, view_mode)
            if day_filter is not None and te.day != day_filter:
                continue
            events.append(te)
        elif view_mode == ViewMode.PLAYER_VIEW and viewer is not None:
            if not _event_visible_to_player(event, viewer_id or "", viewer.role):
                continue
            te = _build_event(event, view_mode)
            if day_filter is not None and te.day != day_filter:
                continue
            events.append(te)

    return TimelineResponse(
        game_id=game_state.game_id,
        view_mode=view_mode,
        events=events,
        total_count=len(events),
    )


_PUBLIC_EVENT_TYPES = {
    "player_died", "player_exiled", "idiot_revealed",
    "werewolf_self_destructed", "sheriff_elected",
    "sheriff_registered", "sheriff_withdraw", "sheriff_vote_tie",
    "sheriff_vote_tie_first",
    "badge_transferred", "badge_torn", "sheriff_no_election",
    "speech", "vote",
}


def _event_visible_to_player(event: Any, viewer_id: str, viewer_role: str) -> bool:
    if event.type in _PUBLIC_EVENT_TYPES:
        return True
    if event.type in {"wolf_discussion", "wolf_kill_selected", "wolf_no_kill_declared", "wolf_no_kill_timeout"}:
        return viewer_role == "werewolf"
    if event.type == "seer_check":
        return event.payload.get("seer_id") == viewer_id
    if event.type in {"witch_antidote_used", "witch_poison_used"}:
        return viewer_role == "witch"
    if event.type == "hybrid_master_chosen":
        return event.payload.get("hybrid_id") == viewer_id
    return False


def _build_public_event(event: Any) -> TimelineEvent:
    payload = dict(event.payload)
    # Strip private info from public view
    payload.pop("actual_role", None)
    payload.pop("wolf_teammates", None)
    payload.pop("private_intent", None)
    if event.type == "player_died":
        payload = {"player_id": payload.get("player_id")}

    return TimelineEvent(
        event_type=event.type,
        day=payload.get("day_number", 0),
        phase=payload.get("phase", ""),
        data=payload,
        is_public=True,
    )


def _build_event(event: Any, view_mode: ViewMode) -> TimelineEvent:
    payload = dict(event.payload)
    # moderator_full sees everything
    if view_mode != ViewMode.MODERATOR_FULL:
        payload.pop("actual_role", None)
        payload.pop("wolf_teammates", None)
    # Never include private_intent in any API response
    payload.pop("private_intent", None)

    return TimelineEvent(
        event_type=event.type,
        day=payload.get("day_number", 0),
        phase=payload.get("phase", ""),
        data=payload,
        is_public=(event.type in _PUBLIC_EVENT_TYPES),
    )


def _current_wolf_kill_target(game_state: GameState) -> str | None:
    for event in reversed(game_state.events):
        if event.type == "wolf_kill_selected" and event.payload.get("night_number") == game_state.night_number:
            return event.payload.get("target_id")
        if event.type in {"wolf_no_kill_declared", "wolf_no_kill_timeout"} and event.payload.get("night_number") == game_state.night_number:
            return None
    return None


# ---------------------------------------------------------------------------
# Replay view
# ---------------------------------------------------------------------------

def build_replay(
    game_state: GameState,
    view_mode: ViewMode,
    viewer_id: str | None = None,
) -> ReplayResponse:
    snapshots = []
    # Build snapshot from current state
    public = build_public_state(game_state)
    snapshot = ReplaySnapshot(
        day=game_state.day_number,
        phase=game_state.phase,
        public_state=public.model_dump(),
    )

    if view_mode == ViewMode.PLAYER_VIEW:
        if viewer_id in game_state.players:
            pv = build_private_state(game_state, viewer_id, view_mode)
            snapshot.player_views[viewer_id] = pv.model_dump()
    elif view_mode == ViewMode.MODERATOR_FULL:
        for pid in game_state.players:
            pv = build_private_state(game_state, pid, view_mode)
            snapshot.player_views[pid] = pv.model_dump()

    if view_mode == ViewMode.MODERATOR_FULL:
        snapshot.moderator_full = {
            "hybrid_master_id": game_state.hybrid_master_id,
            "hybrid_master_faction": game_state.hybrid_master_faction,
            "antidote_used": game_state.antidote_used,
            "poison_used": game_state.poison_used,
            "private_intents": "[redacted in api - audit only]",
            "all_roles": {pid: p.role for pid, p in game_state.players.items()},
        }

    snapshots.append(snapshot)
    return ReplayResponse(
        game_id=game_state.game_id,
        view_mode=view_mode,
        snapshots=snapshots,
    )


# ---------------------------------------------------------------------------
# Evaluation view
# ---------------------------------------------------------------------------

def build_evaluation(
    game_state: GameState,
    view_mode: ViewMode,
    audit_events: list[dict[str, Any]] | None = None,
) -> EvaluationResponse:
    events = audit_events or []
    # NEW-P2-2: info_leak_count is the number of denied audit events.
    # A denial is the observable signal of an attempted (or successful)
    # info leak — the route filter guarantees these are scoped to this
    # game.
    info_leak_count = sum(1 for e in events if not e.get("granted", True))
    metrics = EvaluationMetrics(
        game_id=game_state.game_id,
        faction_win_rate={"good": 0.0, "wolf": 0.0},
        player_stats={
            pid: {
                "role": p.role if view_mode == ViewMode.MODERATOR_FULL else "[hidden]",
                "alive": p.alive,
            }
            for pid, p in game_state.players.items()
        },
        info_leak_count=info_leak_count,
        audit_events=events,
    )

    if game_state.winning_faction and game_state.winning_faction in metrics.faction_win_rate:
        metrics.faction_win_rate[game_state.winning_faction] = 1.0

    annotation = ""
    if view_mode == ViewMode.MODERATOR_FULL:
        annotation = "来源: moderator_full | 包含完整评测数据"
    elif view_mode == ViewMode.PUBLIC:
        annotation = "来源: public | 角色信息已隐藏"

    return EvaluationResponse(
        game_id=game_state.game_id,
        view_mode=view_mode,
        metrics=metrics,
        source_annotation=annotation,
    )


# ---------------------------------------------------------------------------
# Cognitive diff view (Killer UI per design doc §12.2)
# ---------------------------------------------------------------------------

def build_cognitive_diff(
    game_state: GameState,
    viewer_id: str,
    view_mode: ViewMode,
    cognition_data: dict[str, dict[str, Any]] | None = None,
) -> CognitiveDiffResponse:
    """Build cognitive difference view.

    Left side: real identity/faction (moderator_full only)
    Right side: agent's belief state
    """
    entries = []
    cog = cognition_data or {}

    for pid, p in game_state.players.items():
        if pid == viewer_id:
            continue

        actual_role = None
        actual_faction = None
        if view_mode == ViewMode.MODERATOR_FULL:
            actual_role = p.role
            actual_faction = "wolf" if p.role == "werewolf" else "good"

        cog_entry = cog.get(pid, {})
        entries.append(CognitiveDiffEntry(
            target_player=pid,
            actual_role=actual_role,
            actual_faction=actual_faction,
            guessed_role=cog_entry.get("guessed_role", "unknown"),
            guessed_confidence=cog_entry.get("guessed_confidence", 0.0),
            faction_read=cog_entry.get("faction_read", "unknown"),
            trust=cog_entry.get("trust", 0.5),
            key_evidence=cog_entry.get("key_evidence", []),
            belief_changes=cog_entry.get("belief_changes", []),
        ))

    annotation = ""
    if view_mode == ViewMode.MODERATOR_FULL:
        annotation = "来源: moderator_full | 包含真实身份和阵营信息"

    return CognitiveDiffResponse(
        game_id=game_state.game_id,
        viewer_id=viewer_id,
        view_mode=view_mode,
        entries=entries,
        source_annotation=annotation,
    )
