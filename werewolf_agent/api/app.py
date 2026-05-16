"""FastAPI app: API endpoints with permission enforcement.

Design doc §12.1 endpoints:
- POST /games — create game
- POST /games/{game_id}/start — start game
- POST /games/{game_id}/pause — pause
- POST /games/{game_id}/resume — resume
- GET /games/{game_id}/public-state — public state
- GET /games/{game_id}/players/{player_id}/private-state — private state
- GET /games/{game_id}/timeline — timeline
- GET /games/{game_id}/replay — replay
- GET /games/{game_id}/evaluation — evaluation
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query

from werewolf_agent.api.permissions import PermissionChecker, PermissionDenied
from werewolf_agent.api.schemas import (
    CallerRole,
    CognitiveDiffRequest,
    CognitiveDiffResponse,
    CreateGameRequest,
    ErrorResponse,
    EvaluationRequest,
    EvaluationResponse,
    GameActionRequest,
    GameActionResponse,
    GameCreateResponse,
    GameInfo,
    PrivateStateRequest,
    PrivateStateResponse,
    PublicStateResponse,
    ReplayRequest,
    ReplayResponse,
    TimelineRequest,
    TimelineResponse,
    ViewMode,
)
from werewolf_agent.api.views import (
    build_cognitive_diff,
    build_evaluation,
    build_private_state,
    build_public_state,
    build_replay,
    build_timeline,
)
from werewolf_agent.core.models import GameState, PlayerState


def create_app() -> FastAPI:
    app = FastAPI(title="Werewolf Agent API", version="1.0")
    checker = PermissionChecker()
    games: dict[str, GameState] = {}
    authorized_callers: dict[str, CallerRole] = {
        "mod1": CallerRole.MODERATOR,
        "dbg1": CallerRole.DEBUGGER,
    }

    @app.post("/games", response_model=GameCreateResponse)
    def create_game(req: CreateGameRequest) -> GameCreateResponse:
        game_id = req.seed is not None and f"game_{req.seed}" or str(uuid4())[:8]
        game_id = f"g_{game_id}" if not game_id.startswith("g_") else game_id
        state = GameState(
            game_id=game_id,
            ruleset_id=req.ruleset_id,
            phase="setup",
        )
        games[game_id] = state
        return GameCreateResponse(
            game=GameInfo(
                game_id=game_id,
                ruleset_id=req.ruleset_id,
                status="created",
                player_count=req.player_count,
            ),
            message="Game created",
        )

    @app.post("/games/{game_id}/start", response_model=GameActionResponse)
    def start_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        state = _get_game(games, game_id)
        if state.phase != "setup":
            raise HTTPException(400, "Game already started")
        # Initialize with 12 players
        roles = ["werewolf"] * 4 + ["villager"] * 3 + [
            "seer", "witch", "hunter", "idiot", "hybrid",
        ]
        for i, role in enumerate(roles):
            pid = f"p{i + 1:02d}"
            state.players[pid] = PlayerState(id=pid, role=role)
        object.__setattr__(state, "phase", "night")
        return GameActionResponse(
            game_id=game_id, action="start", success=True,
            message="Game started",
        )

    @app.post("/games/{game_id}/pause", response_model=GameActionResponse)
    def pause_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        state = _get_game(games, game_id)
        if state.paused:
            raise HTTPException(400, "Already paused")
        object.__setattr__(state, "paused", True)
        return GameActionResponse(
            game_id=game_id, action="pause", success=True,
            message="Game paused",
        )

    @app.post("/games/{game_id}/resume", response_model=GameActionResponse)
    def resume_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        state = _get_game(games, game_id)
        if not state.paused:
            raise HTTPException(400, "Not paused")
        object.__setattr__(state, "paused", False)
        return GameActionResponse(
            game_id=game_id, action="resume", success=True,
            message="Game resumed",
        )

    @app.get("/games/{game_id}/public-state", response_model=PublicStateResponse)
    def get_public_state(game_id: str) -> PublicStateResponse:
        state = _get_game(games, game_id)
        return build_public_state(state)

    @app.get(
        "/games/{game_id}/players/{player_id}/private-state",
        response_model=PrivateStateResponse,
    )
    def get_private_state(
        game_id: str,
        player_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.PLAYER_AGENT),
        view_mode: ViewMode = Query(ViewMode.PLAYER_VIEW),
    ) -> PrivateStateResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(authorized_callers, caller_id, caller_role)

        try:
            allowed_view = checker.check_private_state(
                caller_id=caller_id,
                caller_role=caller_role,
                target_player_id=player_id,
                game_id=game_id,
                endpoint="private-state",
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)

        return build_private_state(state, player_id, allowed_view)

    @app.get("/games/{game_id}/timeline", response_model=TimelineResponse)
    def get_timeline(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
        view_mode: ViewMode = Query(ViewMode.PUBLIC),
        day_filter: int | None = Query(None),
    ) -> TimelineResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(authorized_callers, caller_id, caller_role)

        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="timeline",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)

        return build_timeline(state, allowed_view, day_filter, viewer_id=caller_id)

    @app.get("/games/{game_id}/replay", response_model=ReplayResponse)
    def get_replay(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
    ) -> ReplayResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(authorized_callers, caller_id, caller_role)

        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="replay",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)

        return build_replay(state, allowed_view, viewer_id=caller_id)

    @app.get("/games/{game_id}/evaluation", response_model=EvaluationResponse)
    def get_evaluation(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
    ) -> EvaluationResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(authorized_callers, caller_id, caller_role)

        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="evaluation",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)

        return build_evaluation(
            state, allowed_view,
            audit_events=[e.model_dump() for e in checker.audit_log()],
        )

    @app.get("/games/{game_id}/cognitive-diff", response_model=CognitiveDiffResponse)
    def get_cognitive_diff(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.DEBUGGER),
        player_id: str = Query(""),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
    ) -> CognitiveDiffResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(authorized_callers, caller_id, caller_role)

        try:
            allowed_view = checker.check_cognitive_diff(
                caller_id=caller_id,
                caller_role=caller_role,
                game_id=game_id,
                endpoint="cognitive-diff",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)

        return build_cognitive_diff(state, player_id or "p01", allowed_view)

    @app.get("/games", response_model=list[GameInfo])
    def list_games() -> list[GameInfo]:
        return [
            GameInfo(
                game_id=gid,
                ruleset_id=s.ruleset_id,
                status="active" if s.winning_faction is None else "ended",
                player_count=len(s.players),
            )
            for gid, s in games.items()
        ]

    # Expose for testing
    app.state.games = games
    app.state.checker = checker
    app.state.authorized_callers = authorized_callers

    return app


def _get_game(games: dict[str, GameState], game_id: str) -> GameState:
    state = games.get(game_id)
    if state is None:
        raise HTTPException(404, f"Game {game_id} not found")
    return state


def _resolve_caller_role(
    authorized_callers: dict[str, CallerRole],
    caller_id: str,
    requested_role: CallerRole,
) -> CallerRole:
    if requested_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        if caller_id and authorized_callers.get(caller_id) == requested_role:
            return requested_role
        raise HTTPException(403, "Elevated caller role is not authorized")
    return requested_role
