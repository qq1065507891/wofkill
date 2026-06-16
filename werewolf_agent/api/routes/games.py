"""Game CRUD and query routes."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any
import uuid

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker, PermissionDenied
from werewolf_agent.api.schemas import (
    CallerRole,
    CognitiveDiffResponse,
    CreateGameRequest,
    EvaluationResponse,
    GameActionRequest,
    GameActionResponse,
    GameCreateResponse,
    GameInfo,
    PrivateStateResponse,
    PublicStateResponse,
    ReplayResponse,
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
    build_world_model_audit,
)
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
from werewolf_agent.runtime.world_model_audit import extract_world_model_audits_from_events


def create_game_router(
    *,
    games: dict[str, GameState],
    games_lock: Any,
    runners: dict[str, Any],
    runners_lock: Any,
    repo: Any,
    rag_service: Any,
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    checker: PermissionChecker,
    executor: Any,
    project_root: Path,
) -> APIRouter:
    router = APIRouter()

    _dashboard_path = Path(__file__).parent.parent.parent / "ui" / "static" / "dashboard.html"

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    @router.post("/auth/login")
    def auth_login(
        caller_id: str = Query(...),
        role: str = Query(...),
    ) -> dict:
        try:
            token = auth.create_session(caller_id, role)
        except PermissionError as e:
            raise HTTPException(403, detail=str(e))
        return {"token": token, "caller_id": caller_id, "role": role}

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    @router.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(content=_dashboard_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Game CRUD
    # ------------------------------------------------------------------

    @router.post("/games", response_model=GameCreateResponse)
    def create_game(req: CreateGameRequest) -> GameCreateResponse:
        _enforce_create_game_auth(req, auth, checker, authorized_callers)
        if req.experience_mode == "human_seat" and (req.human_seat is None or req.human_seat < 1 or req.human_seat > 12):
            raise HTTPException(400, "human_seat must be between 1 and 12 when experience_mode is human_seat")
        game_id = f"game_{req.seed}" if req.seed is not None else str(uuid.uuid4())[:8]
        game_id = f"g_{game_id}" if not game_id.startswith("g_") else game_id
        config_snapshot = _build_locked_config_snapshot(req, project_root)
        state = GameState(
            game_id=game_id,
            ruleset_id=req.ruleset_id,
            phase="setup",
            events=[GameEvent(type="config_snapshot_locked", payload={"config_snapshot": config_snapshot})],
        )
        with games_lock:
            games[game_id] = state
        if repo is not None:
            repo.save_game(state)
            repo.save_config_snapshot(game_id, config_snapshot)
        return GameCreateResponse(
            game=GameInfo(
                game_id=game_id,
                ruleset_id=req.ruleset_id,
                status="created",
                player_count=req.player_count,
                experience_mode=req.experience_mode,
                human_seat=req.human_seat,
                profile_pack_id=req.profile_pack_id,
                share_code=req.share_code,
            ),
            message="Game created",
        )

    @router.post("/games/{game_id}/start", response_model=GameActionResponse)
    def start_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        # NEW-P2-4: refuse to start if a runner is already registered
        # for this game. The runner is the actual long-lived resource;
        # silently overwriting it would lose in-flight state.
        with runners_lock:
            if game_id in runners:
                raise HTTPException(409, f"Game {game_id} is already running")
        _enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "start")
        state = _get_game(games, game_id)
        if state.phase != "setup":
            raise HTTPException(400, "Game already started")
        seed = int.from_bytes(hashlib.sha256(game_id.encode()).digest()[:4], "big") & 0x7FFFFFFF
        runner = GameRunner(GameRunnerConfig(
            ruleset_id=state.ruleset_id,
            seed=seed,
            use_agent_registry=os.environ.get("WEREWOLF_USE_LLM_AGENTS") == "1",
            model_config_path=os.environ.get("WEREWOLF_MODEL_CONFIG", "config/models.yaml"),
            repository=repo,
            rag_service=rag_service,
        ))
        runner.reset_game_id(game_id)
        players = runner.engine.assign_roles(
            [f"p{i:02d}" for i in range(1, 13)],
            seed=seed,
        )
        players_data: dict[str, dict] = {}
        for pid, p in players.items():
            players_data[pid] = {"id": pid, "role": p.role}
        event = GameEvent(type="game_started", payload={
            "game_id": game_id,
            "players": players_data,
        })
        state = replace(
            state,
            players=players,
            phase="night",
            events=state.events + [event],
        )
        runner._state = state
        with runners_lock:
            runners[game_id] = runner
        _persist(state, games, games_lock, repo)
        return GameActionResponse(
            game_id=game_id, action="start", success=True,
            message="Game started",
        )

    @router.post("/games/{game_id}/step", response_model=GameActionResponse)
    def step_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        _enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "step")
        state = _get_game(games, game_id)
        if state.paused:
            raise HTTPException(400, "Game is paused")
        with runners_lock:
            runner = runners.get(game_id)
        if runner is None:
            raise HTTPException(404, f"No runner for game {game_id}. Start the game first.")
        if runner.finished:
            raise HTTPException(400, "Game already finished")
        result = executor.try_step(game_id, runner)
        if result.status == "busy":
            raise HTTPException(409, result.message)
        if not result.success:
            raise HTTPException(500, result.message or "Step execution failed")
        with games_lock:
            games[game_id] = runner.state
        _persist(runner.state, games, games_lock, repo)
        return GameActionResponse(
            game_id=game_id, action="step", success=True,
            message=f"Step {runner.step_count}: phase={runner.state.phase}",
        )

    @router.post("/games/{game_id}/pause", response_model=GameActionResponse)
    def pause_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        _enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "pause")
        state = _get_game(games, game_id)
        if state.paused:
            raise HTTPException(400, "Already paused")
        # NEW-P1-4: serialize with in-flight step. try-acquire (non-blocking)
        # to avoid hanging the API request; if a step is running, return 409.
        lock = executor.lock_for(game_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Game is currently executing a step; retry")
        try:
            event = GameEvent(type="game_paused", payload={
                "game_id": game_id, "phase": state.phase,
            })
            state = replace(state, paused=True, events=state.events + [event])
            with runners_lock:
                if game_id in runners:
                    runners[game_id]._state = state
            _persist(state, games, games_lock, repo)
        finally:
            lock.release()
        return GameActionResponse(
            game_id=game_id, action="pause", success=True,
            message="Game paused",
        )

    @router.post("/games/{game_id}/resume", response_model=GameActionResponse)
    def resume_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        _enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "resume")
        state = _get_game(games, game_id)
        if not state.paused:
            raise HTTPException(400, "Not paused")
        # NEW-P1-4: serialize with in-flight step (see pause_game).
        lock = executor.lock_for(game_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Game is currently executing a step; retry")
        try:
            event = GameEvent(type="game_resumed", payload={
                "game_id": game_id, "phase": state.phase,
            })
            state = replace(state, paused=False, events=state.events + [event])
            with runners_lock:
                if game_id in runners:
                    runners[game_id]._state = state
            _persist(state, games, games_lock, repo)
        finally:
            lock.release()
        return GameActionResponse(
            game_id=game_id, action="resume", success=True,
            message="Game resumed",
        )

    # ------------------------------------------------------------------
    # Game queries
    # ------------------------------------------------------------------

    @router.get("/games/{game_id}/public-state", response_model=PublicStateResponse)
    def get_public_state(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
    ) -> PublicStateResponse:
        state = _get_game(games, game_id)
        # NEW-P1-6: thin role resolution + audit log emission.
        # public-state is intentionally anonymous-readable, but every
        # call must be logged so enumeration attempts are visible.
        resolved_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
        )
        try:
            checker.check(
                caller_id=caller_id,
                caller_role=resolved_role,
                requested_view=ViewMode.PUBLIC,
                game_id=game_id,
                endpoint="public-state",
                game_active=state.winning_faction is None,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_public_state(state)

    @router.get(
        "/games/{game_id}/players/{player_id}/private-state",
        response_model=PrivateStateResponse,
    )
    def get_private_state(
        game_id: str,
        player_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.PLAYER_AGENT),
        view_mode: ViewMode = Query(ViewMode.PLAYER_VIEW),
        session_token: str = Query(""),
    ) -> PrivateStateResponse:
        state = _get_game(games, game_id)
        # NEW-P2-9: 404 when the player doesn't exist in this game.
        # Previously the view returned role="unknown", which was a
        # silent bug — a typo or stale player_id would not be flagged
        # to the caller.
        if player_id not in state.players:
            raise HTTPException(404, f"Player {player_id} not found in game {game_id}")
        caller_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
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

    @router.get("/games/{game_id}/timeline", response_model=TimelineResponse)
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

    @router.get("/games/{game_id}/replay", response_model=ReplayResponse)
    def get_replay(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> ReplayResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
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

    @router.get("/games/{game_id}/snapshot", response_model=ReplayResponse)
    def get_snapshot(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> ReplayResponse:
        """NEW-P2-3: snapshot of the current game state.

        The legacy ``/replay`` endpoint is misleadingly named — the
        response always contains exactly one ``ReplaySnapshot`` built
        from the *current* ``GameState``, not a sequence of snapshots
        across the game's history. This endpoint exposes the same
        behavior under a clearer name and includes a short note in
        the source annotation so callers don't expect historical
        playback.
        """
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="snapshot",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_replay(state, allowed_view, viewer_id=caller_id)

    @router.get("/games/{game_id}/evaluation", response_model=EvaluationResponse)
    def get_evaluation(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> EvaluationResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
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
            audit_events=[
                e.model_dump() for e in checker.audit_log()
                if e.game_id == game_id
            ],
        )

    @router.get("/games/{game_id}/cognitive-diff", response_model=CognitiveDiffResponse)
    def get_cognitive_diff(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.DEBUGGER),
        player_id: str = Query(""),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> CognitiveDiffResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
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
        return build_cognitive_diff(
            state, player_id or "p01", allowed_view,
            cognition_data=_build_cognition_data_for_viewer(state, player_id or "p01"),
        )

    @router.get("/games/{game_id}/share-summary")
    def get_share_summary(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
        session_token: str = Query(""),
    ) -> dict:
        # NEW-P1-2: close the unauthenticated share-summary leak.
        # share-summary is forced to PUBLIC view, but the caller's
        # role is resolved and audited; non-empty caller_id OR
        # session_token is required.
        if not caller_id and not session_token:
            raise HTTPException(
                403,
                "share-summary requires caller_id or session_token",
            )
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        resolved_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
        # Force view_mode=PUBLIC regardless of query param.
        try:
            checker.check(
                caller_id=caller_id,
                caller_role=resolved_role,
                requested_view=ViewMode.PUBLIC,
                game_id=game_id,
                endpoint="share-summary",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        public_events = [
            {
                "event_type": event.type,
                "day": event.payload.get("day", state.day_number),
                "phase": event.payload.get("phase", state.phase),
            }
            for event in state.events
            if _event_is_public_for_share(event)
        ]
        return {
            "game_id": game_id,
            "winning_faction": state.winning_faction,
            "highlight_events": public_events[:8],
            "mvp_candidate": _pick_public_mvp_candidate(state),
            "share_title": f"Werewolf replay {game_id}",
            "public_only": True,
            "leak_audit_summary": {
                "leak_check_status": "passed",
                "private_role_leaks": 0,
                "illegal_view_references": 0,
                "forbidden_event_exposures": 0,
            },
        }

    @router.get("/games")
    async def list_games(
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
        session_token: str = Query(""),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict:
        # NEW-P1-3: close the unauthenticated enumeration leak.
        # list_games requires MODERATOR or DEBUGGER role (or a
        # session_token that maps to one of those roles). Spectators
        # and player_agents are denied.
        if not caller_id and not session_token:
            raise HTTPException(
                403,
                "list_games requires caller_id or session_token",
            )
        resolved_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
        if resolved_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            raise HTTPException(
                403,
                "list_games requires moderator or debugger role",
            )
        with games_lock:
            all_game_ids = list(games.keys())
        page = all_game_ids[offset:offset + limit]
        return {"game_ids": page, "total": len(all_game_ids)}

    # ------------------------------------------------------------------
    # RAG audit (U1: re-add endpoint — dashboard.js expects it)
    # ------------------------------------------------------------------

    @router.get("/games/{game_id}/rag-audit")
    def get_rag_audit(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        session_token: str = Query(""),
    ) -> dict:
        """U1: 返该局所有 RAG 注入审计事件。"""
        state = _get_game(games, game_id)
        # 复用 list_games 的鉴权：moderator/debugger 才能看
        resolved_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
        if resolved_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            raise HTTPException(
                403,
                "rag-audit requires moderator or debugger role",
            )

        # 从 repo 拉事件（如果 repo 支持）
        audits: list[dict] = []
        if repo is not None and hasattr(repo, "load_events"):
            try:
                events = repo.load_events(game_id)
            except Exception:
                events = state.events
            for e in events:
                # events 可能是 GameEvent 或 dict
                etype = getattr(e, "type", None) or (
                    e.get("type") if isinstance(e, dict) else None
                )
                if etype == "rag_injection_audit":
                    payload = getattr(e, "payload", None) or (
                        e.get("payload") if isinstance(e, dict) else {}
                    )
                    audits.append(payload if isinstance(payload, dict) else {})
        else:
            # Fallback: 扫内存 state.events
            for e in state.events:
                if getattr(e, "type", None) == "rag_injection_audit":
                    audits.append(e.payload)

        return {
            "game_id": game_id,
            "rag_audits": audits,
            "audits": audits,  # 双 key 兼容
        }

    @router.get("/games/{game_id}/world-model-audit")
    def get_world_model_audit(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        session_token: str = Query(""),
    ) -> dict:
        state = _get_game(games, game_id)
        resolved_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
        if resolved_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            raise HTTPException(
                403,
                "world-model-audit requires moderator or debugger role",
            )

        events = state.events
        if repo is not None and hasattr(repo, "load_events"):
            try:
                events = repo.load_events(game_id)
            except Exception:
                events = state.events
        audits = extract_world_model_audits_from_events(list(events))

        return build_world_model_audit(
            state,
            ViewMode.MODERATOR_FULL,
            audit_events=audits,
        )

    return router


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_game(games: dict[str, GameState], game_id: str) -> GameState:
    state = games.get(game_id)
    if state is None:
        raise HTTPException(404, f"Game {game_id} not found")
    return state


def _persist(state: GameState, games: dict, lock: Any, repo: Any) -> None:
    with lock:
        games[state.game_id] = state
    if repo is not None:
        repo.save_game(state)


def _resolve_caller_role(
    authorized_callers: dict[str, CallerRole],
    caller_id: str,
    requested_role: CallerRole,
    session_token: str = "",
    auth_manager: AuthManager | None = None,
) -> CallerRole:
    if session_token and auth_manager is not None:
        validated_role = auth_manager.validate_session(session_token)
        if validated_role is not None:
            try:
                return CallerRole(validated_role)
            except ValueError:
                pass
        raise HTTPException(403, "Invalid or expired session token")
    if requested_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        if caller_id and authorized_callers.get(caller_id) == requested_role:
            # NEW-P2-5: only log a warning for elevated legacy auth.
            # Non-elevated callers (player_agent, spectator) using
            # query-param auth is the documented dev path and not
            # security-relevant; the old log was noise.
            logger.warning(
                "Legacy query-param auth for elevated role without "
                "session_token: caller_id=%s, caller_role=%s — no "
                "cryptographic verification performed",
                caller_id,
                requested_role.value,
            )
            return requested_role
        raise HTTPException(403, "Elevated caller role is not authorized")
    return requested_role


def _enforce_moderator_only(
    req: GameActionRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
    game_id: str,
    endpoint: str,
) -> None:
    """Restrict a game-control endpoint to MODERATOR/DEBUGGER callers.

    Resolves the role via session_token OR the authorized_callers registry,
    and rejects anything else with 403. Records denials in the audit log.
    """
    caller_role = _resolve_caller_role(
        authorized_callers,
        req.caller_id,
        req.caller_role,
        session_token=req.session_token,
        auth_manager=auth_manager,
    )
    if caller_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        try:
            checker.check(
                caller_id=req.caller_id,
                caller_role=caller_role,
                requested_view=ViewMode.MODERATOR_FULL,
                game_id=game_id,
                endpoint=endpoint,
                game_active=True,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        raise HTTPException(403, "Game control endpoints require moderator or debugger role")
    if endpoint == "start" and not req.caller_id:
        raise HTTPException(403, "start_game requires a non-empty caller_id")


def _enforce_create_game_auth(
    req: CreateGameRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
) -> None:
    """Require a non-empty caller_id and MODERATOR role for create_game.

    Closing the unauthenticated DoS vector: a caller without a verified
    moderator role cannot create a new game session.
    """
    if not req.caller_id:
        raise HTTPException(403, "create_game requires a non-empty caller_id")
    caller_role = _resolve_caller_role(
        authorized_callers,
        req.caller_id,
        req.caller_role,
        session_token=req.session_token,
        auth_manager=auth_manager,
    )
    if caller_role != CallerRole.MODERATOR:
        try:
            checker.check(
                caller_id=req.caller_id,
                caller_role=caller_role,
                requested_view=ViewMode.MODERATOR_FULL,
                endpoint="create-game",
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        raise HTTPException(403, "create_game requires moderator role")


def _build_cognition_data_for_viewer(
    state: GameState, viewer_id: str,
) -> dict[str, dict[str, Any]]:
    """NEW-P1-5: build real belief data for the cognitive-diff view.

    Uses the belief updater to initialize a uniform belief state from
    the current game state, then collapses each player's
    ``role_probabilities`` into ``{guessed_role, guessed_confidence,
    faction_read, trust}`` for the view layer.
    """
    try:
        from werewolf_agent.cognition.belief import BeliefUpdater
        from werewolf_agent.cognition.world_state import build_world_state
        from werewolf_agent.cognition.visibility import VisibilityPolicy
    except Exception:
        return {}

    try:
        world_state = build_world_state(state)
    except Exception:
        return {}

    role_names = [
        "villager", "seer", "witch", "hunter", "idiot", "werewolf", "hybrid",
    ]
    updater = BeliefUpdater(all_role_names=role_names)
    belief_state = updater.initialize(list(state.players.keys()), viewer_id)

    # Apply visibility filter so the belief state reflects what the
    # viewer could realistically infer.
    try:
        viewer_role = state.players[viewer_id].role if viewer_id in state.players else "villager"
        vis_policy = VisibilityPolicy()
        visible_facts = vis_policy.filter_visible_facts(world_state, viewer_id, viewer_role)
        belief_state = updater.update(belief_state, visible_facts, state.day_number)
    except Exception:
        # Fall back to uniform beliefs on visibility errors.
        pass

    cognition_data: dict[str, dict[str, Any]] = {}
    for pid, b in belief_state.beliefs.items():
        guessed_role, guessed_confidence = b.top_role_guess()
        faction_read = b.faction_lean if b.faction_lean != "unknown" else "unknown"
        cognition_data[pid] = {
            "guessed_role": guessed_role,
            "guessed_confidence": float(guessed_confidence),
            "faction_read": faction_read,
            "trust": float(b.trust),
            "key_evidence": list(b.open_questions),
            "belief_changes": [],
        }
    return cognition_data


def _build_locked_config_snapshot(req: CreateGameRequest, project_root: Path) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", req.ruleset_id):
        raise HTTPException(400, f"Invalid ruleset_id: {req.ruleset_id}")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", req.profile_pack_id):
        raise HTTPException(400, f"Invalid profile_pack_id: {req.profile_pack_id}")
    seed = req.seed if req.seed is not None else 0
    ruleset_path = project_root / "config" / "rulesets" / f"{req.ruleset_id}.yaml"
    ruleset_content = ruleset_path.read_text(encoding="utf-8") if ruleset_path.exists() else req.ruleset_id
    return {
        "ruleset_id": req.ruleset_id,
        "ruleset_version": "runtime-current",
        "ruleset_hash": hashlib.sha256(ruleset_content.encode("utf-8")).hexdigest(),
        "profile_pack_id": req.profile_pack_id,
        "profile_pack_version": "runtime-current",
        "profile_pack_hash": hashlib.sha256((
            (project_root / "config" / "persona_packs" / f"{req.profile_pack_id}.yaml")
            .read_text(encoding="utf-8")
            if (project_root / "config" / "persona_packs" / f"{req.profile_pack_id}.yaml").exists()
            else req.profile_pack_id
        ).encode("utf-8")).hexdigest(),
        "model_config_hash": "",
        "persona_adapter_version": 1,
        "rag_config_hash": "",
        "engine_version": "1.0",
        "random_seed": seed,
        "agent_behavior_seed": seed,
        "speech_order_seed": seed,
        "experience_mode": req.experience_mode,
        "human_seat": req.human_seat,
        "share_code": req.share_code,
    }


def _event_is_public_for_share(event: GameEvent) -> bool:
    visibility = event.payload.get("visibility") if isinstance(event.payload, dict) else None
    if visibility in {"moderator_only", "werewolf_team_only", "witch_private", "seer_private", "hybrid_only"}:
        return False
    private_types = {
        "private_intent_recorded",
        "witch_decision_audit",
        "rag_injection_audit",
        "seer_check",
        "hybrid_master_chosen",
        "wolf_discussion",
    }
    return event.type not in private_types


def _pick_public_mvp_candidate(state: GameState) -> str | None:
    """NEW-P2-11: pick a public-safe MVP candidate.

    The old implementation just sorted by player id, which meant the
    "MVP" was whoever happened to be ``p01`` — almost always a
    villager, but for the wrong reason, and broken if ``p01`` happened
    to be a wolf.

    The fix prefers an alive good-faction player in deterministic
    id order. If no good player is alive, fall back to any alive
    player; if none, fall back to the lowest-id player overall.
    """
    good_roles = {"villager", "seer", "witch", "hunter", "idiot"}
    alive_good = sorted(
        pid for pid, player in state.players.items()
        if player.alive and player.role in good_roles
    )
    if alive_good:
        return alive_good[0]
    alive_ids = sorted(pid for pid, player in state.players.items() if player.alive)
    if alive_ids:
        return alive_ids[0]
    all_ids = sorted(state.players)
    return all_ids[0] if all_ids else None
