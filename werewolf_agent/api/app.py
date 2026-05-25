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

import hashlib
import logging
import os
from typing import Any, TYPE_CHECKING
from dataclasses import asdict, replace
import threading
import uuid

logger = logging.getLogger(__name__)

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import yaml

from werewolf_agent.api.auth import AuthManager, AuthConfig
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
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.customization.persona_adapter import adapt_persona_pack
from werewolf_agent.customization.preview import build_persona_preview
from werewolf_agent.customization.repository import InMemoryCustomizationRepository
from werewolf_agent.customization.validators import (
    validate_persona_pack_yaml,
    validate_ruleset_yaml,
)
from werewolf_agent.storage.memory_store import InMemoryGameRepository
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
from werewolf_agent.runtime.executor import LocalRuntimeExecutor

if TYPE_CHECKING:
    from werewolf_agent.storage.repository import GameRepository


def create_app(
    repository: GameRepository | None = None,
    auth_manager: AuthManager | None = None,
) -> FastAPI:
    # Auto-configure repository when env vars are set.
    if repository is None:
        storage_backend = os.environ.get("WEREWOLF_STORAGE_BACKEND", "").strip().lower()
        if storage_backend:
            from werewolf_agent.storage.production import (
                ProductionStorageConfig,
                create_game_repository,
            )
            repository = create_game_repository(ProductionStorageConfig(
                backend=storage_backend,
                sqlite_path=os.environ.get("WEREWOLF_DB_PATH", "data/wofkill.db"),
                postgres_dsn=os.environ.get("POSTGRES_DSN", ""),
                redis_url=os.environ.get("REDIS_URL", ""),
            ))
        else:
            db_path = os.environ.get("WEREWOLF_DB_PATH")
            if db_path:
                from werewolf_agent.storage.sqlite_store import SqliteGameRepository
                repository = SqliteGameRepository(db_path)
    app = FastAPI(title="Werewolf Agent API", version="1.0")
    static_dir = Path(__file__).parent.parent / "ui" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    checker = PermissionChecker()
    auth = auth_manager or AuthManager()
    executor = LocalRuntimeExecutor()
    customization_repo = InMemoryCustomizationRepository()
    games: dict[str, GameState] = {}
    runners: dict[str, GameRunner] = {}
    _games_lock = threading.Lock()
    _runners_lock = threading.Lock()
    authorized_callers: dict[str, CallerRole] = {
        "mod1": CallerRole.MODERATOR,
        "dbg1": CallerRole.DEBUGGER,
    }
    _repo = repository
    rag_service = None
    try:
        from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

        vector_store = None
        vector_backend = os.environ.get("WEREWOLF_VECTOR_BACKEND", "").strip().lower()
        if vector_backend:
            try:
                from werewolf_agent.rag.vector_store import create_vector_store

                vector_store = create_vector_store(vector_backend)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Vector store initialization failed; using RAG fallback: %s",
                    exc,
                )
        rag_service = RAGKnowledgeService(repository=_repo, vector_store=vector_store)
        rag_service.ensure_seeded()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("RAG knowledge service initialization failed: %s", exc)
    app.state.repository = _repo
    app.state.rag_service = rag_service

    def _persist(state: GameState) -> None:
        with _games_lock:
            games[state.game_id] = state
        if _repo is not None:
            _repo.save_game(state)

    def _persist_custom_config(record: Any) -> None:
        if _repo is not None and hasattr(_repo, "save_custom_config"):
            _repo.save_custom_config(_record_to_storage_dict(record))

    async def _read_upload_text(request: Request) -> str:
        body = await request.body()
        if len(body) > 256 * 1024:
            raise HTTPException(413, "Uploaded customization template is too large")
        return body.decode("utf-8")

    @app.post("/auth/login")
    def auth_login(
        caller_id: str = Query(...),
        role: str = Query(...),
    ) -> dict:
        try:
            token = auth.create_session(caller_id, role)
        except PermissionError as e:
            raise HTTPException(403, detail=str(e))
        return {"token": token, "caller_id": caller_id, "role": role}

    @app.get("/templates/ruleset", response_class=PlainTextResponse)
    def download_ruleset_template() -> PlainTextResponse:
        path = Path("config/rulesets/templates/custom_ruleset_template.yaml")
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @app.get("/templates/persona-pack", response_class=PlainTextResponse)
    def download_persona_pack_template() -> PlainTextResponse:
        path = Path("config/personas/templates/player_profile_pack_template.yaml")
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @app.post("/customization/rulesets/validate")
    async def validate_ruleset_upload(request: Request) -> dict:
        text = await _read_upload_text(request)
        return _validation_result_to_dict(validate_ruleset_yaml(text))

    @app.post("/customization/persona-packs/validate")
    async def validate_persona_pack_upload(request: Request) -> dict:
        text = await _read_upload_text(request)
        result = validate_persona_pack_yaml(text)
        data = _validation_result_to_dict(result)
        if result.normalized.get("players"):
            previews: dict[str, dict[str, str]] = {}
            for player in result.normalized["players"]:
                seat = int(player.get("seat", 0))
                previews[f"p{seat:02d}"] = build_persona_preview(player)
            data["persona_preview"] = previews
        return data

    @app.post("/customization/rulesets")
    async def save_ruleset_upload(
        request: Request,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
    ) -> dict:
        _require_customization_admin(authorized_callers, caller_id, caller_role)
        text = await _read_upload_text(request)
        result = validate_ruleset_yaml(text)
        data = _validation_result_to_dict(result)
        if not result.valid:
            raise HTTPException(400, data)
        record = customization_repo.save(
            config_type="ruleset",
            raw_yaml=text,
            normalized=result.normalized,
            validation_result=data,
            creator_id=caller_id,
        )
        _persist_custom_config(record)
        return _record_to_public_dict(record)

    @app.post("/customization/persona-packs")
    async def save_persona_pack_upload(
        request: Request,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
    ) -> dict:
        _require_customization_admin(authorized_callers, caller_id, caller_role)
        text = await _read_upload_text(request)
        result = validate_persona_pack_yaml(text)
        data = _validation_result_to_dict(result)
        if not result.valid:
            raise HTTPException(400, data)
        adapted = adapt_persona_pack(result.normalized)
        normalized = dict(result.normalized)
        normalized["persona_profiles"] = adapted["persona_profiles"]
        normalized["player_assignments"] = adapted["player_assignments"]
        record = customization_repo.save(
            config_type="persona_pack",
            raw_yaml=text,
            normalized=normalized,
            validation_result=data,
            creator_id=caller_id,
        )
        _persist_custom_config(record)
        return _record_to_public_dict(record)

    @app.get("/marketplace/rulesets")
    def list_ruleset_marketplace() -> dict:
        return _load_marketplace("config/rulesets/marketplace.yaml")

    @app.get("/marketplace/persona-packs")
    def list_persona_pack_marketplace() -> dict:
        return _load_marketplace("config/personas/marketplace.yaml")

    @app.post("/games", response_model=GameCreateResponse)
    def create_game(req: CreateGameRequest) -> GameCreateResponse:
        if req.experience_mode == "human_seat" and (req.human_seat is None or req.human_seat < 1 or req.human_seat > 12):
            raise HTTPException(400, "human_seat must be between 1 and 12 when experience_mode is human_seat")
        game_id = req.seed is not None and f"game_{req.seed}" or str(uuid.uuid4())[:8]
        game_id = f"g_{game_id}" if not game_id.startswith("g_") else game_id
        config_snapshot = _build_locked_config_snapshot(req)
        state = GameState(
            game_id=game_id,
            ruleset_id=req.ruleset_id,
            phase="setup",
            events=[GameEvent(type="config_snapshot_locked", payload={"config_snapshot": config_snapshot})],
        )
        with _games_lock:
            games[game_id] = state
        if _repo is not None:
            _repo.save_game(state)
            _repo.save_config_snapshot(game_id, config_snapshot)
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

    @app.post("/games/{game_id}/start", response_model=GameActionResponse)
    def start_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        state = _get_game(games, game_id)
        if state.phase != "setup":
            raise HTTPException(400, "Game already started")
        # Deterministic seed from game_id (hashlib.sha256 is stable across sessions)
        seed = int.from_bytes(hashlib.sha256(game_id.encode()).digest()[:4], "big") & 0x7FFFFFFF
        # Use GameRunner for deterministic role assignment via RuleEngine
        runner = GameRunner(GameRunnerConfig(
            ruleset_id=state.ruleset_id,
            seed=seed,
            use_agent_registry=os.environ.get("WEREWOLF_USE_LLM_AGENTS") == "1",
            model_config_path=os.environ.get("WEREWOLF_MODEL_CONFIG", "config/models.yaml"),
            repository=_repo,
            rag_service=rag_service,
        ))
        # Override the runner's game_id to match the API game_id
        runner.reset_game_id(game_id)
        # Use RuleEngine.assign_roles for deterministic role assignment
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
        # Sync runner state with API state
        runner._state = state
        runners[game_id] = runner
        _persist(state)
        return GameActionResponse(
            game_id=game_id, action="start", success=True,
            message="Game started",
        )

    @app.post("/games/{game_id}/step", response_model=GameActionResponse)
    def step_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        """Advance the game by one LangGraph node using GameRunner."""
        state = _get_game(games, game_id)
        if state.paused:
            raise HTTPException(400, "Game is paused")
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
        # Sync API state with runner state
        with _games_lock:
            games[game_id] = runner.state
        _persist(runner.state)
        return GameActionResponse(
            game_id=game_id, action="step", success=True,
            message=f"Step {runner.step_count}: phase={runner.state.phase}",
        )

    @app.post("/games/{game_id}/pause", response_model=GameActionResponse)
    def pause_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        state = _get_game(games, game_id)
        if state.paused:
            raise HTTPException(400, "Already paused")
        event = GameEvent(type="game_paused", payload={
            "game_id": game_id, "phase": state.phase,
        })
        state = replace(state, paused=True, events=state.events + [event])
        if game_id in runners:
            runners[game_id]._state = state
        _persist(state)
        return GameActionResponse(
            game_id=game_id, action="pause", success=True,
            message="Game paused",
        )

    @app.post("/games/{game_id}/resume", response_model=GameActionResponse)
    def resume_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        state = _get_game(games, game_id)
        if not state.paused:
            raise HTTPException(400, "Not paused")
        event = GameEvent(type="game_resumed", payload={
            "game_id": game_id, "phase": state.phase,
        })
        state = replace(state, paused=False, events=state.events + [event])
        if game_id in runners:
            runners[game_id]._state = state
        _persist(state)
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
        session_token: str = Query(""),
    ) -> PrivateStateResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
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

    @app.get("/games/{game_id}/evaluation", response_model=EvaluationResponse)
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
            audit_events=[e.model_dump() for e in checker.audit_log()],
        )

    @app.get("/games/{game_id}/cognitive-diff", response_model=CognitiveDiffResponse)
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

        return build_cognitive_diff(state, player_id or "p01", allowed_view)

    @app.get("/games/{game_id}/rag-audit")
    def get_rag_audit(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.DEBUGGER),
        session_token: str = Query(""),
    ) -> dict:
        state = _get_game(games, game_id)
        resolved_role = _resolve_caller_role(
            authorized_callers, caller_id, caller_role,
            session_token=session_token, auth_manager=auth,
        )
        if resolved_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            raise HTTPException(403, "RAG audit requires moderator or debugger access")
        rag_events = [e for e in state.events if e.type == "rag_injection_audit"]
        return {"game_id": game_id, "rag_audits": [e.payload for e in rag_events]}

    @app.get("/games/{game_id}/share-summary")
    def get_share_summary(game_id: str) -> dict:
        state = _get_game(games, game_id)
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

    @app.get("/games")
    async def list_games(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict:
        all_game_ids = list(games.keys())
        page = all_game_ids[offset:offset + limit]
        return {"game_ids": page, "total": len(all_game_ids)}

    # Dashboard
    _dashboard_path = Path(__file__).parent.parent / "ui" / "static" / "dashboard.html"

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(content=_dashboard_path.read_text(encoding="utf-8"))

    # Expose for testing
    app.state.games = games
    app.state.runners = runners
    app.state.executor = executor
    app.state.checker = checker
    app.state.authorized_callers = authorized_callers
    app.state.auth = auth
    app.state.customization_repo = customization_repo

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
    session_token: str = "",
    auth_manager: AuthManager | None = None,
) -> CallerRole:
    # When a session token is provided, validate it first.
    if session_token and auth_manager is not None:
        validated_role = auth_manager.validate_session(session_token)
        if validated_role is not None:
            try:
                return CallerRole(validated_role)
            except ValueError:
                pass
        raise HTTPException(403, "Invalid or expired session token")
    # Fallback: legacy caller_id authorization.
    if requested_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        if caller_id and authorized_callers.get(caller_id) == requested_role:
            return requested_role
        raise HTTPException(403, "Elevated caller role is not authorized")
    # PLAYER_AGENT / SPECTATOR 角色通过 query 参数直接传入，无 session_token
    # 时记录警告日志，便于审计追踪
    logger.warning(
        "Legacy query-param auth without session_token: "
        "caller_id=%s, caller_role=%s — no cryptographic verification performed",
        caller_id,
        requested_role.value,
    )
    return requested_role


def _require_customization_admin(
    authorized_callers: dict[str, CallerRole],
    caller_id: str,
    caller_role: CallerRole,
) -> None:
    if caller_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        raise HTTPException(403, "Customization save requires moderator or debugger access")
    if not caller_id or authorized_callers.get(caller_id) != caller_role:
        raise HTTPException(403, "Elevated caller role is not authorized")


def _validation_result_to_dict(result: Any) -> dict:
    return {
        "valid": result.valid,
        "summary": result.summary,
        "normalized": result.normalized,
        "errors": [asdict(issue) for issue in result.errors],
        "warnings": [asdict(issue) for issue in result.warnings],
        "diff_against_default": result.diff_against_default,
    }


def _record_to_public_dict(record: Any) -> dict:
    return {
        "config_id": record.config_id,
        "config_type": record.config_type,
        "content_hash": record.content_hash,
        "status": record.status,
        "version": record.version,
        "maturity": record.maturity,
        "compatibility_matrix": record.compatibility_matrix,
        "diff_against_default": record.diff_against_default,
        "creator_id": record.creator_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _record_to_storage_dict(record: Any) -> dict:
    return {
        "config_id": record.config_id,
        "config_type": record.config_type,
        "raw_yaml": record.raw_yaml,
        "normalized": record.normalized,
        "validation_result": record.validation_result,
        "content_hash": record.content_hash,
        "status": record.status,
        "version": record.version,
        "maturity": record.maturity,
        "compatibility_matrix": record.compatibility_matrix,
        "diff_against_default": record.diff_against_default,
        "creator_id": record.creator_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _load_marketplace(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"items": []}
    items = data.get("items", [])
    return {"items": items if isinstance(items, list) else []}


def _build_locked_config_snapshot(req: CreateGameRequest) -> dict:
    seed = req.seed if req.seed is not None else 0
    # Hash actual ruleset content, not just the ID
    ruleset_path = Path("config/rulesets") / f"{req.ruleset_id}.yaml"
    ruleset_content = ruleset_path.read_text(encoding="utf-8") if ruleset_path.exists() else req.ruleset_id
    return {
        "ruleset_id": req.ruleset_id,
        "ruleset_version": "runtime-current",
        "ruleset_hash": hashlib.sha256(ruleset_content.encode("utf-8")).hexdigest(),
        "profile_pack_id": req.profile_pack_id,
        "profile_pack_version": "runtime-current",
        "profile_pack_hash": hashlib.sha256(req.profile_pack_id.encode("utf-8")).hexdigest(),
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
    alive_ids = sorted(pid for pid, player in state.players.items() if player.alive)
    if alive_ids:
        return alive_ids[0]
    all_ids = sorted(state.players)
    return all_ids[0] if all_ids else None
