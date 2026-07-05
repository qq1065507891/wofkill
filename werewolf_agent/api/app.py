# -*- coding: utf-8 -*-
"""
功能描述：FastAPI 应用工厂，集成存储、鉴权、权限检查与路由挂载。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：app = create_app(repository=repo, auth_manager=auth)
"""

from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker
from werewolf_agent.api.schemas import CallerRole
from werewolf_agent.customization.repository import InMemoryCustomizationRepository
from werewolf_agent.runtime.executor import LocalRuntimeExecutor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from werewolf_agent.storage.repository import GameRepository


def _persist_custom_config_impl(repo: Any, record: Any) -> None:
    """P-A3: persist a custom-config record via the game repository.

    Surfaces a WARNING when the repo lacks ``save_custom_config`` so the
    silent data loss is visible during ops review — otherwise the
    customization save API would return 200 OK while the record is
    dropped on the next restart.
    """
    from werewolf_agent.api.routes.customization import _record_to_storage_dict

    if repo is not None and hasattr(repo, "save_custom_config"):
        repo.save_custom_config(_record_to_storage_dict(record))
        logger.info("Persisted custom config %s", getattr(record, "config_id", "?"))
    else:
        logger.warning(
            "Repository %s lacks save_custom_config — custom config %s "
            "will be lost on restart",
            type(repo).__name__ if repo is not None else "None",
            getattr(record, "config_id", "?"),
        )


def create_app(
    repository: GameRepository | None = None,
    auth_manager: AuthManager | None = None,
) -> FastAPI:
    """Build the FastAPI application with all routes mounted."""

    # --- Storage ---
    repo = repository
    if repo is None:
        storage_backend = os.environ.get("WEREWOLF_STORAGE_BACKEND", "").strip().lower()
        if storage_backend:
            from werewolf_agent.storage.production import (
                ProductionStorageConfig, create_game_repository,
            )
            repo = create_game_repository(ProductionStorageConfig(
                backend=storage_backend,
                sqlite_path=os.environ.get("WEREWOLF_DB_PATH", "data/wofkill.db"),
                postgres_dsn=os.environ.get("POSTGRES_DSN", ""),
                redis_url=os.environ.get("REDIS_URL", ""),
            ))
        else:
            db_path = os.environ.get("WEREWOLF_DB_PATH")
            if db_path:
                from werewolf_agent.storage.sqlite_store import SqliteGameRepository
                repo = SqliteGameRepository(db_path)
            else:
                logger.warning("No storage backend configured — using in-memory (data lost on restart)")

    # --- App ---
    app = FastAPI(title="Werewolf Agent API", version="1.0")

    # NEW-P2-7: configurable CORS middleware. Origins come from a
    # comma-separated env var so production deployments can restrict
    # access without code changes; the default is permissive for
    # local dev parity with the previous no-CORS behavior.
    cors_origins_raw = os.environ.get(
        "WEREWOLF_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).strip()
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    from pathlib import Path
    _static_dir = Path(__file__).parent.parent / "ui" / "static"
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    # --- Components ---
    import threading
    checker = PermissionChecker()
    auth = auth_manager or AuthManager()
    executor = LocalRuntimeExecutor()
    customization_repo = InMemoryCustomizationRepository()
    games: dict = {}
    runners: dict = {}
    _games_lock = threading.Lock()
    _runners_lock = threading.Lock()
    authorized_callers: dict[str, CallerRole] = {
        "mod1": CallerRole.MODERATOR,
        "dbg1": CallerRole.DEBUGGER,
    }

    # --- RAG ---
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
                logger.warning("Vector store initialization failed; using RAG fallback: %s", exc)
        # N1: auto-init reranker when SILICONFLOW_API_KEY is set, mirroring
        # the auto-detect in ``retriever.create_retriever()``. Without this
        # wiring the reranker was dead code in production — RAGKnowledgeService
        # was constructed without a reranker, so R1's wiring was never
        # exercised. If the API key is missing or the client fails to
        # construct, fall through to reranker=None (rule-based-only path).
        reranker = None
        if os.environ.get("SILICONFLOW_API_KEY"):
            try:
                from werewolf_agent.rag.reranker_client import SiliconFlowRerankerClient
                reranker = SiliconFlowRerankerClient()
            except Exception as exc:
                logger.warning(
                    "SiliconFlow reranker initialization failed; "
                    "continuing with rule-based retrieval only: %s", exc,
                )
        rag_service = RAGKnowledgeService(
            repository=repo, vector_store=vector_store, reranker=reranker,
        )
        rag_service.ensure_seeded()
    except Exception as exc:
        logger.warning("RAG knowledge service initialization failed: %s", exc)

    # --- Persistence helper ---
    # P-A3: the real work is in the module-level _persist_custom_config_impl
    # so tests can exercise it without spinning up the full app factory.
    # The closure below keeps the original ``persist_fn(record)`` shape
    # that the customization router expects.
    def _persist_custom_config(record: Any) -> None:
        _persist_custom_config_impl(repo, record)

    # --- Project root ---
    _project_root = Path(__file__).resolve().parent.parent.parent

    # --- Mount routers ---
    from werewolf_agent.api.routes.games import create_game_router
    game_router = create_game_router(
        games=games,
        games_lock=_games_lock,
        runners=runners,
        runners_lock=_runners_lock,
        repo=repo,
        rag_service=rag_service,
        authorized_callers=authorized_callers,
        auth=auth,
        checker=checker,
        executor=executor,
        project_root=_project_root,
    )
    app.include_router(game_router)

    from werewolf_agent.api.routes.customization import create_customization_router
    customization_router = create_customization_router(
        authorized_callers=authorized_callers,
        customization_repo=customization_repo,
        persist_fn=_persist_custom_config,
        project_root=_project_root,
    )
    app.include_router(customization_router)

    # --- State exposure (for tests) ---
    app.state.repository = repo
    app.state.rag_service = rag_service
    app.state.games = games
    app.state.runners = runners
    app.state.executor = executor
    app.state.checker = checker
    app.state.authorized_callers = authorized_callers
    app.state.auth = auth
    app.state.customization_repo = customization_repo

    return app
