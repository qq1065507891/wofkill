# -*- coding: utf-8 -*-
"""
GameRunner 兼容入口，负责组合配置、初始化、执行、HITL 事件盖章和记忆 mixin。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-15

使用示例:
    >>> runner = GameRunner(GameRunnerConfig(seed=42))
    >>> runner.run_step()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from typing import Any, Iterator

from werewolf_agent.agents.judge import JudgeAgent
from werewolf_agent.agents.judge_hitl import JudgeHITLInterface
from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.core.models import GameState
from werewolf_agent.customization.ruleset_registry import RulesetRegistry
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
from werewolf_agent.persona_runtime.router import PersonaRouter
from werewolf_agent.runtime.agent_adapter import SimpleAgentRegistry
from werewolf_agent.runtime.cognition_state import CognitionStateManager
from werewolf_agent.runtime.game_runner_config import GameRunnerConfig
from werewolf_agent.runtime.game_runner_execution import GameRunnerExecutionMixin
from werewolf_agent.runtime.event_metadata import stamp_new_events
from werewolf_agent.runtime.game_runner_memory import GameRunnerMemoryMixin
from werewolf_agent.runtime.game_runner_setup import GameRunnerSetupMixin
from werewolf_agent.runtime.graph import RuntimeState, build_game_graph


logger = logging.getLogger(__name__)


class GameRunner(
    GameRunnerSetupMixin,
    GameRunnerExecutionMixin,
    GameRunnerMemoryMixin,
):
    """组织完整游戏，从 setup 执行到 finish。"""

    def __init__(self, config: GameRunnerConfig) -> None:
        self._config = config
        self._game_id = config.game_id or (
            f"g_{config.seed if config.seed is not None else uuid.uuid4().hex[:8]}"
        )
        self._ruleset_registry: RulesetRegistry = config.ruleset_registry or RulesetRegistry()
        self._ruleset_entry = self._ruleset_registry.require_playable(config.ruleset_id)
        self._engine = RuleEngine.from_yaml(self._ruleset_entry.path)
        self._state = GameState(
            game_id=self._game_id,
            ruleset_id=config.ruleset_id,
        )
        self._graph = build_game_graph()
        self._step_count: int = 0
        self._finished: bool = False
        self._stream_gen: Iterator | None = None
        self._restored_memory: Any = None
        self._restored_rag: list[Any] | None = None
        self._cognition_state_manager = CognitionStateManager(
            MemoryStore(repo=config.repository)
        )
        self._model_router: ModelRouter | None = None
        self._persona_router: PersonaRouter | None = None
        self._rag_service: Any = config.rag_service
        if self._rag_service is None and config.enable_default_rag_service:
            self._rag_service = self._build_default_rag_service()
        self._agent_registry: SimpleAgentRegistry | None = self._build_agent_registry()
        self._judge_agent: JudgeAgent | None = None
        if self._model_router is not None:
            profile_router = self._load_judge_profile_router()
            self._judge_agent = JudgeAgent(
                model_router=self._model_router,
                profile_router=profile_router,
                profile_id=self._config.judge_persona_profile_id,
            )
        elif self._config.judge_llm_enabled:
            logger.warning(
                "judge_llm_enabled=True but use_agent_registry=False: "
                "JudgeAgent requires an agent registry. Set use_agent_registry=True "
                "to enable LLM-powered judge broadcasts."
            )
        self._hitl_interface: JudgeHITLInterface | None = None
        if self._config.judge_hitl_enabled:
            auto_pause = set(self._config.judge_hitl_auto_pause_triggers or [])
            self._hitl_interface = JudgeHITLInterface(
                auto_pause_phases=auto_pause,
            )
        self._restore_memory_if_configured()

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def engine(self) -> RuleEngine:
        return self._engine

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def config(self) -> GameRunnerConfig:
        return self._config

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def restored_memory(self) -> Any:
        """MemoryStore restored from a previous game snapshot, or None."""
        return self._restored_memory

    @property
    def restored_rag(self) -> list[Any] | None:
        """RAG entries restored from a previous snapshot, or None."""
        return self._restored_rag

    @property
    def hitl_interface(self) -> JudgeHITLInterface | None:
        """Layer 4: Judge HITL interface, if enabled."""
        return self._hitl_interface

    def pause(self) -> str | None:
        """暂停到下一检查点。"""
        if self._hitl_interface is None:
            return None
        self._hitl_interface.pause()
        return f"游戏已暂停。步骤: {self._step_count}"

    def resume(self, steps: int = 0) -> str | None:
        """恢复游戏执行，可指定 N 步后自动暂停。"""
        if self._hitl_interface is None:
            return None
        self._hitl_interface.resume(steps)
        return f"游戏已恢复。{'将执行' + str(steps) + '步后暂停' if steps else ''}"

    def send_command(self, raw: str) -> str | None:
        """向运行中的 HITL 游戏发送命令。"""
        if self._hitl_interface is None:
            return None
        self._hitl_interface.send_command(raw)
        if self._hitl_interface.is_paused:
            from werewolf_agent.agents.judge_hitl import HITLCommand

            cmd = self._hitl_interface._pending_command
            if cmd is None:
                cmd = HITLCommand.parse(raw)
            previous_events = self._state.events
            result = self._hitl_interface.handle_command(cmd, self._state)
            if "game_state" in result:
                self._state = result["game_state"]
            hitl_events = self._hitl_interface.flush_events()
            if hitl_events:
                self._state = replace(self._state, events=self._state.events + hitl_events)
            self._state = replace(
                self._state,
                events=stamp_new_events(
                    self._state.game_id,
                    previous_events,
                    self._state.events,
                ),
            )
            return result.get("response", "OK")
        return f"命令已排队: {raw}"

    def reset_game_id(self, game_id: str) -> None:
        """更新 API 层 game_id，并同步到当前 GameState。"""
        self._game_id = game_id
        self._state = replace(self._state, game_id=game_id)


__all__ = [
    "GameRunner",
    "GameRunnerConfig",
    "JudgeAgent",
    "JudgeHITLInterface",
    "JudgeProfileRouter",
    "MemoryStore",
    "ModelRouter",
    "PersonaRouter",
    "PlayerAgent",
    "RuleEngine",
    "RulesetRegistry",
    "RuntimeState",
    "SimpleAgentRegistry",
    "build_game_graph",
    "logger",
]
