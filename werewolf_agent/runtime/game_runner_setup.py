# -*- coding: utf-8 -*-
"""
GameRunner 初始化依赖和 RuntimeState 构造逻辑。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.game_runner_setup import GameRunnerSetupMixin
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from werewolf_agent.runtime.agent_adapter import SimpleAgentRegistry
from werewolf_agent.runtime.graph import RuntimeState


logger = logging.getLogger("werewolf_agent.runtime.game_runner")


def _legacy_dependency(name: str) -> Any:
    """从旧 facade 读取依赖，保留测试和外部 monkeypatch 入口。"""
    from werewolf_agent.runtime import game_runner

    return getattr(game_runner, name)


class GameRunnerSetupMixin:
    def _build_runtime_state(
        self,
        *,
        wolf_kill_target_id: str | None = None,
        use_antidote: bool = False,
        poison_target_id: str | None = None,
        seer_target_id: str | None = None,
    ) -> RuntimeState:
        """从当前 runner 状态和配置构造 RuntimeState。"""
        rt: RuntimeState = {
            "game_state": self._state,
            "engine": self._engine,
            "wolf_kill_target_id": wolf_kill_target_id,
            "wolf_action": "kill" if wolf_kill_target_id else None,
            "wolf_action_reason": "",
            "use_antidote": use_antidote,
            "poison_target_id": poison_target_id,
            "seer_target_id": seer_target_id,
            "hybrid_master_target_id": None,
            "self_destruct_wolf_id": None,
            "exile_votes": {},
            "revote": False,
            "sheriff_candidates": [],
            "sheriff_votes": {},
            "sheriff_withdrawing": [],
            "badge_decision": None,
            "badge_target_id": None,
            "hunter_shot_target_id": None,
            "action_index_by_game": {},
            "pending_exposure_events_by_trace": {},
            "prompt_proof_key_provider": self._prompt_proof_key_provider,
        }
        if self._agent_registry is not None:
            rt["agent_registry"] = self._agent_registry
        if self._judge_agent is not None:
            rt["judge_agent"] = self._judge_agent
            rt["judge_llm_enabled"] = self._config.judge_llm_enabled
        if self._hitl_interface is not None:
            rt["judge_hitl"] = self._hitl_interface
            rt["judge_hitl_enabled"] = True
            rt["hitl_auto_pause_after"] = self._config.judge_hitl_auto_pause_triggers or []
        rt["agent_call_delay_ms"] = self._config.agent_call_delay_ms
        if self._rag_service is not None:
            rt["rag_service"] = self._rag_service
        if self._config.agent_call_timeout > 0:
            rt["agent_call_timeout"] = self._config.agent_call_timeout
        rt["cognition_state_manager"] = self._cognition_state_manager
        if self._restored_memory is not None:
            rt["restored_memory"] = self._restored_memory
        if self._config.repository is not None:
            rt["repository"] = self._config.repository
        return rt

    def _build_default_rag_service(self) -> Any | None:
        """构造无 Docker 的默认种子 RAG 服务。"""
        try:
            from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

            return RAGKnowledgeService()
        except Exception:
            logger.warning("Default RAG service initialization failed", exc_info=True)
            return None

    def _build_agent_registry(self) -> SimpleAgentRegistry | None:
        """真实 agent 模式下构造 PlayerAgent registry。"""
        if not self._config.use_agent_registry:
            return None
        model_config_path = self._config.model_config_path or str(
            Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"
        )
        model_router_cls = _legacy_dependency("ModelRouter")
        router = model_router_cls.from_yaml(model_config_path, register_env_providers=True)
        self._model_router = router
        if self._config.probe_tool_call_support:
            probe = router.probe_tool_call_support("p01", "speech")
            if not probe.get("supported"):
                raise RuntimeError(f"tool call probe failed: {probe}")
        persona_map = self._load_persona_names()
        persona_path = self._player_persona_path()
        if persona_path is not None and persona_map:
            persona_router_cls = _legacy_dependency("PersonaRouter")
            self._persona_router = persona_router_cls.from_yaml(persona_path)
            self._persona_router.load_assignments({
                player_id: persona_key
                for player_id, (_, persona_key) in persona_map.items()
                if persona_key
            })
        registry = SimpleAgentRegistry()
        player_agent_cls = _legacy_dependency("PlayerAgent")
        for i in range(1, self._config.player_count + 1):
            player_id = f"p{i:02d}"
            name, pkey = persona_map.get(player_id, (player_id, None))
            registry.register(player_id, player_agent_cls(
                agent_id=player_id, model_router=router,
                player_name=name, persona_key=pkey,
                persona_router=self._persona_router,
            ))
        return registry

    def _player_persona_path(self) -> Path | None:
        configured = self._config.persona_config_path
        path = Path(configured) if configured else (
            Path(__file__).resolve().parent.parent.parent
            / "config" / "personas" / "jingcheng_style_prototypes.yaml"
        )
        return path if path.exists() else None

    def _load_persona_names(self) -> dict[str, tuple[str, str | None]]:
        """从 persona 配置轮转加载玩家名。"""
        p = self._player_persona_path()
        if p is None:
            return {}
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        profiles = data.get("persona_profiles", {})
        names = []
        for key, prof in profiles.items():
            pname = prof.get("player_name", key)
            names.append((key, pname))
        if not names:
            return {}
        result: dict[str, tuple[str, str | None]] = {}
        for i in range(1, self._config.player_count + 1):
            pid = f"p{i:02d}"
            pkey, pname = names[(i - 1) % len(names)]
            result[pid] = (pname, pkey)
        return result

    def _load_judge_profile_router(self) -> Any | None:
        """从配置路径加载裁判 persona profile。"""
        profile_path = self._config.judge_persona_config_path
        if not profile_path:
            default_path = (
                Path(__file__).resolve().parent.parent.parent
                / "config" / "personas" / "judge_profiles.yaml"
            )
            if default_path.exists():
                profile_path = str(default_path)
            else:
                return None
        try:
            judge_profile_router_cls = _legacy_dependency("JudgeProfileRouter")
            return judge_profile_router_cls.from_yaml(profile_path)
        except Exception:
            logger.warning("Failed to load judge profile router", exc_info=True)
            return None


__all__ = ["GameRunnerSetupMixin"]
