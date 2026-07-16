# -*- coding: utf-8 -*-
"""
GameRunner 的配置数据结构。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.game_runner_config import GameRunnerConfig
    >>> GameRunnerConfig(seed=42)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SAFE_GAME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass
class GameRunnerConfig:
    """GameRunner 实例的配置。"""

    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    player_count: int = 12
    seed: int | None = None
    use_agent_registry: bool = False
    model_config_path: str = ""
    persona_config_path: str = ""
    agent_call_timeout: float = 0
    ruleset_registry: Any = None
    repository: Any = None
    memory_coordinator: Any = None
    rag_service: Any = None
    enable_default_rag_service: bool = True
    probe_tool_call_support: bool = False
    judge_llm_enabled: bool = False
    judge_persona_profile_id: str = "tournament_referee"
    judge_persona_config_path: str = ""
    judge_hitl_enabled: bool = False
    judge_hitl_auto_pause_triggers: list[str] | None = None
    agent_call_delay_ms: int = 0
    game_id: str = ""
    emergency_artifact_dir: str | Path = Path("artifacts/emergency_game_aborts")

    def __post_init__(self) -> None:
        if not isinstance(self.game_id, str):
            raise ValueError("game_id must be a string")
        if self.game_id and (
            ".." in self.game_id or _SAFE_GAME_ID.fullmatch(self.game_id) is None
        ):
            raise ValueError(
                "game_id must start with an ASCII letter or digit, contain only "
                "letters, digits, '.', '_', or '-', exclude '..', and be at most "
                "128 characters"
            )
        if self.seed is None:
            import secrets

            self.seed = secrets.randbits(32)


__all__ = ["GameRunnerConfig"]
