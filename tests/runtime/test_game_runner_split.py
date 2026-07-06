# -*- coding: utf-8 -*-
"""
GameRunner 拆分后的兼容导入测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/runtime/test_game_runner_split.py
"""

from __future__ import annotations

from werewolf_agent.runtime import game_runner
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
from werewolf_agent.runtime.game_runner_config import GameRunnerConfig as SplitConfig


def test_game_runner_config_is_reexported_from_legacy_module() -> None:
    assert GameRunnerConfig is SplitConfig
    assert game_runner.GameRunnerConfig is SplitConfig


def test_game_runner_keeps_private_method_surface_after_split() -> None:
    runner = GameRunner(GameRunnerConfig(seed=42, enable_default_rag_service=False))

    for method_name in (
        "_build_runtime_state",
        "_build_agent_registry",
        "_build_default_rag_service",
        "_process_chunk",
        "_persist_if_configured",
        "_restore_memory_if_configured",
        "_save_memory_snapshot",
        "_latest_self_reviews",
    ):
        assert callable(getattr(runner, method_name))
