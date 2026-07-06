# -*- coding: utf-8 -*-
"""
提供运行时玩家 Agent 注册表。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.agent_registry import SimpleAgentRegistry
    >>> registry = SimpleAgentRegistry()
"""

from __future__ import annotations

from typing import Protocol

from werewolf_agent.agents.player import PlayerAgent


class AgentRegistry(Protocol):
    """按 player_id 查找 PlayerAgent；返回 None 表示走脚本回退。"""

    def get_agent(self, player_id: str) -> PlayerAgent | None: ...


class SimpleAgentRegistry:
    """简单注册表：维护 player_id 到 PlayerAgent 的映射。"""

    def __init__(self, agents: dict[str, PlayerAgent] | None = None) -> None:
        self._agents: dict[str, PlayerAgent] = agents or {}

    def register(self, player_id: str, agent: PlayerAgent) -> None:
        self._agents[player_id] = agent

    def get_agent(self, player_id: str) -> PlayerAgent | None:
        return self._agents.get(player_id)
