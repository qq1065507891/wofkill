# -*- coding: utf-8 -*-
"""
狼人夜晚节点兼容 facade。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.nodes.wolf_night_nodes import wolf_discussion
"""

from __future__ import annotations

from werewolf_agent.runtime.nodes.wolf_consensus import (
    _legacy_wolf_consensus,
    wolf_consensus,
)
from werewolf_agent.runtime.nodes.wolf_discussion import (
    _build_fallback_wolf_team_plan,
    wolf_discussion,
    wolf_team_plan_node,
)
from werewolf_agent.runtime.wolf_no_kill_policy import (
    NoKillDecision,
    NoKillPolicy,
)

__all__ = [
    "NoKillDecision",
    "NoKillPolicy",
    "_build_fallback_wolf_team_plan",
    "_legacy_wolf_consensus",
    "wolf_consensus",
    "wolf_discussion",
    "wolf_team_plan_node",
]
