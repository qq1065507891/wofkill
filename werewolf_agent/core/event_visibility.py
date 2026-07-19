# -*- coding: utf-8 -*-
"""
统一定义并读取游戏事件可见性，兼容 V1 payload-only 事件。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-20

使用示例:
    >>> from werewolf_agent.core.models import GameEvent
    >>> event_visibility(GameEvent(type="speech"))
    <EventVisibility.PUBLIC: 'public'>
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from werewolf_agent.core.models import GameEvent


class EventVisibility(str, Enum):
    """事件可见性规范值。"""

    PUBLIC = "public"
    MODERATOR_ONLY = "moderator_only"
    MODERATOR_FULL = "moderator_full"
    MODERATOR_POSTGAME = "moderator_postgame"
    WEREWOLF_TEAM_ONLY = "werewolf_team_only"
    SEER_PRIVATE = "seer_private"
    SEER_ONLY = "seer_only"
    WITCH_PRIVATE = "witch_private"
    HYBRID_ONLY = "hybrid_only"
    HYBRID_PRIVATE = "hybrid_private"
    ROLE_PRIVATE = "role_private"
    ACTOR_PRIVATE = "actor_private"
    PRIVATE = "private"
    PLAYER_ONLY = "player_only"

    @classmethod
    def from_legacy(cls, value: Any) -> EventVisibility:
        """把旧 payload 值规范化；缺失值按历史语义视为公开。"""
        if isinstance(value, cls):
            return value
        if value in (None, ""):
            return cls.PUBLIC
        aliases = {
            "moderator": cls.MODERATOR_ONLY,
            "wolf_team": cls.WEREWOLF_TEAM_ONLY,
        }
        if isinstance(value, str) and value in aliases:
            return aliases[value]
        try:
            return cls(str(value))
        except (TypeError, ValueError):
            # 未知旧值不得打断公开消费者，但必须按最严格级别封闭。
            return cls.MODERATOR_ONLY


def event_visibility(event: GameEvent) -> EventVisibility:
    """读取事件可见性；V2 非空顶层字段优先，V1 回退 payload。"""
    if event.visibility not in (None, ""):
        return EventVisibility.from_legacy(event.visibility)
    return EventVisibility.from_legacy(event.payload.get("visibility", "public"))


__all__ = ["EventVisibility", "event_visibility"]
