# -*- coding: utf-8 -*-
"""
游戏公开分享摘要的事件过滤与 MVP 候选 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.api.routes.game_public_share import _event_is_public_for_share
    >>> _event_is_public_for_share(event)
"""

from __future__ import annotations

from werewolf_agent.core.models import GameEvent, GameState


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
    """选择公开分享中可展示的 MVP 候选。"""
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


__all__ = [
    "_event_is_public_for_share",
    "_pick_public_mvp_candidate",
]
