# -*- coding: utf-8 -*-
"""
RuleEngine 的私有意图记录和可见上下文 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.engine.rule_visibility import build_visible_context
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameState, VisibleContext


def record_private_intent(
    state: GameState,
    *,
    player_id: str,
    private_intent: dict[str, Any],
) -> GameState:
    new_intents = {**state.private_intents, player_id: private_intent}
    return replace(state, private_intents=new_intents)


def build_visible_context(
    raw: dict[str, Any],
    state: GameState,
    *,
    viewer_id: str,
    view_mode: str,
) -> VisibleContext:
    forbidden = set(raw["information_visibility"]["forbidden_for_player_agents"])
    sections: set[str] = set()
    if view_mode == "player_view":
        sections.add("public_state")
        sections.add("own_private_state")
        if viewer_id in state.private_intents:
            sections.add(f"{viewer_id}.private_intent")
        viewer = state.players.get(viewer_id)
        if viewer:
            role_private = raw["information_visibility"].get("private", {})
            role_sections = role_private.get(viewer.role, [])
            sections.update(role_sections)
        sections -= forbidden
    elif view_mode == "moderator_full":
        sections.add("moderator_full")
        sections.add("all_private_states")
    return VisibleContext(view_mode=view_mode, visible_sections=sections)


__all__ = ["build_visible_context", "record_private_intent"]
