# -*- coding: utf-8 -*-
"""
渲染公开关键事件 salience prompt 片段。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.prompt_salience import _slim_salience_item
"""

from __future__ import annotations

from typing import Any

_MAX_SALIENCE_ITEMS = 3

# 只允许公开字段进入玩家可见 prompt，防止未来新增私有字段时静默泄漏。
_SALIENCE_PUBLIC_FIELDS: frozenset[str] = frozenset({
    "weight", "bucket", "fact_type", "source", "target", "value",
    "day", "phase", "event_type",
    "speaker", "result", "alignment",
    "id", "summary",
})
_SALIENCE_PRIVATE_KEYS: frozenset[str] = frozenset({
    "seer_result", "witch_target", "wolf_team",
    "private_intent", "moderator_full",
})


class PromptSalienceMixin:
    def _build_salience_events(self) -> str:
        ctx = self.context
        if not ctx.salience_items:
            return ""
        # P0-2 (defense in depth): explicitly whitelist public fields
        # so a future change that leaks a private key (seer_result,
        # witch_target, wolf_team, private_intent, moderator_full) into
        # ctx.salience_items cannot end up in the player-visible prompt.
        # The runtime (runtime/context.py:build_agent_context) does not
        # currently populate these, but the renderer should still
        # enforce the boundary.
        slimmed = [_slim_salience_item(item) for item in ctx.salience_items[:_MAX_SALIENCE_ITEMS]]
        slimmed = [item for item in slimmed if item is not None]
        if not slimmed:
            return ""
        return "关键事件: " + self._compact_json(slimmed)



def _slim_salience_item(item: Any) -> dict[str, Any] | None:
    """Return a salience item with only public fields, or None if it contains any private key."""
    if not isinstance(item, dict):
        return None
    if any(key in item for key in _SALIENCE_PRIVATE_KEYS):
        return None
    return {k: v for k, v in item.items() if k in _SALIENCE_PUBLIC_FIELDS}
