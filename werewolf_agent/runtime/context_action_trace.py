# -*- coding: utf-8 -*-
"""
构造 Agent action trace 的兼容 payload。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.context_action_trace import _action_trace_payload
"""

from __future__ import annotations

from typing import Any


def _action_trace_payload(action: Any) -> dict[str, Any] | None:
    trace = getattr(action, "trace", None)
    return trace.model_dump() if trace else None
