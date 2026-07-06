# -*- coding: utf-8 -*-
"""
提取 PlayerAgent 使用的生成延迟读取 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.player_latency import latency_from_result
    >>> latency_from_result(result)
"""

from __future__ import annotations

from typing import Any


def latency_from_result(result: Any) -> int:
    """Best-effort latency extraction from a GenerateResult.

    Returns 0 when usage metadata is unavailable (e.g. the router returned
    an empty GenerateResult after primary+fallback failures). The
    categorizer treats 0 as "no signal" so it will not falsely report
    ``timeout``.
    """
    usage = getattr(result, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "latency_ms", 0) or 0)
