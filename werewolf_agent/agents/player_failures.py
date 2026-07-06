# -*- coding: utf-8 -*-
"""
整理 PlayerAgent 的失败分类与 fallback reason helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.player_failures import fallback_reason
    >>> fallback_reason(action)
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import ActionType, FallbackAction


def fallback_reason(action: FallbackAction) -> str:
    """Return a fallback reason that does NOT embed the target_id.

    The caller is responsible for substituting the actual target into the
    log display. This prevents the audit trail from showing "chose p07" while
    the actual ``vote_target`` is a different player (the LLM's choice may
    later override the fallback target in ``agent_day_vote``).
    """
    if action.action_type == ActionType.VOTE and not action.target_id:
        return "fallback: 结构化输出失败，无足够公开证据补票"
    return "fallback: 结构化输出失败，按当前可见线索选择默认目标"


def categorize_failure_category(
    *,
    latency_ms: int,
    raw_error: str | None,
    http_status: int = 0,
) -> str | None:
    """Bridge from player-side signals to the failure_category string.

    Imported lazily so that importing player.py does not require the
    model_gateway.providers package (some test harnesses mock the
    router). When the categorizer is unavailable we conservatively
    return None — the field in RetryInfo will simply be unset.

    R3-MG-2: ``http_status`` is plumbed through from ``GenerateResult``
    so 4xx/5xx responses classify as ``provider_error`` rather than
    silently falling through to ``unknown``.
    """
    try:
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
    except ImportError:
        return None
    return categorize_empty_response(
        response_text="",
        latency_ms=latency_ms,
        http_status=http_status,
        raw_error=raw_error,
    )
