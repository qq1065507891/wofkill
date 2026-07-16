# -*- coding: utf-8 -*-
"""
整理 PlayerAgent 的失败分类与 fallback reason helper。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.agents.player_failures import fallback_reason
    >>> fallback_reason(action)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from werewolf_agent.agents.schemas import ActionType, FallbackAction
from werewolf_agent.runtime.decision_outcomes import normalize_terminal_failure_code


_TASK_FAILURE_CODE_ALIASES = {
    "agent_exception": "model_generation_failed",
    "captain_agent_missing": "fallback_route_unavailable",
    "generate_error": "model_generation_failed",
    "json_parse_failed": "parse_error",
    "membership_validation_failed": "illegal_action",
    "no_alive_wolves": "policy_rejection",
    "no_registry": "fallback_route_unavailable",
    "provider_tool_choice_unsupported": "structured_output_unsupported",
    "schema_validation_failed": "schema_validation",
    "speech_timeout": "timeout",
    "pre_supplied_speech_text": "policy_rejection",
    "agent_dispatch_error": "model_generation_failed",
    "self_destruct_before_speech": "policy_rejection",
    "missing_action_trace": "invalid_output",
    "agent_unavailable": "fallback_route_unavailable",
}

TERMINAL_FAILURE_STAGES = frozenset({
    "provider",
    "protocol",
    "schema",
    "semantic",
    "model_output",
    "registry",
    "runtime",
})
TERMINAL_FALLBACK_KINDS = frozenset({
    "ordinary_speech",
    "sheriff_speech",
    "night_legal_action",
    "night_explicit_abstain",
    "reflection_not_generated",
    "last_words_not_generated",
    "wolf_team_plan_structured_stance",
    "wolf_discussion_speech",
    "safe_action",
    "badge_transfer",
    "badge_tear",
    "badge_unavailable",
})


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


def terminal_failure_code_for_task_failure(reason: object) -> str:
    """把任务局部 reason 映射为不含原始错误正文的 V2 稳定码。"""
    candidate = _TASK_FAILURE_CODE_ALIASES.get(reason, reason)
    return normalize_terminal_failure_code(candidate)


def is_informative_terminal_failure_code(value: object) -> bool:
    """仅认可稳定且能表达根因的终退码；unknown/空值不算覆盖。"""
    return (
        isinstance(value, str)
        and bool(value)
        and value != "unknown"
        and normalize_terminal_failure_code(value) == value
    )


def is_complete_terminal_fallback_v2(
    row: Mapping[str, object],
    *,
    source_type: str,
) -> bool:
    """校验终退 V2 的闭集字段、最终动作和事实执行证据。"""
    terminal_code = row.get("terminal_failure_code")
    original_code = row.get("original_failure_code")
    final_action = row.get("final_action")
    if not (
        row.get("generated_by") == "terminal_fallback"
        and row.get("decision_outcome") == "terminal_fallback"
        and is_informative_terminal_failure_code(terminal_code)
        and original_code == terminal_code
        and row.get("failure_stage") in TERMINAL_FAILURE_STAGES
        and row.get("fallback_kind") in TERMINAL_FALLBACK_KINDS
        and isinstance(final_action, Mapping)
        and isinstance(final_action.get("action_type"), str)
        and bool(final_action.get("action_type"))
        and "target_id" in final_action
        and isinstance(final_action.get("reason"), str)
    ):
        return False
    final_action_type = row.get("final_action_type")
    if (
        isinstance(final_action_type, str)
        and final_action_type
        and final_action.get("action_type") != final_action_type
    ):
        return False
    if source_type == "wolf_team_plan_fallback":
        return _is_non_negative_int(row.get("attempts"))
    attempts = row.get("execution_attempts")
    return (
        isinstance(attempts, Sequence)
        and not isinstance(attempts, (str, bytes, bytearray))
        and _is_non_negative_int(row.get("attempt_count"))
        and row.get("attempt_count") == len(attempts)
        and _is_non_negative_int(row.get("retry_count"))
        and _is_non_negative_int(row.get("provider_fallback_count"))
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
