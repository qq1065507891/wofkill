# -*- coding: utf-8 -*-
"""
封装 PlayerAgent 的重复错误签名检测 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.player_retry import check_repeat_error_signature
    >>> check_repeat_error_signature(retry, raw_text, 2, last_signature)
"""

from __future__ import annotations

from collections.abc import Callable

from werewolf_agent.agents.schemas import ActionType, AgentContext, FallbackAction, RetryInfo, TaskType

_TARGET_REQUIRED_FALLBACK_ACTIONS = {
    ActionType.VOTE,
    ActionType.WOLF_KILL,
    ActionType.USE_POISON,
    ActionType.CHECK_ALIGNMENT,
    ActionType.CHOOSE_MASTER,
    ActionType.HUNTER_SHOT,
    ActionType.BADGE_TRANSFER,
    ActionType.SHERIFF_VOTE,
}


def check_repeat_error_signature(
    retry: RetryInfo,
    raw_text: str,
    attempt: int,
    last_signature: tuple[str, str, str] | None,
    *,
    structured_output_mode: str = "",
) -> tuple[bool, tuple[str, str, str] | None]:
    """Pipeline-optimization Task 1: detect repeated retry failures.

    When two consecutive attempts in the same structured-output mode
    produce the same ``(error_code, mode, raw_text[:50])`` signature,
    the LLM is almost certainly stuck. A protocol change gets its own
    attempt even when the returned text is identical.

    Skill-tool nudges (where ``retry.error_code`` is ``None``) bypass the
    check so the existing skill-skip retry budget is preserved.
    """
    if retry.error_code is None:
        return False, last_signature
    raw_text_snippet = (raw_text or "")[:50]
    current_sig = (
        retry.error_code,
        structured_output_mode,
        raw_text_snippet,
    )
    if last_signature is not None and last_signature == current_sig:
        retry.early_exit_reason = (
            f"repeat_error_signature: {retry.error_code} on attempts "
            f"{attempt - 1} and {attempt}"
        )
        return True, current_sig
    return False, current_sig


def fallback_vote_target_from_context(
    context: AgentContext,
    candidates: list[str],
) -> str | None:
    for item in context.salience_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or item.get("event")
        target = item.get("target") or item.get("target_id")
        result = str(item.get("result") or item.get("alignment") or "").lower()
        if (
            item_type == "seer_claim"
            and target in candidates
            and ("wolf" in result or "狼" in result or "查杀" in result)
        ):
            return target
    return None


def build_fallback_action(
    context: AgentContext,
    *,
    fallback_reason: Callable[[FallbackAction], str],
    fallback_speech: Callable[[AgentContext], str],
) -> FallbackAction:
    """Compute a safe fallback action from legal sets."""
    if context.legal_actions:
        safe_action = context.legal_actions[0]
    else:
        safe_action = ActionType.NO_ACTION

    safe_target = None
    if safe_action in _TARGET_REQUIRED_FALLBACK_ACTIONS and context.legal_targets:
        # vote fallback 必须基于证据或显式策略，不按座位顺序补票。
        if safe_action == ActionType.VOTE:
            non_self = [target for target in context.legal_targets if target != context.agent_id]
            fb = (
                context.strategy_directive.get("_vote_fallback_target")
                if context.strategy_directive else None
            )
            if fb and fb in non_self:
                safe_target = fb
            else:
                safe_target = fallback_vote_target_from_context(context, non_self)
        else:
            safe_target = context.legal_targets[0]

    speech = ""
    if safe_action == ActionType.SPEECH and context.task_type == TaskType.WOLF_DISCUSSION:
        speech = fallback_speech(context)

    # reason 在拿到完整 FallbackAction 后再生成，避免字符串里写死 target_id。
    fallback = FallbackAction(
        action_type=safe_action,
        target_id=safe_target,
        speech=speech,
        reason="",
    )
    return fallback.model_copy(update={"reason": fallback_reason(fallback)})
