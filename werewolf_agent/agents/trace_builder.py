"""ActionTrace construction helper extracted from PlayerAgent.

Builds the ``ActionTrace`` audit object that records every attempt the
LLM makes — raw text, parsed payload, retry metadata, fallback flags,
structured-output failure reasons, and so on. Pulled out of player.py
so the retry loop stays focused on orchestration.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import ActionTrace, AgentContext, PlayerAction, RetryInfo


def build_action_trace(
    context: AgentContext,
    *,
    raw_text: str,
    parsed_action: PlayerAction | dict[str, Any] | None,
    final_action_type: Any,
    retry: RetryInfo | None,
    fallback_reason: str | None = None,
    fallback_target_used: bool = False,
    fallback_target_id: str | None = None,
    tool_call_required: bool = False,
    tool_call_received: bool = False,
    parse_success: bool = False,
    parse_error: str | None = None,
    retry_count: int = 0,
    structured_failure_reason: str | None = None,
) -> ActionTrace:
    """Build an ActionTrace from the current attempt's state.

    The ``final_action_type`` is normalized to its ``.value`` string when
    an ``ActionType`` enum is supplied so downstream audit code can
    treat the field uniformly.
    """
    final_type_value = (
        final_action_type.value
        if hasattr(final_action_type, "value")
        else final_action_type
    )
    return ActionTrace(
        raw_text=raw_text,
        parsed_action=(
            parsed_action.model_dump(exclude={"trace"})
            if isinstance(parsed_action, PlayerAction)
            else parsed_action
        ),
        final_action_type=final_type_value,
        legal_actions=[action.value for action in context.legal_actions],
        legal_targets=list(context.legal_targets),
        retry=retry.model_dump() if retry else None,
        fallback_reason=fallback_reason,
        fallback_target_used=fallback_target_used,
        fallback_target_id=fallback_target_id,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        tool_call_name="submit_player_action" if tool_call_required else "",
        parse_success=parse_success,
        parse_error=parse_error,
        retry_count=retry_count,
        structured_failure_reason=structured_failure_reason,
    )
