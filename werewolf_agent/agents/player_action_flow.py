# -*- coding: utf-8 -*-
"""
封装 PlayerAgent 的结构化行动生成、重试和 fallback 流程。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.player_action_flow import run_player_action_flow
    >>> run_player_action_flow(agent, context)
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from werewolf_agent.agents.parse_dispatch import (
    parse_choice_action as _parse_choice_action,
    parse_speech_intent_action as _parse_speech_intent_action,
    select_output_mode as _select_output_mode,
)
from werewolf_agent.agents.player_failures import (
    categorize_failure_category as _categorize_failure_category,
)
from werewolf_agent.agents.player_generation_request import (
    build_player_generation_request,
    call_player_generation_request,
)
from werewolf_agent.agents.player_action_result import (
    finalize_fallback_player_action,
    finalize_successful_player_action,
)
from werewolf_agent.agents.player_latency import latency_from_result as _latency_from_result
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    OutputMode,
    PlayerAction,
    RetryInfo,
)
from werewolf_agent.model_gateway.structured_output import (
    StructuredFailureStage,
    StructuredOutputMode,
    StructuredOutputPolicy,
    classify_structured_failure,
)

logger = logging.getLogger(__name__)


def run_player_action_flow(
    agent: Any,
    context: AgentContext,
) -> tuple[PlayerAction | FallbackAction, RetryInfo]:
    """生成带重试和 fallback 的受约束玩家行动。"""
    context = agent._attach_persona_snapshot(context)
    retry = RetryInfo(max_retries=agent.max_retries)
    raw_text = ""
    parsed_action: PlayerAction | None = None
    # Task 9: track structured output metadata across retries
    tool_call_required = True  # We always pass tools and tool_choice
    tool_call_received = False
    parse_success = False
    parse_error_str: str | None = None
    structured_failure_reason: str | None = None
    structured_failure_stage: str | None = None
    structured_output_mode = ""

    attempt = 0
    # Pipeline-optimization Task 1: track previous attempt's error signature
    # ``(error_code, protocol_mode, raw_text[:50])`` to short-circuit only
    # when the same protocol repeats an identical failure.
    last_error_signature: tuple[str, str, str] | None = None
    # Resolve the protocol order once; semantic retries stay on the same
    # mode while protocol/schema failures may advance through the policy.
    _fb_config, _fb = agent.model_router.resolve_config(
        agent.agent_id, context.task_type.value,
    )
    structured_policy = StructuredOutputPolicy.from_config(_fb_config)
    active_structured_mode = structured_policy.primary_mode
    structured_output_mode = active_structured_mode.value

    while attempt < agent.max_retries:
        # Exponential-backoff jitter between successive attempts:
        # attempt 2 → ~ 0.5-1.5 s, attempt 3 → ~ 1-3 s, attempt 4+ → cap ~ 4-12 s.
        # Gives transient API errors / spikes a chance to clear; no delay on
        # the first attempt.
        if attempt > 0:
            delay = min(1.0 * (2 ** (attempt - 1)), 8.0) * random.uniform(0.5, 1.5)
            time.sleep(delay)

        attempt += 1
        retry = RetryInfo(
            attempt=attempt,
            max_retries=agent.max_retries,
            error_code=retry.error_code,
            error_message=retry.error_message,
            correction_hint=retry.correction_hint,
        )

        generation_request = build_player_generation_request(
            agent,
            context,
            retry,
            active_structured_mode,
        )
        tool_call_required = generation_request.tool_call_required
        try:
            result = call_player_generation_request(
                agent,
                context,
                generation_request,
            )
        except NotImplementedError:
            # Provider does not support tool_choice
            structured_failure_reason = "structured_output_unsupported"
            structured_failure_stage = StructuredFailureStage.PROTOCOL.value
            fallback = finalize_fallback_player_action(
                agent=agent,
                context=context,
                fallback=agent._fallback_action(context),
                retry=retry,
                raw_text="",
                parsed_action=None,
                tool_call_required=tool_call_required,
                tool_call_received=False,
                parse_success=False,
                parse_error="provider does not support tool_choice",
                retry_count=attempt,
                structured_failure_reason=structured_failure_reason,
                structured_output_mode=structured_output_mode,
                structured_failure_stage=structured_failure_stage,
                metrics_error_code=structured_failure_reason,
            )
            return fallback, retry

        raw_text = result.text or ""

        structured_output_mode = (
            getattr(result, "structured_output_mode", "")
            or active_structured_mode.value
        )
        tool_call_required = (
            active_structured_mode == StructuredOutputMode.NATIVE_TOOL
        )
        tool_call_received = bool(getattr(result, "tool_call_received", False))
        output_mode = _select_output_mode(
            legal_actions=context.legal_actions,
            legal_targets=context.legal_targets,
            task_type=context.task_type,
            speech_intent_tasks=agent._SPEECH_INTENT_TASKS,
        )

        if not result.text:
            failure_reason = agent._latest_generation_failure_reason()
            if failure_reason and "empty_response" not in failure_reason:
                if "NotImplementedError" in failure_reason:
                    structured_failure_reason = "structured_output_unsupported"
                    structured_failure_stage = StructuredFailureStage.PROTOCOL.value
                else:
                    structured_failure_reason = "model_generation_failed"
                    structured_failure_stage = StructuredFailureStage.PROVIDER.value
                failure_category = _categorize_failure_category(
                    latency_ms=_latency_from_result(result),
                    raw_error=failure_reason,
                    http_status=int(getattr(result, "http_status", 0) or 0),
                )
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=agent.max_retries,
                    error_code="model_generation_failed",
                    error_message=failure_reason,
                    failure_category=failure_category,
                    correction_hint=(
                        f"Provider generation failed (category={failure_category}); "
                        "using fallback action."
                    ),
                )
                fallback = finalize_fallback_player_action(
                    agent=agent,
                    context=context,
                    fallback=agent._fallback_action(context),
                    retry=retry,
                    raw_text="",
                    parsed_action=None,
                    tool_call_required=tool_call_required,
                    tool_call_received=False,
                    parse_success=False,
                    parse_error=failure_reason,
                    retry_count=attempt,
                    structured_failure_reason=structured_failure_reason,
                    structured_output_mode=structured_output_mode,
                    structured_failure_stage=structured_failure_stage,
                    metrics_error_code="model_generation_failed",
                )
                return fallback, retry
            failure_category = _categorize_failure_category(
                latency_ms=_latency_from_result(result),
                raw_error=None,
                http_status=int(getattr(result, "http_status", 0) or 0),
            )
            category_hint = (
                f" (cause: {failure_category})" if failure_category else ""
            )
            # P0-R2: when the empty_response is categorized as a
            # timeout, the LLM needs explicit permission to take
            # a safe no-op. Without it, the model either retries
            # and times out again or fabricates a vote target.
            # Game trace g_3528592081 Action 57: seer p03 vote hit
            # 3 empty retries and fell back to a default target —
            # a '如果超时, 返回 no_action' hint would have let it
            # safely no-op the second time around.
            # D4-3: but only suggest ``no_action`` when it's actually
            # legal. For VOTE-only contexts, ``no_action`` is not in
            # legal_actions — the LLM would copy the hint, the
            # validator would reject the action, and we'd loop
            # forever. Fall back to a target-suggestion hint for
            # those cases.
            timeout_hint = ""
            if failure_category == "timeout":
                can_emit_no_action = (
                    ActionType.NO_ACTION in context.legal_actions
                    and output_mode == OutputMode.FULL_ACTION
                )
                if can_emit_no_action:
                    timeout_hint = (
                        " 如果超时，请直接返回 no_action 而非空响应"
                        "（action_type='no_action', target_id=null,"
                        "reason='timeout - safe no-op'）。"
                    )
                elif context.legal_targets:
                    first_target = context.legal_targets[0]
                    timeout_hint = (
                        f" 如果超时，请直接选择一个合法目标 "
                        f"（例如 {first_target}）并提交结构化JSON。"
                    )
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code="empty_response",
                error_message="Model returned empty text",
                failure_category=failure_category,
                correction_hint=(
                    f"Please provide a valid JSON action{category_hint}. "
                    f"If the model timed out, consider shorter reasoning."
                    f"{timeout_hint}"
                ),
            )
            structured_failure_reason = "empty_response"
            structured_failure_stage = StructuredFailureStage.PROTOCOL.value
            active_structured_mode = structured_policy.next_mode(
                active_structured_mode,
                structured_failure_reason,
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                break
            continue

        allow_text_tool_fallback = bool(
            active_structured_mode != StructuredOutputMode.NATIVE_TOOL
            or (
                getattr(result, "allow_text_tool_fallback", False)
                and getattr(result, "text_fallback_used", False)
            )
        )
        if (
            tool_call_required
            and not tool_call_received
            and not allow_text_tool_fallback
        ):
            structured_failure_reason = (
                getattr(result, "structured_failure_reason", None)
                or "missing_tool_call"
            )
            structured_failure_stage = StructuredFailureStage.PROTOCOL.value
            parse_error_str = "missing required tool call: submit_player_action"
            correction_hint = (
                "必须通过 submit_player_action 工具调用提交结构化参数；"
                "不要把JSON写在普通文本内容里。"
            )
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code=structured_failure_reason,
                error_message=parse_error_str,
                correction_hint=correction_hint,
            )
            active_structured_mode = structured_policy.next_mode(
                active_structured_mode,
                structured_failure_reason,
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                break
            continue

        # Parse JSON. Mandatory vote tasks may use a narrower choice schema;
        # the program maps that choice back into a legal PlayerAction.
        choice_data: dict[str, Any] | None = None
        action, parse_error, choice_data = agent._parse_planning_action(
            result.text,
            context,
        )
        if action is not None or parse_error:
            pass
        elif output_mode == OutputMode.TARGET_CHOICE:
            action, parse_error, choice_data = _parse_choice_action(
                result.text,
                context,
            )
            if parse_error and action is None:
                action, parse_error = agent._parse_action(result.text)
        elif output_mode == OutputMode.SPEECH_INTENT:
            action, parse_error = agent._parse_action(result.text)
            if parse_error and action is None:
                action, parse_error, choice_data = _parse_speech_intent_action(
                    result.text,
                    context,
                )
        else:
            action, parse_error = agent._parse_action(result.text)
        parsed_action = action
        if parse_error:
            parse_error_str = parse_error
            structured_failure_reason = (
                "schema_validation"
                if parse_error.startswith("Schema validation error:")
                else "parse_error"
            )
            failure_stage = classify_structured_failure(
                structured_failure_reason
            )
            structured_failure_stage = (
                failure_stage.value if failure_stage else None
            )
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code=structured_failure_reason,
                error_message=parse_error,
                correction_hint=agent._parse_correction_hint(context, parse_error) or (
                    "只输出JSON，不要解释、不要Markdown代码块。必须包含action_type、target_id、speech、"
                    "reason、confidence；action_type必须来自合法动作，target_id必须来自合法目标或null。"
                ),
            )
            active_structured_mode = structured_policy.next_mode(
                active_structured_mode,
                structured_failure_reason,
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                break
            continue

        parse_success = True

        # Validate against legal sets
        valid, validation_error = agent.validator.validate(
            action.action_type, action.target_id,
            context.legal_actions, context.legal_targets,
        )
        if not valid:
            structured_failure_reason = "illegal_action"
            structured_failure_stage = StructuredFailureStage.SEMANTIC.value
            # P3-7: indirect the hint — don't expose the full enum
            # list (LLM was copying the hint verbatim into the
            # action_type field, then the validator rejected it
            # for the wrong reason).  Instead, describe WHAT the
            # validator expects in the game's terms (role/action)
            # and let "最终输出协议" carry the exact enum values.
            # P4-10: if the error came from a speech_quality or
            # vote_quality gate (different check), the generic
            # "action_type 不在合法动作内" can mislead the LLM
            # into changing its action when the real problem was
            # the content.  Gate the hint: only expose the
            # action-type hint when validation_error originates
            # from the validator itself, not from downstream gates.
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code="illegal_action",
                error_message=validation_error,
                correction_hint=(
                    "你提交的 action_type 不在当前回合合法动作内。"
                    "请查看上方「最终输出协议」段的 action_type 枚举"
                    "和 target_id 约束（合法目标或 null）。"
                ),
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                break
            continue
        speech_quality_err = agent._speech_quality_error(context, action)
        if speech_quality_err:
            structured_failure_reason = "speech_quality"
            structured_failure_stage = StructuredFailureStage.SEMANTIC.value
            # P1-S6 (residual): error_message keeps the full field-missing
            # enumeration (for the audit log + prompt snippet via
            # _build_retry_hint), but correction_hint is a short
            # action-oriented line so the LLM knows what KIND of action
            # to take. The detailed enumeration is too noisy to copy
            # back into the LLM as a "do this" instruction.
            # P3-3: executable hint — g_3528592081 trace showed the
            # LLM was copying the meta-description ("发言必须包含:")
            # directly into the speech field instead of following it.
            # The new hint lists the SPECIFIC fields the speech
            # must mention and gives a concrete anti-pattern.
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code="speech_quality",
                error_message=speech_quality_err,
                correction_hint=(
                    f"发言缺少以下必填字段: {speech_quality_err}。"
                    f"请基于公开记录重写发言，在 speech 字段中体现："
                    f"1) 你的身份立场（至少引用一处公开事实）；"
                    f"2) 攻击或防御的明确论点（PK 阶段必填）。"
                    f"不要写「按公开信息判断」之类的占位文本。"
                ),
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                break
            continue
        vote_quality_err = agent._vote_quality_error(context, action)
        if vote_quality_err:
            structured_failure_reason = "vote_quality"
            structured_failure_stage = StructuredFailureStage.SEMANTIC.value
            # P3-3: executable hint — g_3528592081 trace showed the
            # LLM was copying the meta-description into the vote
            # reason field.  The new hint names SPECIFIC public
            # evidence sources the vote must cite and gives an
            # anti-pattern.
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code="vote_quality",
                error_message=vote_quality_err,
                correction_hint=(
                    f"投票理由缺少以下必填字段: {vote_quality_err}。"
                    f"请基于以下公开来源重写 vote reason："
                    f"1) 预言家查杀声明（金水/查杀 + 报验人+夜数）；"
                    f"2) 票型异常（谁跟谁、票型突变）；"
                    f"3) 警徽流状态（撕徽/未撕）；"
                    f"4) 公开记录里的具体发言引用。"
                    f"不要写「综合分析」之类的占位文本。"
                ),
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                break
            continue

        return finalize_successful_player_action(
            agent=agent,
            context=context,
            action=action,
            retry=retry,
            raw_text=raw_text,
            parsed_action=choice_data or action,
            tool_call_required=tool_call_required,
            tool_call_received=tool_call_received,
            parse_success=parse_success,
            retry_count=attempt,
            structured_output_mode=structured_output_mode,
        ), retry

    # Fallback
    exit_reason = f" early_exit={retry.early_exit_reason}" if retry and retry.early_exit_reason else ""
    logger.warning(
        "Agent %s exhausted retries (task=%s, attempts=%d, last_error=%s) → fallback%s",
        context.agent_id, context.task_type, attempt,
        retry.error_code if retry else "none",
        exit_reason,
    )
    fallback = finalize_fallback_player_action(
        agent=agent,
        context=context,
        fallback=agent._fallback_action(context),
        retry=retry,
        raw_text=raw_text,
        parsed_action=parsed_action,
        fallback_target_used=True,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        parse_success=parse_success,
        parse_error=parse_error_str,
        retry_count=attempt,
        structured_failure_reason=structured_failure_reason,
        structured_output_mode=structured_output_mode,
        structured_failure_stage=structured_failure_stage,
    )
    return fallback, retry
