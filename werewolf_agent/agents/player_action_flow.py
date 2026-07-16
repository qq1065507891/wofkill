# -*- coding: utf-8 -*-
"""
封装 PlayerAgent 的结构化行动生成、重试和 fallback 流程。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-16

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
from werewolf_agent.agents.player_retry_hints import (
    build_empty_response_retry,
    build_missing_tool_call_retry,
)
from werewolf_agent.agents.player_quality_retries import (
    build_speech_quality_retry,
    build_vote_quality_retry,
)
from werewolf_agent.agents.player_generation_request import (
    build_player_generation_request,
    call_player_generation_request,
)
from werewolf_agent.agents.player_action_result import (
    finalize_fallback_player_action,
    finalize_successful_player_action,
)
from werewolf_agent.agents.semantic_repair_audit import (
    build_semantic_repair_audit,
    preserve_verified_claim_in_fallback,
    semantic_repair_retains_verified_claim,
)
from werewolf_agent.agents.player_fallback_speech import generic_fallback_speech_used
from werewolf_agent.agents.player_latency import latency_from_result as _latency_from_result
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    OutputMode,
    PlayerAction,
    RetryInfo,
    TaskType,
)
from werewolf_agent.model_gateway.structured_output import (
    StructuredFailureStage,
    StructuredOutputMode,
    StructuredOutputPolicy,
    classify_structured_failure,
)
from werewolf_agent.model_gateway.generation_attempt_context import GenerationAttemptContext
from werewolf_agent.runtime.decision_outcomes import summarize_attempt_counts

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
    generation_attempt_context = GenerationAttemptContext(run_scope=context.agent_id)
    semantic_repair_source: PlayerAction | None = None

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
                generation_attempt_context,
            )
        except NotImplementedError:
            # Provider does not support tool_choice
            structured_failure_reason = "structured_output_unsupported"
            structured_failure_stage = StructuredFailureStage.PROTOCOL.value
            generation_attempt_context.append_terminal_fallback()
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
                retry_count=summarize_attempt_counts(
                    generation_attempt_context.attempts
                ).retry_count,
                structured_failure_reason=structured_failure_reason,
                structured_output_mode=structured_output_mode,
                structured_failure_stage=structured_failure_stage,
                metrics_error_code=structured_failure_reason,
                execution_attempts=generation_attempt_context.attempts,
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
            if any(
                item.root_cause.value == "invalid_output"
                for item in getattr(result, "attempts", ())
            ):
                failure_reason = "empty_response"
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
                generation_attempt_context.append_terminal_fallback()
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
                    retry_count=summarize_attempt_counts(
                        generation_attempt_context.attempts
                    ).retry_count,
                    structured_failure_reason=structured_failure_reason,
                    structured_output_mode=structured_output_mode,
                    structured_failure_stage=structured_failure_stage,
                    metrics_error_code="model_generation_failed",
                    execution_attempts=generation_attempt_context.attempts,
                )
                return fallback, retry
            failure_category = _categorize_failure_category(
                latency_ms=_latency_from_result(result),
                raw_error=None,
                http_status=int(getattr(result, "http_status", 0) or 0),
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
            retry = build_empty_response_retry(
                context=context,
                attempt=attempt,
                max_retries=agent.max_retries,
                failure_category=failure_category,
                output_mode=output_mode,
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
                generation_attempt_context.reject_latest_output()
                break
            generation_attempt_context.reject_latest_output()
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
            retry = build_missing_tool_call_retry(
                attempt=attempt,
                max_retries=agent.max_retries,
                structured_failure_reason=structured_failure_reason,
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
                generation_attempt_context.reject_latest_output()
                break
            generation_attempt_context.reject_latest_output()
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
                action, parse_error = agent._parse_action(result.text, context=context)
        elif output_mode == OutputMode.SPEECH_INTENT:
            action, parse_error = agent._parse_action(result.text, context=context)
            if parse_error and action is None:
                action, parse_error, choice_data = _parse_speech_intent_action(
                    result.text,
                    context,
                )
        else:
            action, parse_error = agent._parse_action(result.text, context=context)
            # 完整行动 JSON 也要复用投票字段修复器：模型经常遗漏
            # candidate_comparison 等审计字段，但已有 target_id 时可以
            # 基于当前上下文安全补齐，而不应直接降级成 FallbackAction。
            if (
                action is None
                and parse_error
                and parse_error.startswith("Schema validation error:")
                and ActionType.VOTE in context.legal_actions
            ):
                repaired_action, repaired_error, repaired_data = _parse_choice_action(
                    result.text,
                    context,
                )
                if repaired_action is not None:
                    action, parse_error, choice_data = (
                        repaired_action,
                        None,
                        repaired_data,
                    )
        parsed_action = action
        if parse_error:
            parse_error_str = parse_error
            if parse_error.startswith("Schema validation error:"):
                structured_failure_reason = "schema_validation"
            elif parse_error.startswith("truncated_json:"):
                structured_failure_reason = "truncated_json"
            else:
                structured_failure_reason = "parse_error"
            failure_stage = classify_structured_failure(
                structured_failure_reason
            )
            structured_failure_stage = (
                failure_stage.value if failure_stage else None
            )
            correction_hint = agent._parse_correction_hint(context, parse_error)
            if structured_failure_reason == "truncated_json":
                correction_hint = (
                    "上次输出的JSON没有闭合。请缩短发言和reason，"
                    "只输出一个完整JSON对象，确保以}结尾；"
                    "不要输出private_intent长列表或多余解释。"
                )
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code=structured_failure_reason,
                error_message=parse_error,
                correction_hint=correction_hint or (
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
                generation_attempt_context.reject_latest_output()
                break
            generation_attempt_context.reject_latest_output()
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
                generation_attempt_context.reject_latest_output()
                break
            generation_attempt_context.reject_latest_output()
            continue
        speech_quality_err = agent._speech_quality_error(context, action)
        if speech_quality_err:
            if semantic_repair_source is None:
                semantic_repair_source = action
            structured_failure_reason = "speech_quality"
            structured_failure_stage = StructuredFailureStage.SEMANTIC.value
            retry = build_speech_quality_retry(
                speech_quality_err,
                attempt=attempt,
                max_retries=agent.max_retries,
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                generation_attempt_context.reject_latest_output()
                break
            generation_attempt_context.reject_latest_output()
            continue
        vote_quality_err = agent._vote_quality_error(context, action)
        if vote_quality_err:
            structured_failure_reason = "vote_quality"
            structured_failure_stage = StructuredFailureStage.SEMANTIC.value
            retry = build_vote_quality_retry(
                vote_quality_err,
                attempt=attempt,
                max_retries=agent.max_retries,
            )
            should_short_circuit, last_error_signature = agent._check_repeat_error_signature(
                retry, raw_text, attempt, last_error_signature,
                structured_output_mode=structured_output_mode,
            )
            if should_short_circuit:
                generation_attempt_context.reject_latest_output()
                break
            generation_attempt_context.reject_latest_output()
            continue

        if (
            semantic_repair_source is not None
            and not semantic_repair_retains_verified_claim(
                context, semantic_repair_source, action
            )
        ):
            structured_failure_reason = "speech_quality"
            structured_failure_stage = StructuredFailureStage.SEMANTIC.value
            retry = RetryInfo(
                attempt=attempt,
                max_retries=agent.max_retries,
                error_code="semantic_claim_retention",
                error_message="修复结果未完整保持目标与公开论点边界。",
                correction_hint=(
                    "保持源目标和全部已有公开来源支撑的论点，且不得新增事实声明。"
                ),
            )
            should_short_circuit, last_error_signature = (
                agent._check_repeat_error_signature(
                    retry,
                    raw_text,
                    attempt,
                    last_error_signature,
                    structured_output_mode=structured_output_mode,
                )
            )
            if should_short_circuit:
                generation_attempt_context.reject_latest_output()
                break
            generation_attempt_context.reject_latest_output()
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
            retry_count=summarize_attempt_counts(
                generation_attempt_context.attempts
            ).retry_count,
            structured_output_mode=structured_output_mode,
            execution_attempts=generation_attempt_context.attempts,
            semantic_repair_audit=(
                build_semantic_repair_audit(
                    context,
                    semantic_repair_source,
                    action,
                    success=True,
                )
                if semantic_repair_source is not None else None
            ),
        ), retry

    # Fallback
    exit_reason = f" early_exit={retry.early_exit_reason}" if retry and retry.early_exit_reason else ""
    logger.warning(
        "Agent %s exhausted retries (task=%s, attempts=%d, last_error=%s) → fallback%s",
        context.agent_id, context.task_type, attempt,
        retry.error_code if retry else "none",
        exit_reason,
    )
    generation_attempt_context.append_terminal_fallback()
    fallback_action = agent._fallback_action(context)
    if context.task_type is not TaskType.WOLF_DISCUSSION:
        fallback_action = fallback_action.model_copy(update={"speech": ""})
    if semantic_repair_source is not None:
        fallback_action = preserve_verified_claim_in_fallback(
            context, semantic_repair_source, fallback_action
        )
    fallback = finalize_fallback_player_action(
        agent=agent,
        context=context,
        fallback=fallback_action,
        retry=retry,
        raw_text=raw_text,
        parsed_action=parsed_action,
        fallback_target_used=True,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        parse_success=parse_success,
        parse_error=parse_error_str,
        retry_count=summarize_attempt_counts(
            generation_attempt_context.attempts
        ).retry_count,
        structured_failure_reason=structured_failure_reason,
        structured_output_mode=structured_output_mode,
        structured_failure_stage=structured_failure_stage,
        execution_attempts=generation_attempt_context.attempts,
        semantic_repair_audit=(
            build_semantic_repair_audit(
                context,
                semantic_repair_source,
                fallback_action,
                success=False,
                generic_template_used=generic_fallback_speech_used(
                    context, fallback_action.speech
                ),
            )
            if semantic_repair_source is not None else None
        ),
    )
    return fallback, retry
