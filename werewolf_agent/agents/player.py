"""Player Agent: schema-constrained output with illegal-output retry/fallback.

Player agents propose actions, reasons, and speech. They MUST NOT:
- Mutate GameState directly
- See moderator_full or other players' private state
- Bypass RuleEngine legal action sets
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from html import unescape
from typing import Any, Protocol

from pydantic import ValidationError

from werewolf_agent.agents.schemas import (
    ActionTrace,
    ActionType,
    AgentContext,
    FallbackAction,
    FactionGoal,
    OutputMode,
    PlayerAction,
    PrivateIntent,
    RetryInfo,
    RiskFlag,
    SeerStance,
    TaskType,
    VoteBasis,
)
from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.output_parser import (
    repair_json_text as _repair_json_impl,
    extract_json_object_candidates as _extract_json_impl,
    extract_parameter_tag_action as _extract_param_impl,
    normalize_action_data as _normalize_impl,
    clean_enum_value as _clean_enum_impl,
    clean_reason as _clean_reason_impl,
    sanitize_optional_private_fields as _sanitize_impl,
    action_from_data as _action_from_data_impl,
    parse_action as _parse_action_impl,
    extract_decision_data as _extract_decision_impl,
    repair_vote_decision as _repair_vote_impl,
    repair_target_decision as _repair_target_impl,
    repair_speech_intent_decision as _repair_speech_impl,
    vote_choice_map as _vote_choice_map_impl,
    target_from_vote_decision as _target_from_vote_impl,
    choice_for_target as _choice_for_target_impl,
    vote_candidate_summary as _vote_summary_impl,
    target_candidate_summary as _target_summary_impl,
    infer_speech_intent as _infer_intent_impl,
    speech_target_from_decision as _speech_target_impl,
    synthesize_intent_speech as _synthesize_impl,
    ensure_speech_quality_components as _ensure_quality_impl,
    speech_pressure_target as _pressure_target_impl,
    speech_intent_reason as _intent_reason_impl,
    infer_standing_with_seer as _infer_standing_impl,
    infer_seer_stance as _infer_stance_impl,
    infer_vote_basis as _infer_basis_impl,
    default_not_voting_reason as _default_not_voting_impl,
    parse_choice_action as _parse_choice_impl,
    parse_speech_intent_action as _parse_speech_impl,
    uses_choice_pipeline as _uses_choice_impl,
    uses_speech_intent_pipeline as _uses_speech_impl,
)
from werewolf_agent.agents.tool_schema import (
    player_action_tool as _tool_impl,
    vote_audit_tool_properties as _vote_audit_impl,
    speech_quality_error as _speech_quality_impl,
    speech_quality_phase as _speech_phase_impl,
    vote_quality_error as _vote_quality_impl,
    all_legal_actions_require_target as _all_target_impl,
)
from werewolf_agent.model_gateway.router import ModelRouter

logger = logging.getLogger(__name__)


class ActionValidator(Protocol):
    """Protocol for validating actions against RuleEngine legal sets."""

    def validate(
        self,
        action_type: ActionType,
        target_id: str | None,
        legal_actions: list[ActionType],
        legal_targets: list[str],
    ) -> tuple[bool, str | None]:
        ...


def _fallback_reason(action: FallbackAction) -> str:
    """Return a fallback reason that does NOT embed the target_id.

    The caller is responsible for substituting the actual target into the
    log display. This prevents the audit trail from showing "chose p07" while
    the actual ``vote_target`` is a different player (the LLM's choice may
    later override the fallback target in ``agent_day_vote``).
    """
    return "fallback: 结构化输出失败，按当前可见线索选择默认目标"


class DefaultActionValidator:
    """Validates agent output against RuleEngine-provided legal sets."""

    _TARGET_REQUIRED_ACTIONS = {
        ActionType.VOTE,
        ActionType.WOLF_KILL,
        ActionType.USE_POISON,
        ActionType.CHECK_ALIGNMENT,
        ActionType.CHOOSE_MASTER,
        ActionType.HUNTER_SHOT,
        ActionType.BADGE_TRANSFER,
        ActionType.SHERIFF_VOTE,
    }

    def validate(
        self,
        action_type: ActionType,
        target_id: str | None,
        legal_actions: list[ActionType],
        legal_targets: list[str],
    ) -> tuple[bool, str | None]:
        if legal_actions and action_type not in legal_actions:
            return False, f"action_type={action_type.value} not in legal_actions"
        if (
            target_id is not None
            and action_type in self._TARGET_REQUIRED_ACTIONS
            and legal_actions
            and not legal_targets
        ):
            return False, f"no legal_targets provided for action_type={action_type.value}"
        if target_id is not None and legal_targets and target_id not in legal_targets:
            return False, f"target_id={target_id} not in legal_targets"
        return True, None


class PlayerAgent:
    """Schema-constrained player agent with retry and fallback.

    Flow:
    1. Build prompt from AgentContext
    2. Call ModelRouter for LLM generation
    3. Parse into PlayerAction via Pydantic schema
    4. Validate against RuleEngine legal sets
    5. Retry with correction hints on failure
    6. Fallback to safe action after max retries
    """

    _CHOICE_TARGET_ACTIONS = DefaultActionValidator._TARGET_REQUIRED_ACTIONS
    _SPEECH_INTENT_TASKS = {
        TaskType.SPEECH,
        TaskType.SHERIFF_SPEECH,
        TaskType.DEFENSE_SPEECH,
        TaskType.PK_SPEECH,
        TaskType.LAST_WORDS,
    }

    def __init__(
        self,
        agent_id: str,
        model_router: ModelRouter,
        validator: ActionValidator | None = None,
        max_retries: int = 3,
        player_name: str | None = None,
        persona_key: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.player_name = player_name or agent_id
        self.persona_key = persona_key
        self.model_router = model_router
        self.validator = validator or DefaultActionValidator()
        self.max_retries = max_retries

    def act(self, context: AgentContext) -> tuple[PlayerAction | FallbackAction, RetryInfo]:
        """Generate a constrained player action with retry/fallback."""
        retry = RetryInfo(max_retries=self.max_retries)
        raw_text = ""
        parsed_action: PlayerAction | None = None
        # Task 9: track structured output metadata across retries
        tool_call_required = True  # We always pass tools and tool_choice
        tool_call_received = False
        parse_success = False
        parse_error_str: str | None = None
        structured_failure_reason: str | None = None
        skill_call_count = 0  # cap total on-demand skill injections per act()
        skill_skip_count = 0  # track times LLM skipped skill tools

        attempt = 0
        _MAX_SKILL_SKIP = 3
        # Resolve model config once to check text-fallback support.
        # Models that reliably return plain-text JSON don't need forced
        # tool_choice — we let them respond in their native format and
        # parse the text directly, skipping wasted retries.
        _fb_config, _fb = self.model_router.resolve_config(
            self.agent_id, context.task_type.value,
        )
        _model_text_fallback = _fb_config.allow_text_tool_fallback

        while attempt < self.max_retries:
            attempt += 1
            retry = RetryInfo(
                attempt=attempt,
                max_retries=self.max_retries,
                error_code=retry.error_code,
                error_message=retry.error_message,
                correction_hint=retry.correction_hint,
            )

            # Build tool list: always include submit_player_action; optionally
            # include skill analysis tools for the LLM to call on-demand.
            tools = [self._player_action_tool(context)]
            has_skill_tools = bool(context.skill_tools)
            if has_skill_tools:
                tools.extend(context.skill_tools)
                tool_choice_val: dict[str, Any] | None = {"type": "auto"}
            elif _model_text_fallback:
                # This model returns plain-text JSON natively — no need
                # to force tool_choice. Let the provider decide format.
                tool_choice_val = None
            else:
                tool_choice_val = {"type": "tool", "name": "submit_player_action"}

            # Generate LLM output
            prompt = self._build_prompt(context, retry)
            try:
                result = self.model_router.generate(
                    agent_id=self.agent_id,
                    task_type=context.task_type.value,
                    prompt=prompt,
                    system_prompt=self._build_system_prompt(context),
                    tools=tools,
                    tool_choice=tool_choice_val,
                )
            except NotImplementedError:
                # Provider does not support tool_choice
                structured_failure_reason = "structured_output_unsupported"
                fallback = self._fallback_action(context)
                trace = self._build_action_trace(
                    context,
                    raw_text="",
                    parsed_action=None,
                    final_action_type=fallback.action_type,
                    retry=retry,
                    fallback_reason=fallback.reason,
                    tool_call_required=tool_call_required,
                    tool_call_received=False,
                    parse_success=False,
                    parse_error="provider does not support tool_choice",
                    retry_count=attempt,
                    structured_failure_reason=structured_failure_reason,
                )
                fallback = fallback.model_copy(update={"trace": trace})
                return fallback, retry

            raw_text = result.text or ""

            tool_call_received = bool(getattr(result, "tool_call_received", False))

            # ── On-demand skill loading (load_skill tool pattern) ──
            # The model calls a skill analysis tool → inject analysis body.
            called_tool = getattr(result, "tool_call_name", "") or ""
            if called_tool and called_tool in context.skill_analyses and skill_call_count < 2:
                skill_call_count += 1
                analysis = context.skill_analyses.get(called_tool, "")
                if analysis:
                    hint = (
                        f"【技能分析结果】\n{analysis}\n\n"
                        "请基于以上分析，通过 submit_player_action 提交你的行动。"
                    )
                else:
                    hint = "技能分析暂无结果，请直接通过 submit_player_action 提交你的行动。"
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    correction_hint=hint,
                )
                attempt -= 1  # skill tool 调用不消耗重试次数
                continue

            # ── Skill skip detection: LLM submitted action without calling any skill tool ──
            # When skill tools are available but the LLM went straight to submit_player_action,
            # nudge it to call an analysis tool first.  After _MAX_SKIP attempts, force-inject
            # all skill analyses directly into the prompt so good players still get the info.
            _called_skill_tool = called_tool and called_tool in context.skill_analyses
            if (
                has_skill_tools
                and tool_call_received
                and not _called_skill_tool
                and skill_call_count == 0
                and context.skill_analyses
                and skill_skip_count < _MAX_SKILL_SKIP
            ):
                skill_skip_count += 1
                if skill_skip_count >= _MAX_SKILL_SKIP:
                    # Fallback: LLM won't call skill tools — inject analyses directly
                    all_analyses = "\n\n".join(
                        v for v in context.skill_analyses.values() if v
                    )
                    if not all_analyses:
                        break  # nothing to inject, proceed normally
                    retry = RetryInfo(
                        attempt=attempt,
                        max_retries=self.max_retries,
                        correction_hint=(
                            f"【技能分析结果】\n{all_analyses}\n\n"
                            "请基于以上分析，通过 submit_player_action 提交你的行动。"
                        ),
                    )
                    attempt -= 1
                    continue
                skill_names = ", ".join(
                    t.get("name", "") for t in context.skill_tools
                )
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    correction_hint=(
                        f"请先调用分析工具（{skill_names}）获取局势分析，"
                        "然后再通过 submit_player_action 提交行动。"
                        "不要跳过分析直接提交。"
                    ),
                )
                attempt -= 1  # skill skip retry 不消耗重试次数
                continue

            if not result.text:
                failure_reason = self._latest_generation_failure_reason()
                if failure_reason:
                    if "NotImplementedError" in failure_reason:
                        structured_failure_reason = "structured_output_unsupported"
                    else:
                        structured_failure_reason = "model_generation_failed"
                    retry = RetryInfo(
                        attempt=attempt,
                        max_retries=self.max_retries,
                        error_code="model_generation_failed",
                        error_message=failure_reason,
                        correction_hint="Provider generation failed; using fallback action.",
                    )
                    fallback = self._fallback_action(context)
                    trace = self._build_action_trace(
                        context,
                        raw_text="",
                        parsed_action=None,
                        final_action_type=fallback.action_type,
                        retry=retry,
                        fallback_reason=fallback.reason,
                        tool_call_required=tool_call_required,
                        tool_call_received=False,
                        parse_success=False,
                        parse_error=failure_reason,
                        retry_count=attempt,
                        structured_failure_reason=structured_failure_reason,
                    )
                    fallback = fallback.model_copy(update={"trace": trace})
                    return fallback, retry
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="empty_response",
                    error_message="Model returned empty text",
                    correction_hint="Please provide a valid JSON action.",
                )
                continue

            allow_text_tool_fallback = bool(
                getattr(result, "allow_text_tool_fallback", False)
                and getattr(result, "text_fallback_used", False)
            )
            if (
                tool_call_required
                and not tool_call_received
                and not allow_text_tool_fallback
                and not _model_text_fallback
            ):
                structured_failure_reason = (
                    getattr(result, "structured_failure_reason", None)
                    or "missing_tool_call"
                )
                parse_error_str = "missing required tool call: submit_player_action"
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code=structured_failure_reason,
                    error_message=parse_error_str,
                    correction_hint=(
                        "必须通过 submit_player_action 工具调用提交结构化参数；"
                        "不要把JSON写在普通文本内容里。"
                    ),
                )
                continue

            # Parse JSON. Mandatory vote tasks may use a narrower choice schema;
            # the program maps that choice back into a legal PlayerAction.
            choice_data: dict[str, Any] | None = None
            output_mode = self._select_output_mode(context)
            if output_mode == OutputMode.TARGET_CHOICE:
                action, parse_error, choice_data = self._parse_choice_action(
                    result.text,
                    context,
                )
                if parse_error and action is None:
                    action, parse_error = self._parse_action(result.text)
            elif output_mode == OutputMode.SPEECH_INTENT:
                action, parse_error = self._parse_action(result.text)
                if parse_error and action is None:
                    action, parse_error, choice_data = self._parse_speech_intent_action(
                        result.text,
                        context,
                    )
            else:
                action, parse_error = self._parse_action(result.text)
            parsed_action = action
            if parse_error:
                parse_error_str = parse_error
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="parse_error",
                    error_message=parse_error,
                    correction_hint=(
                        "只输出JSON，不要解释、不要Markdown代码块。必须包含action_type、target_id、speech、"
                        "reason、confidence；action_type必须来自合法动作，target_id必须来自合法目标或null。"
                    ),
                )
                continue

            parse_success = True

            # Validate against legal sets
            valid, validation_error = self.validator.validate(
                action.action_type, action.target_id,
                context.legal_actions, context.legal_targets,
            )
            if not valid:
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="illegal_action",
                    error_message=validation_error,
                    correction_hint=f"Legal actions: {[a.value for a in context.legal_actions]}. "
                                    f"Legal targets: {context.legal_targets}",
                )
                continue
            speech_quality_err = self._speech_quality_error(context, action)
            if speech_quality_err:
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="speech_quality",
                    error_message=speech_quality_err,
                    correction_hint=speech_quality_err,
                )
                continue
            vote_quality_err = self._vote_quality_error(context, action)
            if vote_quality_err:
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="vote_quality",
                    error_message=vote_quality_err,
                    correction_hint=vote_quality_err,
                )
                continue

            # Private intent is stored but never written to public timeline
            trace = self._build_action_trace(
                context,
                raw_text=raw_text,
                parsed_action=choice_data or action,
                final_action_type=action.action_type,
                retry=retry,
                tool_call_required=tool_call_required,
                tool_call_received=tool_call_received,
                parse_success=parse_success,
                retry_count=attempt,
            )
            return action.model_copy(update={"trace": trace}), retry

        # Fallback
        logger.warning(
            "Agent %s exhausted retries (task=%s, attempts=%d, last_error=%s) → fallback",
            context.agent_id, context.task_type, attempt,
            retry.error_code if retry else "none",
        )
        fallback = self._fallback_action(context)
        trace = self._build_action_trace(
            context,
            raw_text=raw_text,
            parsed_action=parsed_action,
            final_action_type=fallback.action_type,
            retry=retry,
            fallback_reason=fallback.reason,
            fallback_target_used=True,
            fallback_target_id=fallback.target_id,
            tool_call_required=tool_call_required,
            tool_call_received=tool_call_received,
            parse_success=parse_success,
            parse_error=parse_error_str,
            retry_count=self.max_retries,
            structured_failure_reason=structured_failure_reason,
        )
        fallback = fallback.model_copy(update={"trace": trace})
        return fallback, retry

    def _latest_generation_failure_reason(self) -> str | None:
        get_usage_log = getattr(self.model_router, "get_usage_log", None)
        if get_usage_log is None:
            return None
        usage_log = get_usage_log()
        if not usage_log:
            return None
        last_record = usage_log[-1]
        if last_record.success or not last_record.fallback_reason:
            return None
        return str(last_record.fallback_reason)

    def _build_action_trace(
        self,
        context: AgentContext,
        *,
        raw_text: str,
        parsed_action: PlayerAction | dict[str, Any] | None,
        final_action_type: ActionType,
        retry: RetryInfo,
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
        return ActionTrace(
            raw_text=raw_text,
            parsed_action=(
                parsed_action.model_dump(exclude={"trace"})
                if isinstance(parsed_action, PlayerAction)
                else parsed_action
            ),
            final_action_type=final_action_type.value,
            legal_actions=[action.value for action in context.legal_actions],
            legal_targets=list(context.legal_targets),
            retry=retry.model_dump(),
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

    # ── Delegated to output_parser.py ──

    @staticmethod
    def _repair_json_text(raw: str) -> str:
        return _repair_json_impl(raw)

    def _parse_action(self, text: str) -> tuple[PlayerAction | None, str | None]:
        return _parse_action_impl(text)

    def _action_from_data(self, data: Any) -> tuple[PlayerAction | None, str | None]:
        return _action_from_data_impl(data)

    def _normalize_action_data(self, data: Any) -> Any:
        return _normalize_impl(data)

    def _extract_parameter_tag_action(self, text: str) -> dict[str, Any] | None:
        return _extract_param_impl(text)

    def _uses_choice_pipeline(self, context: AgentContext) -> bool:
        return _uses_choice_impl(context.legal_actions, context.legal_targets)

    def _uses_speech_intent_pipeline(self, context: AgentContext) -> bool:
        return _uses_speech_impl(context.legal_actions, context.task_type, self._SPEECH_INTENT_TASKS)

    def _select_output_mode(self, context: AgentContext) -> OutputMode:
        if self._uses_choice_pipeline(context):
            return OutputMode.TARGET_CHOICE
        if self._uses_speech_intent_pipeline(context):
            return OutputMode.SPEECH_INTENT
        return OutputMode.FULL_ACTION

    def _parse_choice_action(
        self,
        text: str,
        context: AgentContext,
    ) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
        return _parse_choice_impl(
            text,
            context.legal_actions,
            context.legal_targets,
            context.salience_items,
        )

    def _parse_speech_intent_action(
        self,
        text: str,
        context: AgentContext,
    ) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
        return _parse_speech_impl(
            text,
            context.agent_id,
            context.own_role,
            context.legal_targets,
            context.salience_items,
            context.visible_world_state,
            context.recent_transcript,
        )

    def _extract_decision_data(self, text: str) -> tuple[dict[str, Any] | None, str | None]:
        return _extract_decision_impl(text)

    def _repair_vote_decision(
        self,
        data: dict[str, Any],
        context: AgentContext,
    ) -> dict[str, Any] | None:
        return _repair_vote_impl(
            data, context.legal_actions, context.legal_targets, context.salience_items,
        )

    def _repair_target_decision(
        self,
        data: dict[str, Any],
        context: AgentContext,
    ) -> dict[str, Any] | None:
        return _repair_target_impl(
            data, context.legal_actions, context.legal_targets, context.salience_items,
        )

    def _repair_speech_intent_decision(
        self,
        data: dict[str, Any],
        context: AgentContext,
    ) -> dict[str, Any]:
        return _repair_speech_impl(
            data,
            context.agent_id,
            context.own_role,
            context.legal_targets,
            context.salience_items,
            context.visible_world_state,
            context.recent_transcript,
        )

    def _vote_choice_map(self, context: AgentContext) -> dict[str, str]:
        return _vote_choice_map_impl(context.legal_targets)

    def _target_from_vote_decision(
        self,
        data: dict[str, Any],
        choice_map: dict[str, str],
        legal_targets: list[str],
    ) -> str | None:
        return _target_from_vote_impl(data, choice_map, legal_targets)

    def _choice_for_target(self, choice_map: dict[str, str], target_id: str) -> str:
        return _choice_for_target_impl(choice_map, target_id)

    def _clean_reason(self, value: Any) -> str:
        return _clean_reason_impl(value)

    def _vote_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        return _vote_summary_impl(context.salience_items, target_id)

    def _target_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        return _target_summary_impl(context.legal_actions, context.salience_items, target_id)

    def _infer_speech_intent(self, data: dict[str, Any], context: AgentContext) -> str:
        return _infer_intent_impl(data, context.legal_targets)

    def _speech_target_from_decision(
        self,
        data: dict[str, Any],
        legal_targets: list[str],
    ) -> str | None:
        return _speech_target_impl(data, legal_targets)

    def _synthesize_intent_speech(
        self,
        intent: str,
        target_id: str | None,
        context: AgentContext,
    ) -> str:
        return _synthesize_impl(
            intent, target_id,
            context.salience_items, context.visible_world_state,
            context.recent_transcript, context.legal_targets,
        )

    def _ensure_speech_quality_components(
        self,
        speech: str,
        intent: str,
        target_id: str | None,
        context: AgentContext,
    ) -> str:
        return _ensure_quality_impl(
            speech, intent, target_id,
            context.own_role, context.agent_id, context.legal_targets,
        )

    def _speech_pressure_target(
        self,
        intent: str,
        target_id: str | None,
        context: AgentContext,
    ) -> str | None:
        return _pressure_target_impl(intent, target_id, context.legal_targets)

    def _speech_intent_reason(self, intent: str, target_id: str | None) -> str:
        return _intent_reason_impl(intent, target_id)

    def _infer_standing_with_seer(self, context: AgentContext) -> str:
        return _infer_standing_impl(context.salience_items)

    def _infer_seer_stance(self, context: AgentContext, standing_with_seer: str) -> str:
        return _infer_stance_impl(context.salience_items, standing_with_seer)

    def _infer_vote_basis(self, *texts: str) -> str:
        return _infer_basis_impl(*texts)

    def _clean_enum_value(self, value: Any, allowed: set[str]) -> str | None:
        return _clean_enum_impl(value, allowed)

    def _default_not_voting_reason(self, context: AgentContext, target_id: str) -> str:
        return _default_not_voting_impl(context.legal_targets, target_id)

    def _extract_json_object_candidates(self, text: str) -> list[str]:
        return _extract_json_impl(text)

    def _sanitize_optional_private_fields(self, data: Any) -> Any:
        return _sanitize_impl(data)

    # ── Delegated to tool_schema.py ──

    def _player_action_tool(self, context: AgentContext) -> dict[str, Any]:
        return _tool_impl(context.legal_actions, context.legal_targets, context.task_type)

    def _vote_audit_tool_properties(self) -> dict[str, Any]:
        return _vote_audit_impl()

    def _speech_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        return _speech_quality_impl(
            context.task_type,
            action,
            context.recent_transcript,
            context.public_summary,
            context.strategy_directive,
        )

    def _speech_quality_phase(self, task_type: TaskType) -> str | None:
        return _speech_phase_impl(task_type)

    def _vote_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        return _vote_quality_impl(
            context.task_type,
            action,
            context.strategy_directive,
            context.salience_items,
            context.recent_transcript,
        )

    def _all_legal_actions_require_target(self, context: AgentContext) -> bool:
        return _all_target_impl(context.legal_actions)

    # ── Remaining PlayerAgent methods ──

    def _fallback_action(self, context: AgentContext) -> FallbackAction:
        """Compute a safe fallback action from legal sets."""
        if context.legal_actions:
            safe_action = context.legal_actions[0]
        else:
            safe_action = ActionType.NO_ACTION

        safe_target = None
        if safe_action in {
            ActionType.VOTE, ActionType.WOLF_KILL, ActionType.USE_POISON,
            ActionType.CHECK_ALIGNMENT, ActionType.CHOOSE_MASTER,
            ActionType.HUNTER_SHOT, ActionType.BADGE_TRANSFER,
            ActionType.SHERIFF_VOTE,
        } and context.legal_targets:
            # For vote actions, exclude self and use evidence-based fallback
            if safe_action == ActionType.VOTE:
                non_self = [t for t in context.legal_targets if t != context.agent_id]
                if non_self:
                    fb = context.strategy_directive.get("_vote_fallback_target") if context.strategy_directive else None
                    if fb and fb in non_self:
                        safe_target = fb
                    else:
                        safe_target = non_self[0]
            else:
                safe_target = context.legal_targets[0]

        speech = ""
        if safe_action == ActionType.SPEECH:
            speech = self._fallback_speech(context)

        # Build a generic fallback action. Reason is computed at the end once
        # we have the full FallbackAction (which is target-aware) so that the
        # reason string never embeds the target_id.
        fallback = FallbackAction(
            action_type=safe_action,
            target_id=safe_target,
            speech=speech,
            reason="",
        )
        fallback = fallback.model_copy(update={"reason": _fallback_reason(fallback)})
        return fallback

    def _context_clues(self, context: AgentContext) -> str:
        clues: list[str] = []
        sheriff_id = context.visible_world_state.get("sheriff_id")
        alive_players = context.visible_world_state.get("alive_players", [])
        if sheriff_id and sheriff_id in alive_players:
            clues.append(f"当前警长是{sheriff_id}")
        for item in context.salience_items[:3]:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or item.get("event")
            if item_type == "seer_claim":
                speaker = item.get("speaker") or item.get("seer_id")
                target = item.get("target") or item.get("target_id")
                result = item.get("result") or item.get("alignment")
                if speaker and target and result:
                    clues.append(f"{speaker}报{target}为{result}")
            elif item_type in {"player_died", "death"}:
                player = item.get("player_id") or item.get("target_id")
                reason = item.get("reason")
                if player:
                    clues.append(f"{player}死亡" + (f"({reason})" if reason else ""))
            elif item_type == "vote_resolved":
                exiled = item.get("exiled")
                if exiled:
                    clues.append(f"上一轮放逐{exiled}")
        if context.recent_transcript:
            last = context.recent_transcript[-1]
            speaker = last.get("speaker")
            text = str(last.get("text") or "").strip()
            if speaker and text:
                clues.append(f"{speaker}最近发言：{text[:24]}")
        return "；".join(clues[:3])

    def _fallback_speech(self, context: AgentContext) -> str:
        import hashlib
        import logging
        _log = logging.getLogger(__name__)

        # Hash-based target selection: avoids all agents converging on legal_targets[0]
        seed_str = f"{context.agent_id}:{context.day_number}:{context.phase}"
        seed_hash = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        # For wolves, prefer non-teammate targets to avoid self-damage
        legal = list(context.legal_targets) if context.legal_targets else []
        own_role = context.own_role or ""
        if own_role == "werewolf" and context.strategy_directive:
            wolf_plan = context.strategy_directive.get("wolf_team_plan", {})
            teammates = set()
            for k in ("fake_seer", "pusher", "deep_hook"):
                t = wolf_plan.get(k, "")
                if t and t in legal:
                    teammates.add(t)
            non_teammate = [t for t in legal if t not in teammates]
            if non_teammate:
                legal = non_teammate
        target = legal[seed_hash % len(legal)] if legal else None

        # Template index uses hash, not sum(ord(...)) — better distribution
        tmpl_idx = seed_hash % 7
        prefix = "[FALLBACK]"

        if context.task_type == TaskType.WOLF_DISCUSSION:
            if target:
                templates = [
                    f"{prefix}狼队夜聊我建议优先刀{{target}}，这个位置如果是神职能压缩好人信息。",
                    f"{prefix}狼队视角我倾向先处理{{target}}，今晚统一刀口，明天白天把压力转出去。",
                    f"{prefix}我建议本轮刀{{target}}，后续容易被好人认证，统一行动再分配冲锋倒钩。",
                    f"{prefix}从狼队视角看{{target}}威胁较大，建议集中票型统一处理，避免分刀。",
                    f"{prefix}今晚目标{{target}}，理由是这个位置存活越久越难处理，尽早解决。",
                    f"{prefix}狼队今晚刀{{target}}，明天我们安排一人引领讨论方向，一人补位配合。",
                    f"{prefix}我倾向于刀{{target}}，它在神职概率较高的位置，赌中收益很大。",
                ]
                return templates[tmpl_idx].format(target=target)
            templates = [
                f"{prefix}狼队夜聊先统一刀口，再分配明天的冲锋位和倒钩位，避免发言互相打架。",
                f"{prefix}狼队视角今晚先别分散意见，优先找神职或强势带队位，明天顺着信息推人。",
                f"{prefix}建议先整理每人明天的站位：一人带节奏，一人补逻辑，一人适度倒钩保护团队。",
                f"{prefix}今晚不宜空刀——连续空刀会暴露战术意图，至少制造一个刀口给女巫压力。",
                f"{prefix}狼队需要确定今晚行动，刀口一致才能最大化信息不对称优势。",
                f"{prefix}提醒队友注意发言一致性，不同人的站边不要互相矛盾以免被好人抓住破绽。",
                f"{prefix}狼队视角先定今晚目标，再决定明天谁冲锋谁潜伏，分工明确胜率更高。",
            ]
            return templates[tmpl_idx]

        if context.task_type in (TaskType.SHERIFF_SPEECH, TaskType.PK_SPEECH):
            templates = [
                f"{prefix}我上警是想给出自己的独立判断视角，重点关注前几位发言的逻辑一致性。",
                f"{prefix}我参加警长竞选，希望通过观察和提问帮好人理清局势。",
                f"{prefix}上警是为了确保好人阵营有人能带节奏，我会根据后续发言调整站边。",
                f"{prefix}我是好人视角上警，主要是防止狼人控场，请大家根据发言质量判断。",
                f"{prefix}上警竞选，我有信心带队——我会认真分析每个人的发言和投票逻辑。",
                f"{prefix}参选警长不是为了秀存在感，而是要让好人阵营有一个清晰的发言方向。",
                f"{prefix}我上警是对局势负责，不想看到警徽落入可疑玩家手中。",
            ]
            return templates[tmpl_idx]

        if context.task_type == TaskType.DEFENSE_SPEECH:
            templates = [
                f"{prefix}我确实不是狼人，请大家仔细分析我的发言和投票逻辑。",
                f"{prefix}我没有理由被推，关注我的人应该先看看自己的视角是否正确。",
                f"{prefix}我是好人，我的选择都是基于公开信息，没有任何隐藏动机。",
                f"{prefix}回顾我的发言和投票，没有任何矛盾之处，被推可能是狼人在带节奏。",
                f"{prefix}如果你们仔细看我的逻辑链，会发现我的站边和推理都是连贯且合理的。",
                f"{prefix}被质疑很正常，但我希望大家关注推我的人背后的动机——可能是狼人抗推。",
                f"{prefix}请好人看清局势，我不是狼，真正的问题可能在那些急于归票的人身上。",
            ]
            return templates[tmpl_idx]

        if context.task_type == TaskType.LAST_WORDS:
            if target:
                templates = [
                    f"{prefix}遗言重点关注{{target}}，发言逻辑存在明显矛盾，请大家后续留意。",
                    f"{prefix}走了，提醒大家注意{{target}}的立场和行为不一致，我对此有较大疑虑。",
                    f"{prefix}最后说一句，{{target}}的发言中有些关键点没有解释清楚，值得深挖。",
                    f"{prefix}遗言不多说，但{{target}}的投票路线和发言立场严重不符，建议重点观察。",
                    f"{prefix}我注意到{{target}}在关键轮次的站边突变，这不正常——好人阵营请留意。",
                    f"{prefix}临别前提醒一句：{{target}}可能是突破口，其逻辑链有明显断裂。",
                    f"{prefix}遗言：关注{{target}}，其行为模式与好人视角不符，建议后续深入盘查。",
                ]
                return templates[tmpl_idx].format(target=target)
            templates = [
                f"{prefix}遗言不多说了，请大家仔细分析每个人的站边逻辑和投票记录。",
                f"{prefix}我相信好人阵营能通过票型和发言找出狼人，加油。",
                f"{prefix}最后提醒一下，注意观察谁在关键投票中立场摇摆。",
                f"{prefix}遗言：希望大家冷静分析，不要被情绪化发言带偏，聚焦票型和逻辑链。",
                f"{prefix}走了。好人阵营请重点复盘关键轮的投票分布，那里有答案。",
                f"{prefix}我的身份是好人，希望我的出局能让你们更清晰地看清局势。",
                f"{prefix}遗言简短：信任逻辑不信任直觉，仔细对比每个人的发言与投票是否一致。",
            ]
            return templates[tmpl_idx]

        # SPEECH and other day discussion types
        if target:
            templates = [
                f"{prefix}我目前对{{target}}有较大疑虑，其发言逻辑不够连贯，需要进一步观察。",
                f"{prefix}从现有信息来看，{{target}}的立场和行为有矛盾，我倾向于关注这个方向。",
                f"{prefix}我分析了一下，{{target}}的发言中有些观点缺乏依据，我对此保持警惕。",
                f"{prefix}综合来看{{target}}在关键节点的表现比较可疑，值得进一步深挖其动机。",
                f"{prefix}{{target}}的投票行为和发言内容之间存在落差，这一点不太对劲。",
                f"{prefix}目前关注{{target}}——其逻辑推理链中有几处跳跃，不像自然的好人思维。",
                f"{prefix}{{target}}的站边轨迹值得关注，在关键轮次的变化缺乏充分解释。",
            ]
            return templates[tmpl_idx].format(target=target)
        templates = [
            f"{prefix}我目前还在整理信息，请大家注意分析发言中的逻辑矛盾和票型走向。",
            f"{prefix}暂时没有确定的目标，但我会重点关注后续发言中立场摇摆的人。",
            f"{prefix}根据现有公开信息，我建议大家都仔细梳理一下各人的站边逻辑。",
            f"{prefix}这一轮信息量较大，我需要时间消化——建议大家关注投票链和发言一致性。",
            f"{prefix}本轮我先听大家的分析，重点观察谁的逻辑链最严密、谁的立场有突变。",
            f"{prefix}我倾向于保持开放态度，不急于站边——让子弹飞一会儿，看后续发言质量。",
            f"{prefix}好人阵营需要团结，但也要警惕跟风——独立思考是我给大家的建议。",
        ]
        _log.warning(
            "fallback speech used for agent=%s day=%s phase=%s task=%s",
            context.agent_id, context.day_number, context.phase, context.task_type,
        )
        return templates[tmpl_idx]

    # ── Prompt building delegated to PlayerPromptBuilder (s10 pipeline) ──

    def _build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt via s10 pipeline: core + rules + role_guide + skills + output_contract."""
        return PlayerPromptBuilder(context, self.player_name).build_system_prompt()

    def _build_prompt(self, context: AgentContext, retry: RetryInfo) -> str:
        """Build user prompt via s10 pipeline: dynamic per-turn context and task instructions."""
        return PlayerPromptBuilder(context, self.player_name).build_user_prompt(retry)
