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
    PlayerAction,
    PrivateIntent,
    RetryInfo,
    RiskFlag,
    SeerStance,
    TaskType,
    VoteBasis,
)
from werewolf_agent.model_gateway.router import ModelRouter

logger = logging.getLogger(__name__)


class OutputMode(str, Enum):
    FULL_ACTION = "full_action"
    TARGET_CHOICE = "target_choice"
    SPEECH_INTENT = "speech_intent"


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

    _MAX_JSON_CONTEXT_CHARS = 1800
    _MAX_TRANSCRIPT_ITEMS = 4
    _MAX_TRANSCRIPT_TEXT_CHARS = 220
    _MAX_SALIENCE_ITEMS = 4
    _CHOICE_TARGET_ACTIONS = {
        ActionType.VOTE,
        ActionType.WOLF_KILL,
        ActionType.USE_POISON,
        ActionType.CHECK_ALIGNMENT,
        ActionType.CHOOSE_MASTER,
        ActionType.HUNTER_SHOT,
        ActionType.BADGE_TRANSFER,
        ActionType.SHERIFF_VOTE,
    }
    _SPEECH_INTENT_TASKS = {
        TaskType.SPEECH,
        TaskType.SHERIFF_SPEECH,
        TaskType.DEFENSE_SPEECH,
        TaskType.PK_SPEECH,
        TaskType.LAST_WORDS,
    }
    _SPEECH_INTENTS = {
        "self_clear": "表水",
        "question_target": "质疑/追问目标",
        "stand_with_seer": "站边预言家或逻辑线",
        "respond_pressure": "回应质疑",
        "push_vote": "提出投票倾向",
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
        skill_tools_consumed = False

        attempt = 0
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
            has_skill_tools = bool(context.skill_tools) and not skill_tools_consumed
            if has_skill_tools:
                tools.extend(context.skill_tools)
                tool_choice_val: dict[str, Any] = {"type": "auto"}
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

            # Detect on-demand skill tool call (LLM requested tactical analysis)
            called_tool = getattr(result, "tool_call_name", "") or ""
            if called_tool and called_tool in context.skill_analyses:
                skill_tools_consumed = True
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
            if tool_call_required and not tool_call_received and not allow_text_tool_fallback:
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
            speech_quality_error = self._speech_quality_error(context, action)
            if speech_quality_error:
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="speech_quality",
                    error_message=speech_quality_error,
                    correction_hint=speech_quality_error,
                )
                continue
            vote_quality_error = self._vote_quality_error(context, action)
            if vote_quality_error:
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="vote_quality",
                    error_message=vote_quality_error,
                    correction_hint=vote_quality_error,
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
        fallback = self._fallback_action(context)
        trace = self._build_action_trace(
            context,
            raw_text=raw_text,
            parsed_action=parsed_action,
            final_action_type=fallback.action_type,
            retry=retry,
            fallback_reason=fallback.reason,
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
            tool_call_required=tool_call_required,
            tool_call_received=tool_call_received,
            tool_call_name="submit_player_action" if tool_call_required else "",
            parse_success=parse_success,
            parse_error=parse_error,
            retry_count=retry_count,
            structured_failure_reason=structured_failure_reason,
        )

    def _parse_action(self, text: str) -> tuple[PlayerAction | None, str | None]:
        """Parse LLM output into PlayerAction. Returns (action, error)."""
        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Try direct parse first
        try:
            data = json.loads(cleaned)
            return self._action_from_data(data)
        except json.JSONDecodeError as direct_error:
            parameter_data = self._extract_parameter_tag_action(cleaned)
            if parameter_data is not None:
                action, parse_error = self._action_from_data(parameter_data)
                if action is not None:
                    return action, None
                return None, parse_error

            candidates = self._extract_json_object_candidates(cleaned)
            if not candidates:
                return None, f"No JSON object found in output"
            first_error: str | None = None
            for candidate in candidates:
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError as e:
                    if first_error is None:
                        first_error = f"JSON parse error: {e}"
                    continue
                action, parse_error = self._action_from_data(data)
                if action is not None:
                    return action, None
                if first_error is None:
                    first_error = parse_error
            return None, first_error or f"JSON parse error: {direct_error}"

    def _action_from_data(self, data: Any) -> tuple[PlayerAction | None, str | None]:
        data = self._normalize_action_data(data)
        try:
            return PlayerAction(**data), None
        except ValidationError as e:
            sanitized = self._sanitize_optional_private_fields(data)
            if sanitized != data:
                try:
                    return PlayerAction(**sanitized), None
                except ValidationError:
                    pass
            return None, f"Schema validation error: {e}"

    def _normalize_action_data(self, data: Any) -> Any:
        """Normalize provider quirks before schema validation."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if isinstance(normalized.get("target_id"), str) and normalized["target_id"].strip().lower() in {
            "",
            "null",
            "none",
        }:
            normalized["target_id"] = None
        if "confidence" in normalized and isinstance(normalized["confidence"], str):
            try:
                normalized["confidence"] = float(normalized["confidence"].strip())
            except ValueError:
                pass
        return normalized

    def _extract_parameter_tag_action(self, text: str) -> dict[str, Any] | None:
        """Extract MiniMax-style <parameter name="...">value</parameter> tool payloads."""
        pairs = re.findall(
            r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</parameter>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not pairs:
            return None

        data: dict[str, Any] = {}
        for key, raw_value in pairs:
            value = unescape(raw_value.strip())
            if value.lower() in {"null", "none"}:
                data[key] = None
            elif key == "confidence":
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
            else:
                data[key] = value
        return data if "action_type" in data else None

    def _uses_choice_pipeline(self, context: AgentContext) -> bool:
        return (
            len(context.legal_actions) == 1
            and context.legal_actions[0] in self._CHOICE_TARGET_ACTIONS
            and bool(context.legal_targets)
        )

    def _uses_speech_intent_pipeline(self, context: AgentContext) -> bool:
        return (
            context.task_type in self._SPEECH_INTENT_TASKS
            and context.legal_actions == [ActionType.SPEECH]
        )

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
        data, parse_error = self._extract_decision_data(text)
        if data is None:
            return None, parse_error, None
        if "choice" not in data and "target_id" not in data:
            return None, "Choice output must include choice or target_id", data

        if context.legal_actions == [ActionType.VOTE]:
            repaired = self._repair_vote_decision(data, context)
        else:
            repaired = self._repair_target_decision(data, context)
        if repaired is None:
            return None, "Could not map choice to legal target", data

        action = PlayerAction(
            action_type=context.legal_actions[0],
            target_id=repaired["target_id"],
            speech="",
            reason=repaired["reason"],
            confidence=repaired["confidence"],
            seer_stance=repaired.get("seer_stance", SeerStance.UNDECIDED.value),
            vote_basis=repaired.get("vote_basis", VoteBasis.FALLBACK.value),
            standing_with_seer=repaired.get("standing_with_seer", ""),
            suspect_reason=repaired.get("suspect_reason", ""),
            not_voting_reason=repaired.get("not_voting_reason", ""),
            private_reason=repaired.get("private_reason", ""),
        )
        return action, None, repaired

    def _parse_speech_intent_action(
        self,
        text: str,
        context: AgentContext,
    ) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
        data, parse_error = self._extract_decision_data(text)
        if data is None:
            return None, parse_error, None
        if "intent" not in data and "speech" not in data:
            return None, "Speech intent output must include intent or speech", data

        repaired = self._repair_speech_intent_decision(data, context)
        action = PlayerAction(
            action_type=ActionType.SPEECH,
            target_id=repaired["target_id"],
            speech=repaired["speech"],
            reason=repaired["reason"],
            confidence=repaired["confidence"],
        )
        return action, None, repaired

    def _extract_decision_data(self, text: str) -> tuple[dict[str, Any] | None, str | None]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return self._normalize_action_data(data), None
            return None, "Decision JSON must be an object"
        except json.JSONDecodeError as direct_error:
            parameter_data = self._extract_parameter_tag_action(cleaned)
            if parameter_data is not None:
                return self._normalize_action_data(parameter_data), None
            candidates = self._extract_json_object_candidates(cleaned)
            for candidate in candidates:
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    return self._normalize_action_data(data), None
            return None, f"No JSON object found in output: {direct_error}"

    def _repair_vote_decision(
        self,
        data: dict[str, Any],
        context: AgentContext,
    ) -> dict[str, Any] | None:
        choice_map = self._vote_choice_map(context)
        target_id = self._target_from_vote_decision(data, choice_map, context.legal_targets)
        if target_id is None:
            return None

        summary = self._vote_candidate_summary(context, target_id)
        reason = self._clean_reason(data.get("reason")) or summary
        suspect_reason = self._clean_reason(data.get("suspect_reason")) or summary
        standing = self._clean_reason(data.get("standing_with_seer")) or self._infer_standing_with_seer(context)
        not_voting = self._clean_reason(data.get("not_voting_reason")) or self._default_not_voting_reason(
            context,
            target_id,
        )
        private_reason = self._clean_reason(data.get("private_reason")) or (
            f"结构化投票修复：在合法候选中选择{target_id}。依据：{reason}"
        )
        vote_basis = self._clean_enum_value(
            data.get("vote_basis"),
            {basis.value for basis in VoteBasis},
        )
        if vote_basis is None:
            vote_basis = self._infer_vote_basis(reason, suspect_reason, private_reason)
        seer_stance = self._clean_enum_value(
            data.get("seer_stance"),
            {stance.value for stance in SeerStance},
        )
        if seer_stance is None:
            seer_stance = self._infer_seer_stance(context, standing)
        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        return {
            "choice": self._choice_for_target(choice_map, target_id),
            "target_id": target_id,
            "reason": reason,
            "seer_stance": seer_stance,
            "vote_basis": vote_basis,
            "standing_with_seer": standing,
            "suspect_reason": suspect_reason,
            "not_voting_reason": not_voting,
            "private_reason": private_reason,
            "confidence": confidence,
        }

    def _repair_target_decision(
        self,
        data: dict[str, Any],
        context: AgentContext,
    ) -> dict[str, Any] | None:
        choice_map = self._vote_choice_map(context)
        target_id = self._target_from_vote_decision(data, choice_map, context.legal_targets)
        if target_id is None:
            return None

        reason = self._clean_reason(data.get("reason")) or self._target_candidate_summary(
            context,
            target_id,
        )
        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        return {
            "choice": self._choice_for_target(choice_map, target_id),
            "target_id": target_id,
            "reason": reason,
            "confidence": confidence,
        }

    def _repair_speech_intent_decision(
        self,
        data: dict[str, Any],
        context: AgentContext,
    ) -> dict[str, Any]:
        intent = str(data.get("intent") or "").strip()
        if intent not in self._SPEECH_INTENTS:
            intent = self._infer_speech_intent(data, context)
        target_id = self._speech_target_from_decision(data, context.legal_targets)
        speech = self._clean_reason(data.get("speech"))
        reason = self._clean_reason(data.get("reason"))
        if not speech:
            speech = self._synthesize_intent_speech(intent, target_id, context)
        speech = self._ensure_speech_quality_components(speech, intent, target_id, context)
        if not reason:
            reason = self._speech_intent_reason(intent, target_id)
        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return {
            "intent": intent,
            "target_id": target_id,
            "speech": speech,
            "reason": reason,
            "confidence": confidence,
        }

    def _vote_choice_map(self, context: AgentContext) -> dict[str, str]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {
            letters[idx]: target
            for idx, target in enumerate(context.legal_targets[:len(letters)])
        }

    def _target_from_vote_decision(
        self,
        data: dict[str, Any],
        choice_map: dict[str, str],
        legal_targets: list[str],
    ) -> str | None:
        choice = str(data.get("choice") or "").strip().upper()
        if choice in choice_map:
            return choice_map[choice]
        target = data.get("target_id")
        if isinstance(target, str) and target in legal_targets:
            return target
        haystack = " ".join(str(value) for value in data.values())
        for candidate in legal_targets:
            if candidate in haystack:
                return candidate
        if len(legal_targets) == 1:
            return legal_targets[0]
        return None

    def _choice_for_target(self, choice_map: dict[str, str], target_id: str) -> str:
        for choice, mapped_target in choice_map.items():
            if mapped_target == target_id:
                return choice
        return ""

    def _clean_reason(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text or text in {"未说明", "无", "none", "null"}:
            return ""
        return text

    def _vote_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        clues: list[str] = []
        for item in context.salience_items:
            if not isinstance(item, dict):
                continue
            target = item.get("target") or item.get("target_id") or item.get("player_id")
            if target != target_id:
                continue
            item_type = item.get("type") or item.get("event")
            if item_type == "seer_claim":
                speaker = item.get("speaker") or item.get("seer_id")
                result = item.get("result") or item.get("alignment")
                if speaker and result:
                    clues.append(f"{speaker}报{target_id}为{result}")
            elif item_type in {"vote_resolved", "vote"}:
                clues.append(f"{target_id}出现在关键票型中")
            elif item_type in {"player_died", "death"}:
                clues.append(f"{target_id}关联死亡事件")
        if clues:
            return "；".join(clues[:2])
        return f"{target_id}是当前合法投票候选，需要基于发言、票型和站边继续施压"

    def _target_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        action = context.legal_actions[0] if context.legal_actions else ActionType.NO_ACTION
        action_reasons = {
            ActionType.WOLF_KILL: "作为狼队夜间击杀目标",
            ActionType.USE_POISON: "作为女巫毒药目标",
            ActionType.CHECK_ALIGNMENT: "作为预言家查验目标",
            ActionType.CHOOSE_MASTER: "作为混血儿主人选择目标",
            ActionType.HUNTER_SHOT: "作为猎人开枪目标",
            ActionType.BADGE_TRANSFER: "作为警徽移交目标",
            ActionType.SHERIFF_VOTE: "作为警长投票目标",
        }
        clues: list[str] = []
        for item in context.salience_items:
            if not isinstance(item, dict):
                continue
            item_text = json.dumps(item, ensure_ascii=False)
            if target_id in item_text:
                clues.append(item_text[:80])
        basis = f"；依据：{'；'.join(clues[:2])}" if clues else ""
        return f"{target_id}{action_reasons.get(action, '作为当前合法目标')}较合适{basis}"

    def _infer_speech_intent(self, data: dict[str, Any], context: AgentContext) -> str:
        text = " ".join(str(value) for value in data.values())
        if any(word in text for word in ("站边", "预言家", "查验")):
            return "stand_with_seer"
        if any(word in text for word in ("投", "归票", "出")):
            return "push_vote"
        if any(word in text for word in ("回应", "解释", "表水")):
            return "respond_pressure"
        if context.legal_targets:
            return "question_target"
        return "self_clear"

    def _speech_target_from_decision(
        self,
        data: dict[str, Any],
        legal_targets: list[str],
    ) -> str | None:
        target = data.get("target_id")
        if isinstance(target, str) and target in legal_targets:
            return target
        haystack = " ".join(str(value) for value in data.values())
        for candidate in legal_targets:
            if candidate in haystack:
                return candidate
        return legal_targets[0] if len(legal_targets) == 1 else None

    def _synthesize_intent_speech(
        self,
        intent: str,
        target_id: str | None,
        context: AgentContext,
    ) -> str:
        target = target_id or (context.legal_targets[0] if context.legal_targets else "")
        clue = self._context_clues(context)
        basis = f"结合{clue}，" if clue else ""
        if intent == "stand_with_seer":
            return (
                f"{basis}我现在需要明确站边和逻辑线。"
                f"{('我更倾向相信' + target + '这边的信息，') if target else ''}"
                "接下来会继续核对查验、票型和发言是否能互相印证。"
            )
        if intent == "respond_pressure":
            return (
                f"{basis}我先回应当前压力：我的判断不是跟票，"
                "而是基于发言前后、票型变化和关键事件来排顺序。"
            )
        if intent == "push_vote":
            if target:
                return (
                    f"{basis}我这轮会把投票压力先放到{target}。"
                    f"{target}需要解释自己的站边、票型和对关键事件的回应。"
                )
            return f"{basis}我这轮会明确给出投票倾向，不接受继续模糊站边。"
        if intent == "question_target" and target:
            return (
                f"{basis}我想追问{target}：你的站边、票型和关键发言需要正面解释。"
                "如果解释仍然空泛，我会继续把你放在重点怀疑位。"
            )
        return (
            f"{basis}我先把自己的视角说清楚：我会按查验、死亡、票型和发言一致性来判断，"
            "不会只跟随场上声音。"
        )

    def _ensure_speech_quality_components(
        self,
        speech: str,
        intent: str,
        target_id: str | None,
        context: AgentContext,
    ) -> str:
        target = self._speech_pressure_target(intent, target_id, context)
        additions: list[str] = []
        if not re.search(r"好人|我是.*?(?:村民|预言家|女巫|猎人|p\d{2})", speech):
            if context.own_role in {"werewolf", "hybrid"}:
                additions.append(f"我是{context.agent_id}视角。")
            else:
                additions.append("我是好人视角。")
        if target and not re.search(r"(?:怀疑\s*p\d{2}|p\d{2}\s*有问题|投\s*p\d{2})", speech):
            additions.append(f"我怀疑{target}有问题。")
        if target and not re.search(r"(?:投|投票|归票|倾向).*?p\d{2}", speech):
            additions.append(f"我倾向投{target}。")
        if not re.search(r"矛盾|前后不一|不合理|查杀|查验|警徽流|对跳|票数|之前说|刚才说", speech):
            additions.append("依据是查验、票型和前后发言矛盾需要继续对上。")
        if additions:
            return speech.rstrip("。") + "。" + "".join(additions)
        return speech

    def _speech_pressure_target(
        self,
        intent: str,
        target_id: str | None,
        context: AgentContext,
    ) -> str | None:
        if intent == "stand_with_seer" and target_id:
            for candidate in context.legal_targets:
                if candidate != target_id:
                    return candidate
        return target_id or (context.legal_targets[0] if context.legal_targets else None)

    def _speech_intent_reason(self, intent: str, target_id: str | None) -> str:
        intent_label = self._SPEECH_INTENTS.get(intent, "补充发言")
        if target_id:
            return f"按发言意图“{intent_label}”围绕{target_id}组织公开发言"
        return f"按发言意图“{intent_label}”组织公开发言"

    def _infer_standing_with_seer(self, context: AgentContext) -> str:
        for item in context.salience_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or item.get("event")
            speaker = item.get("speaker") or item.get("seer_id")
            if item_type == "seer_claim" and speaker:
                return str(speaker)
        return ""

    def _infer_seer_stance(self, context: AgentContext, standing_with_seer: str) -> str:
        if standing_with_seer:
            return SeerStance.TRUST.value
        for item in context.salience_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or item.get("event")
            if item_type == "seer_claim":
                return SeerStance.UNDECIDED.value
        return SeerStance.NO_CLAIM.value

    def _infer_vote_basis(self, *texts: str) -> str:
        try:
            from werewolf_agent.runtime.vote_quality import (
                extract_vote_basis,
                normalize_vote_basis,
            )

            detected = extract_vote_basis(" ".join(text for text in texts if text))
            return normalize_vote_basis(detected)
        except Exception:
            logger.debug("Vote basis inference failed", exc_info=True)
            return VoteBasis.FALLBACK.value

    def _clean_enum_value(self, value: Any, allowed: set[str]) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned if cleaned in allowed else None

    def _default_not_voting_reason(self, context: AgentContext, target_id: str) -> str:
        others = [target for target in context.legal_targets if target != target_id]
        if not others:
            return "本轮只有一个合法投票目标，没有其他可排除候选。"
        return f"暂不投{', '.join(others[:4])}，因为当前可见线索优先指向{target_id}。"

    def _extract_json_object_candidates(self, text: str) -> list[str]:
        """Extract balanced JSON object candidates from mixed model text."""
        candidates: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
                continue
            if ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:idx + 1])
                    start = None

        action_candidates = [
            candidate for candidate in candidates
            if '"action_type"' in candidate or "'action_type'" in candidate
        ]
        return action_candidates or candidates

    def _sanitize_optional_private_fields(self, data: Any) -> Any:
        """Drop malformed optional audit fields without invalidating core action."""
        if not isinstance(data, dict):
            return data
        private_intent = data.get("private_intent")
        if not isinstance(private_intent, dict):
            return data

        sanitized = dict(data)
        sanitized_intent = dict(private_intent)

        valid_goals = {goal.value for goal in FactionGoal}
        if sanitized_intent.get("faction_goal") not in valid_goals:
            true_role = str(sanitized_intent.get("true_role") or sanitized.get("action_type") or "")
            sanitized_intent["faction_goal"] = (
                FactionGoal.CONFUSE_GOOD.value
                if true_role == "werewolf"
                else FactionGoal.FIND_WOLVES.value
            )

        valid_flags = {flag.value for flag in RiskFlag}
        flags = sanitized_intent.get("risk_flags")
        if isinstance(flags, list):
            sanitized_intent["risk_flags"] = [
                flag for flag in flags
                if isinstance(flag, str) and flag in valid_flags
            ]
        else:
            sanitized_intent["risk_flags"] = []

        sanitized["private_intent"] = sanitized_intent
        return sanitized

    def _player_action_tool(self, context: AgentContext) -> dict[str, Any]:
        action_values = [action.value for action in context.legal_actions]
        if not action_values:
            action_values = [action.value for action in ActionType]
        target_values: list[str | None] = list(context.legal_targets)
        if not self._all_legal_actions_require_target(context) and None not in target_values:
            target_values.append(None)
        target_schema: dict[str, Any] = {
            "type": ["string", "null"],
            "description": "Target player id when required; null otherwise.",
        }
        if context.legal_targets:
            target_schema["enum"] = target_values
        properties: dict[str, Any] = {
            "action_type": {
                "type": "string",
                "enum": action_values,
                "description": "Must be one of the currently legal actions.",
            },
            "target_id": target_schema,
            "speech": {
                "type": "string",
                "description": "Public Chinese speech. Empty string for private night actions if no speech is needed.",
            },
            "reason": {
                "type": "string",
                "description": "Short Chinese reason for the action.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        }
        if context.task_type == TaskType.VOTE:
            properties.update(self._vote_audit_tool_properties())
        elif context.task_type == TaskType.WOLF_DISCUSSION:
            properties["private_intent"] = {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "true_role": {"type": "string"},
                    "faction_goal": {
                        "type": "string",
                        "enum": [
                            "push_good_player_out",
                            "protect_teammate",
                            "find_wolves",
                            "survive",
                            "help_master_faction",
                            "confuse_good",
                            "deep_hook",
                            "aggressive_push",
                        ],
                    },
                    "claimed_view": {"type": "string"},
                    "pressure_target": {"enum": target_values},
                    "risk_flags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "avoid_night_kill_leak",
                                "avoid_teammate_exposure",
                                "high_visibility",
                                "low_trust",
                                "suspected",
                            ],
                        },
                    },
                },
                "required": ["true_role", "faction_goal", "claimed_view"],
            }
        required = ["action_type", "target_id", "speech", "reason", "confidence"]
        if context.task_type == TaskType.VOTE:
            required.extend([
                "seer_stance",
                "vote_basis",
                "standing_with_seer",
                "suspect_reason",
                "not_voting_reason",
                "private_reason",
            ])
        return {
            "name": "submit_player_action",
            "description": "Submit exactly one legal Werewolf player action.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            },
        }

    def _vote_audit_tool_properties(self) -> dict[str, Any]:
        return {
            "seer_stance": {
                "type": "string",
                "enum": [stance.value for stance in SeerStance],
                "description": "Vote stance enum about seer logic: trust, distrust, undecided, or no_claim.",
            },
            "vote_basis": {
                "type": "string",
                "enum": [basis.value for basis in VoteBasis],
                "description": "Primary vote basis enum.",
            },
            "standing_with_seer": {
                "type": "string",
                "description": "Private moderator-only vote audit: seer or logic line you stand with; empty if none.",
            },
            "suspect_reason": {
                "type": "string",
                "description": "Private moderator-only vote audit: why the final vote target is suspicious.",
            },
            "not_voting_reason": {
                "type": "string",
                "description": "Private moderator-only vote audit: why you are not voting other major candidates.",
            },
            "private_reason": {
                "type": "string",
                "description": "Private moderator-only vote audit: full reasoning for the moderator; never public speech.",
            },
        }

    def _speech_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        quality_phase = self._speech_quality_phase(context.task_type)
        if quality_phase is None or action.action_type != ActionType.SPEECH:
            return None
        try:
            from werewolf_agent.runtime.speech_quality import validate_public_speech

            result = validate_public_speech(
                action.speech,
                phase=quality_phase,
                context={
                    "recent_transcript": list(context.recent_transcript),
                    "public_summary": context.public_summary,
                    "must_address_alerts": context.strategy_directive.get("must_address_alerts", [])
                    if context.strategy_directive else [],
                },
            )
        except Exception:
            logger.debug("Speech quality validation failed unexpectedly", exc_info=True)
            return None
        if result.get("valid"):
            return None
        return str(result.get("hint") or "发言质量不足，请补充立场、怀疑对象、投票倾向和依据。")

    def _speech_quality_phase(self, task_type: TaskType) -> str | None:
        phase_by_task = {
            TaskType.SPEECH: "day_discussion",
            TaskType.SHERIFF_SPEECH: "sheriff_speech",
            TaskType.DEFENSE_SPEECH: "pk_speech",
            TaskType.PK_SPEECH: "pk_speech",
        }
        return phase_by_task.get(task_type)

    def _vote_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        if (
            context.task_type != TaskType.VOTE
            or action.action_type != ActionType.VOTE
            or not context.strategy_directive.get("require_vote_quality")
        ):
            return None
        try:
            from werewolf_agent.runtime.vote_quality import validate_structured_vote_action

            result = validate_structured_vote_action(
                action.model_dump(exclude={"trace"}),
                context={
                    "strategy_directive": context.strategy_directive,
                    "salience_items": list(context.salience_items),
                    "recent_transcript": list(context.recent_transcript),
                },
            )
        except Exception:
            logger.debug("Vote quality validation failed unexpectedly", exc_info=True)
            return None
        if result.get("valid"):
            return None
        return str(result.get("hint") or "投票必须包含预言家立场、投票基点和具体理由。")

    def _all_legal_actions_require_target(self, context: AgentContext) -> bool:
        return bool(context.legal_actions) and all(
            action in DefaultActionValidator._TARGET_REQUIRED_ACTIONS
            for action in context.legal_actions
        )

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
        reason = self._fallback_reason(context, safe_action, safe_target)

        return FallbackAction(
            action_type=safe_action,
            target_id=safe_target,
            speech=speech,
            reason=reason,
        )

    def _fallback_reason(
        self,
        context: AgentContext,
        action_type: ActionType,
        target_id: str | None,
    ) -> str:
        """Provide an audit-friendly fallback reason instead of an empty/default vote reason."""
        if action_type in {ActionType.VOTE, ActionType.SHERIFF_VOTE} and target_id:
            clues = self._context_clues(context)
            if clues:
                return f"fallback: 结构化输出失败，按当前可见线索优先选择{target_id}；依据：{clues}"
            return f"fallback: 结构化输出失败，按合法候选顺序选择{target_id}，后续需要补充站边和排除理由"
        if action_type in {
            ActionType.WOLF_KILL,
            ActionType.USE_POISON,
            ActionType.CHECK_ALIGNMENT,
            ActionType.CHOOSE_MASTER,
            ActionType.HUNTER_SHOT,
            ActionType.BADGE_TRANSFER,
        } and target_id:
            return f"fallback: 结构化输出失败，按当前合法目标选择{target_id}"
        return "fallback: retries exhausted"

    def _context_clues(self, context: AgentContext) -> str:
        clues: list[str] = []
        sheriff_id = context.visible_world_state.get("sheriff_id")
        if sheriff_id:
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
        target = context.legal_targets[0] if context.legal_targets else None
        salt = sum(ord(ch) for ch in f"{context.agent_id}:{context.day_number}")
        if context.task_type == TaskType.WOLF_DISCUSSION:
            if target:
                templates = [
                    (
                        "我是{agent_id}，狼队夜聊我建议优先刀{target}。"
                        "这个位置如果是神职能压缩好人信息，如果是民也方便我们白天做抗推布局。"
                    ),
                    (
                        "狼队视角我倾向先处理{target}。"
                        "今晚需要统一刀口，明天白天再围绕票型和发言把压力转出去。"
                    ),
                    (
                        "我建议本轮刀{target}，理由是这个位置后续容易被好人阵营认证。"
                        "我们先统一行动，再安排明天谁冲锋、谁倒钩。"
                    ),
                ]
                return templates[salt % len(templates)].format(
                    agent_id=context.agent_id,
                    target=target,
                )
            templates = [
                "狼队夜聊我建议先统一刀口，再分配明天的冲锋位和倒钩位，避免白天发言互相打架。",
                "狼队视角今晚先别分散意见，优先找神职或强势带队位，明天再顺着公共信息做推人路线。",
                "我这里建议先整理每个人明天的站位：一人带节奏，一人补逻辑，一人适度倒钩保护团队。",
            ]
            return templates[salt % len(templates)]
        if target:
            clues = self._context_clues(context)
            if clues:
                return (
                    f"我这轮先把视角压到{target}身上。依据是{clues}，"
                    f"我需要{target}正面回应自己的站边、票型和关键事件解释；"
                    "如果回应仍然空泛，我会把投票优先放在这里。"
                )
            templates = [
                (
                    "我这轮先重点听{target}的解释。"
                    "{target}需要交代站边、投票依据和不投其他人的理由；"
                    "如果这些信息补不出来，我会把他作为优先投票对象。"
                ),
                (
                    "我会先追问{target}的视角来源。"
                    "{target}需要把怀疑对象、站边逻辑和票型判断讲清楚，"
                    "否则这个位置容易成为今天的主要焦点。"
                ),
                (
                    "今天我先把{target}放进重点观察位。"
                    "我需要听到他对关键发言和投票选择的正面解释，"
                    "如果继续回避，我会考虑把票压过去。"
                ),
            ]
            return templates[salt % len(templates)].format(target=target)
        templates = [
            "我这轮会重点看发言前后是否一致、投票是否跟逻辑匹配，不接受纯划水过麦。",
            "现在信息还不够一锤定音，但每个人都要交代自己的站边和投票理由。",
            "后续我会优先跟进票型变化和谁在回避关键问题，不能只靠情绪归票。",
        ]
        return templates[salt % len(templates)]

    # Role name mapping for Chinese prompts
    _ROLE_NAMES = {
        "werewolf": "狼人", "villager": "村民", "seer": "预言家",
        "witch": "女巫", "hunter": "猎人", "idiot": "白痴",
        "hybrid": "混血儿",
    }

    def _build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt with role identity and constraints (Chinese)."""
        role_cn = self._ROLE_NAMES.get(context.own_role or "", context.own_role or "")
        parts = [
            "你是一场狼人杀游戏的玩家。请用中文发言和思考。",
            "【禁止事项】本局只有以下7种角色：狼人、村民、预言家、女巫、猎人、白痴、混血儿。"
            "绝对禁止提及守卫、恋人、丘比特、白狼王、熊、乌鸦、狐狸、盗贼、吹笛者等任何不存在的角色。"
            "没有守卫，不存在被守护的可能。没有平安夜是由守卫造成的。平安夜只有两种可能：狼人空刀，或女巫使用解药救人。",
            "【平安夜与女巫规则硬约束】平安夜不等于无人被刀，只代表公开结果无人死亡。"
            "除狼人外，普通玩家不知道狼人是否空刀；除女巫外，普通玩家不知道女巫是否救人。"
            "不能用“平安夜没人死”反驳女巫知道刀口，也不能把“不公开救谁”直接等同于假女巫。"
            "可以质疑跳女巫玩家是否用药、为什么暂不公开银水、以及发言前后是否矛盾。"
            "不要跟风复述已有指控；每次发言必须给出独立证据、明确区分事实和推测。",
            "【公开记录引用约束】只有游戏概况、可见状态、关键事件、近期发言中明确出现的信息，才能称为公开记录。"
            "不要编造某玩家曾经说过的话、声称过的身份、投票理由或查验结论；不确定时必须写成推测或质疑。",
            f"你的玩家ID: {context.agent_id}",
            f"你的名字: {self.player_name}",
        ]
        if context.own_role:
            parts.append(f"你的角色: {role_cn}（{context.own_role}）")
            # Role-specific rules
            role_rules = {
                "hunter": "猎人规则：被狼人杀死或被放逐时可以开枪带走一人；被女巫毒杀时不能开枪。夜间无法自保。",
                "idiot": "白痴规则：被放逐时亮出身份免死，但失去投票权且不能再被放逐；之后被狼人杀死才算真正死亡。夜间无法自保。",
                "witch": "女巫规则：有一瓶解药和一瓶毒药，不能在同一夜同时使用。解药不能自救。N1 / 首夜大概率应该救人。",
                "seer": "预言家规则：每晚可查验一人身份（好人/狼人），查验混血儿结果为好人。上警时必须留两夜警徽流。",
                "werewolf": "狼人规则：夜间与队友讨论击杀目标。可以悍跳预言家上警对抗真预言家。",
                "hybrid": "混血儿规则：N1 / 首夜选择一名主人，跟随主人阵营获胜。主人死亡后阵营不再改变。",
            }
            if context.own_role in role_rules:
                parts.append(role_rules[context.own_role])
        # Inject belief state: who I suspect and trust
        if context.belief_state:
            suspects = context.belief_state.get("my_suspects", [])
            trusted = context.belief_state.get("my_trusted", [])
            belief_lines = []
            if suspects:
                suspect_desc = ", ".join(
                    f"{s['player']}(嫌疑{s['faction_lean']}, 猜{s['top_role_guess']})"
                    for s in suspects[:5]
                )
                belief_lines.append(f"我怀疑的玩家: {suspect_desc}")
            if trusted:
                trust_desc = ", ".join(
                    f"{t['player']}(倾向{t['faction_lean']}, 信任{t['trust']})"
                    for t in trusted[:5]
                )
                belief_lines.append(f"我信任的玩家: {trust_desc}")
            if belief_lines:
                parts.append("【我的判断（基于已有信息的推理，可能是错的）】" + " ".join(belief_lines))
        parts.append(f"当前阶段: {context.phase}")
        if context.legal_actions:
            parts.append(
                f"可用操作: {[a.value for a in context.legal_actions]}"
            )
        if context.legal_targets:
            parts.append(f"可选目标: {context.legal_targets}")
        # Mandatory vote pressure hints
        if context.legal_actions and ActionType.VOTE in context.legal_actions:
            if ActionType.NO_ACTION not in context.legal_actions:
                parts.append("重要：本轮投票必须选择一名玩家放逐，不能弃票！")
            if context.legal_actions == [ActionType.VOTE] and context.legal_targets:
                parts.append("你必须投出选票，从可选目标中选择一人。")
            parts.append(
                "投票时必须先在心里完成判断，并在JSON中额外给出这些私有字段："
                "seer_stance（枚举：trust/distrust/undecided/no_claim）、"
                "vote_basis（枚举：seer_check/seer_siding/speech_logic/vote_pattern/pressure_test/anti_herd/fallback）、"
                "standing_with_seer（你站边哪个预言家/逻辑线，没有则写空字符串）、"
                "suspect_reason（为什么怀疑最终投票对象）、"
                "not_voting_reason（为什么不投其他主要候选人）、"
                "private_reason（完整内心活动：为什么投他、担心什么、最终如何决定）。"
                "这些字段不会公开发言，只给主持人审计。"
            )
        parts.append(
            "请优先通过 submit_player_action 工具提交结构化行动。"
            "如果当前模型无法调用工具，则只输出一个JSON对象，不要解释、不要Markdown。"
            "字段必须包含 action_type、target_id、speech、reason、confidence。"
        )
        parts.append("重要：speech字段必须使用中文，这是你在游戏中的公开发言。")
        parts.append("")

        # Show context-appropriate examples based on legal actions
        if context.legal_actions and any(
            a in (ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL) for a in context.legal_actions
        ):
            example_target = context.legal_targets[0] if context.legal_targets else "p05"
            parts.append("示例输出（狼人击杀场景）：")
            parts.append(
                f'{{"action_type": "wolf_kill", "target_id": "{example_target}", '
                f'"speech": "", '
                f'"reason": "选择击杀目标", "confidence": 0.8, '
                f'"private_intent": {{"true_role": "werewolf", '
                f'"faction_goal": "push_good_player_out", "claimed_view": "我是好人", '
                f'"pressure_target": "{example_target}", "risk_flags": []}}}}'
            )
            parts.append("示例输出（狼人空刀场景）：")
            parts.append(
                '{"action_type": "wolf_no_kill", "target_id": null, '
                '"speech": "", '
                '"reason": "本轮空刀策略", "confidence": 0.6, '
                '"private_intent": {"true_role": "werewolf", '
                '"faction_goal": "confuse_good", "claimed_view": "我是好人", '
                '"pressure_target": null, "risk_flags": []}}'
            )
        elif context.legal_actions and ActionType.SHERIFF_REGISTER in context.legal_actions:
            parts.append("示例输出（上警报名场景）：")
            parts.append(
                '{"action_type": "sheriff_register", "target_id": null, '
                '"speech": "我报名竞选警长。", '
                '"reason": "希望参与警上发言并争取带队", "confidence": 0.6}'
            )
            if ActionType.NO_ACTION in context.legal_actions:
                parts.append("示例输出（不上警场景）：")
                parts.append(
                    '{"action_type": "no_action", "target_id": null, '
                    '"speech": "我不上警，先听警上发言再判断。", '
                    '"reason": "当前信息不足，先观察警上格局", "confidence": 0.6}'
                )
        else:
            parts.append("示例输出（发言场景）：")
            parts.append('{"action_type": "speech", "target_id": null, '
                         '"speech": "我觉得p05很可疑，昨晚他的发言前后矛盾。", '
                         '"reason": "根据发言分析", "confidence": 0.7, '
                         '"private_intent": {"true_role": "villager", '
                         '"faction_goal": "find_wolves", "claimed_view": "我是好人", '
                         '"pressure_target": "p05", "risk_flags": []}}')
            parts.append("示例输出（投票场景）：")
            parts.append('{"action_type": "vote", "target_id": "p05", '
                         '"speech": "", '
                         '"reason": "公开理由：p05发言可疑", '
                         '"seer_stance": "trust", '
                         '"vote_basis": "seer_check", '
                         '"standing_with_seer": "p03", '
                         '"suspect_reason": "p05没有回应p03的查杀逻辑，发言前后不一致", '
                         '"not_voting_reason": "p07虽然被踩，但目前没有明确查验或票型证据", '
                         '"private_reason": "心里活动：我更信p03的预言家线，p05像狼队抗推失败后的防守位，所以投p05。", '
                         '"confidence": 0.8, '
                         '"private_intent": {"true_role": "seer", '
                         '"faction_goal": "find_wolves", "claimed_view": "我是预言家", '
                         '"pressure_target": "p05", "risk_flags": []}}')
        return "\n".join(parts)

    def _build_prompt(self, context: AgentContext, retry: RetryInfo) -> str:
        """Build the user prompt with context and retry hints (Chinese)."""
        parts: list[str] = []

        if context.public_summary:
            parts.append(f"游戏概况:\n{self._truncate_text(context.public_summary, self._MAX_JSON_CONTEXT_CHARS)}")

        if context.visible_world_state:
            parts.append(
                "可见状态: "
                + self._compact_json_for_prompt(context.visible_world_state, self._MAX_JSON_CONTEXT_CHARS)
            )

        if context.salience_items:
            parts.append(
                "关键事件: "
                + self._compact_json_for_prompt(
                    context.salience_items[:self._MAX_SALIENCE_ITEMS],
                    self._MAX_JSON_CONTEXT_CHARS,
                )
            )

        if context.strategy_directive:
            parts.append(
                "策略建议: "
                + self._compact_json_for_prompt(context.strategy_directive, self._MAX_JSON_CONTEXT_CHARS)
            )

        if context.persona_snapshot:
            parts.append(
                "人格设定: "
                + self._compact_json_for_prompt(context.persona_snapshot, self._MAX_JSON_CONTEXT_CHARS)
            )

        if context.recent_transcript:
            parts.append("近期发言:\n" + self._format_recent_transcript(context.recent_transcript))

        if retry.correction_hint:
            parts.append(f"\n纠正提示（第{retry.attempt}/{retry.max_retries}次尝试）: {retry.correction_hint}")
            parts.append(f"错误信息: {retry.error_message}")

        output_mode = self._select_output_mode(context)
        if output_mode == OutputMode.TARGET_CHOICE:
            parts.append(self._format_choice_prompt(context))
        elif output_mode == OutputMode.SPEECH_INTENT:
            parts.append(self._format_speech_intent_prompt(context))

        parts.append(self._strict_output_contract(context, output_mode))
        return "\n".join(parts)

    def _compact_json_for_prompt(self, value: Any, max_chars: int) -> str:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return self._truncate_text(text, max_chars)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"...（已截断，原长度{len(text)}）"

    def _format_recent_transcript(self, transcript: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in transcript[-self._MAX_TRANSCRIPT_ITEMS:]:
            speaker = item.get("speaker", "?")
            text = self._truncate_text(
                str(item.get("text", "")),
                self._MAX_TRANSCRIPT_TEXT_CHARS,
            )
            lines.append(f"  [{speaker}] {text}")
        return "\n".join(lines)

    def _strict_output_contract(self, context: AgentContext, output_mode: OutputMode | None = None) -> str:
        legal_actions = [action.value for action in context.legal_actions]
        legal_targets = list(context.legal_targets)
        output_mode = output_mode or self._select_output_mode(context)
        if output_mode == OutputMode.TARGET_CHOICE:
            output_fields = "choice、reason、confidence"
            if context.legal_actions == [ActionType.VOTE]:
                output_fields = (
                    "choice、reason、seer_stance、vote_basis、standing_with_seer、suspect_reason、"
                    "not_voting_reason、private_reason、confidence"
                )
            lines = [
                "",
                "最终输出协议（必须遵守）：",
                "1. 只输出一个choice决策JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
                "2. JSON必须以{开头、以}结尾，且只能有一个对象。",
                f"3. 最终输出字段：{output_fields}。",
                "4. choice只能取上方候选枚举中的字母，不要直接编写target_id。",
            ]
            if context.legal_actions == [ActionType.VOTE]:
                lines.append(
                    "5. 投票还必须包含seer_stance、vote_basis、standing_with_seer、"
                    "suspect_reason、not_voting_reason、private_reason，理由字段不能写“未说明”。"
                )
            lines.append("现在提交行动。")
            return "\n".join(lines)
        if output_mode == OutputMode.SPEECH_INTENT:
            lines = [
                "",
                "最终输出协议（必须遵守）：",
                "1. 只输出一个发言意图JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
                "2. JSON必须以{开头、以}结尾，且只能有一个对象。",
                "3. 最终输出字段：intent、target_id、speech、reason、confidence。",
                "4. target_id没有目标时必须是null，不要写字符串\"null\"。",
            ]
            if legal_targets:
                lines.append(f"5. target_id只能取这些玩家之一或null：{legal_targets}。")
            lines.append("现在提交行动。")
            return "\n".join(lines)
        lines = [
            "",
            "最终输出协议（必须遵守）：",
            "1. 首选 submit_player_action 工具调用提交结构化参数。",
            "2. 如果当前模型无法工具调用，只输出一个JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
            "3. JSON必须以{开头、以}结尾，且只能有一个对象。",
            "4. target_id没有目标时必须是null，不要写字符串\"null\"。",
            "5. 必填字段：action_type、target_id、speech、reason、confidence。",
        ]
        if legal_actions:
            lines.append(f"6. action_type只能取：{legal_actions}。")
        if legal_targets:
            lines.append(f"7. target_id只能取这些玩家之一或null：{legal_targets}。")
        if ActionType.VOTE in context.legal_actions:
            lines.append(
                "8. 投票还必须包含seer_stance、vote_basis、standing_with_seer、"
                "suspect_reason、not_voting_reason、private_reason，理由字段不能写“未说明”。"
            )
        lines.append("现在提交行动。")
        return "\n".join(lines)

    def _format_choice_prompt(self, context: AgentContext) -> str:
        is_vote = context.legal_actions == [ActionType.VOTE]
        header = "投票候选枚举" if is_vote else "目标候选枚举"
        lines = [f"{header}（必须从中选择一个choice，不要直接编写target_id）："]
        for choice, target_id in self._vote_choice_map(context).items():
            summary = (
                self._vote_candidate_summary(context, target_id)
                if is_vote
                else self._target_candidate_summary(context, target_id)
            )
            lines.append(f"{choice} = {target_id}，摘要：{summary}")
        if is_vote:
            example = (
                '{"choice":"A","reason":"投票公开理由",'
                '"seer_stance":"trust",'
                '"vote_basis":"seer_check",'
                '"standing_with_seer":"站边的预言家或逻辑线",'
                '"suspect_reason":"为什么怀疑该候选",'
                '"not_voting_reason":"为什么不投其他候选",'
                '"private_reason":"完整内心理由",'
                '"confidence":0.7}'
            )
        else:
            example = '{"choice":"A","reason":"选择该目标的简明理由","confidence":0.7}'
        lines.extend([
            "只需要输出choice决策JSON，程序会把choice映射为target_id并组装PlayerAction。",
            "示例：",
            example,
        ])
        return "\n".join(lines)

    def _format_speech_intent_prompt(self, context: AgentContext) -> str:
        lines = ["发言意图枚举（先选intent，再写speech；不要输出分析过程）："]
        for intent, label in self._SPEECH_INTENTS.items():
            lines.append(f"- {intent}: {label}")
        if context.legal_targets:
            lines.append(f"可围绕的目标玩家: {context.legal_targets}")
        lines.extend([
            "发言阶段只需要输出intent决策JSON，程序会组装为speech行动。",
            "speech必须是公开发言正文，不能留空，不能写“未发言”。",
            "示例：",
            (
                '{"intent":"question_target","target_id":"p05",'
                '"speech":"我想追问p05，你的站边和投票理由需要讲清楚。",'
                '"reason":"围绕可疑目标施压","confidence":0.7}'
            ),
        ])
        return "\n".join(lines)
