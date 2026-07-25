# -*- coding: utf-8 -*-
"""
玩家 Agent public facade，保留身份、persona、fallback 和解析辅助入口。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-25
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

if TYPE_CHECKING:
    from werewolf_agent.model_gateway.generation_attempt_context import (
        GenerationAttemptContext,
    )

from werewolf_agent.agents.action_validation import (
    ActionValidator,
    DefaultActionValidator,
)
from werewolf_agent.agents.discussion_summary import (
    DiscussionSummary,
    DiscussionSummaryGenerationError,
    discussion_summary_tool,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    PlayerAction,
    RetryInfo,
    TaskType,
)
from werewolf_agent.agents.parse_dispatch import (
    select_output_mode as _select_output_mode,
)
from werewolf_agent.agents.player_failures import (
    # 兼容旧模块私有 helper 导入路径，实际流程已迁入 player_action_flow。
    categorize_failure_category as _categorize_failure_category,  # noqa: F401
    fallback_reason as _fallback_reason,
)
from werewolf_agent.agents.player_generation import (
    latest_generation_failure_reason as _latest_generation_failure_reason,
)
from werewolf_agent.agents.player_fallback_speech import (
    build_fallback_speech,
    context_clues,
)
from werewolf_agent.agents.player_latency import (  # noqa: F401
    latency_from_result as _latency_from_result,
)
from werewolf_agent.agents.player_persona import (
    attach_persona_snapshot,
    record_persona_exposure,
)
from werewolf_agent.agents.player_retry import (
    build_fallback_action as _build_fallback_action,
    check_repeat_error_signature as _check_repeat_error_signature,
    fallback_vote_target_from_context as _fallback_vote_target_from_context,
)
from werewolf_agent.agents.player_retry_hints import build_schema_validation_hint
from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.prompt_output import discussion_summary_prompts
from werewolf_agent.agents.planning import planning_envelope_to_action
from werewolf_agent.agents.metrics_collector import MetricsCollector
from werewolf_agent.agents.output_parser import (
    parse_action as _parse_action_impl,
    extract_decision_data as _extract_decision_impl,
    vote_choice_map as _vote_choice_map_impl,
    vote_candidate_summary as _vote_summary_impl,
    target_candidate_summary as _target_summary_impl,
)
from werewolf_agent.agents.tool_schema import (
    player_action_tool as _tool_impl,
    speech_quality_error as _speech_quality_impl,
    vote_quality_error as _vote_quality_impl,
)
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.persona_runtime.router import PersonaRouter

_SHERIFF_VOTE_FORBIDDEN_AUDIT_FIELDS = (
    "seer_stance",
    "vote_basis",
    "standing_with_seer",
    "suspect_reason",
    "not_voting_reason",
    "private_reason",
)

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
        persona_router: PersonaRouter | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.player_name = player_name or agent_id
        self.persona_key = persona_key
        self.persona_router = persona_router
        self.model_router = model_router
        self.validator = validator or DefaultActionValidator()
        self.max_retries = max_retries
        # Per-player failure profile: aggregates per-attempt outcomes so
        # developers can identify which persona's prompt template needs
        # tuning. Memory-only; not persisted across sessions.
        self.metrics_collector = MetricsCollector()

    def act(
        self,
        context: AgentContext,
        *,
        generation_attempt_context: GenerationAttemptContext | None = None,
    ) -> tuple[PlayerAction | FallbackAction, RetryInfo]:
        """Generate a constrained player action with retry/fallback.

        2026-07-21 R4: 加 ``generation_attempt_context`` kwarg 让 e2e 测试 / 外部 caller
        显式注入 attempt 上下文以便观测 mode_downgrades / attempts 累积. 默认 None,
        仍走 ``run_player_action_flow`` 内部 fresh 上下文, 行为不变.
        """
        from werewolf_agent.agents.player_action_flow import run_player_action_flow

        if generation_attempt_context is not None:
            return run_player_action_flow(
                self, context,
                generation_attempt_context=generation_attempt_context,
            )
        return run_player_action_flow(self, context)

    def summarize_discussion(self, context: AgentContext) -> DiscussionSummary:
        """通过独立窄 Schema 生成内部讨论摘要。"""

        if context.task_type is not TaskType.DISCUSSION_SUMMARY:
            raise DiscussionSummaryGenerationError("task_contract_mismatch")
        system_prompt, prompt = discussion_summary_prompts(context)
        tool = discussion_summary_tool()
        try:
            result = self.model_router.generate(
                agent_id=self.agent_id,
                task_type=TaskType.DISCUSSION_SUMMARY.value,
                prompt=prompt,
                system_prompt=system_prompt,
                tools=[tool],
                tool_choice={
                    "type": "tool",
                    "name": "submit_discussion_summary",
                },
            )
        except Exception as exc:
            raise DiscussionSummaryGenerationError(
                "model_generation_failed"
            ) from exc

        raw_text = str(getattr(result, "text", "") or "").strip()
        if not raw_text:
            raise DiscussionSummaryGenerationError("empty_response")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DiscussionSummaryGenerationError("invalid_json") from exc
        try:
            return DiscussionSummary.model_validate(payload)
        except ValidationError as exc:
            raise DiscussionSummaryGenerationError(
                "schema_validation_failed"
            ) from exc

    _attach_persona_snapshot = attach_persona_snapshot
    _record_persona_exposure = staticmethod(record_persona_exposure)

    def _latest_generation_failure_reason(self) -> str | None:
        return _latest_generation_failure_reason(self.model_router)

    def _check_repeat_error_signature(
        self,
        retry: RetryInfo,
        raw_text: str,
        attempt: int,
        last_signature: tuple[str, str, str] | None,
        *,
        structured_output_mode: str = "",
    ) -> tuple[bool, tuple[str, str, str] | None]:
        return _check_repeat_error_signature(
            retry,
            raw_text,
            attempt,
            last_signature,
            structured_output_mode=structured_output_mode,
        )

    # ── Delegated to output_parser.py ──

    def _parse_correction_hint(
        self,
        context: AgentContext,
        parse_error: str,
    ) -> str | None:
        # 2026-07-21 R1: Pydantic schema-validation 失败时, 把字段级违规细节 (loc /
        # msg / input) 直接灌入 hint, 让 LLM 第二次输出能定位到具体字段。
        # 退路: 解析失败或非 Pydantic 类错误时, 返回 None 让 caller 走空泛兜底语。
        if parse_error.startswith("Schema validation error:"):
            schema_hint = build_schema_validation_hint(parse_error)
            if schema_hint:
                return schema_hint
        if not self._is_sheriff_vote_audit_field_error(context, parse_error):
            return None
        forbidden = ", ".join(_SHERIFF_VOTE_FORBIDDEN_AUDIT_FIELDS)
        allowed = "action_type, target_id, speech, reason, confidence"
        return (
            "sheriff_vote is sheriff election voting, not exile voting. "
            f"Remove exile-vote audit fields: {forbidden}. "
            f"Only keep these fields: {allowed}."
        )

    @staticmethod
    def _is_sheriff_vote_audit_field_error(
        context: AgentContext,
        parse_error: str,
    ) -> bool:
        if ActionType.SHERIFF_VOTE not in context.legal_actions:
            return False
        if ActionType.VOTE in context.legal_actions:
            return False
        return any(
            field in parse_error
            for field in _SHERIFF_VOTE_FORBIDDEN_AUDIT_FIELDS
        )

    def _parse_action(
        self,
        text: str,
        *,
        context: AgentContext | None = None,
    ) -> tuple[PlayerAction | None, str | None]:
        return _parse_action_impl(
            text,
            task_type=context.task_type if context is not None else None,
        )

    def _parse_planning_action(
        self,
        text: str,
        context: AgentContext,
    ) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
        data, parse_error = _extract_decision_impl(text)
        if data is None:
            return None, None, None
        has_decision = "decision_plan" in data
        has_dialogue = "dialogue_plan" in data
        if not has_decision and not has_dialogue:
            return None, None, None
        if not has_decision or not has_dialogue:
            return None, "Planning envelope requires decision_plan and dialogue_plan", None
        try:
            action, audit = planning_envelope_to_action(data, context)
        except Exception as exc:
            return None, f"Planning envelope validation error: {exc}", None
        return action, None, audit

    def _vote_choice_map(self, context: AgentContext) -> dict[str, str]:
        return _vote_choice_map_impl(context.legal_targets)

    def _vote_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        return _vote_summary_impl(context.salience_items, target_id)

    def _target_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        return _target_summary_impl(context.legal_actions, context.salience_items, target_id)

    # ── Delegated to tool_schema.py ──

    def _player_action_tool(self, context: AgentContext) -> dict[str, Any]:
        output_mode = _select_output_mode(
            legal_actions=context.legal_actions,
            legal_targets=context.legal_targets,
            task_type=context.task_type,
            speech_intent_tasks=self._SPEECH_INTENT_TASKS,
        )
        return _tool_impl(
            context.legal_actions,
            context.legal_targets,
            context.task_type,
            output_mode,
        )

    def _speech_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        return _speech_quality_impl(
            context.task_type,
            action,
            context.recent_transcript,
            context.public_summary,
            context.strategy_directive,
            context.day_number,
        )

    def _vote_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        return _vote_quality_impl(
            context.task_type,
            action,
            context.strategy_directive,
            context.salience_items,
            context.recent_transcript,
        )

    # ── Remaining PlayerAgent methods ──

    def _fallback_action(self, context: AgentContext) -> FallbackAction:
        return _build_fallback_action(
            context,
            fallback_reason=_fallback_reason,
            fallback_speech=self._fallback_speech,
        )

    def _fallback_vote_target_from_context(
        self,
        context: AgentContext,
        candidates: list[str],
    ) -> str | None:
        return _fallback_vote_target_from_context(context, candidates)

    _context_clues = staticmethod(context_clues)
    _fallback_speech = staticmethod(build_fallback_speech)

    # ── Prompt building delegated to PlayerPromptBuilder (s10 pipeline) ──

    def _build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt via s10 pipeline: core + rules + role_guide + skills + output_contract."""
        return PlayerPromptBuilder(context, self.player_name).build_system_prompt()

    def _build_prompt(self, context: AgentContext, retry: RetryInfo) -> str:
        """Build user prompt via s10 pipeline: dynamic per-turn context and task instructions."""
        return PlayerPromptBuilder(context, self.player_name).build_user_prompt(retry)
