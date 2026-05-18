"""Player Agent: schema-constrained output with illegal-output retry/fallback.

Player agents propose actions, reasons, and speech. They MUST NOT:
- Mutate GameState directly
- See moderator_full or other players' private state
- Bypass RuleEngine legal action sets
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from pydantic import ValidationError

from werewolf_agent.agents.schemas import (
    ActionTrace,
    ActionType,
    AgentContext,
    FallbackAction,
    PlayerAction,
    PrivateIntent,
    RetryInfo,
    TaskType,
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

    def __init__(
        self,
        agent_id: str,
        model_router: ModelRouter,
        validator: ActionValidator | None = None,
        max_retries: int = 3,
    ) -> None:
        self.agent_id = agent_id
        self.model_router = model_router
        self.validator = validator or DefaultActionValidator()
        self.max_retries = max_retries

    def act(self, context: AgentContext) -> tuple[PlayerAction | FallbackAction, RetryInfo]:
        """Generate a constrained player action with retry/fallback."""
        retry = RetryInfo(max_retries=self.max_retries)
        raw_text = ""
        parsed_action: PlayerAction | None = None

        for attempt in range(1, self.max_retries + 1):
            retry = RetryInfo(
                attempt=attempt,
                max_retries=self.max_retries,
                error_code=retry.error_code,
                error_message=retry.error_message,
            )

            # Generate LLM output
            prompt = self._build_prompt(context, retry)
            result = self.model_router.generate(
                agent_id=self.agent_id,
                task_type=context.task_type.value,
                prompt=prompt,
                system_prompt=self._build_system_prompt(context),
            )
            raw_text = result.text or ""

            if not result.text:
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="empty_response",
                    error_message="Model returned empty text",
                    correction_hint="Please provide a valid JSON action.",
                )
                continue

            # Parse JSON
            action, parse_error = self._parse_action(result.text)
            parsed_action = action
            if parse_error:
                retry = RetryInfo(
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error_code="parse_error",
                    error_message=parse_error,
                    correction_hint="Output must be valid JSON matching the PlayerAction schema.",
                )
                continue

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

            # Private intent is stored but never written to public timeline
            trace = self._build_action_trace(
                context,
                raw_text=raw_text,
                parsed_action=action,
                final_action_type=action.action_type,
                retry=retry,
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
        )
        fallback = fallback.model_copy(update={"trace": trace})
        return fallback, retry

    def _build_action_trace(
        self,
        context: AgentContext,
        *,
        raw_text: str,
        parsed_action: PlayerAction | None,
        final_action_type: ActionType,
        retry: RetryInfo,
        fallback_reason: str | None = None,
    ) -> ActionTrace:
        return ActionTrace(
            raw_text=raw_text,
            parsed_action=parsed_action.model_dump(exclude={"trace"}) if parsed_action else None,
            final_action_type=final_action_type.value,
            legal_actions=[action.value for action in context.legal_actions],
            legal_targets=list(context.legal_targets),
            retry=retry.model_dump(),
            fallback_reason=fallback_reason,
        )

    def _parse_action(self, text: str) -> tuple[PlayerAction | None, str | None]:
        """Parse LLM output into PlayerAction. Returns (action, error)."""
        import re

        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Try direct parse first
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Extract first JSON object from surrounding text
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError as e:
                    return None, f"JSON parse error: {e}"
            else:
                return None, f"No JSON object found in output"

        try:
            return PlayerAction(**data), None
        except ValidationError as e:
            return None, f"Schema validation error: {e}"

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
            safe_target = context.legal_targets[0]

        return FallbackAction(
            action_type=safe_action,
            target_id=safe_target,
            reason="fallback: retries exhausted",
        )

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
            f"你的玩家ID: {context.agent_id}",
        ]
        if context.own_role:
            parts.append(f"你的角色: {role_cn}（{context.own_role}）")
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
            "请严格输出以下格式的JSON（不要输出其他内容）: "
            '{"action_type": string, "target_id": string|null, '
            '"speech": "中文发言内容", "reason": "中文原因说明", '
            '"confidence": float, '
            '"private_intent": {"true_role": string, "faction_goal": string, '
            '"claimed_view": string, "pressure_target": string|null, '
            '"risk_flags": [string]}}'
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
                f'"faction_goal": "eliminate_villager", "claimed_view": "我是好人", '
                f'"pressure_target": "{example_target}", "risk_flags": []}}}}'
            )
            parts.append("示例输出（狼人空刀场景）：")
            parts.append(
                '{"action_type": "wolf_no_kill", "target_id": null, '
                '"speech": "", '
                '"reason": "本轮空刀策略", "confidence": 0.6, '
                '"private_intent": {"true_role": "werewolf", '
                '"faction_goal": "frame_villager", "claimed_view": "我是好人", '
                '"pressure_target": null, "risk_flags": []}}'
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
                         '"speech": "我投p05，他逻辑不通。", '
                         '"reason": "发言可疑", "confidence": 0.8, '
                         '"private_intent": {"true_role": "seer", '
                         '"faction_goal": "find_wolves", "claimed_view": "我是预言家", '
                         '"pressure_target": "p05", "risk_flags": []}}')
        return "\n".join(parts)

    def _build_prompt(self, context: AgentContext, retry: RetryInfo) -> str:
        """Build the user prompt with context and retry hints (Chinese)."""
        parts: list[str] = []

        if context.public_summary:
            parts.append(f"游戏概况:\n{context.public_summary}")

        if context.visible_world_state:
            parts.append(f"可见状态: {json.dumps(context.visible_world_state, ensure_ascii=False)}")

        if context.salience_items:
            parts.append(f"关键事件: {json.dumps(context.salience_items[:5], ensure_ascii=False)}")

        if context.strategy_directive:
            parts.append(f"策略建议: {json.dumps(context.strategy_directive, ensure_ascii=False)}")

        if context.persona_snapshot:
            parts.append(f"人格设定: {json.dumps(context.persona_snapshot, ensure_ascii=False)}")

        if context.recent_transcript:
            parts.append(f"近期发言:\n" + "\n".join(
                f"  [{t.get('speaker', '?')}] {t.get('text', '')}"
                for t in context.recent_transcript[-6:]
            ))

        if retry.correction_hint:
            parts.append(f"\n纠正提示（第{retry.attempt}/{retry.max_retries}次尝试）: {retry.correction_hint}")
            parts.append(f"错误信息: {retry.error_message}")

        parts.append("\n请输出你的行动JSON:")
        return "\n".join(parts)
