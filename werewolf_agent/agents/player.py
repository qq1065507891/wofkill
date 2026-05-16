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
            return action, retry

        # Fallback
        fallback = self._fallback_action(context)
        return fallback, retry

    def _parse_action(self, text: str) -> tuple[PlayerAction | None, str | None]:
        """Parse LLM output into PlayerAction. Returns (action, error)."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"

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

    def _build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt with role identity and constraints."""
        parts = [
            "You are a player in a Werewolf game.",
            f"Your player ID: {context.agent_id}",
        ]
        if context.own_role:
            parts.append(f"Your role: {context.own_role}")
        parts.append(f"Current phase: {context.phase}")
        if context.legal_actions:
            parts.append(
                f"Legal actions: {[a.value for a in context.legal_actions]}"
            )
        if context.legal_targets:
            parts.append(f"Legal targets: {context.legal_targets}")
        parts.append(
            "Output ONLY valid JSON matching this schema: "
            '{"action_type": string, "target_id": string|null, '
            '"speech": string, "reason": string, "confidence": float, '
            '"private_intent": {"true_role": string, "faction_goal": string, '
            '"claimed_view": string, "pressure_target": string|null, '
            '"risk_flags": [string]}}'
        )
        return "\n".join(parts)

    def _build_prompt(self, context: AgentContext, retry: RetryInfo) -> str:
        """Build the user prompt with context and retry hints."""
        parts: list[str] = []

        if context.public_summary:
            parts.append(f"Game summary:\n{context.public_summary}")

        if context.visible_world_state:
            parts.append(f"Visible state: {json.dumps(context.visible_world_state, ensure_ascii=False)}")

        if context.salience_items:
            parts.append(f"Key events: {json.dumps(context.salience_items[:5], ensure_ascii=False)}")

        if context.strategy_directive:
            parts.append(f"Strategy: {json.dumps(context.strategy_directive, ensure_ascii=False)}")

        if context.persona_snapshot:
            parts.append(f"Persona: {json.dumps(context.persona_snapshot, ensure_ascii=False)}")

        if context.recent_transcript:
            parts.append(f"Recent speech:\n" + "\n".join(
                f"  [{t.get('speaker', '?')}] {t.get('text', '')}"
                for t in context.recent_transcript[-6:]
            ))

        if retry.correction_hint:
            parts.append(f"\nCORRECTION (attempt {retry.attempt}/{retry.max_retries}): {retry.correction_hint}")
            parts.append(f"Error was: {retry.error_message}")

        parts.append("\nProvide your action as JSON:")
        return "\n".join(parts)
