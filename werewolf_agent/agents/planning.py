"""Private decision and public dialogue planning contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from werewolf_agent.agents.schemas import ActionType, AgentContext, PlayerAction
from werewolf_agent.persona_runtime.policy import PersonaPolicyPrior


_REFERENCE_PREFIXES = ("rag:", "reflection:", "profile:", "memory:")
_PRIVATE_PUBLIC_MARKERS = (
    "wolf teammate",
    "my teammate",
    "night kill",
    "private goal",
    "hidden role",
    "狼队友",
    "我的队友",
    "夜刀",
    "夜间击杀",
    "真实身份",
    "私密目标",
)
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


class DecisionPlan(BaseModel):
    """Private, moderator-audit decision plan."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    target_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    private_goal: str
    evidence_refs: list[str] = Field(default_factory=list)
    reference_refs: list[str] = Field(default_factory=list)
    selected_world_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_not_current_evidence(self) -> "DecisionPlan":
        bad_refs = [
            ref for ref in self.evidence_refs
            if ref.startswith(_REFERENCE_PREFIXES)
        ]
        if bad_refs:
            raise ValueError(
                "evidence_refs must cite current-game evidence; "
                "move historical/RAG refs to reference_refs: "
                + ", ".join(bad_refs)
            )
        return self


class DialoguePlan(BaseModel):
    """Public expression plan derived from a private decision."""

    model_config = ConfigDict(extra="forbid")

    public_intent: str
    public_target_id: str | None = None
    talking_points: list[str] = Field(default_factory=list)
    conceal: list[str] = Field(default_factory=list)
    tone: str = ""


def render_dialogue_plan(dialogue: DialoguePlan, *, max_points: int = 3) -> str:
    """Render only public fields from a dialogue plan."""
    points = [
        _clean_public_text(point)
        for point in dialogue.talking_points[:max_points]
        if str(point or "").strip()
    ]
    if points:
        return " ".join(points)
    if dialogue.public_target_id:
        return f"Focus {dialogue.public_target_id}."
    return _clean_public_text(dialogue.public_intent)


def planning_envelope_to_action(
    data: dict[str, Any],
    context: AgentContext,
) -> tuple[PlayerAction, dict[str, Any]]:
    """Convert an explicit decision/dialogue envelope into an action."""
    decision_data = data.get("decision_plan")
    dialogue_data = data.get("dialogue_plan")
    if not isinstance(decision_data, dict) or not isinstance(dialogue_data, dict):
        raise ValueError("planning envelope requires decision_plan and dialogue_plan objects")

    decision = DecisionPlan.model_validate(decision_data)
    dialogue = DialoguePlan.model_validate(dialogue_data)
    action = decision_and_dialogue_to_action(decision, dialogue, context)
    prior = PersonaPolicyPrior.from_snapshot(
        context.persona_snapshot,
        own_role=context.own_role,
        task_type=context.task_type.value,
    )
    audit = {
        "planning_mode": "decision_dialogue",
        "decision_plan": decision.model_dump(mode="json"),
        "dialogue_plan": dialogue.model_dump(mode="json"),
        "persona_policy_prior": {
            "vote_threshold": planning_threshold_for_action(
                decision.action_type,
                prior,
                base=0.7,
            ),
            "claim_risk_threshold": prior.claim_risk_threshold(0.7),
            "speech_directness": prior.speech_directness,
            "deception_allowed": prior.deception_allowed,
        },
    }
    return action, audit


def decision_and_dialogue_to_action(
    decision: DecisionPlan,
    dialogue: DialoguePlan,
    context: AgentContext,
) -> PlayerAction:
    """Validate a plan pair and convert it into a schema-constrained action."""
    _validate_against_context(decision, context)
    _validate_public_dialogue(dialogue)
    public_reason = render_dialogue_plan(dialogue)
    target_id = decision.target_id or dialogue.public_target_id

    if decision.action_type == ActionType.VOTE:
        return PlayerAction(
            action_type=ActionType.VOTE,
            target_id=target_id,
            speech="",
            reason=public_reason,
            confidence=decision.confidence,
            suspect_reason=public_reason or "current-game evidence supports this target",
            not_voting_reason="other legal targets have weaker current-game evidence",
            private_reason=_private_audit_reason(decision),
        )

    return PlayerAction(
        action_type=decision.action_type,
        target_id=target_id,
        speech=public_reason if decision.action_type == ActionType.SPEECH else "",
        reason=public_reason,
        confidence=decision.confidence,
    )


def planning_threshold_for_action(
    action_type: ActionType,
    prior: PersonaPolicyPrior | None,
    *,
    base: float = 0.7,
) -> float:
    """Return a persona-adjusted planning threshold for an action."""
    if prior is None:
        return base
    if action_type in {ActionType.VOTE, ActionType.SHERIFF_VOTE}:
        return prior.vote_threshold(base)
    return base


def _validate_against_context(decision: DecisionPlan, context: AgentContext) -> None:
    if context.legal_actions and decision.action_type not in context.legal_actions:
        raise ValueError(
            f"action_type={decision.action_type.value} not in legal_actions"
        )
    if decision.action_type in _TARGET_REQUIRED_ACTIONS and not decision.target_id:
        raise ValueError(f"action_type={decision.action_type.value} requires target_id")
    if (
        decision.target_id is not None
        and context.legal_targets
        and decision.target_id not in context.legal_targets
    ):
        raise ValueError(f"target_id={decision.target_id} not in legal_targets")


def _private_audit_reason(decision: DecisionPlan) -> str:
    parts: list[str] = [decision.private_goal]
    if decision.evidence_refs:
        parts.append("evidence=" + ",".join(decision.evidence_refs[:5]))
    if decision.reference_refs:
        parts.append("reference=" + ",".join(decision.reference_refs[:3]))
    if decision.selected_world_ids:
        parts.append("worlds=" + ",".join(decision.selected_world_ids[:3]))
    return "; ".join(part for part in parts if part)


def _validate_public_dialogue(dialogue: DialoguePlan) -> None:
    public_parts = [
        dialogue.public_intent,
        *(dialogue.talking_points or []),
    ]
    public_text = "\n".join(_clean_public_text(part) for part in public_parts).lower()
    if not public_text:
        return
    for secret in dialogue.conceal:
        secret_text = _clean_public_text(secret).lower()
        if len(secret_text) >= 4 and secret_text in public_text:
            raise ValueError("private dialogue content leaked into public text")
    for marker in _PRIVATE_PUBLIC_MARKERS:
        if marker in public_text:
            raise ValueError("private dialogue content leaked into public text")


def _clean_public_text(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())
