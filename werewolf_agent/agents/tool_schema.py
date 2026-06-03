"""Standalone tool schema generation functions extracted from PlayerAgent.

These functions generate JSON schema definitions for the
submit_player_action tool and related quality checks.
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import (
    ActionType,
    PlayerAction,
    SeerStance,
    TaskType,
    VoteBasis,
)

logger = logging.getLogger(__name__)

# Target-requiring action types (duplicated from DefaultActionValidator to
# avoid circular imports between player.py and this module).
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


def all_legal_actions_require_target(legal_actions: list[ActionType]) -> bool:
    return bool(legal_actions) and all(
        action in _TARGET_REQUIRED_ACTIONS
        for action in legal_actions
    )


def vote_audit_tool_properties() -> dict[str, Any]:
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


def build_action_tool_schema(
    task_type: TaskType,
    action_values: list[str],
    target_values: list[str | None],
) -> dict[str, Any]:
    """Build a task-specific input_schema body for the submit_player_action tool.

    Returns only the fields relevant to the current ``task_type``, so the LLM
    sees a narrower schema (the whole point of the PlayerAction union refactor
    in Task 5). Callers wrap the result in the standard
    ``{"name": ..., "description": ..., "input_schema": ...}`` envelope.

    ``action_values`` and ``target_values`` are passed in by the caller because
    they are derived from per-call ``legal_actions`` / ``legal_targets``; this
    helper stays purely about *which* fields to advertise for a given task.
    """
    target_schema: dict[str, Any] = {
        "type": ["string", "null"],
        "description": "Target player id when required; null otherwise.",
    }
    # Only constrain target_id to an enum when concrete (non-None) targets
    # are known. A bare [None] means "any target allowed" and should fall
    # through to a plain nullable string so the LLM can omit target_id.
    concrete_targets = [t for t in target_values if t is not None]
    if concrete_targets:
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
    required = ["action_type", "target_id", "speech", "reason", "confidence"]
    if task_type == TaskType.VOTE:
        properties.update(vote_audit_tool_properties())
        required.extend([
            "seer_stance",
            "vote_basis",
            "standing_with_seer",
            "suspect_reason",
            "not_voting_reason",
            "private_reason",
        ])
    elif task_type == TaskType.WOLF_DISCUSSION:
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
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def player_action_tool(
    legal_actions: list[ActionType],
    legal_targets: list[str],
    task_type: TaskType,
) -> dict[str, Any]:
    action_values = [action.value for action in legal_actions]
    if not action_values:
        action_values = [action.value for action in ActionType]
    target_values: list[str | None] = list(legal_targets)
    if not all_legal_actions_require_target(legal_actions) and None not in target_values:
        target_values.append(None)
    return {
        "name": "submit_player_action",
        "description": "Submit exactly one legal Werewolf player action.",
        "input_schema": build_action_tool_schema(task_type, action_values, target_values),
    }


def speech_quality_phase(task_type: TaskType) -> str | None:
    phase_by_task = {
        TaskType.SPEECH: "day_discussion",
        TaskType.SHERIFF_SPEECH: "sheriff_speech",
        TaskType.DEFENSE_SPEECH: "pk_speech",
        TaskType.PK_SPEECH: "pk_speech",
    }
    return phase_by_task.get(task_type)


def speech_quality_error(
    task_type: TaskType,
    action: PlayerAction,
    recent_transcript: list[dict[str, Any]],
    public_summary: str,
    strategy_directive: dict[str, Any] | None,
) -> str | None:
    quality_phase = speech_quality_phase(task_type)
    if quality_phase is None or action.action_type != ActionType.SPEECH:
        return None
    try:
        from werewolf_agent.runtime.speech_quality import validate_public_speech

        result = validate_public_speech(
            action.speech,
            phase=quality_phase,
            context={
                "recent_transcript": list(recent_transcript),
                "public_summary": public_summary,
                "must_address_alerts": (strategy_directive or {}).get("must_address_alerts", []),
            },
        )
    except Exception:
        logger.debug("Speech quality validation failed unexpectedly", exc_info=True)
        return None
    if result.get("valid"):
        return None
    return str(result.get("hint") or "发言质量不足，请补充立场、怀疑对象、投票倾向和依据。")


def vote_quality_error(
    task_type: TaskType,
    action: PlayerAction,
    strategy_directive: dict[str, Any] | None,
    salience_items: list[dict[str, Any]],
    recent_transcript: list[dict[str, Any]],
) -> str | None:
    if (
        task_type != TaskType.VOTE
        or action.action_type != ActionType.VOTE
        or not (strategy_directive or {}).get("require_vote_quality")
    ):
        return None
    try:
        from werewolf_agent.runtime.vote_quality import validate_structured_vote_action

        result = validate_structured_vote_action(
            action.model_dump(exclude={"trace"}),
            context={
                "strategy_directive": strategy_directive or {},
                "salience_items": list(salience_items),
                "recent_transcript": list(recent_transcript),
            },
        )
    except Exception:
        logger.debug("Vote quality validation failed unexpectedly", exc_info=True)
        return None
    if result.get("valid"):
        return None
    hint = result.get("hint")
    if hint:
        return str(hint)
    # Fallback: include valid enum values so the LLM retry has actionable info.
    try:
        from werewolf_agent.runtime.vote_quality import (
            VALID_VOTE_BASIS_VALUES,
            VALID_SEER_STANCE_VALUES,
        )
        basis_list = sorted(VALID_VOTE_BASIS_VALUES)
        stance_list = sorted(VALID_SEER_STANCE_VALUES)
    except Exception:
        basis_list = [b.value for b in VoteBasis]
        stance_list = [s.value for s in SeerStance]
    return (
        "投票必须包含预言家立场、投票基点和具体理由。"
        f"有效 vote_basis: {basis_list}。"
        f"有效 seer_stance: {stance_list}。"
    )
