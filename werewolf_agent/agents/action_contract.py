# -*- coding: utf-8 -*-
"""
功能描述：**：为不同输出模式（FULL_ACTION / TARGET_CHOICE / SPEECH_INTENT）生成结构化输出契约。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-09
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from werewolf_agent.agents.schemas import (
    ActionType,
    OutputMode,
    SeerStance,
    TaskType,
    VoteBasis,
)


SPEECH_INTENTS: tuple[str, ...] = (
    "self_clear",
    "question_target",
    "stand_with_seer",
    "respond_pressure",
    "push_vote",
    "info_synthesis",
    "anti_herd_call",
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


def all_legal_actions_require_target(legal_actions: list[ActionType]) -> bool:
    return bool(legal_actions) and all(
        action in _TARGET_REQUIRED_ACTIONS
        for action in legal_actions
    )


def vote_audit_properties() -> dict[str, Any]:
    return {
        "seer_stance": {
            "type": "string",
            "enum": [stance.value for stance in SeerStance],
            "description": "Vote stance enum about seer logic.",
        },
        "vote_basis": {
            "type": "string",
            "enum": [basis.value for basis in VoteBasis],
            "description": "Primary vote basis enum.",
        },
        "standing_with_seer": {
            "type": "string",
            "description": "Moderator-only seer or logic line selected by the player.",
        },
        "suspect_reason": {
            "type": "string",
            "description": "Moderator-only reason the final target is suspicious.",
        },
        "not_voting_reason": {
            "type": "string",
            "description": "Moderator-only reason other major candidates were rejected.",
        },
        "candidate_comparison": {
            "type": "string",
            "description": "Moderator-only comparison between at least two vote candidates.",
        },
        "private_reason": {
            "type": "string",
            "description": "Moderator-only full vote reasoning; never public speech.",
        },
    }


def build_full_action_schema(
    task_type: TaskType,
    action_values: list[str],
    target_values: list[str | None],
    *,
    include_vote_audit: bool = False,
) -> dict[str, Any]:
    target_schema: dict[str, Any] = {
        "type": ["string", "null"],
        "description": "Target player id when required; null otherwise.",
    }
    concrete_targets = [target for target in target_values if target is not None]
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
            "description": "Public Chinese speech; empty for private actions.",
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
    if include_vote_audit:
        properties.update(vote_audit_properties())
        required.extend([
            "seer_stance",
            "vote_basis",
            "standing_with_seer",
            "suspect_reason",
            "not_voting_reason",
            "candidate_comparison",
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


@dataclass(frozen=True)
class ActionContract:
    """One output-mode-specific schema shared by prompt and provider."""

    output_mode: OutputMode
    task_type: TaskType
    json_schema: dict[str, Any]

    @classmethod
    def build(
        cls,
        *,
        output_mode: OutputMode,
        task_type: TaskType,
        legal_actions: list[ActionType],
        legal_targets: list[str],
    ) -> "ActionContract":
        if output_mode == OutputMode.TARGET_CHOICE:
            schema = _target_choice_schema(
                task_type,
                legal_targets,
                include_vote_audit=_is_exile_vote_contract(
                    task_type,
                    legal_actions,
                ),
            )
        elif output_mode == OutputMode.SPEECH_INTENT:
            schema = _speech_intent_schema(legal_targets)
        else:
            action_values = [action.value for action in legal_actions]
            if not action_values:
                action_values = [action.value for action in ActionType]
            target_values: list[str | None] = list(legal_targets)
            if (
                not all_legal_actions_require_target(legal_actions)
                and None not in target_values
            ):
                target_values.append(None)
            schema = build_full_action_schema(
                task_type,
                action_values,
                target_values,
                include_vote_audit=_is_exile_vote_contract(
                    task_type,
                    legal_actions,
                ),
            )
        return cls(
            output_mode=output_mode,
            task_type=task_type,
            json_schema=schema,
        )

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(self.json_schema.get("required", ()))

    @property
    def tool(self) -> dict[str, Any]:
        return {
            "name": "submit_player_action",
            "description": "Submit exactly one legal Werewolf player action.",
            "input_schema": self.json_schema,
        }


def _target_choice_schema(
    task_type: TaskType,
    legal_targets: list[str],
    *,
    include_vote_audit: bool = False,
) -> dict[str, Any]:
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(legal_targets)])
    properties: dict[str, Any] = {
        "choice": {
            "type": "string",
            "enum": letters,
            "description": "Candidate letter from the current prompt.",
        },
        "reason": {"type": "string"},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    }
    required = ["choice", "reason"]
    if include_vote_audit:
        properties.update(vote_audit_properties())
        required.extend([
            "seer_stance",
            "vote_basis",
            "standing_with_seer",
            "suspect_reason",
            "not_voting_reason",
            "candidate_comparison",
            "private_reason",
        ])
    required.append("confidence")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _is_exile_vote_contract(
    task_type: TaskType,
    legal_actions: list[ActionType],
) -> bool:
    return task_type == TaskType.VOTE and ActionType.VOTE in legal_actions


def _speech_intent_schema(legal_targets: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(SPEECH_INTENTS),
            },
            "target_id": {
                "type": ["string", "null"],
                "enum": [*legal_targets, None],
            },
            "speech": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": [
            "intent",
            "target_id",
            "speech",
            "reason",
            "confidence",
        ],
    }
