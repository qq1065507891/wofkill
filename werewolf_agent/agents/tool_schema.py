"""Standalone tool schema generation functions extracted from PlayerAgent.

These functions generate JSON schema definitions for the
submit_player_action tool and related quality checks.
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.action_contract import (
    ActionContract,
    all_legal_actions_require_target as _contract_requires_target,
    vote_audit_properties,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    OutputMode,
    PlayerAction,
    TaskType,
)

logger = logging.getLogger(__name__)

def all_legal_actions_require_target(legal_actions: list[ActionType]) -> bool:
    return _contract_requires_target(legal_actions)


def vote_audit_tool_properties() -> dict[str, Any]:
    return vote_audit_properties()


def player_action_tool(
    legal_actions: list[ActionType],
    legal_targets: list[str],
    task_type: TaskType,
    output_mode: OutputMode = OutputMode.FULL_ACTION,
) -> dict[str, Any]:
    return ActionContract.build(
        output_mode=output_mode,
        task_type=task_type,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
    ).tool


def wolf_team_plan_tool(
    alive_wolves: list[str],
    alive_non_wolves: list[str],
) -> dict[str, Any]:
    """OpenAI tool schema for the wolf team captain's nightly plan output.

    Mirrors `WolfTeamPlan` (agents/schemas.py): 4 role-slot fields are
    constrained to alive werewolves (or null); kill targets to alive
    non-wolves (or null). Schema-level structural rules (no duplicate
    roles, kills not overlapping roles) are re-checked by Pydantic on
    parse — the enum constraints here help the LLM choose well-formed
    values up-front, reducing retry cost.
    """
    wolf_enum: list[str | None] = [*alive_wolves, None]
    target_enum: list[str | None] = [*alive_non_wolves, None]
    return {
        "name": "submit_wolf_team_plan",
        "description": (
            "由狼队队长一次性提交本夜完整战术计划：4 角色分工(悍跳/冲票/倒钩/深水)、"
            "击杀目标(主+备)、白天对外口径、决策依据。所有 player_id 字段必须从合法 enum 中选择，"
            "未分配位置填 null。不要把同一位狼塞进两个角色;不要把击杀目标设为狼队成员。"
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "night_number": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "本夜编号 (gs.night_number)",
                },
                "night_kill_primary": {
                    "type": ["string", "null"],
                    "enum": target_enum,
                    "description": "首选击杀目标 player_id; null 表示主动空刀",
                },
                "night_kill_backup": {
                    "type": ["string", "null"],
                    "enum": target_enum,
                    "description": "备选击杀目标; null 表示无备选",
                },
                "fake_seer": {
                    "type": ["string", "null"],
                    "enum": wolf_enum,
                    "description": "悍跳预言家位 (alive werewolf 或 null)",
                },
                "pusher": {
                    "type": ["string", "null"],
                    "enum": wolf_enum,
                    "description": "冲票位",
                },
                "hooker": {
                    "type": ["string", "null"],
                    "enum": wolf_enum,
                    "description": "倒钩位",
                },
                "deep_cover": {
                    "type": ["string", "null"],
                    "enum": wolf_enum,
                    "description": "深水位",
                },
                "public_story": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 120,
                    "description": "白天对外统一口径 / 抗推叙事 (1~120 字)",
                },
                "evidence_quality": {
                    "type": "string",
                    "enum": ["strong", "weak", "none"],
                    "description": "对夜聊共识度的评估",
                },
                "reasoning": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "队长决策依据 (1~200 字, werewolf_team_only)",
                },
            },
            "required": [
                "night_number",
                "night_kill_primary",
                "night_kill_backup",
                "fake_seer",
                "pusher",
                "hooker",
                "deep_cover",
                "public_story",
                "evidence_quality",
                "reasoning",
            ],
        },
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
                "intent": getattr(action, "intent", ""),
                "target_id": action.target_id,
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
