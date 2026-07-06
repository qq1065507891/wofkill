# -*- coding: utf-8 -*-
"""
处理 LLM 原始输出到 PlayerAction 的解析入口，并兼容导出拆分后的 helper。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.output_parser import parse_action
    >>> parse_action(raw_text)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from werewolf_agent.agents.action_data_extraction import (
    extract_decision_data,
    extract_parameter_tag_action,
    extract_partial_decision_data,
)
from werewolf_agent.agents.action_normalization import (
    _REASON_PLACEHOLDERS,
    _TYPO_ALIASES,
    _VALID_CLAIMED_VIEW_VALUES,
    _normalize_typos,
    _safe_default_claimed_view,
    clean_enum_value,
    clean_reason,
    normalize_action_data,
    sanitize_optional_private_fields,
)
from werewolf_agent.agents.action_repair import (
    choice_for_target,
    default_not_voting_reason,
    repair_speech_intent_decision,
    repair_target_decision,
    repair_vote_decision,
    target_candidate_summary,
    target_consistent_reason,
    target_from_vote_decision,
    vote_candidate_summary,
    vote_choice_map,
)
from werewolf_agent.agents.choice_pipeline import (
    CHOICE_TARGET_ACTIONS,
    _target_choice_action,
    parse_choice_action,
    parse_speech_intent_action,
    uses_choice_pipeline,
    uses_speech_intent_pipeline,
)
from werewolf_agent.agents.json_repair import (
    extract_json_object_candidates,
    repair_json_text,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    FactionGoal,
    PlayerAction,
    RiskFlag,
    SeerStance,
    VoteBasis,
)
from werewolf_agent.agents.speech_intent_parser import (
    SPEECH_INTENTS,
    _context_clues,
    ensure_speech_quality_components,
    infer_seer_stance,
    infer_speech_intent,
    infer_standing_with_seer,
    infer_vote_basis,
    speech_intent_reason,
    speech_pressure_target,
    speech_target_from_decision,
    synthesize_intent_speech,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ActionType",
    "CHOICE_TARGET_ACTIONS",
    "FactionGoal",
    "PlayerAction",
    "RiskFlag",
    "SPEECH_INTENTS",
    "SeerStance",
    "VoteBasis",
    "_REASON_PLACEHOLDERS",
    "_TYPO_ALIASES",
    "_VALID_CLAIMED_VIEW_VALUES",
    "_context_clues",
    "_normalize_typos",
    "_safe_default_claimed_view",
    "_target_choice_action",
    "action_from_data",
    "choice_for_target",
    "clean_enum_value",
    "clean_reason",
    "default_not_voting_reason",
    "ensure_speech_quality_components",
    "extract_decision_data",
    "extract_json_object_candidates",
    "extract_parameter_tag_action",
    "extract_partial_decision_data",
    "infer_seer_stance",
    "infer_speech_intent",
    "infer_standing_with_seer",
    "infer_vote_basis",
    "logger",
    "normalize_action_data",
    "parse_action",
    "parse_choice_action",
    "parse_speech_intent_action",
    "repair_json_text",
    "repair_speech_intent_decision",
    "repair_target_decision",
    "repair_vote_decision",
    "sanitize_optional_private_fields",
    "speech_intent_reason",
    "speech_pressure_target",
    "speech_target_from_decision",
    "synthesize_intent_speech",
    "target_candidate_summary",
    "target_consistent_reason",
    "target_from_vote_decision",
    "uses_choice_pipeline",
    "uses_speech_intent_pipeline",
    "vote_candidate_summary",
    "vote_choice_map",
]


def action_from_data(data: Any) -> tuple[PlayerAction | None, str | None]:
    # PlayerAction is a discriminated Union of 16 action-type variants
    # (pipeline-optimization Task 5). ``model_validate`` is overridden on
    # the base class to route the data through the Union's TypeAdapter,
    # which dispatches on the ``action_type`` discriminator.
    if isinstance(data, dict):
        # P1-G3223805846-5: 解析入口归一常见 LLM 字段名 typo
        # (如 `not_vading_reason` → `not_voting_reason`)。
        data = _normalize_typos(data)
    data = normalize_action_data(data)
    # P1-S7 (residual): always sanitize private_intent before validation.
    # The previous code only sanitized on validation failure, which let
    # free-form Chinese claimed_view strings (e.g., "我是好人，混水摸鱼")
    # pass through cleanly. Sanitizing first normalizes claimed_view to
    # an enum-style identifier so the audit log / dashboard see only
    # clean values.
    data = sanitize_optional_private_fields(data)
    try:
        return PlayerAction.model_validate(data), None
    except ValidationError as e:
        return None, f"Schema validation error: {e}"


def parse_action(text: str) -> tuple[PlayerAction | None, str | None]:
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
        return action_from_data(data)
    except json.JSONDecodeError as direct_error:
        # Repair and retry
        repaired = repair_json_text(cleaned)
        if repaired != cleaned:
            try:
                data = json.loads(repaired)
                return action_from_data(data)
            except json.JSONDecodeError:
                pass  # fall through to extraction

        parameter_data = extract_parameter_tag_action(cleaned)
        if parameter_data is not None:
            action, parse_error = action_from_data(parameter_data)
            if action is not None:
                return action, None
            return None, parse_error

        candidates = extract_json_object_candidates(cleaned)
        if not candidates:
            return None, f"No JSON object found in output"
        first_error: str | None = None
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as e:
                # Try repair on candidate
                repaired = repair_json_text(candidate)
                if repaired != candidate:
                    try:
                        data = json.loads(repaired)
                    except json.JSONDecodeError:
                        if first_error is None:
                            first_error = f"JSON parse error: {e}"
                        continue
                else:
                    if first_error is None:
                        first_error = f"JSON parse error: {e}"
                    continue
            action, parse_error = action_from_data(data)
            if action is not None:
                return action, None
            if first_error is None:
                first_error = parse_error
        return None, first_error or f"JSON parse error: {direct_error}"
