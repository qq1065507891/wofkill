# -*- coding: utf-8 -*-
"""
定义讨论摘要 V2 契约、解析模型输出，并兼容读取旧版字符串 checkpoint。

作者: Project contributors
创建日期: 2026-07-25
修改日期: 2026-07-27

使用示例:
    >>> state = {"discussion_positions": {"p01": "我怀疑p03"}}
    >>> discussion_summary_for_player(state, "p01").summary
    '我怀疑p03'
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from werewolf_agent.agents.json_repair import (
    extract_balanced_json_objects,
    repair_json_text,
)


_SAFE_FAILURE_CODES = frozenset({
    "task_contract_mismatch",
    "model_generation_failed",
    "empty_response",
    "invalid_json",
    "schema_validation_failed",
    "agent_unavailable",
    "model_failure",
    "missing_tool_call",
})

_SAFE_AUDIT_FIELDS = frozenset({
    "structured_output_mode",
    "tool_call_required",
    "tool_call_received",
    "response_shape",
    "json_candidate_count",
    "failure_stage",
})
_SAFE_STRUCTURED_OUTPUT_MODES = frozenset({
    "",
    "native_tool",
    "json_schema",
    "json_object",
    "text_json",
})
_SAFE_RESPONSE_SHAPES = frozenset({
    "empty",
    "text",
    "json_object",
    "json_array",
    "multiple_json_objects",
    "unknown",
})
_SAFE_FAILURE_STAGES = frozenset({"protocol", "schema", "provider"})


class DiscussionSummary(BaseModel):
    """单个玩家对当天公开讨论的内部结构化整理。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    summary: str
    suspected_players: list[str] = Field(default_factory=list)
    trusted_players: list[str] = Field(default_factory=list)
    vote_target: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class DiscussionSummaryGenerationError(RuntimeError):
    """携带可审计安全码的讨论摘要生成失败。"""

    def __init__(
        self,
        failure_code: str,
        *,
        audit: dict[str, Any] | None = None,
    ) -> None:
        safe_code = (
            failure_code
            if failure_code in _SAFE_FAILURE_CODES
            else "model_failure"
        )
        super().__init__(safe_code)
        self.failure_code = safe_code
        self.audit = {
            "failure_code": safe_code,
            **_sanitize_summary_audit(audit or {}),
        }


def _sanitize_summary_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """只保留固定类型与固定枚举值，避免任意字符串进入审计。"""

    sanitized: dict[str, Any] = {}
    for key in _SAFE_AUDIT_FIELDS:
        value = audit.get(key)
        if key == "structured_output_mode":
            if isinstance(value, str) and value in _SAFE_STRUCTURED_OUTPUT_MODES:
                sanitized[key] = value
        elif key in {"tool_call_required", "tool_call_received"}:
            if type(value) is bool:
                sanitized[key] = value
        elif key == "response_shape":
            if isinstance(value, str) and value in _SAFE_RESPONSE_SHAPES:
                sanitized[key] = value
        elif key == "json_candidate_count":
            if type(value) is int and 0 <= value <= 100:
                sanitized[key] = value
        elif (
            key == "failure_stage"
            and isinstance(value, str)
            and value in _SAFE_FAILURE_STAGES
        ):
            sanitized[key] = value
    return sanitized


def discussion_summary_response_shape(raw_text: str) -> tuple[str, int]:
    """不保留原文地返回响应形状与平衡 JSON 对象数量。"""

    if not raw_text.strip():
        return "empty", 0
    repaired = repair_json_text(raw_text)
    candidates = extract_balanced_json_objects(repaired)
    candidate_count = len(candidates)
    if _contains_object_in_array(repaired) or repaired.lstrip().startswith("["):
        return "json_array", candidate_count
    if candidate_count > 1:
        return "multiple_json_objects", candidate_count
    if candidate_count == 1:
        return "json_object", 1
    return "text", 0


def parse_discussion_summary_text(raw_text: str) -> DiscussionSummary:
    """修复并严格解析包含 DiscussionSummary 的模型文本。"""

    repaired = repair_json_text(raw_text)
    if _contains_object_in_array(repaired):
        return DiscussionSummary.model_validate([])
    if repaired.lstrip().startswith("["):
        payload = json.loads(repaired)
        return DiscussionSummary.model_validate(payload)
    candidates = extract_balanced_json_objects(repaired)
    if not candidates:
        payload = json.loads(repaired)
        return DiscussionSummary.model_validate(payload)

    validated: list[DiscussionSummary] = []
    validation_error: ValidationError | None = None
    for candidate in candidates:
        payload = json.loads(candidate)
        try:
            validated.append(DiscussionSummary.model_validate(payload))
        except ValidationError as exc:
            validation_error = validation_error or exc

    if len(validated) == 1:
        return validated[0]
    if len(validated) > 1:
        raise ValueError(
            "ambiguous_discussion_summary: multiple objects validated"
        )
    if validation_error is not None:
        raise validation_error
    raise ValueError("no_discussion_summary_object")


def _contains_object_in_array(text: str) -> bool:
    """识别字符串外数组容器中的对象，避免提取其内部摘要。"""

    array_depth = 0
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            array_depth += 1
        elif char == "]" and array_depth:
            array_depth -= 1
        elif char == "{" and array_depth:
            return True
    return False


def discussion_summary_tool() -> dict[str, Any]:
    """返回仅包含 DiscussionSummary 字段的强制工具契约。"""

    return {
        "name": "submit_discussion_summary",
        "description": "提交对当天公开讨论的内部结构化摘要。",
        "input_schema": DiscussionSummary.model_json_schema(),
    }


def discussion_summary_for_player(
    state: MutableMapping[str, Any],
    player_id: str,
) -> DiscussionSummary | None:
    """按版本优先级读取玩家摘要，并在无冲突时原子升级状态。"""

    positions = state.get("discussion_positions")
    if not isinstance(positions, Mapping) or player_id not in positions:
        return None

    if "discussion_positions_version" in state:
        version = state.get("discussion_positions_version")
        if type(version) is not int or version != 2:
            return None
        validated = _validate_v2_positions(positions)
        if validated is None:
            return None
        return validated[player_id]

    upgraded = _upgrade_unversioned_positions(positions)
    if upgraded is not None:
        state["discussion_positions"] = upgraded
        state["discussion_positions_version"] = 2
        return DiscussionSummary.model_validate(upgraded[player_id])

    value = positions[player_id]
    if isinstance(value, str):
        return DiscussionSummary(summary=value)
    return _validate_summary(value)


def discussion_summary_text(summary: DiscussionSummary | None) -> str:
    """以固定字段顺序生成旧 prompt 可消费的纯文本投影。"""

    if summary is None:
        return ""
    lines = [summary.summary]
    if summary.suspected_players:
        lines.append(f"怀疑玩家: {', '.join(summary.suspected_players)}")
    if summary.trusted_players:
        lines.append(f"信任玩家: {', '.join(summary.trusted_players)}")
    if summary.vote_target is not None:
        lines.append(f"投票目标: {summary.vote_target}")
    if summary.evidence_refs:
        lines.append(f"证据引用: {', '.join(summary.evidence_refs)}")
    return "\n".join(lines)


def _upgrade_unversioned_positions(
    positions: Mapping[str, Any],
) -> dict[str, dict[str, Any]] | None:
    """仅当整个映射可无歧义解释时，构造规范 V2 快照。"""

    if not positions:
        return {}
    if all(isinstance(value, str) for value in positions.values()):
        return {
            player_id: DiscussionSummary(summary=value).model_dump()
            for player_id, value in positions.items()
        }
    validated = _validate_v2_positions(positions)
    if validated is None:
        return None

    return {
        player_id: summary.model_dump()
        for player_id, summary in validated.items()
    }


def _validate_v2_positions(
    positions: Mapping[str, Any],
) -> dict[str, DiscussionSummary] | None:
    """严格校验整个显式 V2 映射，任一冲突即全局 fail closed。"""

    if not all(isinstance(value, Mapping) for value in positions.values()):
        return None
    validated: dict[str, DiscussionSummary] = {}
    for player_id, value in positions.items():
        summary = _validate_summary(value)
        if summary is None:
            return None
        validated[player_id] = summary
    return validated


def _validate_summary(value: Any) -> DiscussionSummary | None:
    try:
        return DiscussionSummary.model_validate(value)
    except ValidationError:
        return None


__all__ = [
    "DiscussionSummary",
    "DiscussionSummaryGenerationError",
    "discussion_summary_for_player",
    "discussion_summary_response_shape",
    "discussion_summary_text",
    "discussion_summary_tool",
    "parse_discussion_summary_text",
]
