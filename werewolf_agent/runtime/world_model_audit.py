# -*- coding: utf-8 -*-
"""World-model audit extraction and sanitization helpers.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any


FORBIDDEN_WORLD_MODEL_KEYS = {
    "roles",
    "supporting_evidence",
    "private_goal",
    "conceal",
    "raw_prompt",
    "raw_response",
}

ALLOWED_WORLD_MODEL_KEYS = {
    "player_id",
    "phase",
    "day_number",
    "night_number",
    "planning_mode",
    "belief",
    "possible_worlds",
    "simulation_predictions",
    "decision_plan",
    "dialogue_plan",
    "persona_policy_prior",
    "metrics",
}


def build_world_model_audit_from_context(
    context: Any,
    *,
    parsed_action: Any = None,
) -> dict[str, Any]:
    """Build a prompt-safe audit payload from AgentContext and parsed planning data."""
    payload: dict[str, Any] = {
        "player_id": str(getattr(context, "agent_id", "") or ""),
        "belief": getattr(context, "belief_state", {}) or {},
        "possible_worlds": getattr(context, "possible_worlds", {}) or {},
        "simulation_predictions": getattr(context, "simulation_predictions", {}) or {},
    }
    parsed = _as_dict(parsed_action)
    if parsed:
        for key in (
            "planning_mode",
            "decision_plan",
            "dialogue_plan",
            "persona_policy_prior",
        ):
            if key in parsed:
                payload[key] = parsed[key]
    return sanitize_world_model_audit(payload)


def extract_world_model_audits_from_events(events: list[Any]) -> list[dict[str, Any]]:
    """Extract sanitized world-model audits from explicit or action-trace events."""
    audits: list[dict[str, Any]] = []
    for event in events:
        event_type = _event_type(event)
        payload = _event_payload(event)
        if not isinstance(payload, dict):
            continue
        if event_type == "world_model_audit":
            audits.append(sanitize_world_model_audit(payload))
            continue
        if event_type != "action_trace_audit":
            continue
        trace = payload.get("action_trace")
        if not isinstance(trace, dict):
            continue
        audit = trace.get("world_model_audit")
        if not isinstance(audit, dict) or not audit:
            continue
        merged = {
            "player_id": payload.get("player_id") or audit.get("player_id"),
            "phase": payload.get("phase") or audit.get("phase"),
            "day_number": payload.get("day_number") or audit.get("day_number"),
            "night_number": payload.get("night_number") or audit.get("night_number"),
            **audit,
        }
        audits.append(sanitize_world_model_audit(merged))
    return audits


def sanitize_world_model_audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only moderator-audit fields that are safe to expose in audit APIs."""
    return {
        key: strip_world_model_private_fields(value)
        for key, value in payload.items()
        if key in ALLOWED_WORLD_MODEL_KEYS
    }


def strip_world_model_private_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_world_model_private_fields(child)
            for key, child in value.items()
            if key not in FORBIDDEN_WORLD_MODEL_KEYS
        }
    if isinstance(value, list):
        return [strip_world_model_private_fields(item) for item in value]
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude={"trace"})
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", "") or "")


def _event_payload(event: Any) -> Any:
    if isinstance(event, dict):
        return event.get("payload") or {}
    return getattr(event, "payload", {}) or {}
