# -*- coding: utf-8 -*-
"""Runtime audit events for module exposure, call monitoring, and prompt injection.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-16
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping
from typing import Any

from werewolf_agent.core.models import GameEvent
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.decision_outcomes import (
    DecisionGeneratedBy,
    normalize_terminal_failure_code,
)

_RAG_KEYS = frozenset({
    "entry_id",
    "rank",
    "relevance_score",
    "prompt_visible",
    "title",
    "situation_signature",
    "retrieval_reason",
})
_REFLECTION_KEYS = frozenset({
    "entry_id",
    "rank",
    "quality_score",
    "prompt_visible",
    "lesson_key",
    "quality_status",
})
_PERSONA_KEYS = frozenset({
    "profile_id",
    "prompt_visible",
    "policy_keys",
    "sanitized",
})
_SKILL_TOOL_CALL_KEYS = frozenset({
    "call_kind",
    "call_name",
    "skill_name",
    "tool_name",
    "provider_name",
    "status",
    "success",
    "required",
    "received",
    "prompt_visible",
    "result_available_to_decision",
    "decision_usage",
    "fallback_triggered",
    "error_type",
    "error_message",
    "structured_failure_reason",
    "structured_failure_stage",
    "structured_output_mode",
    "parse_success",
    "retry_count",
    "attempt_count",
    "provider_fallback_count",
    "generated_by",
    "terminal_failure_code",
    "duration_ms",
})
_SKILL_TOOL_INPUT_KEYS = frozenset({
    "role",
    "phase",
    "task_type",
    "day",
    "night",
    "legal_target_count",
    "candidate_count",
    "has_wolf_team_plan",
})
_SKILL_TOOL_OUTPUT_KEYS = frozenset({
    "confidence",
    "has_prompt_injectable",
    "risk_alert_count",
    "evidence_ref_count",
    "summary_hash",
    "reasoning_hash",
    "tool_call_name",
})
_PROMPT_INJECTION_KEYS = frozenset({
    "module_name",
    "field_path",
    "injection_kind",
    "injected",
    "prompt_visible",
    "visibility_scope",
    "item_count",
    "char_count",
    "content_hash",
    "decision_usage",
    "sanitized",
})
_PERSONA_PROOF_KEYS = frozenset({
    "final_system_location",
    "final_system_message_index",
    "message_char_count",
    "run_scoped_fingerprint",
    "confirmed_injection",
    "attempt_kind",
    "attempt_ordinal",
    "provider",
    "model",
    "sanitized",
})
_PROMPT_INJECTION_FIELDS = (
    ("public_summary", "public_summary", "text_section", "public"),
    ("recent_transcript", "recent_transcript", "transcript_section", "public"),
    ("visible_world_state", "visible_world_state", "structured_state", "viewer_visible"),
    ("rag_hints", "rag_hints", "retrieval_section", "viewer_visible"),
    ("private_memory_hints", "private_memory_hints", "memory_section", "viewer_private"),
    ("reflection_memory_hints", "reflection_memory_hints", "memory_section", "viewer_private"),
    ("profile_memory_hint", "profile_memory_hint", "memory_section", "viewer_private"),
    ("cognition_matrix_hint", "cognition_matrix_hint", "memory_section", "viewer_private"),
    ("error_pattern_hint", "error_pattern_hint", "memory_section", "viewer_private"),
    ("belief_state", "belief_state", "cognition_section", "viewer_visible"),
    ("contradiction_alerts", "contradiction_alerts", "cognition_section", "viewer_visible"),
    ("seer_credibility", "seer_credibility", "cognition_section", "viewer_visible"),
    ("possible_worlds", "possible_worlds", "world_model_section", "viewer_visible"),
    ("simulation_predictions", "simulation_predictions", "world_model_section", "viewer_visible"),
    ("strategy_directive", "strategy_directive", "strategy_section", "viewer_visible"),
    ("skill_analyses", "skill_analyses", "skill_section", "viewer_visible"),
    ("persona", "persona_snapshot", "persona_section", "viewer_visible"),
)
_PRIVATE_WOLF_STANCE_KEYS = frozenset({
    "wolf_id",
    "target_stance",
    "source_event_id",
})


def _identity_payload(identity: DecisionIdentity) -> dict[str, Any]:
    return {
        "trace_id": identity.trace_id(),
        "game_id": identity.game_id,
        "player_id": identity.player_id,
        "phase": identity.phase,
        "day_number": identity.day_number,
        "night_number": identity.night_number,
        "task_type": identity.task_type,
        "action_index": identity.action_index,
        "visibility": "moderator_only",
    }


def _sanitize_allowed(value: Any, allowed_keys: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_allowed(item, allowed_keys)
            for key, item in value.items()
            if (
                str(key) in allowed_keys
                and str(key) not in _PRIVATE_WOLF_STANCE_KEYS
            )
        }
    if isinstance(value, list):
        return [_sanitize_allowed(item, allowed_keys) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_allowed(item, allowed_keys) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summary_hash(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _truncate(value: Any, limit: int = 300) -> str:
    text = str(value or "")
    return text[:limit]


def _skill_rows(analyses: Any) -> list[dict[str, Any]]:
    if isinstance(analyses, Mapping):
        items = [
            {
                "skill_name": str(name),
                "rank": index + 1,
                "prompt_visible": True,
                "summary_hash": _summary_hash(summary),
                "advice_type": "tactical",
            }
            for index, (name, summary) in enumerate(analyses.items())
            if summary
        ]
        return items

    if not isinstance(analyses, list):
        return []

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(analyses):
        if not isinstance(item, Mapping):
            continue
        summary = (
            item.get("summary")
            or item.get("analysis")
            or item.get("advice")
            or item.get("prompt_visible")
            or ""
        )
        row: dict[str, Any] = {
            "skill_name": str(item.get("skill_name") or item.get("name") or item.get("skill") or ""),
            "rank": item.get("rank", index + 1),
            "prompt_visible": item.get("prompt_visible", True),
            "summary_hash": _summary_hash(summary),
        }
        advice_type = item.get("advice_type")
        if advice_type:
            row["advice_type"] = str(advice_type)
        rows.append(row)
    return rows


def _sanitize_skill_tool_rows(calls: Any) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in calls:
        if not isinstance(item, Mapping):
            continue
        row = {
            str(key): _sanitize_allowed(value, _SKILL_TOOL_CALL_KEYS)
            for key, value in item.items()
            if str(key) in _SKILL_TOOL_CALL_KEYS
        }
        if "error_message" in row:
            row["error_message"] = _truncate(row["error_message"])
        _sanitize_decision_trace_values(row)
        input_summary = item.get("input_summary")
        if isinstance(input_summary, Mapping):
            clean_input = _sanitize_allowed(input_summary, _SKILL_TOOL_INPUT_KEYS)
            if clean_input:
                row["input_summary"] = clean_input
        output_summary = item.get("output_summary")
        if isinstance(output_summary, Mapping):
            clean_output = _sanitize_allowed(output_summary, _SKILL_TOOL_OUTPUT_KEYS)
            if clean_output:
                row["output_summary"] = clean_output
        if row:
            rows.append(row)
    return rows


def _sanitize_decision_trace_values(row: dict[str, Any]) -> None:
    """对决策 trace 值执行封闭集合校验，拒绝把任意正文带入审计。"""
    flags: list[str] = []
    generated_by = row.get("generated_by")
    allowed_generated = {item.value for item in DecisionGeneratedBy}
    if "generated_by" in row and (
        not isinstance(generated_by, str) or generated_by not in allowed_generated
    ):
        row["generated_by"] = "unknown"
        flags.append("generated_by_invalid")

    terminal_code = row.get("terminal_failure_code")
    if "terminal_failure_code" in row and not isinstance(terminal_code, str):
        row["terminal_failure_code"] = "unknown"
        flags.append("terminal_failure_code_invalid")
    elif "terminal_failure_code" in row:
        normalized = normalize_terminal_failure_code(terminal_code)
        row["terminal_failure_code"] = normalized
        if normalized == "unknown" and terminal_code != "unknown":
            flags.append("terminal_failure_code_invalid")
    if flags:
        row["value_sanitization"] = flags


def _context_value(context: Any, field_path: str) -> Any:
    if isinstance(context, Mapping):
        return context.get(field_path)
    return getattr(context, field_path, None)


def _is_injected(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple, set)):
        return bool(value)
    return value is not None


def _item_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        return 1 if value.strip() else 0
    return 1 if value is not None else 0


def _prompt_injection_rows(context: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module_name, field_path, injection_kind, visibility_scope in _PROMPT_INJECTION_FIELDS:
        value = _context_value(context, field_path)
        injected = _is_injected(value)
        rows.append({
            "module_name": module_name,
            "field_path": field_path,
            "injection_kind": injection_kind,
            "injected": injected,
            "prompt_visible": True,
            "visibility_scope": visibility_scope,
            "item_count": _item_count(value),
            "char_count": len(str(value or "")),
            "content_hash": _summary_hash(value) if injected else "",
            "decision_usage": (
                "prompt_context_available" if injected else "not_available_empty"
            ),
            "sanitized": True,
        })
    return rows


class ModuleExposureAuditCollector:
    """Collect sanitized module exposure audit events for one agent action."""

    def __init__(self) -> None:
        self._events: list[GameEvent] = []
        self._persona_proof_secret = secrets.token_bytes(32)
        self._persona_proof_keys: set[tuple[str, str, int | None, str]] = set()

    def record_rag(self, identity: DecisionIdentity, hits: list[dict[str, Any]] | None) -> None:
        if not hits:
            return
        sanitized = _sanitize_allowed(hits, _RAG_KEYS)
        if not any(row for row in sanitized if row):
            return
        self._append(
            "rag_exposure_audit",
            identity,
            {"hits": sanitized},
        )

    def record_reflection(
        self,
        identity: DecisionIdentity,
        cards: list[dict[str, Any]] | None,
    ) -> None:
        if not cards:
            return
        sanitized = _sanitize_allowed(cards, _REFLECTION_KEYS)
        if not any(row for row in sanitized if row):
            return
        self._append(
            "reflection_exposure_audit",
            identity,
            {"cards": sanitized},
        )

    def record_skill(self, identity: DecisionIdentity, analyses: Any) -> None:
        rows = _skill_rows(analyses)
        if not rows:
            return
        self._append("skill_exposure_audit", identity, {"analyses": rows})

    def record_skill_tool_calls(
        self,
        identity: DecisionIdentity,
        calls: list[dict[str, Any]] | None,
    ) -> None:
        rows = _sanitize_skill_tool_rows(calls or [])
        if not rows:
            return
        self._append("skill_tool_call_audit", identity, {"calls": rows})

    def record_prompt_injections(
        self,
        identity: DecisionIdentity,
        context: Any,
    ) -> None:
        rows = [
            _sanitize_allowed(row, _PROMPT_INJECTION_KEYS)
            for row in _prompt_injection_rows(context)
        ]
        if not rows:
            return
        self._append("prompt_injection_audit", identity, {"injections": rows})

    def record_action_tool_call(
        self,
        identity: DecisionIdentity,
        action_trace: Mapping[str, Any] | None,
    ) -> None:
        if not action_trace:
            return
        required = bool(action_trace.get("tool_call_required"))
        received = bool(action_trace.get("tool_call_received"))
        call_name = str(action_trace.get("tool_call_name") or "")
        if not call_name and required:
            call_name = "submit_player_action"
        if not (required or received or call_name):
            return

        fallback_triggered = bool(
            action_trace.get("fallback_reason")
            or action_trace.get("fallback_target_used")
        )
        parse_success = bool(action_trace.get("parse_success"))
        if required and not received:
            status = "missing"
            success = False
        elif received and parse_success:
            status = "success"
            success = True
        elif received:
            status = "parse_failed"
            success = False
        else:
            status = "not_required"
            success = True

        if received and parse_success:
            decision_usage = "tool_response_parsed"
            result_available = True
        elif parse_success:
            decision_usage = "text_fallback_parsed"
            result_available = True
        elif fallback_triggered:
            decision_usage = "not_used_fallback"
            result_available = False
        elif received and not parse_success:
            decision_usage = "not_used_parse_failed"
            result_available = False
        else:
            decision_usage = "not_used"
            result_available = False

        self.record_skill_tool_calls(
            identity,
            [{
                "call_kind": "tool",
                "call_name": call_name,
                "tool_name": call_name,
                "status": status,
                "success": success,
                "required": required,
                "received": received,
                "fallback_triggered": fallback_triggered,
                "result_available_to_decision": result_available,
                "decision_usage": decision_usage,
                "parse_success": parse_success,
                "retry_count": int(action_trace.get("retry_count") or 0),
                "attempt_count": int(action_trace.get("attempt_count") or 0),
                "provider_fallback_count": int(
                    action_trace.get("provider_fallback_count") or 0
                ),
                "generated_by": action_trace.get("generated_by"),
                "terminal_failure_code": action_trace.get(
                    "terminal_failure_code"
                ),
                "structured_failure_reason": action_trace.get("structured_failure_reason"),
                "structured_failure_stage": action_trace.get("structured_failure_stage"),
                "structured_output_mode": action_trace.get("structured_output_mode"),
                "output_summary": {"tool_call_name": call_name},
            }],
        )

    def record_persona(self, identity: DecisionIdentity, snapshot: Mapping[str, Any] | None) -> None:
        if not snapshot:
            return
        sanitized = _sanitize_allowed(snapshot, _PERSONA_KEYS)
        if not sanitized:
            return
        sanitized.setdefault("sanitized", True)
        self._append("persona_exposure_audit", identity, {"snapshot": sanitized})

    def record_persona_prompt_proof(
        self,
        identity: DecisionIdentity,
        messages: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        persona_text: str | None,
        attempt_kind: str,
        *,
        attempt_ordinal: int | None = None,
        confirmed_injection: bool | None = None,
    ) -> None:
        """兼容旧调用：只记录 request assembly 调试证据，不确认 provider 注入。"""
        self.record_persona_request_assembly(
            identity,
            messages,
            persona_text,
            attempt_kind,
            attempt_ordinal=attempt_ordinal,
        )

    def record_persona_request_assembly(
        self,
        identity: DecisionIdentity,
        messages: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        persona_text: str | None,
        attempt_kind: str,
        *,
        attempt_ordinal: int | None = None,
    ) -> None:
        """记录 provider 之前的组装调试信息；该事件不进入 confirmed 指标。"""
        system_index: int | None = None
        system_content = ""
        for index, message in enumerate(messages):
            if message.get("role") == "system":
                system_index = index
                system_content = str(message.get("content") or "")
        proof = {
            "final_system_location": "messages",
            "final_system_message_index": system_index,
            "message_char_count": len(system_content),
            "run_scoped_fingerprint": "",
            "confirmed_injection": False,
            "attempt_kind": str(attempt_kind),
            "attempt_ordinal": attempt_ordinal,
            "sanitized": True,
        }
        self._append(
            "persona_request_assembly_audit",
            identity,
            {"proof": _sanitize_allowed(proof, _PERSONA_PROOF_KEYS)},
        )

    def record_provider_persona_prompt_proof(
        self,
        identity: DecisionIdentity,
        system_bytes: bytes,
        persona_text: str,
        attempt_kind: str,
        *,
        attempt_ordinal: int | None,
        provider: str,
        model: str,
        final_system_location: str,
        final_system_message_index: int | None,
    ) -> None:
        """在 provider HTTP 前把真实 system 字节转为 run-scoped HMAC 证明。"""
        digest = hmac.new(
            self._persona_proof_secret,
            system_bytes,
            hashlib.sha256,
        ).hexdigest()[:24]
        fingerprint = f"run_hmac_{digest}"
        persona_bytes = persona_text.encode("utf-8") if persona_text else b""
        proof = {
            "final_system_location": str(final_system_location),
            "final_system_message_index": final_system_message_index,
            "message_char_count": len(system_bytes.decode("utf-8")),
            "run_scoped_fingerprint": fingerprint,
            "confirmed_injection": bool(persona_bytes and persona_bytes in system_bytes),
            "attempt_kind": str(attempt_kind),
            "attempt_ordinal": attempt_ordinal,
            "provider": str(provider),
            "model": str(model),
            "sanitized": True,
        }
        dedupe_key = (
            identity.trace_id(),
            str(attempt_kind),
            attempt_ordinal,
            fingerprint,
        )
        if dedupe_key in self._persona_proof_keys:
            return
        self._persona_proof_keys.add(dedupe_key)
        self._append(
            "persona_prompt_injection_audit",
            identity,
            {"proof": _sanitize_allowed(proof, _PERSONA_PROOF_KEYS)},
        )

    def flush_events(self) -> list[GameEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def _append(
        self,
        event_type: str,
        identity: DecisionIdentity,
        module_payload: dict[str, Any],
    ) -> None:
        payload = _identity_payload(identity)
        payload.update(module_payload)
        self._events.append(GameEvent(type=event_type, payload=payload))


def summarize_persona_prompt_confirmation(
    events: list[Any],
) -> dict[str, Any]:
    """按 decision identity 连接 persona 曝光和最终 system 注入证明。"""
    if events and hasattr(events[0], "module_exposures"):
        configured = {
            str(trace.trace_id)
            for trace in events
            if any(exposure.module == "persona" for exposure in trace.module_exposures)
        }
        confirmed = {
            str(trace.trace_id)
            for trace in events
            if str(trace.trace_id) in configured
            and any(
                exposure.module == "persona_prompt_confirmation"
                and exposure.score == 1.0
                for exposure in trace.module_exposures
            )
        }
        return _persona_confirmation_summary(configured, confirmed)

    def _type(event: Any) -> str:
        if isinstance(event, Mapping):
            return str(event.get("type") or "")
        return str(getattr(event, "type", ""))

    def _payload(event: Any) -> dict[str, Any]:
        if isinstance(event, Mapping):
            payload = event.get("payload", event)
        else:
            payload = getattr(event, "payload", {})
        return payload if isinstance(payload, Mapping) else {}

    configured = {
        str(_payload(event).get("trace_id") or "")
        for event in events
        if _type(event) == "persona_exposure_audit"
        and _payload(event).get("trace_id")
    }
    confirmed = {
        str(_payload(event).get("trace_id") or "")
        for event in events
        if _type(event) == "persona_prompt_injection_audit"
        and _payload(event).get("trace_id") in configured
        and bool((_payload(event).get("proof") or {}).get("confirmed_injection"))
    }
    return _persona_confirmation_summary(configured, confirmed)


def _persona_confirmation_summary(
    configured: set[str],
    confirmed: set[str],
) -> dict[str, Any]:
    denominator = len(configured)
    return {
        "supported": denominator > 0,
        "configured_action_count": denominator,
        "confirmed_action_count": len(confirmed),
        "confirmation_rate": len(confirmed) / denominator if denominator else None,
    }
