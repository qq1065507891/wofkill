# -*- coding: utf-8 -*-
"""
从评估对局结果构建归一化反馈轨迹，并稳定合并运行时与兼容侧通道曝光审计。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-16
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from werewolf_agent.evaluation.decision_helpers import (
    decision_is_legal_from_trace,
    dialogue_leaked_from_trace,
)
from werewolf_agent.evaluation.feedback_schemas import (
    DecisionOutcome,
    DecisionSnapshot,
    EvaluationTrace,
    MetricSupport,
    ModuleExposure,
)
from werewolf_agent.evaluation.schemas import GameResult
from werewolf_agent.evaluation.trace_identity import make_trace_id

_SKILL_TOOL_CALL_METADATA_KEYS = frozenset({
    "call_kind",
    "status",
    "success",
    "required",
    "received",
    "result_available_to_decision",
    "decision_usage",
    "fallback_triggered",
    "error_type",
    "structured_failure_reason",
    "structured_failure_stage",
    "original_failure_code",
    "failure_stage",
    "fallback_kind",
    "structured_output_mode",
    "parse_success",
    "retry_count",
})
_SKILL_TOOL_INPUT_METADATA_KEYS = frozenset({
    "role",
    "phase",
    "task_type",
    "day",
    "night",
    "legal_target_count",
    "candidate_count",
    "has_wolf_team_plan",
})
_SKILL_TOOL_OUTPUT_METADATA_KEYS = frozenset({
    "confidence",
    "has_prompt_injectable",
    "risk_alert_count",
    "evidence_ref_count",
    "summary_hash",
    "reasoning_hash",
    "tool_call_name",
})
_PROMPT_INJECTION_METADATA_KEYS = frozenset({
    "module_name",
    "field_path",
    "injection_kind",
    "injected",
    "visibility_scope",
    "item_count",
    "char_count",
    "content_hash",
    "decision_usage",
    "sanitized",
})
_PERSONA_PROOF_METADATA_KEYS = frozenset({
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


class EvaluationTraceBuilder:
    """Convert game audit records into per-decision feedback traces."""

    def build(
        self,
        result: GameResult,
        exposure_audits: list[dict[str, Any]] | None = None,
    ) -> list[EvaluationTrace]:
        runtime_exposure_audits = self._collect_exposure_events(result.event_log)
        side_channel_audits = exposure_audits or []
        exposure_by_trace = self._merge_exposure_audits(
            runtime_exposure_audits,
            side_channel_audits,
        )
        exposure_sources_provided = bool(runtime_exposure_audits or side_channel_audits)
        traces: list[EvaluationTrace] = []
        action_index = 0
        for event_index, event in enumerate(result.event_log):
            if _event_type(event) != "action_trace_audit":
                continue
            payload = _event_payload(event)
            if not isinstance(payload, dict):
                continue
            action_trace = payload.get("action_trace")
            if not isinstance(action_trace, dict):
                continue
            trace = self._trace_from_action_event(
                result=result,
                payload=payload,
                action_trace=action_trace,
                action_index=action_index,
                event_index=event_index,
                exposure_by_trace=exposure_by_trace,
                exposure_sources_provided=exposure_sources_provided,
            )
            traces.append(trace)
            action_index += 1
        return traces

    def _trace_from_action_event(
        self,
        *,
        result: GameResult,
        payload: dict[str, Any],
        action_trace: dict[str, Any],
        action_index: int,
        event_index: int,
        exposure_by_trace: dict[str, list[ModuleExposure]],
        exposure_sources_provided: bool,
    ) -> EvaluationTrace:
        player_id = str(payload.get("player_id") or "")
        phase = str(payload.get("phase") or "")
        day_number = _int(payload.get("day_number"))
        night_number = _int(payload.get("night_number"))
        parsed_action = action_trace.get("parsed_action")
        parsed_action = parsed_action if isinstance(parsed_action, dict) else {}
        decision_action = _effective_decision_action(action_trace, parsed_action)
        task_type = str(
            action_trace.get("task_type")
            or decision_action.get("task_type")
            or decision_action.get("action_type")
            or action_trace.get("final_action_type")
            or ""
        )
        trace_id = str(payload.get("trace_id") or "")
        if not trace_id:
            trace_id = make_trace_id(
                game_id=result.game_id,
                player_id=player_id,
                phase=phase,
                day_number=day_number,
                night_number=night_number,
                task_type=task_type,
                action_index=action_index,
            )
        decision = _decision_snapshot(action_trace, decision_action)
        exposures = []
        exposures.extend(_world_model_exposures(action_trace.get("world_model_audit")))
        exposures.extend(exposure_by_trace.get(trace_id, []))
        if not exposure_sources_provided:
            exposures.extend([
                ModuleExposure(
                    module="rag",
                    item_id="missing_source",
                    support=MetricSupport.UNSUPPORTED,
                ),
                ModuleExposure(
                    module="reflection",
                    item_id="missing_source",
                    support=MetricSupport.UNSUPPORTED,
                ),
            ])
        return EvaluationTrace(
            trace_id=trace_id,
            game_id=result.game_id,
            player_id=player_id,
            role=result.player_roles.get(player_id, ""),
            faction=result.player_factions.get(player_id, ""),
            phase=phase,
            day_number=day_number,
            night_number=night_number,
            task_type=task_type,
            legal_actions=[str(item) for item in action_trace.get("legal_actions", []) or []],
            legal_targets=[str(item) for item in action_trace.get("legal_targets", []) or []],
            module_exposures=exposures,
            decision=decision,
            outcome=_decision_outcome(result, decision, action_trace),
            source_refs=[f"event:{event_index}:action_trace_audit"],
        )

    @staticmethod
    def _group_exposure_audits(
        audits: list[dict[str, Any]],
    ) -> dict[str, list[ModuleExposure]]:
        grouped: dict[str, list[ModuleExposure]] = defaultdict(list)
        for audit in audits:
            if not isinstance(audit, dict):
                continue
            trace_id = str(audit.get("trace_id") or "")
            if not trace_id:
                continue
            audit_type = str(audit.get("type") or "")
            if audit_type == "rag_exposure_audit":
                grouped[trace_id].extend(_rag_exposures(audit))
            elif audit_type == "reflection_exposure_audit":
                grouped[trace_id].extend(_reflection_exposures(audit))
            elif audit_type == "skill_exposure_audit":
                grouped[trace_id].extend(_skill_exposures(audit))
            elif audit_type == "skill_tool_call_audit":
                grouped[trace_id].extend(_skill_tool_call_exposures(audit))
            elif audit_type == "prompt_injection_audit":
                grouped[trace_id].extend(_prompt_injection_exposures(audit))
            elif audit_type == "persona_exposure_audit":
                grouped[trace_id].extend(_persona_exposures(audit))
            elif audit_type == "persona_prompt_injection_audit":
                grouped[trace_id].extend(_persona_prompt_proof_exposures(audit))
        return grouped

    @classmethod
    def _merge_exposure_audits(
        cls,
        runtime_audits: list[dict[str, Any]],
        side_channel_audits: list[dict[str, Any]],
    ) -> dict[str, list[ModuleExposure]]:
        """以运行时记录为准，稳定去除兼容侧通道的重复曝光。"""
        runtime_grouped = cls._group_exposure_audits(runtime_audits)
        side_grouped = cls._group_exposure_audits(side_channel_audits)
        merged: dict[str, list[ModuleExposure]] = defaultdict(list)
        for trace_id in dict.fromkeys([*runtime_grouped, *side_grouped]):
            seen: set[tuple[Any, ...]] = set()
            for exposure in [
                *runtime_grouped.get(trace_id, []),
                *side_grouped.get(trace_id, []),
            ]:
                identity = _exposure_identity(trace_id, exposure)
                if identity in seen:
                    continue
                seen.add(identity)
                merged[trace_id].append(exposure)
        return merged

    @staticmethod
    def _collect_exposure_events(event_log: list[Any]) -> list[dict[str, Any]]:
        audits: list[dict[str, Any]] = []
        for event in event_log:
            event_type = _event_type(event)
            if event_type not in {
                "rag_exposure_audit",
                "reflection_exposure_audit",
                "skill_exposure_audit",
                "skill_tool_call_audit",
                "prompt_injection_audit",
                "persona_exposure_audit",
                "persona_prompt_injection_audit",
            }:
                continue
            payload = _event_payload(event)
            if not isinstance(payload, dict):
                continue
            audits.append({**payload, "type": event_type})
        return audits


def _rag_exposures(audit: dict[str, Any]) -> list[ModuleExposure]:
    exposures: list[ModuleExposure] = []
    for hit in audit.get("hits", []) or []:
        if not isinstance(hit, dict):
            continue
        entry_id = str(hit.get("entry_id") or "")
        if not entry_id:
            continue
        exposures.append(ModuleExposure(
            module="rag",
            item_id=entry_id,
            rank=_int(hit.get("rank")),
            score=_float(hit.get("relevance_score")),
            prompt_visible=bool(hit.get("prompt_visible")),
            metadata={
                key: value
                for key, value in hit.items()
                if key not in {"entry_id", "rank", "relevance_score", "prompt_visible"}
            },
        ))
    return exposures


def _reflection_exposures(audit: dict[str, Any]) -> list[ModuleExposure]:
    exposures: list[ModuleExposure] = []
    for card in audit.get("cards", []) or []:
        if not isinstance(card, dict):
            continue
        entry_id = str(card.get("entry_id") or "")
        if not entry_id:
            continue
        exposures.append(ModuleExposure(
            module="reflection",
            item_id=entry_id,
            rank=_int(card.get("rank")),
            score=_float(card.get("quality_score")),
            prompt_visible=bool(card.get("prompt_visible")),
            metadata={
                key: value
                for key, value in card.items()
                if key not in {"entry_id", "rank", "quality_score", "prompt_visible"}
            },
        ))
    return exposures


def _skill_exposures(audit: dict[str, Any]) -> list[ModuleExposure]:
    exposures: list[ModuleExposure] = []
    for analysis in audit.get("analyses", []) or []:
        if not isinstance(analysis, dict):
            continue
        skill_name = str(analysis.get("skill_name") or "")
        if not skill_name:
            continue
        exposures.append(ModuleExposure(
            module="skills",
            item_id=skill_name,
            rank=_int(analysis.get("rank")),
            prompt_visible=bool(analysis.get("prompt_visible")),
            metadata={
                key: analysis[key]
                for key in ("summary_hash", "advice_type")
                if key in analysis
            },
        ))
    return exposures


def _skill_tool_call_exposures(audit: dict[str, Any]) -> list[ModuleExposure]:
    exposures: list[ModuleExposure] = []
    for call in audit.get("calls", []) or []:
        if not isinstance(call, dict):
            continue
        call_name = str(
            call.get("call_name")
            or call.get("skill_name")
            or call.get("tool_name")
            or ""
        )
        if not call_name:
            continue
        success = call.get("success")
        exposures.append(ModuleExposure(
            module="skill_tool_calls",
            item_id=call_name,
            score=1.0 if success is True else 0.0,
            prompt_visible=bool(call.get("prompt_visible")),
            metadata=_skill_tool_call_metadata(call),
        ))
    return exposures


def _skill_tool_call_metadata(call: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: call[key]
        for key in _SKILL_TOOL_CALL_METADATA_KEYS
        if key in call
    }
    input_summary = _safe_nested_metadata(
        call.get("input_summary"),
        _SKILL_TOOL_INPUT_METADATA_KEYS,
    )
    if input_summary:
        metadata["input_summary"] = input_summary
    output_summary = _safe_nested_metadata(
        call.get("output_summary"),
        _SKILL_TOOL_OUTPUT_METADATA_KEYS,
    )
    if output_summary:
        metadata["output_summary"] = output_summary
    return metadata


def _safe_nested_metadata(value: Any, allowed_keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if str(key) in allowed_keys
    }


def _prompt_injection_exposures(audit: dict[str, Any]) -> list[ModuleExposure]:
    exposures: list[ModuleExposure] = []
    for row in audit.get("injections", []) or []:
        if not isinstance(row, dict):
            continue
        field_path = str(row.get("field_path") or row.get("module_name") or "")
        if not field_path:
            continue
        injected = row.get("injected")
        exposures.append(ModuleExposure(
            module="prompt_injections",
            item_id=field_path,
            score=1.0 if injected is True else 0.0,
            prompt_visible=bool(row.get("prompt_visible", True)),
            metadata={
                key: row[key]
                for key in _PROMPT_INJECTION_METADATA_KEYS
                if key in row
            },
        ))
    return exposures


def _persona_exposures(audit: dict[str, Any]) -> list[ModuleExposure]:
    snapshot = audit.get("snapshot")
    if not isinstance(snapshot, dict):
        return []
    profile_id = str(snapshot.get("profile_id") or "")
    if not profile_id:
        return []
    return [
        ModuleExposure(
            module="persona",
            item_id=profile_id,
            prompt_visible=bool(snapshot.get("prompt_visible", True)),
            metadata={
                key: snapshot[key]
                for key in ("policy_keys", "sanitized")
                if key in snapshot
            },
        )
    ]


def _persona_prompt_proof_exposures(audit: dict[str, Any]) -> list[ModuleExposure]:
    proof = audit.get("proof")
    if not isinstance(proof, dict):
        return []
    ordinal = _int(proof.get("attempt_ordinal"))
    provider = str(proof.get("provider") or "unknown")
    return [ModuleExposure(
        module="persona_prompt_confirmation",
        item_id=f"{provider}:{ordinal if ordinal is not None else 'unknown'}",
        score=1.0 if proof.get("confirmed_injection") is True else 0.0,
        prompt_visible=False,
        metadata={
            key: proof[key]
            for key in _PERSONA_PROOF_METADATA_KEYS
            if key in proof
        },
    )]


def _exposure_identity(
    trace_id: str,
    exposure: ModuleExposure,
) -> tuple[Any, ...]:
    """构造稳定曝光身份；persona 证明按 provider 最终 payload 位置区分。"""
    if exposure.module == "persona_prompt_confirmation":
        metadata = exposure.metadata
        return (
            trace_id,
            exposure.module,
            metadata.get("attempt_kind"),
            metadata.get("attempt_ordinal"),
            metadata.get("provider"),
            metadata.get("model"),
            metadata.get("run_scoped_fingerprint"),
            metadata.get("final_system_location"),
            metadata.get("final_system_message_index"),
        )
    return (
        trace_id,
        exposure.module,
        exposure.item_id,
        exposure.rank,
        exposure.score,
        exposure.prompt_visible,
        exposure.cited_by_decision,
        exposure.aligned_with_decision,
        exposure.support.value,
        json.dumps(exposure.metadata, ensure_ascii=False, sort_keys=True, default=str),
    )


def _world_model_exposures(audit: Any) -> list[ModuleExposure]:
    if not isinstance(audit, dict):
        return []
    exposures: list[ModuleExposure] = []
    possible_worlds = audit.get("possible_worlds")
    if isinstance(possible_worlds, dict):
        worlds = possible_worlds.get("top_worlds")
    else:
        worlds = possible_worlds
    if isinstance(worlds, list):
        for rank, world in enumerate(worlds, start=1):
            if not isinstance(world, dict):
                continue
            label = str(world.get("label") or f"World {rank}")
            exposures.append(ModuleExposure(
                module="possible_worlds",
                item_id=label,
                rank=rank,
                score=_float(world.get("probability")),
                prompt_visible=True,
                metadata={
                    "key_assignments": dict(world.get("key_assignments") or {}),
                    "rank_scope": "top_k_only",
                },
            ))
    simulation = audit.get("simulation_predictions")
    predictions = simulation.get("predictions") if isinstance(simulation, dict) else None
    if isinstance(predictions, list):
        for rank, prediction in enumerate(predictions, start=1):
            if not isinstance(prediction, dict):
                continue
            event = str(prediction.get("event") or "")
            if not event:
                continue
            exposures.append(ModuleExposure(
                module="simulator",
                item_id=event,
                rank=rank,
                score=_float(prediction.get("probability")),
                prompt_visible=True,
                metadata={
                    "affected_players": list(prediction.get("affected_players") or []),
                },
            ))
    return exposures


def _decision_snapshot(
    action_trace: dict[str, Any],
    parsed_action: dict[str, Any],
) -> DecisionSnapshot:
    decision_plan = parsed_action.get("decision_plan")
    decision_plan = decision_plan if isinstance(decision_plan, dict) else {}
    action_type = str(
        action_trace.get("final_action_type")
        or parsed_action.get("action_type")
        or decision_plan.get("action_type")
        or ""
    )
    target_id = (
        parsed_action.get("target_id")
        if parsed_action.get("target_id") is not None
        else decision_plan.get("target_id")
    )
    return DecisionSnapshot(
        action_type=action_type,
        target_id=str(target_id) if target_id is not None else None,
        reason=str(parsed_action.get("reason") or decision_plan.get("reason") or ""),
        confidence=_float(parsed_action.get("confidence", decision_plan.get("confidence"))),
        raw=dict(parsed_action),
    )


def _effective_decision_action(
    action_trace: dict[str, Any],
    parsed_action: dict[str, Any],
) -> dict[str, Any]:
    """终退只消费实际 fallback；被拒 parsed_action 永远只用于审计。"""
    if (
        action_trace.get("generated_by") == "terminal_fallback"
        or action_trace.get("decision_outcome") == "terminal_fallback"
    ):
        final_action = action_trace.get("final_action")
        return dict(final_action) if isinstance(final_action, dict) else {}
    return parsed_action


def _decision_outcome(
    result: GameResult,
    decision: DecisionSnapshot,
    action_trace: dict[str, Any],
) -> DecisionOutcome:
    target_id = decision.target_id or ""
    target_role = result.player_roles.get(target_id, "")
    target_faction = result.player_factions.get(target_id, "")
    vote_hit_wolf = None
    if decision.action_type == "vote" and target_faction:
        vote_hit_wolf = target_faction == "werewolf"
    legal: bool | None = None
    leaked: bool = False
    if isinstance(action_trace, dict):
        legal = decision_is_legal_from_trace(action_trace)
        leak_decision = dialogue_leaked_from_trace(action_trace)
        leaked = leak_decision is True
    return DecisionOutcome(
        legal=legal,
        target_role=target_role,
        target_faction=target_faction,
        vote_hit_wolf=vote_hit_wolf,
        leaked_hidden_info=leaked,
        outcome_refs=[f"player_roles:{target_id}"] if target_id and target_role else [],
    )


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", "") or "")


def _event_payload(event: Any) -> Any:
    if isinstance(event, dict):
        return event.get("payload") or {}
    return getattr(event, "payload", {}) or {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
