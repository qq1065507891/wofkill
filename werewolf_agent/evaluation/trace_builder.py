"""Build normalized feedback-loop traces from evaluation game results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from werewolf_agent.evaluation.feedback_schemas import (
    DecisionOutcome,
    DecisionSnapshot,
    EvaluationTrace,
    MetricSupport,
    ModuleExposure,
)
from werewolf_agent.evaluation.schemas import GameResult


class EvaluationTraceBuilder:
    """Convert game audit records into per-decision feedback traces."""

    def build(
        self,
        result: GameResult,
        exposure_audits: list[dict[str, Any]] | None = None,
    ) -> list[EvaluationTrace]:
        exposure_by_trace = self._group_exposure_audits(exposure_audits or [])
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
                exposure_sources_provided=bool(exposure_audits),
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
        task_type = str(
            action_trace.get("task_type")
            or parsed_action.get("task_type")
            or parsed_action.get("action_type")
            or action_trace.get("final_action_type")
            or ""
        )
        trace_id = make_trace_id(
            game_id=result.game_id,
            player_id=player_id,
            phase=phase,
            day_number=day_number,
            night_number=night_number,
            task_type=task_type,
            action_index=action_index,
        )
        decision = _decision_snapshot(action_trace, parsed_action)
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
            outcome=_decision_outcome(result, decision),
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
        return grouped


def make_trace_id(
    *,
    game_id: str,
    player_id: str,
    phase: str,
    day_number: int,
    night_number: int,
    task_type: str,
    action_index: int,
) -> str:
    return (
        f"{game_id}:{player_id}:{phase}:"
        f"D{day_number}:N{night_number}:{task_type}:{action_index}"
    )


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


def _decision_outcome(
    result: GameResult,
    decision: DecisionSnapshot,
) -> DecisionOutcome:
    target_id = decision.target_id or ""
    target_role = result.player_roles.get(target_id, "")
    target_faction = result.player_factions.get(target_id, "")
    vote_hit_wolf = None
    if decision.action_type == "vote" and target_faction:
        vote_hit_wolf = target_faction == "werewolf"
    return DecisionOutcome(
        target_role=target_role,
        target_faction=target_faction,
        vote_hit_wolf=vote_hit_wolf,
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

