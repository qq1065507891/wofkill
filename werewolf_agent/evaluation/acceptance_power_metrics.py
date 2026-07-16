# -*- coding: utf-8 -*-
"""
投影猎人和女巫伤害决策的证据完整性验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from werewolf_agent.evaluation.game_projection import (
    normalize_acceptance_games,
    projection_support,
)


def compute_power_acceptance_metrics(
    games: Iterable[Any],
) -> dict[str, Any]:
    """对账伤害事实与动作轨迹并投影神职证据指标。"""
    return _compute_power_acceptance_metrics_from_normalized(
        normalize_acceptance_games(games)
    )


def _compute_power_acceptance_metrics_from_normalized(
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """消费同一调用链刚完成验证的不可变游戏快照。"""
    projection_is_supported, projection_reason = projection_support(games)
    power_decisions: list[dict[str, Any]] = []
    for game in games:
        power_trace_candidates: list[tuple[str, str | None, str, dict[str, Any]]] = []
        power_damage_events: list[tuple[str, str | None, str]] = []
        for event in game.get("events", []):
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "player_died" and payload.get("reason") in {
                "hunter_shot", "witch_poison",
            }:
                power_damage_events.append((
                    str(payload.get("reason")),
                    str(payload.get("source_player_id"))
                    if payload.get("source_player_id") else None,
                    str(payload.get("player_id") or ""),
                ))
            if event_type != "action_trace_audit":
                continue
            trace = payload.get("action_trace")
            if not isinstance(trace, Mapping):
                continue
            if trace.get("final_action_type") in {"hunter_shot", "use_poison"}:
                evidence = trace.get("power_role_evidence")
                power_trace_candidates.append((
                    str(trace.get("final_action_type")),
                    str(payload.get("player_id")) if payload.get("player_id") else None,
                    str((evidence or {}).get("target_id") or "")
                    if isinstance(evidence, Mapping) else "",
                    evidence if isinstance(evidence, Mapping) else {},
                ))

        # deaths 是权威结果；player_died 必须与之逐项一致，不能互相掩盖遗漏。
        authoritative_deaths = [
            (
                str(death["reason"]),
                str(death.get("source_player_id"))
                if death.get("source_player_id") else None,
                str(death.get("player_id") or ""),
            )
            for death in game.get("deaths", [])
            if isinstance(death, Mapping) and death.get("reason") in {
                "hunter_shot", "witch_poison",
            }
        ]
        death_counts = Counter(authoritative_deaths)
        event_counts = Counter(power_damage_events)
        damage_sources_match = death_counts == event_counts
        reconciled_damage_events = list((death_counts | event_counts).elements())
        used_trace_indexes: set[int] = set()
        for reason, source, target in reconciled_damage_events:
            if not damage_sources_match:
                power_decisions.append({})
                continue
            expected_action = "hunter_shot" if reason == "hunter_shot" else "use_poison"
            match_index = next((
                index
                for index, (action, actor, trace_target, _evidence)
                in enumerate(power_trace_candidates)
                if index not in used_trace_indexes
                and action == expected_action
                and trace_target == target
                and (reason != "hunter_shot" or actor == source)
            ), None)
            if match_index is None:
                power_decisions.append({})
            else:
                used_trace_indexes.add(match_index)
                power_decisions.append(power_trace_candidates[match_index][3])

    complete_power_decisions = sum(
        _power_role_evidence_complete(evidence) for evidence in power_decisions
    )
    power_count = len(power_decisions)
    return {
        "power_role_evidence_metrics_supported": (
            projection_is_supported and power_count > 0
        ),
        "power_role_evidence_metrics_unsupported_reason": (
            projection_reason if not projection_is_supported
            else None if power_count else "no_power_role_damage_decisions"
        ),
        "power_role_damage_decision_count": power_count,
        "power_role_evidence_complete_count": complete_power_decisions,
        "power_role_evidence_completeness_rate": (
            complete_power_decisions / power_count
            if projection_is_supported and power_count else None
        ),
    }


def _power_role_evidence_complete(evidence: dict[str, Any]) -> bool:
    comparison = evidence.get("alternative_comparison")
    risk = evidence.get("friendly_fire_risk")
    retain = evidence.get("retain_option")
    if not isinstance(evidence.get("target_id"), str) or not evidence["target_id"]:
        return False
    if not _target_evidence_complete(evidence):
        return False
    if not _friendly_fire_risk_complete(risk):
        return False
    if not _retain_option_complete(retain):
        return False
    if not isinstance(comparison, Mapping):
        return False
    alternatives = comparison.get("legal_alternatives")
    if not isinstance(alternatives, (list, tuple)) or not all(
        isinstance(item, str) and bool(item) for item in alternatives
    ):
        return False
    if not isinstance(comparison.get("no_legal_alternative"), bool):
        return False
    if "alternative_target" not in comparison:
        return False
    alternative_target = comparison["alternative_target"]
    if alternative_target == evidence["target_id"]:
        return False
    if comparison.get("no_legal_alternative") is True:
        return alternative_target is None
    return alternative_target is not None and alternative_target in alternatives


def _target_evidence_complete(evidence: dict[str, Any]) -> bool:
    selected = evidence.get("target_evidence")
    comparison = evidence.get("target_comparison")
    alternatives = evidence.get("alternative_comparison")
    if not isinstance(selected, Mapping) or not isinstance(comparison, Mapping):
        return False
    if not isinstance(alternatives, Mapping):
        return False
    selected_score = selected.get("selected_score")
    selected_signals = selected.get("selected_signals")
    if not _score(selected_score) or not _signals(selected_signals):
        return False
    if (
        comparison.get("selected_score") != selected_score
        or comparison.get("selected_signals") != selected_signals
        or comparison.get("alternative_target")
        != alternatives.get("alternative_target")
        or not isinstance(comparison.get("comparison_basis"), str)
        or not comparison["comparison_basis"].strip()
    ):
        return False
    alternative_target = comparison.get("alternative_target")
    alternative_score = comparison.get("alternative_score")
    alternative_signals = comparison.get("alternative_signals")
    if alternative_target is None:
        return alternative_score is None and _signals(alternative_signals)
    return _score(alternative_score) and _signals(alternative_signals)


def _score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _signals(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, str) and bool(item) for item in value
    )


def _friendly_fire_risk_complete(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    targets = value.get("targets")
    return (
        value.get("status") == "assessed"
        and isinstance(value.get("basis"), str)
        and bool(value["basis"].strip())
        and isinstance(targets, (list, tuple)) and all(
            isinstance(item, str) and bool(item) for item in targets
        )
    )


def _retain_option_complete(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("action") == "no_action"
        and isinstance(value.get("available"), bool)
        and isinstance(value.get("required"), bool)
        and isinstance(value.get("reason"), str)
        and bool(value["reason"].strip())
    )
