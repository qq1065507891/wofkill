# -*- coding: utf-8 -*-
"""
评价指标中的世界模型审计与真实世界排名指标计算。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.evaluation.world_model_metrics import compute_world_model_metrics
    >>> compute_world_model_metrics(snapshot, results)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.decision_helpers import (
    decision_is_legal_from_trace as _decision_is_legal_from_trace,
    dialogue_leaked_from_trace as _dialogue_leaked_from_trace,
)
from werewolf_agent.evaluation.schemas import GameResult, MetricsSnapshot, WorldModelMetrics
from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder
from werewolf_agent.evaluation.world_model_eval import compute_world_model_rank_metrics


def compute_world_model_metrics(
    snap: MetricsSnapshot,
    results: list[GameResult],
) -> None:
    """把世界模型相关指标写入 MetricsSnapshot。"""
    belief_scores: list[float] = []
    possible_world_hits: list[bool] = []
    simulation_hits: list[bool] = []
    decision_legal: list[bool] = []
    dialogue_leaks: list[bool] = []

    for result in results:
        for review in result.reviews:
            audit = review.get("world_model_audit") if isinstance(review, dict) else None
            if not isinstance(audit, dict):
                continue
            _collect_world_model_audit_samples(
                audit,
                player_roles=result.player_roles,
                belief_scores=belief_scores,
                possible_world_hits=possible_world_hits,
                simulation_hits=simulation_hits,
                decision_legal=decision_legal,
                dialogue_leaks=dialogue_leaks,
            )
        for event in result.event_log:
            trace = _action_trace_from_event(event)
            if not trace:
                continue
            audit = trace.get("world_model_audit")
            if isinstance(audit, dict):
                _collect_world_model_audit_samples(
                    audit,
                    player_roles=result.player_roles,
                    belief_scores=belief_scores,
                    possible_world_hits=possible_world_hits,
                    simulation_hits=simulation_hits,
                    decision_legal=decision_legal,
                    dialogue_leaks=dialogue_leaks,
                )
            legal = _decision_is_legal_from_trace(trace)
            if legal is not None:
                decision_legal.append(legal)
            leaked = _dialogue_leaked_from_trace(trace)
            if leaked is not None:
                dialogue_leaks.append(leaked)

    rank_supported = 0
    rank_unsupported = 0
    rank_top1_hits = 0.0
    rank_top3_hits = 0.0
    rank_sum = 0.0
    rank_overconfident = 0.0
    for result in results:
        rank_metrics = compute_world_model_rank_metrics(
            result,
            EvaluationTraceBuilder().build(result, exposure_audits=[]),
        )
        rank_supported += rank_metrics.supported_count
        rank_unsupported += rank_metrics.unsupported_count
        rank_top1_hits += rank_metrics.true_world_top1_rate * rank_metrics.supported_count
        rank_top3_hits += rank_metrics.true_world_top3_rate * rank_metrics.supported_count
        rank_sum += rank_metrics.avg_true_world_rank * rank_metrics.supported_count
        rank_overconfident += (
            rank_metrics.overconfidence_rate * rank_metrics.supported_count
        )

    snap.world_model_metrics = WorldModelMetrics(
        belief_calibration=_avg(belief_scores),
        possible_world_topk_hit_rate=_bool_rate(possible_world_hits),
        simulator_prediction_hit_rate=_bool_rate(simulation_hits),
        decision_legality_rate=_bool_rate(decision_legal),
        dialogue_leakage_rate=_bool_rate(dialogue_leaks),
        true_world_top1_rate=_rate_from_counts(rank_top1_hits, rank_supported),
        true_world_top3_rate=_rate_from_counts(rank_top3_hits, rank_supported),
        avg_true_world_rank=_rate_from_counts(rank_sum, rank_supported),
        world_rank_overconfidence_rate=_rate_from_counts(
            rank_overconfident,
            rank_supported,
        ),
        world_rank_supported_count=rank_supported,
        world_rank_unsupported_count=rank_unsupported,
    )


def _collect_world_model_audit_samples(
    audit: dict[str, Any],
    *,
    player_roles: dict[str, str],
    belief_scores: list[float],
    possible_world_hits: list[bool],
    simulation_hits: list[bool],
    decision_legal: list[bool],
    dialogue_leaks: list[bool],
) -> None:
    for sample in audit.get("belief_calibration_samples", []) or []:
        if not isinstance(sample, dict):
            continue
        predicted = _bounded_float(sample.get("predicted"))
        actual = 1.0 if bool(sample.get("actual")) else 0.0
        belief_scores.append(1.0 - abs(predicted - actual))
    belief_scores.extend(_belief_scores_from_audit(audit, player_roles))

    possible_world_hits.extend(
        bool(item.get("hit"))
        for item in audit.get("possible_world_checks", []) or []
        if isinstance(item, dict)
    )
    world_hit = _possible_world_hit_from_audit(audit, player_roles)
    if world_hit is not None:
        possible_world_hits.append(world_hit)

    simulation_hits.extend(
        bool(item.get("hit"))
        for item in audit.get("simulation_checks", []) or []
        if isinstance(item, dict)
    )
    decision_legal.extend(
        bool(item.get("legal"))
        for item in audit.get("decision_legality_checks", []) or []
        if isinstance(item, dict)
    )
    dialogue_leaks.extend(
        bool(item.get("leaked"))
        for item in audit.get("dialogue_leak_checks", []) or []
        if isinstance(item, dict)
    )


def _belief_scores_from_audit(
    audit: dict[str, Any],
    player_roles: dict[str, str],
) -> list[float]:
    belief = audit.get("belief")
    if not isinstance(belief, dict) or not player_roles:
        return []
    scores: list[float] = []
    for group in ("my_suspects", "my_trusted"):
        for item in belief.get(group, []) or []:
            if not isinstance(item, dict):
                continue
            player_id = str(item.get("player") or "")
            guessed_role = _normalize_role(item.get("top_role_guess"))
            if not player_id or not guessed_role or player_id not in player_roles:
                continue
            predicted = _bounded_float(item.get("top_role_prob"))
            actual = 1.0 if _normalize_role(player_roles[player_id]) == guessed_role else 0.0
            scores.append(1.0 - abs(predicted - actual))
    for player_id, role_probs in belief.items():
        if player_id in {"my_suspects", "my_trusted"}:
            continue
        if not isinstance(role_probs, dict) or player_id not in player_roles:
            continue
        for role, predicted in role_probs.items():
            normalized = _normalize_role(role)
            if not normalized:
                continue
            actual = 1.0 if _normalize_role(player_roles[player_id]) == normalized else 0.0
            scores.append(1.0 - abs(_bounded_float(predicted) - actual))
    return scores


def _possible_world_hit_from_audit(
    audit: dict[str, Any],
    player_roles: dict[str, str],
) -> bool | None:
    possible_worlds = audit.get("possible_worlds")
    if isinstance(possible_worlds, dict):
        worlds = possible_worlds.get("top_worlds")
    else:
        worlds = possible_worlds
    if not isinstance(worlds, list) or not player_roles:
        return None
    saw_assignments = False
    for world in worlds:
        if not isinstance(world, dict):
            continue
        assignments = world.get("key_assignments")
        if not isinstance(assignments, dict) or not assignments:
            continue
        comparable = {
            str(pid): _normalize_role(role)
            for pid, role in assignments.items()
            if str(pid) in player_roles
        }
        if not comparable:
            continue
        saw_assignments = True
        if all(_normalize_role(player_roles[pid]) == role for pid, role in comparable.items()):
            return True
    return False if saw_assignments else None


def _action_trace_from_event(event: Any) -> dict[str, Any] | None:
    if isinstance(event, dict):
        event_type = event.get("type")
        payload = event.get("payload") or {}
    else:
        event_type = getattr(event, "type", None)
        payload = getattr(event, "payload", {}) or {}
    if event_type != "action_trace_audit" or not isinstance(payload, dict):
        return None
    trace = payload.get("action_trace")
    return trace if isinstance(trace, dict) else None


def _normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"wolf", "werewolves"}:
        return "werewolf"
    return role


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _bool_rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _rate_from_counts(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


__all__ = [
    "compute_world_model_metrics",
    "_action_trace_from_event",
    "_avg",
    "_belief_scores_from_audit",
    "_bool_rate",
    "_bounded_float",
    "_collect_world_model_audit_samples",
    "_decision_is_legal_from_trace",
    "_dialogue_leaked_from_trace",
    "_normalize_role",
    "_possible_world_hit_from_audit",
    "_rate_from_counts",
]
