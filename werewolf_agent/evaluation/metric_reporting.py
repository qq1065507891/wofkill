# -*- coding: utf-8 -*-
"""
提供评价指标 snapshot 的比较和报表整形辅助函数。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.evaluation.metric_reporting import compare_snapshots
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.schemas import MetricsSnapshot


__all__ = ["compare_snapshots"]


def compare_snapshots(
    snap_a: MetricsSnapshot,
    snap_b: MetricsSnapshot,
    dimension: str = "",
    label_a: str = "A",
    label_b: str = "B",
) -> list[dict[str, Any]]:
    """Compare two metric snapshots across all metric dimensions."""
    comparisons = []

    def _add(metric_name: str, val_a: float, val_b: float) -> None:
        comparisons.append({
            "dimension": dimension,
            "label_a": label_a,
            "label_b": label_b,
            "metric_name": metric_name,
            "value_a": val_a,
            "value_b": val_b,
            "delta": val_b - val_a,
            "games_a": snap_a.total_games,
            "games_b": snap_b.total_games,
        })

    fm_a, fm_b = snap_a.faction_metrics, snap_b.faction_metrics
    _add("good_win_rate", fm_a.good_win_rate, fm_b.good_win_rate)
    _add("werewolf_win_rate", fm_a.werewolf_win_rate, fm_b.werewolf_win_rate)

    sm_a, sm_b = snap_a.safety_metrics, snap_b.safety_metrics
    _add("leakage_rate", sm_a.leakage_rate, sm_b.leakage_rate)
    _add("illegal_action_rate", sm_a.illegal_action_rate, sm_b.illegal_action_rate)

    qm_a, qm_b = snap_a.quality_metrics, snap_b.quality_metrics
    _add("vote_accuracy", qm_a.vote_accuracy, qm_b.vote_accuracy)
    _add("identity_disguise_rate", qm_a.identity_disguise_rate, qm_b.identity_disguise_rate)
    _add("anti_push_rate", qm_a.anti_push_rate, qm_b.anti_push_rate)
    _add("lie_detection_rate", qm_a.lie_detection_rate, qm_b.lie_detection_rate)
    _add("stance_accuracy", qm_a.stance_accuracy, qm_b.stance_accuracy)
    _add("bold_claim_success_rate", qm_a.bold_claim_success_rate, qm_b.bold_claim_success_rate)
    _add("hybrid_co_win_rate", qm_a.hybrid_co_win_rate, qm_b.hybrid_co_win_rate)
    _add("witch_potion_benefit", qm_a.witch_potion_benefit, qm_b.witch_potion_benefit)
    _add("seer_badge_flow_quality", qm_a.seer_badge_flow_quality, qm_b.seer_badge_flow_quality)
    _add("wolf_consensus_quality", qm_a.wolf_consensus_quality, qm_b.wolf_consensus_quality)
    _add("contradiction_hit_rate", qm_a.contradiction_hit_rate, qm_b.contradiction_hit_rate)

    cm_a, cm_b = snap_a.cost_metrics, snap_b.cost_metrics
    _add("avg_cost_per_game", cm_a.avg_cost_per_game, cm_b.avg_cost_per_game)
    _add("avg_latency_ms", float(cm_a.avg_latency_ms), float(cm_b.avg_latency_ms))

    return comparisons
