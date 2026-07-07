# -*- coding: utf-8 -*-
"""
评价指标公开兼容 facade。

作者: Project contributors
创建日期: 2025-01-15
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.evaluation.metrics import MetricsAggregator
"""

from __future__ import annotations

from werewolf_agent.evaluation.claim_metrics import _CLAIM_ROLE_MAP, _extract_claim_events
from werewolf_agent.evaluation.metric_aggregation import MetricsAggregator
from werewolf_agent.evaluation.pace_metrics import compute_pace_metrics
from werewolf_agent.evaluation.world_model_metrics import (
    _action_trace_from_event,
    _avg,
    _belief_scores_from_audit,
    _bool_rate,
    _bounded_float,
    _collect_world_model_audit_samples,
    _decision_is_legal_from_trace,
    _dialogue_leaked_from_trace,
    _normalize_role,
    _possible_world_hit_from_audit,
    _rate_from_counts,
    compute_world_model_metrics,
)


__all__ = [
    "MetricsAggregator",
    "compute_pace_metrics",
    "compute_world_model_metrics",
    "_CLAIM_ROLE_MAP",
    "_action_trace_from_event",
    "_avg",
    "_belief_scores_from_audit",
    "_bool_rate",
    "_bounded_float",
    "_collect_world_model_audit_samples",
    "_decision_is_legal_from_trace",
    "_dialogue_leaked_from_trace",
    "_extract_claim_events",
    "_normalize_role",
    "_possible_world_hit_from_audit",
    "_rate_from_counts",
]
