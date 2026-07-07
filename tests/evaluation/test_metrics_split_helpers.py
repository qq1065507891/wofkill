# -*- coding: utf-8 -*-
"""
评价指标拆分模块的兼容导入测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/evaluation/test_metrics_split_helpers.py
"""

from __future__ import annotations

from werewolf_agent.evaluation import claim_metrics
from werewolf_agent.evaluation import metric_aggregation
from werewolf_agent.evaluation import metric_reporting
from werewolf_agent.evaluation import metrics
from werewolf_agent.evaluation import pace_metrics
from werewolf_agent.evaluation import world_model_metrics
from werewolf_agent.evaluation.schemas import GameResult, MetricsSnapshot


def test_metrics_aggregator_is_reexported_from_metrics_facade() -> None:
    assert metrics.MetricsAggregator is metric_aggregation.MetricsAggregator
    assert (
        metric_aggregation.MetricsAggregator.compare_snapshots
        is metric_reporting.compare_snapshots
    )


def test_claim_event_extractor_is_reexported_from_metrics_facade() -> None:
    assert metrics._extract_claim_events is claim_metrics._extract_claim_events

    claims = claim_metrics._extract_claim_events([
        {
            "type": "speech",
            "payload": {"speaker": "p01", "text": "我跳 预言家"},
        }
    ])

    assert claims == [{
        "type": "claim_role",
        "payload": {"player_id": "p01", "claimed_role": "seer"},
    }]


def test_pace_metrics_is_reexported_from_metrics_facade() -> None:
    assert metrics.compute_pace_metrics is pace_metrics.compute_pace_metrics

    events = [
        {"type": "vote_resolved", "payload": {"exiled": "p01", "votes": {"p02": "p01"}}},
        {"type": "vote_resolved", "payload": {"exiled": "p01", "votes": {"p02": "p01"}}},
    ]

    result = pace_metrics.compute_pace_metrics(events, finish_night=5)

    assert result["stale_vote_reuse_count"] == 1
    assert result["pace_target_met"] is False


def test_world_model_metric_computer_is_reexported_from_metrics_facade() -> None:
    assert metrics.compute_world_model_metrics is world_model_metrics.compute_world_model_metrics
    assert (
        metrics._decision_is_legal_from_trace
        is world_model_metrics._decision_is_legal_from_trace
    )
    assert (
        metrics._dialogue_leaked_from_trace
        is world_model_metrics._dialogue_leaked_from_trace
    )

    snap = MetricsSnapshot(batch_id="wm", total_games=1)
    world_model_metrics.compute_world_model_metrics(
        snap,
        [GameResult(game_id="g_wm", initial_seed=1, ruleset_id="rules")],
    )

    assert snap.world_model_metrics.world_rank_supported_count == 0
