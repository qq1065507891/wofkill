"""Tests for previously placeholder evaluation metrics."""
import pytest
from werewolf_agent.evaluation.schemas import (
    ActionRecord,
    ActionVerdict,
    GameResult,
    QualityMetrics,
    BatchConfig,
)
from werewolf_agent.evaluation.metrics import MetricsAggregator


def _make_result_with_speech_events():
    """Game result where player_03's speech influenced others' votes."""
    return GameResult(
        game_id="g1",
        initial_seed=1,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        winning_faction="good",
        event_log=[
            {
                "type": "speech",
                "player_id": "p03",
                "day_number": 1,
                "text": "I think p07 is wolf",
                "mentioned_targets": ["p07"],
            },
            {"type": "vote", "player_id": "p04", "target_id": "p07", "day_number": 1},
            {"type": "vote", "player_id": "p05", "target_id": "p07", "day_number": 1},
            {"type": "vote", "player_id": "p06", "target_id": "p08", "day_number": 1},
        ],
        player_roles={"p03": "seer", "p04": "villager", "p07": "werewolf"},
        action_records=[
            ActionRecord(
                player_id="p04", action_type="vote", target_id="p07", phase="day", day_number=1
            ),
            ActionRecord(
                player_id="p05", action_type="vote", target_id="p07", phase="day", day_number=1
            ),
        ],
    )


def test_speech_influence_computed():
    agg = MetricsAggregator()
    agg.add_result(_make_result_with_speech_events())
    snap = agg.compute_snapshot()
    assert 0.0 <= snap.quality_metrics.speech_influence_rate <= 1.0
    assert snap.quality_metrics.speech_influence_rate > 0.0


def test_speech_order_utilization_computed():
    agg = MetricsAggregator()
    agg.add_result(_make_result_with_speech_events())
    snap = agg.compute_snapshot()
    assert 0.0 <= snap.quality_metrics.speech_order_utilization <= 1.0


def test_cognitive_compression_rate_computed():
    agg = MetricsAggregator()
    agg.add_result(
        GameResult(
            game_id="g2",
            initial_seed=2,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            winning_faction="werewolf",
            cognition_snapshots={
                "p01": {"original_fact_count": 50, "compressed_fact_count": 15},
            },
        )
    )
    snap = agg.compute_snapshot()
    assert 0.0 <= snap.quality_metrics.cognitive_compression_rate <= 1.0


def test_empty_results_default_to_zero():
    agg = MetricsAggregator()
    agg.add_result(
        GameResult(game_id="g3", initial_seed=3, ruleset_id="pre_witch_hunter_idiot_mixed")
    )
    snap = agg.compute_snapshot()
    assert snap.quality_metrics.speech_influence_rate == 0.0
    assert snap.quality_metrics.speech_order_utilization == 0.0
    assert snap.quality_metrics.cognitive_compression_rate == 0.0


def test_speech_influence_exact_value():
    """p03 mentions p07, then 3 votes follow: 2 for p07, 1 for p08. Rate = 2/3."""
    agg = MetricsAggregator()
    agg.add_result(_make_result_with_speech_events())
    snap = agg.compute_snapshot()
    assert snap.quality_metrics.speech_influence_rate == pytest.approx(2 / 3, abs=0.01)


def test_speech_order_utilization_with_multiple_speeches():
    """Two speeches on same day: second one should count as referencing prior."""
    agg = MetricsAggregator()
    agg.add_result(
        GameResult(
            game_id="g4",
            initial_seed=4,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            winning_faction="good",
            event_log=[
                {
                    "type": "speech",
                    "player_id": "p01",
                    "day_number": 1,
                    "text": "First speech",
                    "mentioned_targets": [],
                },
                {
                    "type": "speech",
                    "player_id": "p02",
                    "day_number": 1,
                    "text": "I agree with p01",
                    "mentioned_targets": ["p05"],
                },
            ],
        )
    )
    snap = agg.compute_snapshot()
    # 2 speeches, 1 (second) references a prior same-day speech
    assert snap.quality_metrics.speech_order_utilization == pytest.approx(0.5, abs=0.01)


def test_cognitive_compression_rate_multiple_players():
    """Average compression rate across multiple players."""
    agg = MetricsAggregator()
    agg.add_result(
        GameResult(
            game_id="g5",
            initial_seed=5,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            winning_faction="good",
            cognition_snapshots={
                "p01": {"original_fact_count": 100, "compressed_fact_count": 20},
                "p02": {"original_fact_count": 50, "compressed_fact_count": 25},
            },
        )
    )
    snap = agg.compute_snapshot()
    # p01: 20/100=0.2, p02: 25/50=0.5, avg=0.35
    assert snap.quality_metrics.cognitive_compression_rate == pytest.approx(0.35, abs=0.01)


def test_cognitive_compression_skips_zero_original():
    """Players with original_fact_count=0 should be skipped."""
    agg = MetricsAggregator()
    agg.add_result(
        GameResult(
            game_id="g6",
            initial_seed=6,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            winning_faction="good",
            cognition_snapshots={
                "p01": {"original_fact_count": 0, "compressed_fact_count": 0},
                "p02": {"original_fact_count": 50, "compressed_fact_count": 10},
            },
        )
    )
    snap = agg.compute_snapshot()
    # Only p02 counted: 10/50 = 0.2
    assert snap.quality_metrics.cognitive_compression_rate == pytest.approx(0.2, abs=0.01)


def test_speech_influence_no_mentioned_targets():
    """Speech without mentioned_targets should not contribute to influence."""
    agg = MetricsAggregator()
    agg.add_result(
        GameResult(
            game_id="g7",
            initial_seed=7,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            winning_faction="good",
            event_log=[
                {
                    "type": "speech",
                    "player_id": "p01",
                    "day_number": 1,
                    "text": "Hello everyone",
                    "mentioned_targets": [],
                },
                {"type": "vote", "player_id": "p02", "target_id": "p05", "day_number": 1},
            ],
        )
    )
    snap = agg.compute_snapshot()
    assert snap.quality_metrics.speech_influence_rate == 0.0


def test_provenance_includes_new_metrics():
    """All three new metrics should have provenance entries."""
    agg = MetricsAggregator()
    agg.add_result(_make_result_with_speech_events())
    snap = agg.compute_snapshot()
    for key in ("speech_influence_rate", "speech_order_utilization"):
        assert key in snap.provenance, f"Missing provenance for {key}"


def test_provenance_for_cognitive_compression():
    """cognitive_compression_rate should have provenance when cognition data exists."""
    agg = MetricsAggregator()
    agg.add_result(
        GameResult(
            game_id="g8",
            initial_seed=8,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            winning_faction="good",
            cognition_snapshots={
                "p01": {"original_fact_count": 50, "compressed_fact_count": 15},
            },
        )
    )
    snap = agg.compute_snapshot()
    assert "cognitive_compression_rate" in snap.provenance
