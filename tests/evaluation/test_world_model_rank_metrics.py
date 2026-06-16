"""World-model true-world rank metric tests."""

from __future__ import annotations

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    ModuleExposure,
)
from werewolf_agent.evaluation.schemas import GameResult


def _trace(exposures: list[ModuleExposure]) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id="g_world:p01:vote:D2:N1:vote:0",
        game_id="g_world",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        module_exposures=exposures,
    )


def _world(label: str, rank: int, assignments: dict[str, str], score: float = 0.5) -> ModuleExposure:
    return ModuleExposure(
        module="possible_worlds",
        item_id=label,
        rank=rank,
        score=score,
        prompt_visible=True,
        metadata={
            "key_assignments": assignments,
            "rank_scope": "top_k_only",
        },
    )


def test_compute_world_model_rank_metrics_finds_true_world_rank_from_trace_exposures() -> None:
    from werewolf_agent.evaluation.world_model_eval import (
        compute_world_model_rank_metrics,
    )

    result = GameResult(
        game_id="g_world",
        initial_seed=7,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p02": "werewolf", "p03": "seer"},
    )
    traces = [
        _trace([
            _world("wrong_world", 1, {"p02": "seer", "p03": "werewolf"}, 0.72),
            _world("true_world", 2, {"p02": "werewolf", "p03": "seer"}, 0.58),
        ])
    ]

    metrics = compute_world_model_rank_metrics(result, traces)

    assert metrics.supported_count == 1
    assert metrics.unsupported_count == 0
    assert metrics.true_world_top1_rate == 0.0
    assert metrics.true_world_top3_rate == 1.0
    assert metrics.avg_true_world_rank == 2.0
    assert metrics.overconfidence_rate == 1.0
    assert metrics.samples[0].true_world_rank == 2
    assert metrics.samples[0].top_world_score == 0.72


def test_compute_world_model_rank_metrics_marks_missing_comparable_assignments_unsupported() -> None:
    from werewolf_agent.evaluation.world_model_eval import (
        compute_world_model_rank_metrics,
    )

    result = GameResult(
        game_id="g_world",
        initial_seed=7,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p02": "werewolf"},
    )
    traces = [
        _trace([
            _world("not_comparable", 1, {"unknown_player": "seer"}),
        ])
    ]

    metrics = compute_world_model_rank_metrics(result, traces)

    assert metrics.supported_count == 0
    assert metrics.unsupported_count == 1
    assert metrics.true_world_top1_rate == 0.0
    assert metrics.avg_true_world_rank == 0.0
    assert metrics.samples[0].support == "unsupported"
    assert metrics.samples[0].unsupported_reason == "no_comparable_assignments"


def test_compute_world_model_rank_metrics_normalizes_wolf_role_aliases() -> None:
    from werewolf_agent.evaluation.world_model_eval import (
        compute_world_model_rank_metrics,
    )

    result = GameResult(
        game_id="g_world",
        initial_seed=7,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p02": "werewolf"},
    )
    traces = [_trace([_world("alias_world", 1, {"p02": "wolf"})])]

    metrics = compute_world_model_rank_metrics(result, traces)

    assert metrics.supported_count == 1
    assert metrics.true_world_top1_rate == 1.0
    assert metrics.avg_true_world_rank == 1.0


def test_compute_world_model_rank_metrics_uses_top_k_miss_rank_when_truth_absent() -> None:
    from werewolf_agent.evaluation.world_model_eval import (
        compute_world_model_rank_metrics,
    )

    result = GameResult(
        game_id="g_world",
        initial_seed=7,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p02": "werewolf"},
    )
    traces = [
        _trace([
            _world("wrong_1", 1, {"p02": "seer"}),
            _world("wrong_2", 2, {"p02": "villager"}),
            _world("wrong_3", 3, {"p02": "hunter"}),
        ])
    ]

    metrics = compute_world_model_rank_metrics(result, traces)

    assert metrics.supported_count == 1
    assert metrics.true_world_top3_rate == 0.0
    assert metrics.avg_true_world_rank == 4.0
    assert metrics.samples[0].true_world_rank == 4


def test_metrics_aggregator_includes_world_model_true_rank_metrics() -> None:
    from werewolf_agent.evaluation.metrics import MetricsAggregator

    result = GameResult(
        game_id="g_world",
        initial_seed=7,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p02": "werewolf", "p03": "seer"},
        event_log=[
            {
                "type": "action_trace_audit",
                "payload": {
                    "player_id": "p01",
                    "phase": "vote",
                    "day_number": 2,
                    "night_number": 1,
                    "action_trace": {
                        "task_type": "vote",
                        "parsed_action": {"action_type": "vote", "target_id": "p02"},
                        "world_model_audit": {
                            "possible_worlds": {
                                "top_worlds": [
                                    {
                                        "label": "wrong_world",
                                        "probability": 0.72,
                                        "key_assignments": {
                                            "p02": "seer",
                                            "p03": "werewolf",
                                        },
                                    },
                                    {
                                        "label": "true_world",
                                        "probability": 0.58,
                                        "key_assignments": {
                                            "p02": "werewolf",
                                            "p03": "seer",
                                        },
                                    },
                                ]
                            }
                        },
                    },
                },
            }
        ],
    )

    aggregator = MetricsAggregator()
    aggregator.add_result(result)

    snapshot = aggregator.compute_snapshot()

    assert snapshot.world_model_metrics.true_world_top1_rate == 0.0
    assert snapshot.world_model_metrics.true_world_top3_rate == 1.0
    assert snapshot.world_model_metrics.avg_true_world_rank == 2.0
    assert snapshot.world_model_metrics.world_rank_supported_count == 1
    assert snapshot.world_model_metrics.world_rank_unsupported_count == 0
