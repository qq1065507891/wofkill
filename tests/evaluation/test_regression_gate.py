"""Candidate regression gate tests."""

from __future__ import annotations


def test_regression_gate_passes_when_candidate_metrics_are_within_tolerances() -> None:
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig,
        RegressionGate,
    )

    report = RegressionGate().evaluate(
        CandidateRegressionConfig(
            candidate_id="c1",
            target_faction="good",
            vote_quality_drop_tolerance=0.05,
            target_faction_win_rate_drop_tolerance=0.05,
        ),
        baseline_metrics={
            "hidden_info_leak_rate": 0.0,
            "illegal_action_rate": 0.0,
            "vote_quality": 0.8,
            "good_win_rate": 0.6,
            "harmful_transfer_rate": 0.0,
        },
        candidate_metrics={
            "hidden_info_leak_rate": 0.0,
            "illegal_action_rate": 0.0,
            "vote_quality": 0.77,
            "good_win_rate": 0.58,
            "harmful_transfer_rate": 0.0,
        },
        prompt_safe=True,
    )

    assert report.passed is True
    assert report.blocked_reasons == []
    assert all(check.passed for check in report.checks)


def test_regression_gate_blocks_on_all_required_regression_types() -> None:
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig,
        RegressionGate,
    )

    report = RegressionGate().evaluate(
        CandidateRegressionConfig(
            candidate_id="c1",
            target_faction="good",
            vote_quality_drop_tolerance=0.01,
            target_faction_win_rate_drop_tolerance=0.01,
        ),
        baseline_metrics={
            "hidden_info_leak_rate": 0.0,
            "illegal_action_rate": 0.0,
            "vote_quality": 0.9,
            "good_win_rate": 0.7,
            "harmful_transfer_rate": 0.0,
        },
        candidate_metrics={
            "hidden_info_leak_rate": 0.1,
            "illegal_action_rate": 0.2,
            "vote_quality": 0.7,
            "good_win_rate": 0.5,
            "harmful_transfer_rate": 0.3,
        },
        prompt_safe=False,
    )

    assert report.passed is False
    assert set(report.blocked_reasons) == {
        "hidden_info_leak_increased",
        "illegal_action_increased",
        "vote_quality_dropped",
        "good_win_rate_dropped",
        "harmful_transfer_increased",
        "prompt_safety_failed",
    }
    by_metric = {delta.metric: delta for delta in report.metric_deltas}
    assert by_metric["vote_quality"].candidate_minus_baseline == -0.2
    assert by_metric["vote_quality"].regression_amount == 0.2
    assert by_metric["hidden_info_leak_rate"].candidate_minus_baseline == 0.1
    assert by_metric["hidden_info_leak_rate"].regression_amount == 0.1


def test_regression_gate_json_uses_explicit_direction_fields() -> None:
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig,
        RegressionGate,
    )

    report = RegressionGate().evaluate(
        CandidateRegressionConfig(candidate_id="c1"),
        baseline_metrics={"illegal_action_count": 1.0},
        candidate_metrics={"illegal_action_count": 2.0},
        prompt_safe=True,
    )
    data = report.to_json_dict()

    delta = data["metric_deltas"][0]
    assert delta["candidate_minus_baseline"] == 1.0
    assert delta["higher_is_better"] is False
    assert delta["regression_amount"] == 1.0
    assert data["checks"][0]["passed"] is False
