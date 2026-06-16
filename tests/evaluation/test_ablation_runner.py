"""Feedback-loop ablation runner tests."""

from __future__ import annotations

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    ModuleExposure,
)


def _trace(trace_id: str, exposures: list[ModuleExposure]) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id=trace_id,
        game_id="g_ablation",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        module_exposures=exposures,
    )


def test_offline_trace_ablation_removes_selected_module_exposures() -> None:
    from werewolf_agent.evaluation.ablation import OfflineTraceAblationRunner

    traces = [
        _trace(
            "t1",
            [
                ModuleExposure(module="rag", item_id="rag_1"),
                ModuleExposure(module="reflection", item_id="reflection_1"),
            ],
        ),
        _trace(
            "t2",
            [
                ModuleExposure(module="rag", item_id="rag_2"),
                ModuleExposure(module="possible_worlds", item_id="world_1"),
            ],
        ),
    ]

    report = OfflineTraceAblationRunner().run(traces, removed_modules=["rag"])

    assert [exposure.module for exposure in traces[0].module_exposures] == [
        "rag",
        "reflection",
    ]
    assert report.mode == "offline_trace"
    assert report.removed_modules == ["rag"]
    assert report.baseline_trace_count == 2
    assert report.ablated_trace_count == 2
    assert report.baseline_module_metrics["rag"].exposure_count == 2
    assert "rag" not in report.ablated_module_metrics
    assert all(
        exposure.module != "rag"
        for trace in report.ablated_traces
        for exposure in trace.module_exposures
    )
    assert {
        exposure.module
        for trace in report.ablated_traces
        for exposure in trace.module_exposures
    } == {"reflection", "possible_worlds"}


def test_offline_trace_ablation_marks_live_causal_metrics_unsupported() -> None:
    from werewolf_agent.evaluation.ablation import OfflineTraceAblationRunner

    report = OfflineTraceAblationRunner().run(
        [_trace("t1", [ModuleExposure(module="rag", item_id="rag_1")])],
        removed_modules=["rag"],
    )

    assert report.unsupported_metrics["live_win_rate_delta"] == "offline_trace_mode"
    assert report.unsupported_metrics["causal_decision_delta"] == "offline_trace_mode"


def test_runner_exposes_pure_offline_trace_ablation_entrypoint() -> None:
    from werewolf_agent.evaluation.runner import run_offline_trace_ablation

    report = run_offline_trace_ablation(
        [_trace("t1", [ModuleExposure(module="reflection", item_id="r1")])],
        removed_modules=["reflection"],
    )

    assert report.removed_modules == ["reflection"]
    assert "reflection" not in report.ablated_module_metrics
