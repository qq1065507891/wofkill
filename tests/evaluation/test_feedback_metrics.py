"""Feedback-loop module attribution metric tests."""

from __future__ import annotations

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    MetricSupport,
    ModuleExposure,
)


def _trace(trace_id: str, exposures: list[ModuleExposure]) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id=trace_id,
        game_id="g_feedback",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        module_exposures=exposures,
    )


def test_summarize_module_attribution_counts_exposure_support_and_rates() -> None:
    from werewolf_agent.evaluation.feedback_metrics import summarize_module_attribution

    traces = [
        _trace(
            "t1",
            [
                ModuleExposure(
                    module="rag",
                    item_id="rag_1",
                    prompt_visible=True,
                    cited_by_decision=True,
                    aligned_with_decision=True,
                ),
                ModuleExposure(
                    module="rag",
                    item_id="missing_source",
                    support=MetricSupport.UNSUPPORTED,
                ),
            ],
        ),
        _trace(
            "t2",
            [
                ModuleExposure(
                    module="rag",
                    item_id="rag_2",
                    prompt_visible=True,
                    cited_by_decision=False,
                    aligned_with_decision=True,
                ),
            ],
        ),
    ]

    summary = summarize_module_attribution(traces)

    rag = summary["rag"]
    assert rag.exposure_count == 3
    assert rag.supported_count == 2
    assert rag.unsupported_count == 1
    assert rag.prompt_visible_count == 2
    assert rag.cited_count == 1
    assert rag.aligned_count == 2
    assert rag.prompt_visible_rate == 1.0
    assert rag.citation_rate == 0.5
    assert rag.alignment_rate == 1.0


def test_summarize_module_attribution_tracks_harmful_transfer_placeholders() -> None:
    from werewolf_agent.evaluation.feedback_metrics import summarize_module_attribution

    traces = [
        _trace(
            "t1",
            [
                ModuleExposure(
                    module="reflection",
                    item_id="reflection_1",
                    metadata={"harmful_transfer": True},
                ),
                ModuleExposure(
                    module="reflection",
                    item_id="reflection_2",
                    metadata={"harmful": True},
                ),
                ModuleExposure(
                    module="reflection",
                    item_id="reflection_3",
                    metadata={"harmful_transfer": False},
                ),
            ],
        ),
    ]

    reflection = summarize_module_attribution(traces)["reflection"]

    assert reflection.exposure_count == 3
    assert reflection.harmful_count == 2
    assert reflection.harmful_rate == 2 / 3


def test_summarize_module_attribution_returns_empty_summary_for_no_exposures() -> None:
    from werewolf_agent.evaluation.feedback_metrics import summarize_module_attribution

    assert summarize_module_attribution([_trace("t1", [])]) == {}


def test_unsupported_markers_do_not_count_as_rate_denominators() -> None:
    from werewolf_agent.evaluation.feedback_metrics import summarize_module_attribution

    trace = _trace(
        "t1",
        [
            ModuleExposure(
                module="rag",
                item_id="missing_source",
                support=MetricSupport.UNSUPPORTED,
            ),
        ],
    )

    rag = summarize_module_attribution([trace])["rag"]

    assert rag.exposure_count == 1
    assert rag.supported_count == 0
    assert rag.unsupported_count == 1
    assert rag.prompt_visible_rate == 0.0
    assert rag.citation_rate == 0.0
    assert rag.alignment_rate == 0.0
