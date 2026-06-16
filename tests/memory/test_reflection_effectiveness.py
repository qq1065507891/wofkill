"""Reflection effectiveness metrics from feedback traces."""

from __future__ import annotations

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    MetricSupport,
    ModuleExposure,
)


def _trace(trace_id: str, exposures: list[ModuleExposure]) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id=trace_id,
        game_id="g_reflection_eval",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        module_exposures=exposures,
    )


def _reflection(
    entry_id: str,
    *,
    prompt_visible: bool = True,
    cited: bool = False,
    aligned: bool = False,
    stale: bool = False,
    harmful: bool = False,
    support: MetricSupport = MetricSupport.SUPPORTED,
) -> ModuleExposure:
    return ModuleExposure(
        module="reflection",
        item_id=entry_id,
        prompt_visible=prompt_visible,
        cited_by_decision=cited,
        aligned_with_decision=aligned,
        support=support,
        metadata={
            "stale": stale,
            "harmful_transfer": harmful,
        },
    )


def test_evaluate_reflection_effectiveness_counts_injection_citation_alignment_stale_and_harmful() -> None:
    from werewolf_agent.memory.reflection_effectiveness import (
        evaluate_reflection_effectiveness,
    )

    traces = [
        _trace(
            "t1",
            [
                _reflection("reflection_good", cited=True, aligned=True),
                _reflection("reflection_stale", stale=True),
                ModuleExposure(module="rag", item_id="rag_1", prompt_visible=True),
            ],
        ),
        _trace(
            "t2",
            [
                _reflection("reflection_harmful", cited=True, harmful=True),
                _reflection(
                    "missing_source",
                    prompt_visible=False,
                    support=MetricSupport.UNSUPPORTED,
                ),
            ],
        ),
    ]

    report = evaluate_reflection_effectiveness(traces)

    assert report.total_reflection_exposures == 4
    assert report.supported_reflection_exposures == 3
    assert report.unsupported_reflection_exposures == 1
    assert report.injected_count == 3
    assert report.cited_count == 2
    assert report.aligned_count == 1
    assert report.stale_count == 1
    assert report.harmful_count == 1
    assert report.injection_rate == 1.0
    assert report.citation_rate == 2 / 3
    assert report.alignment_rate == 1 / 3
    assert report.stale_rate == 1 / 3
    assert report.harmful_rate == 1 / 3
    assert report.by_entry["reflection_good"].aligned_count == 1


def test_evaluate_reflection_effectiveness_generates_candidate_hints_for_repeated_no_effect_and_harmful_cards() -> None:
    from werewolf_agent.memory.reflection_effectiveness import (
        evaluate_reflection_effectiveness,
    )

    traces = [
        _trace("t1", [_reflection("reflection_no_effect")]),
        _trace("t2", [_reflection("reflection_no_effect")]),
        _trace("t3", [_reflection("reflection_harmful", harmful=True)]),
    ]

    report = evaluate_reflection_effectiveness(
        traces,
        no_effect_threshold=2,
        harmful_threshold=1,
    )

    hints = {hint.entry_id: hint for hint in report.candidate_hints}
    assert hints["reflection_no_effect"].reason == "repeated_no_effect"
    assert hints["reflection_no_effect"].suggested_operation == "downgrade_or_rewrite"
    assert hints["reflection_harmful"].reason == "harmful_transfer"
    assert hints["reflection_harmful"].suggested_operation == "quarantine_or_rewrite"
    assert "p01" not in hints["reflection_harmful"].prompt_safe_note


def test_non_reflection_exposures_do_not_affect_reflection_effectiveness() -> None:
    from werewolf_agent.memory.reflection_effectiveness import (
        evaluate_reflection_effectiveness,
    )

    trace = _trace(
        "t1",
        [
            ModuleExposure(
                module="rag",
                item_id="rag_harmful",
                prompt_visible=True,
                cited_by_decision=True,
                aligned_with_decision=False,
                metadata={"harmful_transfer": True, "stale": True},
            )
        ],
    )

    report = evaluate_reflection_effectiveness([trace])

    assert report.total_reflection_exposures == 0
    assert report.by_entry == {}
    assert report.candidate_hints == []


def test_reflection_effectiveness_only_treats_boolean_true_as_stale_or_harmful() -> None:
    from werewolf_agent.memory.reflection_effectiveness import (
        evaluate_reflection_effectiveness,
    )

    trace = _trace(
        "t1",
        [
            ModuleExposure(
                module="reflection",
                item_id="reflection_dirty_metadata",
                prompt_visible=True,
                metadata={"stale": "true", "harmful_transfer": "true"},
            )
        ],
    )

    report = evaluate_reflection_effectiveness([trace])

    assert report.stale_count == 0
    assert report.harmful_count == 0
    assert report.candidate_hints == []
