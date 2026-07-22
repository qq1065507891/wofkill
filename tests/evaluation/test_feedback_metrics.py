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


def test_llm_cache_summary_built_from_cost_metrics_r9a() -> None:
    """R9a: _llm_cache_summary(cost) 把 CostMetrics 折入 ModuleAttributionSummary 模板.

    验证 exposure_count=1, supported_count=1, cache_creation/cache_read 透传,
    cache_hit_ratio 公式 = cache_read / (cache_creation + cache_read).
    """
    from werewolf_agent.evaluation.feedback_metrics import (
        ModuleAttributionSummary,
        _llm_cache_summary,
    )
    from werewolf_agent.evaluation.metric_aggregation import CostMetrics

    cost = CostMetrics(
        total_prompt_tokens=200,
        total_completion_tokens=80,
        total_cache_creation_tokens=100,
        total_cache_read_tokens=80,
    )
    summary = _llm_cache_summary(cost)
    assert isinstance(summary, ModuleAttributionSummary)
    assert summary.module == "llm_cache"
    assert summary.exposure_count == 1
    assert summary.supported_count == 1
    assert summary.cache_creation_tokens == 100
    assert summary.cache_read_tokens == 80
    # ratio = 80 / (100 + 80) = 0.4444...
    assert abs(summary.cache_hit_ratio - 80 / 180) < 1e-6
    # 其它 exposure 字段默认 0 (cache 不是 exposure 路径).
    assert summary.prompt_visible_count == 0
    assert summary.cited_count == 0
    assert summary.harmful_count == 0


def test_llm_cache_summary_zero_on_no_cost_metrics() -> None:
    """R9a: cost=None 时 _llm_cache_summary 返回空 entry, 不崩."""
    from werewolf_agent.evaluation.feedback_metrics import (
        ModuleAttributionSummary,
        _llm_cache_summary,
    )

    summary = _llm_cache_summary(None)
    assert isinstance(summary, ModuleAttributionSummary)
    assert summary.module == "llm_cache"
    assert summary.exposure_count == 0
    assert summary.cache_creation_tokens == 0
    assert summary.cache_read_tokens == 0
    assert summary.cache_hit_ratio == 0.0


def test_module_metric_to_dict_projects_cache_fields_r9a() -> None:
    """R9a: module_metric_to_dict 自动把 cache_*_tokens + cache_hit_ratio 投影.

    这是 R9a 核心契约: 现有 _module_metric_to_dict 不改, 但因为是
    `summary.__dict__` 自动展开, 新加字段自动出现在 dict.
    """
    from werewolf_agent.evaluation.feedback_metrics import (
        ModuleAttributionSummary,
        _llm_cache_summary,
    )
    from werewolf_agent.evaluation.feedback_report_serialization import (
        module_metric_to_dict,
    )
    from werewolf_agent.evaluation.metric_aggregation import CostMetrics

    cost = CostMetrics(
        total_prompt_tokens=200,
        total_completion_tokens=80,
        total_cache_creation_tokens=100,
        total_cache_read_tokens=80,
    )
    summary = _llm_cache_summary(cost)
    serialized = module_metric_to_dict(summary)
    assert serialized["module"] == "llm_cache"
    assert serialized["cache_creation_tokens"] == 100
    assert serialized["cache_read_tokens"] == 80
    assert abs(serialized["cache_hit_ratio"] - 80 / 180) < 1e-6
