"""Optional LangSmith feedback exporter tests."""

from __future__ import annotations

import builtins

import json

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    FailureDiagnosis,
    ImprovementCandidate,
    ModuleExposure,
)


def _report():
    from werewolf_agent.evaluation.feedback_report import build_feedback_report

    trace = EvaluationTrace(
        trace_id="t1",
        game_id="g_langsmith",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        module_exposures=[ModuleExposure(module="rag", item_id="rag_1")],
    )
    return build_feedback_report(
        report_id="feedback_langsmith",
        batch_id="batch_langsmith",
        traces=[trace],
        generated_at="2026-06-16T12:00:00",
    )


def test_langsmith_exporter_imports_without_langsmith_dependency() -> None:
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    exporter = LangSmithFeedbackExporter(project_name="local-feedback")

    payload = exporter.build_payload(_report())

    assert payload["project_name"] == "local-feedback"
    assert payload["runs"][0]["name"] == "feedback_langsmith"
    assert payload["runs"][0]["outputs"]["trace_count"] == 1
    assert payload["runs"][0]["metadata"]["batch_id"] == "batch_langsmith"


def test_langsmith_payload_build_does_not_import_langsmith(monkeypatch) -> None:
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "langsmith" or name.startswith("langsmith."):
            raise AssertionError("build_payload must not import langsmith")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    payload = LangSmithFeedbackExporter().build_payload(_report())

    assert payload["runs"][0]["run_type"] == "chain"


def test_langsmith_export_uses_injected_client_without_importing_langsmith(monkeypatch) -> None:
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    calls = []

    class FakeClient:
        def create_run(self, **kwargs):
            calls.append(kwargs)
            return {"id": "run_1"}

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "langsmith" or name.startswith("langsmith."):
            raise AssertionError("injected client path must not import langsmith")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = LangSmithFeedbackExporter().export(_report(), client=FakeClient())

    assert result == [{"id": "run_1"}]
    assert calls[0]["name"] == "feedback_langsmith"


def test_langsmith_export_uses_falsy_injected_client_without_importing_langsmith(monkeypatch) -> None:
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    calls = []

    class FalsyClient:
        def __bool__(self):
            return False

        def create_run(self, **kwargs):
            calls.append(kwargs)
            return {"id": "run_falsy"}

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "langsmith" or name.startswith("langsmith."):
            raise AssertionError("falsy injected client path must not import langsmith")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = LangSmithFeedbackExporter().export(_report(), client=FalsyClient())

    assert result == [{"id": "run_falsy"}]
    assert calls[0]["name"] == "feedback_langsmith"


def test_langsmith_export_reports_missing_dependency_only_when_export_called(monkeypatch) -> None:
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    original_import = builtins.__import__

    def missing_langsmith(name, *args, **kwargs):
        if name == "langsmith" or name.startswith("langsmith."):
            raise ImportError("missing langsmith")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_langsmith)

    exporter = LangSmithFeedbackExporter()
    exporter.build_payload(_report())

    try:
        exporter.export(_report())
    except RuntimeError as exc:
        assert "LangSmith is not installed" in str(exc)
    else:
        raise AssertionError("export without LangSmith dependency should fail clearly")


def test_langsmith_payload_scrubs_private_audit_fields_by_default() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    report = build_feedback_report(
        report_id="feedback_private",
        batch_id="batch_langsmith",
        traces=[
            EvaluationTrace(
                trace_id="t_private",
                game_id="g_langsmith",
                player_id="p01",
                role="seer",
                faction="good",
                phase="vote",
                task_type="vote",
                module_exposures=[ModuleExposure(module="rag", item_id="rag_1")],
            )
        ],
        candidates=[
            ImprovementCandidate(
                candidate_id="c_private",
                source_diagnosis_ids=["d_private"],
                target_module="rag",
                operation="review_or_rewrite",
                priority="high",
                prompt_safe_payload={"recommended_use": "Review recurring RAG issue."},
                audit_evidence={
                    "target_role": "werewolf",
                    "target_faction": "werewolf",
                    "key_assignments": {"p02": "werewolf"},
                    "diagnosis_count": 1,
                },
            )
        ],
    )

    payload = LangSmithFeedbackExporter().build_payload(report)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "target_role" not in serialized
    assert "target_faction" not in serialized
    assert "key_assignments" not in serialized
    assert "diagnosis_count" in serialized


def test_langsmith_payload_scrubs_diagnoses_refs_and_moderator_notes() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    trace = EvaluationTrace(
        trace_id="p01:target_role:werewolf",
        game_id="g_langsmith",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        source_refs=["target_faction:werewolf", "trace:p01"],
        module_exposures=[ModuleExposure(module="rag", item_id="rag_1")],
    )
    report = build_feedback_report(
        report_id="feedback_scrub",
        batch_id="batch_langsmith",
        traces=[trace],
        diagnoses=[
            FailureDiagnosis(
                diagnosis_id="d_private",
                trace_id="p01",
                category="hidden_info_leak",
                severity="high",
                primary_module="dialogue",
                evidence_refs=["target_role:werewolf", "trace:p01"],
                explanation="target_faction was werewolf for p01",
            )
        ],
        candidates=[
            ImprovementCandidate(
                candidate_id="c_private",
                source_diagnosis_ids=["d_private"],
                target_module="dialogue",
                operation="harden_prompt_boundary",
                priority="high",
                prompt_safe_payload={"recommended_use": "Harden boundary."},
                moderator_notes="p01 leaked target_role werewolf",
            )
        ],
    )

    payload = LangSmithFeedbackExporter().build_payload(report)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "p01" not in serialized
    assert "target_role" not in serialized
    assert "target_faction" not in serialized
    assert "werewolf" not in serialized
    assert "moderator_notes" not in serialized


def test_langsmith_payload_includes_redacted_monitoring_exposures() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report
    from werewolf_agent.evaluation.langsmith_exporter import LangSmithFeedbackExporter

    trace = EvaluationTrace(
        trace_id="g_langsmith:p01:vote:D2:N1:vote:4",
        game_id="g_langsmith",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        module_exposures=[
            ModuleExposure(
                module="skill_tool_calls",
                item_id="push_vote",
                score=1.0,
                prompt_visible=True,
                metadata={
                    "call_kind": "skill",
                    "status": "success",
                    "success": True,
                    "decision_usage": "prompt_injected",
                    "target_role": "werewolf",
                },
            ),
            ModuleExposure(
                module="prompt_injections",
                item_id="public_summary",
                score=1.0,
                prompt_visible=True,
                metadata={
                    "module_name": "public_summary",
                    "field_path": "public_summary",
                    "injection_kind": "text_section",
                    "injected": True,
                    "content_hash": "abc123",
                    "raw_content": "p01 saw target_role werewolf",
                },
            ),
        ],
    )
    report = build_feedback_report(
        report_id="feedback_monitoring",
        batch_id="batch_langsmith",
        traces=[trace],
    )

    payload = LangSmithFeedbackExporter().build_payload(report)

    monitoring = payload["runs"][0]["outputs"]["monitoring_exposures"]
    assert {
        (row["module"], row["item_id"])
        for row in monitoring
    } == {
        ("skill_tool_calls", "push_vote"),
        ("prompt_injections", "public_summary"),
    }
    skill_row = next(row for row in monitoring if row["module"] == "skill_tool_calls")
    prompt_row = next(row for row in monitoring if row["module"] == "prompt_injections")
    assert skill_row["metadata"]["status"] == "success"
    assert skill_row["metadata"]["success"] is True
    assert prompt_row["metadata"]["content_hash"] == "abc123"
    assert "trace_hash" in skill_row
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "g_langsmith:p01" not in serialized
    assert "p01" not in serialized
    assert "target_role" not in serialized
    assert "werewolf" not in serialized
    assert "raw_content" not in serialized


def test_langsmith_payload_includes_cache_stats_r8() -> None:
    """R8: cache_*_tokens (R7) 上送 LangSmith outputs.cache_stats.

    FeedbackReport.cost_metrics 持有 CostMetrics; build_payload 后
    outputs.cache_stats 含 4 个数字字段 (prompt/completion/creation/read) + 1 个 ratio.
    cache_hit_ratio 在 [0, 1] 闭区间内, 验证 base 公式.
    """
    from dataclasses import asdict
    from werewolf_agent.evaluation.langsmith_exporter import (
        LangSmithFeedbackExporter,
    )
    from werewolf_agent.evaluation.metric_aggregation import CostMetrics

    cost = CostMetrics(
        total_prompt_tokens=200,
        total_completion_tokens=80,
        total_cache_creation_tokens=100,
        total_cache_read_tokens=80,
    )
    report = _build_report(cost_metrics=cost)
    exporter = LangSmithFeedbackExporter(project_name="agent-feedback")
    payload = exporter.build_payload(report)
    cache_stats = payload["runs"][0]["outputs"]["cache_stats"]
    # 4 个数字 + 1 ratio.
    assert cache_stats["prompt_tokens"] == 200
    assert cache_stats["completion_tokens"] == 80
    assert cache_stats["cache_creation_tokens"] == 100
    assert cache_stats["cache_read_tokens"] == 80
    # ratio = 80 / (200 + 80) = 0.2857...
    assert 0.0 <= cache_stats["cache_hit_ratio"] <= 1.0
    assert abs(cache_stats["cache_hit_ratio"] - 80 / 280) < 1e-6
    # R8: cache_stats 与 monitoring_exposures 平级, 不在 _scrub_private_fields 递归范围.
    assert "cache_stats" in payload["runs"][0]["outputs"]


def test_langsmith_payload_cache_stats_zero_when_cost_metrics_none() -> None:
    """R8: cost_metrics=None 时 cache_stats 全 0, 不崩."""
    from werewolf_agent.evaluation.langsmith_exporter import (
        LangSmithFeedbackExporter,
    )

    report = _build_report(cost_metrics=None)
    exporter = LangSmithFeedbackExporter(project_name="agent-feedback")
    payload = exporter.build_payload(report)
    cache_stats = payload["runs"][0]["outputs"]["cache_stats"]
    assert cache_stats["prompt_tokens"] == 0
    assert cache_stats["completion_tokens"] == 0
    assert cache_stats["cache_creation_tokens"] == 0
    assert cache_stats["cache_read_tokens"] == 0
    assert cache_stats["cache_hit_ratio"] == 0.0


def _build_report(cost_metrics=None) -> "FeedbackReport":
    """Helper: 构造一个最小 FeedbackReport (含 cost_metrics 可选)."""
    from werewolf_agent.evaluation.feedback_report import FeedbackReport

    return FeedbackReport(
        report_id="r8-audit",
        batch_id="b8",
        trace_count=0,
        module_metrics={},
        cost_metrics=cost_metrics,
    )
