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
