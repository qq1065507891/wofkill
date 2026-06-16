"""Feedback-loop compact report tests."""

from __future__ import annotations

import json

from werewolf_agent.evaluation.ablation import OfflineTraceAblationRunner
from werewolf_agent.evaluation.feedback_schemas import (
    DecisionOutcome,
    EvaluationTrace,
    FailureDiagnosis,
    ImprovementCandidate,
    ModuleExposure,
)


def _trace(trace_id: str, exposures: list[ModuleExposure]) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id=trace_id,
        game_id="g_report",
        player_id="p01",
        role="seer",
        faction="good",
        phase="vote",
        task_type="vote",
        module_exposures=exposures,
        outcome=DecisionOutcome(target_role="werewolf", target_faction="werewolf"),
    )


def test_feedback_report_serializes_compact_feedback_summary() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report

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
                ModuleExposure(module="reflection", item_id="reflection_1"),
            ],
        )
    ]
    diagnoses = [
        FailureDiagnosis(
            diagnosis_id="d1",
            trace_id="t1",
            category="rag_harmful_transfer",
            severity="medium",
            primary_module="rag",
            evidence_refs=["trace:t1", "exposure:rag_1"],
            explanation="RAG exposure was flagged.",
        )
    ]
    candidates = [
        ImprovementCandidate(
            candidate_id="c1",
            source_diagnosis_ids=["d1"],
            target_module="rag",
            operation="review_or_rewrite",
            priority="medium",
            prompt_safe_payload={"recommended_use": "Review recurring RAG issue."},
            audit_evidence={"diagnosis_count": 1},
        )
    ]
    ablation = OfflineTraceAblationRunner().run(traces, removed_modules=["rag"])

    report = build_feedback_report(
        report_id="feedback_1",
        batch_id="batch_1",
        traces=traces,
        diagnoses=diagnoses,
        candidates=candidates,
        ablation_reports=[ablation],
        generated_at="2026-06-16T12:00:00",
    )
    data = report.to_json_dict()

    assert data["schema_version"] == 1
    assert data["report_id"] == "feedback_1"
    assert data["batch_id"] == "batch_1"
    assert data["trace_count"] == 1
    assert data["module_metrics"]["rag"]["exposure_count"] == 1
    assert data["module_metrics"]["rag"]["citation_rate"] == 1.0
    assert data["diagnoses"][0]["diagnosis_id"] == "d1"
    assert data["failure_clusters"][0]["category"] == "rag_harmful_transfer"
    assert data["failure_clusters"][0]["count"] == 1
    assert data["regression_summary"]["seed_count"] == 0
    assert data["regression_summary"]["status"] == "not_configured"
    assert data["candidates"][0]["candidate_id"] == "c1"
    assert data["ablations"][0]["removed_modules"] == ["rag"]
    assert data["ablations"][0]["unsupported_metrics"] == {
        "causal_decision_delta": "offline_trace_mode",
        "live_win_rate_delta": "offline_trace_mode",
    }

    encoded = report.to_json()
    decoded = json.loads(encoded)
    assert decoded == data


def test_feedback_report_clusters_failures_and_summarizes_regression_seeds() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report

    diagnoses = [
        FailureDiagnosis(
            diagnosis_id="d1",
            trace_id="t1",
            category="rag_harmful_transfer",
            severity="medium",
            primary_module="rag",
            evidence_refs=["trace:t1"],
        ),
        FailureDiagnosis(
            diagnosis_id="d2",
            trace_id="t2",
            category="rag_harmful_transfer",
            severity="medium",
            primary_module="rag",
            evidence_refs=["trace:t2"],
        ),
    ]
    candidates = [
        ImprovementCandidate(
            candidate_id="c1",
            source_diagnosis_ids=["d1", "d2"],
            target_module="rag",
            operation="review_or_rewrite",
            priority="high",
            prompt_safe_payload={"recommended_use": "Review recurring RAG issue."},
            regression_seed_set=[101, 202, 101],
        )
    ]

    report = build_feedback_report(
        report_id="feedback_clusters",
        batch_id="batch_1",
        traces=[],
        diagnoses=diagnoses,
        candidates=candidates,
    )
    data = report.to_json_dict()

    assert data["failure_clusters"] == [
        {
            "cluster_id": "rag:rag_harmful_transfer:medium",
            "category": "rag_harmful_transfer",
            "primary_module": "rag",
            "severity": "medium",
            "count": 2,
            "diagnosis_ids": ["d1", "d2"],
            "trace_refs": ["trace:t1", "trace:t2"],
            "candidate_ids": ["c1"],
        }
    ]
    assert data["regression_summary"] == {
        "status": "pending",
        "seed_count": 2,
        "seed_set": [101, 202],
        "candidate_count": 1,
        "unsupported_metrics": [],
    }


def test_feedback_report_omits_raw_traces_and_hidden_outcome_labels() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report

    report = build_feedback_report(
        report_id="feedback_2",
        batch_id="batch_1",
        traces=[
            _trace(
                "t1",
                [
                    ModuleExposure(
                        module="possible_worlds",
                        item_id="world_1",
                        metadata={"key_assignments": {"p02": "werewolf"}},
                    )
                ],
            )
        ],
    )

    payload = report.to_json_dict()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "traces" not in payload
    assert "target_role" not in serialized
    assert "target_faction" not in serialized
    assert "key_assignments" not in serialized


def test_feedback_report_default_view_scrubs_private_audit_fields() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report

    report = build_feedback_report(
        report_id="feedback_private",
        batch_id="batch_1",
        traces=[
            EvaluationTrace(
                trace_id="p01:target_role",
                game_id="g_report",
                player_id="p01",
                role="seer",
                faction="good",
                phase="vote",
                source_refs=["trace:p01", "target_faction:werewolf"],
            )
        ],
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
                audit_evidence={"target_role": "werewolf"},
                moderator_notes="p01 leaked target_role werewolf",
            )
        ],
    )

    public_payload = report.to_json_dict()
    private_payload = report.to_json_dict(include_private_audit=True)

    assert "p01" not in json.dumps(public_payload, ensure_ascii=False)
    assert "target_role" not in json.dumps(public_payload, ensure_ascii=False)
    assert "target_faction" not in json.dumps(public_payload, ensure_ascii=False)
    assert "werewolf" not in json.dumps(public_payload, ensure_ascii=False)
    assert "target_role" in json.dumps(private_payload, ensure_ascii=False)


def test_feedback_report_serializes_full_game_ablations_separately() -> None:
    from werewolf_agent.evaluation.feedback_report import build_feedback_report
    from werewolf_agent.evaluation.full_game_ablation import (
        FullGameAblationReport,
        FullGameMetricDelta,
    )

    full_game_report = FullGameAblationReport(
        batch_id="batch_1",
        mode="full_game",
        agent_mode="deterministic_fake",
        removed_modules=["rag"],
        pair_count=1,
        metric_deltas={
            "good_win_rate": FullGameMetricDelta(
                metric="good_win_rate",
                baseline=1.0,
                ablated=0.0,
                delta=1.0,
            )
        },
        unsupported_metrics={},
        pairs=[],
    )

    report = build_feedback_report(
        report_id="feedback_full_game",
        batch_id="batch_1",
        traces=[],
        full_game_ablation_reports=[full_game_report],
    )
    data = report.to_json_dict()

    assert data["ablations"] == []
    assert data["full_game_ablations"][0]["mode"] == "full_game"
    assert data["full_game_ablations"][0]["metric_deltas"]["good_win_rate"]["delta"] == 1.0
