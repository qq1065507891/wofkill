"""Candidate materialization workflow tests."""

from __future__ import annotations

from dataclasses import replace

from werewolf_agent.evaluation.feedback_schemas import ImprovementCandidate


class _DraftStore:
    def __init__(self, *, draft_id: str | None = None) -> None:
        self.draft_id = draft_id
        self.drafts: list[tuple[str, dict]] = []
        self.promotions: list[tuple[str, str, dict]] = []
        self.quarantines: list[tuple[str, str, str]] = []

    def save_review_only(self, entry: dict, namespace: str) -> str:
        self.drafts.append((namespace, dict(entry)))
        return self.draft_id or str(entry["candidate_id"])

    def promote(
        self,
        draft_id: str,
        live_namespace: str,
        metadata: dict,
    ) -> str:
        self.promotions.append((draft_id, live_namespace, dict(metadata)))
        return f"{live_namespace}:{draft_id}"

    def quarantine(self, draft_id: str, reason: str, namespace: str = "") -> None:
        self.quarantines.append((namespace, draft_id, reason))


def _candidate(
    *,
    candidate_id: str = "c1",
    target_module: str = "rag",
    payload: dict | None = None,
) -> ImprovementCandidate:
    return ImprovementCandidate(
        candidate_id=candidate_id,
        source_diagnosis_ids=["d1"],
        target_module=target_module,
        operation="review_or_rewrite",
        priority="high",
        prompt_safe_payload=payload or {"recommended_use": "Review recurring issue."},
        audit_evidence={"trace_ids": ["t1"]},
    )


def _approved_store(candidate: ImprovementCandidate):
    from werewolf_agent.evaluation.candidate_store import (
        CandidateStatus,
        InMemoryCandidateStore,
    )

    store = InMemoryCandidateStore(clock=lambda: "2026-06-17T10:00:00")
    store.add(candidate)
    store.transition(
        candidate.candidate_id,
        CandidateStatus.APPROVED,
        reviewer="moderator",
        notes="safe",
    )
    return store


def test_materializer_writes_rag_candidate_to_review_only_draft_namespace() -> None:
    from werewolf_agent.evaluation.candidate_materializer import (
        CandidateMaterializer,
    )
    from werewolf_agent.evaluation.candidate_store import CandidateStatus

    candidate = _candidate(target_module="rag")
    candidate_store = _approved_store(candidate)
    rag_store = _DraftStore()
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        rag_store=rag_store,
        draft_namespace="candidate_review",
        live_namespace="live",
        clock=lambda: "2026-06-17T10:05:00",
    )

    result = materializer.materialize_draft(candidate.candidate_id)

    assert result.status == "materialized"
    assert rag_store.drafts[0][0] == "candidate_review"
    assert rag_store.drafts[0][1]["review_status"] == "review_only"
    assert rag_store.drafts[0][1]["prompt_safe_payload"] == candidate.prompt_safe_payload
    assert candidate_store.get(candidate.candidate_id).status == CandidateStatus.MATERIALIZED


def test_materializer_accepts_approved_candidate_object_and_custom_namespace() -> None:
    from werewolf_agent.evaluation.candidate_materializer import CandidateMaterializer
    from werewolf_agent.evaluation.candidate_store import CandidateStatus, InMemoryCandidateStore

    candidate = replace(_candidate(target_module="rag"), review_status="approved")
    candidate_store = InMemoryCandidateStore(clock=lambda: "2026-06-17T10:00:00")
    rag_store = _DraftStore()
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        rag_store=rag_store,
        draft_namespace="default_draft",
    )

    result = materializer.materialize_draft(candidate, namespace="custom_review")

    assert result.status == "materialized"
    assert result.draft_namespace == "custom_review"
    assert rag_store.drafts[0][0] == "custom_review"
    assert candidate_store.get(candidate.candidate_id).status == CandidateStatus.MATERIALIZED


def test_materializer_writes_reflection_candidate_to_review_only_draft_namespace() -> None:
    from werewolf_agent.evaluation.candidate_materializer import CandidateMaterializer

    candidate = _candidate(candidate_id="c_ref", target_module="reflection")
    candidate_store = _approved_store(candidate)
    reflection_store = _DraftStore()
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        reflection_store=reflection_store,
        draft_namespace="candidate_review",
        live_namespace="live",
    )

    result = materializer.materialize_draft(candidate.candidate_id)

    assert result.status == "materialized"
    assert reflection_store.drafts[0][0] == "candidate_review"
    assert reflection_store.drafts[0][1]["quality_status"] == "review_only"


def test_materializer_rejects_prompt_unsafe_candidate_without_writing_draft() -> None:
    from werewolf_agent.evaluation.candidate_materializer import CandidateMaterializer
    from werewolf_agent.evaluation.candidate_store import (
        CandidateStatus,
        InMemoryCandidateStore,
    )

    unsafe = _candidate(
        payload={"recommended_use": "Use target_role and p03 ground_truth."},
    )
    candidate_store = InMemoryCandidateStore(clock=lambda: "2026-06-17T10:00:00")
    candidate_store.add(unsafe)
    rag_store = _DraftStore()
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        rag_store=rag_store,
    )

    result = materializer.materialize_draft(unsafe.candidate_id)

    assert result.status == "rejected"
    assert rag_store.drafts == []
    assert candidate_store.get(unsafe.candidate_id).status == CandidateStatus.REJECTED


def test_materializer_quarantines_failed_gate_and_never_promotes() -> None:
    from werewolf_agent.evaluation.candidate_materializer import CandidateMaterializer
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig,
        RegressionGate,
    )

    candidate = _candidate(target_module="rag")
    candidate_store = _approved_store(candidate)
    rag_store = _DraftStore()
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        rag_store=rag_store,
        draft_namespace="candidate_review",
        live_namespace="live",
    )
    materializer.materialize_draft(candidate.candidate_id)
    gate_report = RegressionGate().evaluate(
        CandidateRegressionConfig(candidate_id=candidate.candidate_id),
        baseline_metrics={"hidden_info_leak_rate": 0.0},
        candidate_metrics={"hidden_info_leak_rate": 0.2},
        prompt_safe=True,
    )

    result = materializer.rollback(candidate.candidate_id, "regression_gate_failed")

    assert result.status == "rolled_back"
    assert rag_store.promotions == []
    assert rag_store.quarantines == [
        ("candidate_review", candidate.candidate_id, "regression_gate_failed")
    ]


def test_materializer_uses_returned_draft_id_for_rollback_and_promotion() -> None:
    from werewolf_agent.evaluation.candidate_materializer import CandidateMaterializer
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig,
        RegressionGate,
    )

    candidate = _candidate(target_module="rag")
    candidate_store = _approved_store(candidate)
    rag_store = _DraftStore(draft_id="draft:123")
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        rag_store=rag_store,
        draft_namespace="candidate_review",
        live_namespace="live",
    )
    materializer.materialize_draft(candidate.candidate_id)
    pass_report = RegressionGate().evaluate(
        CandidateRegressionConfig(candidate_id=candidate.candidate_id),
        baseline_metrics={"hidden_info_leak_rate": 0.0},
        candidate_metrics={"hidden_info_leak_rate": 0.0},
        prompt_safe=True,
    )

    promoted = materializer.promote(candidate.candidate_id, pass_report)

    assert promoted.status == "promoted"
    assert rag_store.promotions[0][0] == "draft:123"

    second_candidate = _candidate(candidate_id="c2", target_module="rag")
    second_store = _approved_store(second_candidate)
    second_rag_store = _DraftStore(draft_id="draft:456")
    second_materializer = CandidateMaterializer(
        candidate_store=second_store,
        rag_store=second_rag_store,
        draft_namespace="candidate_review",
    )
    second_materializer.materialize_draft(second_candidate.candidate_id)

    rolled_back = second_materializer.rollback(
        second_candidate.candidate_id,
        "regression_gate_failed",
    )

    assert rolled_back.status == "rolled_back"
    assert second_rag_store.quarantines == [
        ("candidate_review", "draft:456", "regression_gate_failed")
    ]


def test_materializer_does_not_promote_before_draft_materialization() -> None:
    from werewolf_agent.evaluation.candidate_materializer import CandidateMaterializer
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig,
        RegressionGate,
    )

    candidate = _candidate(target_module="rag")
    candidate_store = _approved_store(candidate)
    rag_store = _DraftStore()
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        rag_store=rag_store,
    )
    pass_report = RegressionGate().evaluate(
        CandidateRegressionConfig(candidate_id=candidate.candidate_id),
        baseline_metrics={"hidden_info_leak_rate": 0.0},
        candidate_metrics={"hidden_info_leak_rate": 0.0},
        prompt_safe=True,
    )

    result = materializer.promote(candidate.candidate_id, pass_report)

    assert result.status == "not_materialized"
    assert rag_store.promotions == []


def test_materializer_promotes_passed_gate_with_source_and_regression_metadata() -> None:
    from werewolf_agent.evaluation.candidate_materializer import CandidateMaterializer
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig,
        RegressionGate,
    )

    candidate = _candidate(target_module="rag")
    candidate_store = _approved_store(candidate)
    rag_store = _DraftStore()
    materializer = CandidateMaterializer(
        candidate_store=candidate_store,
        rag_store=rag_store,
        draft_namespace="candidate_review",
        live_namespace="live",
    )
    materializer.materialize_draft(candidate.candidate_id)
    gate_report = RegressionGate().evaluate(
        CandidateRegressionConfig(candidate_id=candidate.candidate_id),
        baseline_metrics={
            "hidden_info_leak_rate": 0.0,
            "illegal_action_rate": 0.0,
            "vote_quality": 0.8,
            "good_win_rate": 0.5,
            "harmful_transfer_rate": 0.0,
        },
        candidate_metrics={
            "hidden_info_leak_rate": 0.0,
            "illegal_action_rate": 0.0,
            "vote_quality": 0.81,
            "good_win_rate": 0.55,
            "harmful_transfer_rate": 0.0,
        },
        prompt_safe=True,
    )

    result = materializer.promote(candidate.candidate_id, gate_report)

    assert result.status == "promoted"
    assert rag_store.quarantines == []
    assert rag_store.promotions[0][0] == candidate.candidate_id
    assert rag_store.promotions[0][1] == "live"
    metadata = rag_store.promotions[0][2]
    assert metadata["candidate_id"] == candidate.candidate_id
    assert metadata["source_diagnosis_ids"] == ["d1"]
    assert metadata["regression"]["passed"] is True
