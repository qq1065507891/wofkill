"""Candidate approval workflow store tests."""

from __future__ import annotations

import pytest

from werewolf_agent.evaluation.feedback_schemas import ImprovementCandidate


def _candidate(candidate_id: str = "c1") -> ImprovementCandidate:
    return ImprovementCandidate(
        candidate_id=candidate_id,
        source_diagnosis_ids=["d1"],
        target_module="rag",
        operation="review_or_rewrite",
        priority="high",
        prompt_safe_payload={"recommended_use": "Review recurring RAG issue."},
    )


def test_in_memory_candidate_store_tracks_valid_transitions_and_review_metadata() -> None:
    from werewolf_agent.evaluation.candidate_store import (
        CandidateStatus,
        InMemoryCandidateStore,
    )

    timestamps = iter([
        "2026-06-17T10:00:00",
        "2026-06-17T10:05:00",
        "2026-06-17T10:10:00",
    ])
    store = InMemoryCandidateStore(clock=lambda: next(timestamps))
    store.add(_candidate())

    approved = store.transition(
        "c1",
        CandidateStatus.APPROVED,
        reviewer="moderator",
        notes="Looks safe.",
    )
    materialized = store.transition(
        "c1",
        CandidateStatus.MATERIALIZED,
        reviewer="moderator",
        notes="Draft written.",
    )
    rolled_back = store.transition(
        "c1",
        CandidateStatus.ROLLED_BACK,
        reviewer="gate",
        notes="Regression failed.",
    )

    assert approved.status == CandidateStatus.APPROVED
    assert materialized.status == CandidateStatus.MATERIALIZED
    assert rolled_back.status == CandidateStatus.ROLLED_BACK
    assert rolled_back.reviewer == "gate"
    assert rolled_back.notes == "Regression failed."
    assert rolled_back.updated_at == "2026-06-17T10:10:00"
    assert [item["to_status"] for item in rolled_back.history] == [
        "approved",
        "materialized",
        "rolled_back",
    ]
    assert store.get("c1").candidate.review_status == "rolled_back"


def test_in_memory_candidate_store_rejects_invalid_direct_materialization() -> None:
    from werewolf_agent.evaluation.candidate_store import (
        CandidateStatus,
        InvalidCandidateTransition,
        InMemoryCandidateStore,
    )

    store = InMemoryCandidateStore(clock=lambda: "2026-06-17T10:00:00")
    store.add(_candidate())

    with pytest.raises(InvalidCandidateTransition):
        store.transition(
            "c1",
            CandidateStatus.MATERIALIZED,
            reviewer="moderator",
            notes="Skip approval.",
        )


def test_repository_candidate_store_requires_explicit_repository_methods() -> None:
    from werewolf_agent.evaluation.candidate_store import RepositoryCandidateStore

    store = RepositoryCandidateStore(repository=object())

    with pytest.raises(NotImplementedError, match="save_candidate_record"):
        store.add(_candidate())


def test_repository_candidate_store_delegates_when_methods_exist() -> None:
    from werewolf_agent.evaluation.candidate_store import (
        CandidateStatus,
        RepositoryCandidateStore,
    )

    class Repo:
        def __init__(self) -> None:
            self.saved: dict[str, dict] = {}

        def save_candidate_record(self, record: dict) -> None:
            self.saved[record["candidate"]["candidate_id"]] = record

        def load_candidate_record(self, candidate_id: str) -> dict | None:
            return self.saved.get(candidate_id)

        def list_candidate_records(self, status: str | None = None) -> list[dict]:
            records = list(self.saved.values())
            if status is None:
                return records
            return [record for record in records if record["status"] == status]

        def update_candidate_status(
            self,
            candidate_id: str,
            status: str,
            reviewer: str,
            notes: str,
            timestamp: str,
        ) -> dict:
            record = dict(self.saved[candidate_id])
            history = list(record.get("history", []))
            history.append({
                "from_status": record["status"],
                "to_status": status,
                "reviewer": reviewer,
                "notes": notes,
                "timestamp": timestamp,
            })
            candidate = dict(record["candidate"])
            candidate["review_status"] = status
            record.update({
                "candidate": candidate,
                "status": status,
                "reviewer": reviewer,
                "notes": notes,
                "updated_at": timestamp,
                "history": history,
            })
            self.saved[candidate_id] = record
            return record

    store = RepositoryCandidateStore(repository=Repo(), clock=lambda: "now")
    store.add(_candidate())
    record = store.transition(
        "c1",
        CandidateStatus.APPROVED,
        reviewer="moderator",
        notes="ok",
    )

    assert record.status == CandidateStatus.APPROVED
    assert store.list(status=CandidateStatus.APPROVED)[0].candidate.candidate_id == "c1"
