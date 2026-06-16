"""Review workflow store for improvement candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Protocol

from werewolf_agent.evaluation.feedback_schemas import ImprovementCandidate


class CandidateStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MATERIALIZED = "materialized"
    ROLLED_BACK = "rolled_back"


class InvalidCandidateTransition(ValueError):
    """Raised when a candidate workflow transition is not allowed."""


class CandidateNotFound(KeyError):
    """Raised when a requested candidate does not exist."""


@dataclass(frozen=True)
class CandidateRecord:
    candidate: ImprovementCandidate
    status: CandidateStatus
    reviewer: str = ""
    notes: str = ""
    updated_at: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.candidate),
            "status": self.status.value,
            "reviewer": self.reviewer,
            "notes": self.notes,
            "updated_at": self.updated_at,
            "history": [dict(item) for item in self.history],
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "CandidateRecord":
        candidate_data = dict(data["candidate"])
        status = CandidateStatus(data.get("status") or candidate_data.get("review_status") or "pending")
        candidate_data["review_status"] = status.value
        return cls(
            candidate=ImprovementCandidate(**candidate_data),
            status=status,
            reviewer=str(data.get("reviewer") or ""),
            notes=str(data.get("notes") or ""),
            updated_at=str(data.get("updated_at") or ""),
            history=[dict(item) for item in data.get("history", [])],
        )


class CandidateStore(Protocol):
    def add(
        self,
        candidate: ImprovementCandidate,
        *,
        status: CandidateStatus = CandidateStatus.PENDING,
    ) -> CandidateRecord: ...

    def get(self, candidate_id: str) -> CandidateRecord: ...

    def list(self, *, status: CandidateStatus | None = None) -> list[CandidateRecord]: ...

    def transition(
        self,
        candidate_id: str,
        to_status: CandidateStatus,
        *,
        reviewer: str,
        notes: str,
    ) -> CandidateRecord: ...


_ALLOWED_TRANSITIONS: set[tuple[CandidateStatus, CandidateStatus]] = {
    (CandidateStatus.PENDING, CandidateStatus.APPROVED),
    (CandidateStatus.PENDING, CandidateStatus.REJECTED),
    (CandidateStatus.APPROVED, CandidateStatus.MATERIALIZED),
    (CandidateStatus.MATERIALIZED, CandidateStatus.ROLLED_BACK),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _with_status(
    candidate: ImprovementCandidate,
    status: CandidateStatus,
) -> ImprovementCandidate:
    return replace(candidate, review_status=status.value)


def _validate_transition(
    from_status: CandidateStatus,
    to_status: CandidateStatus,
) -> None:
    if (from_status, to_status) not in _ALLOWED_TRANSITIONS:
        raise InvalidCandidateTransition(
            f"invalid candidate transition {from_status.value} -> {to_status.value}"
        )


class InMemoryCandidateStore:
    """Small deterministic candidate workflow store for tests and local runs."""

    def __init__(self, *, clock: Callable[[], str] | None = None) -> None:
        self._clock = clock or _utc_now
        self._records: dict[str, CandidateRecord] = {}

    def add(
        self,
        candidate: ImprovementCandidate,
        *,
        status: CandidateStatus = CandidateStatus.PENDING,
    ) -> CandidateRecord:
        candidate = _with_status(candidate, status)
        record = CandidateRecord(candidate=candidate, status=status)
        self._records[candidate.candidate_id] = record
        return record

    def get(self, candidate_id: str) -> CandidateRecord:
        try:
            return self._records[candidate_id]
        except KeyError as exc:
            raise CandidateNotFound(candidate_id) from exc

    def list(self, *, status: CandidateStatus | None = None) -> list[CandidateRecord]:
        records = list(self._records.values())
        if status is None:
            return records
        return [record for record in records if record.status == status]

    def transition(
        self,
        candidate_id: str,
        to_status: CandidateStatus,
        *,
        reviewer: str,
        notes: str,
    ) -> CandidateRecord:
        record = self.get(candidate_id)
        _validate_transition(record.status, to_status)
        timestamp = self._clock()
        history = [
            *record.history,
            {
                "from_status": record.status.value,
                "to_status": to_status.value,
                "reviewer": reviewer,
                "notes": notes,
                "timestamp": timestamp,
            },
        ]
        updated = CandidateRecord(
            candidate=_with_status(record.candidate, to_status),
            status=to_status,
            reviewer=reviewer,
            notes=notes,
            updated_at=timestamp,
            history=history,
        )
        self._records[candidate_id] = updated
        return updated


class RepositoryCandidateStore:
    """Candidate store facade for repository-backed applications."""

    def __init__(
        self,
        *,
        repository: Any,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._repo = repository
        self._clock = clock or _utc_now

    def add(
        self,
        candidate: ImprovementCandidate,
        *,
        status: CandidateStatus = CandidateStatus.PENDING,
    ) -> CandidateRecord:
        self._require("save_candidate_record")
        record = CandidateRecord(
            candidate=_with_status(candidate, status),
            status=status,
        )
        self._repo.save_candidate_record(record.to_json_dict())
        return record

    def get(self, candidate_id: str) -> CandidateRecord:
        self._require("load_candidate_record")
        payload = self._repo.load_candidate_record(candidate_id)
        if payload is None:
            raise CandidateNotFound(candidate_id)
        return CandidateRecord.from_json_dict(dict(payload))

    def list(self, *, status: CandidateStatus | None = None) -> list[CandidateRecord]:
        self._require("list_candidate_records")
        raw_records = self._repo.list_candidate_records(
            status.value if status is not None else None
        )
        return [
            CandidateRecord.from_json_dict(dict(record))
            for record in raw_records
        ]

    def transition(
        self,
        candidate_id: str,
        to_status: CandidateStatus,
        *,
        reviewer: str,
        notes: str,
    ) -> CandidateRecord:
        self._require("update_candidate_status")
        current = self.get(candidate_id)
        _validate_transition(current.status, to_status)
        updated = self._repo.update_candidate_status(
            candidate_id,
            to_status.value,
            reviewer,
            notes,
            self._clock(),
        )
        return CandidateRecord.from_json_dict(dict(updated))

    def _require(self, method_name: str) -> None:
        if not hasattr(self._repo, method_name):
            raise NotImplementedError(
                f"repository must implement {method_name} for candidate workflow"
            )
