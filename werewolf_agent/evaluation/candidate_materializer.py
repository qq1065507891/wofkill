# -*- coding: utf-8 -*-
"""
功能描述：**：对已审查候选进行草案具体化与推广
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
from typing import Any, Callable

from werewolf_agent.evaluation.candidate_store import (
    CandidateNotFound,
    CandidateRecord,
    CandidateStatus,
    CandidateStore,
)
from werewolf_agent.evaluation.feedback_schemas import (
    ImprovementCandidate,
    validate_candidate_prompt_safe,
)
from werewolf_agent.evaluation.regression_gate import CandidateRegressionReport


@dataclass(frozen=True)
class CandidateMaterializationResult:
    candidate_id: str
    status: str
    target_module: str
    draft_namespace: str = ""
    live_namespace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CandidateMaterializer:
    """Materialize approved candidates into review-only draft stores."""

    def __init__(
        self,
        *,
        candidate_store: CandidateStore,
        rag_store: Any | None = None,
        reflection_store: Any | None = None,
        draft_namespace: str = "candidate_review",
        live_namespace: str = "live",
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._candidate_store = candidate_store
        self._rag_store = rag_store
        self._reflection_store = reflection_store
        self._draft_namespace = draft_namespace
        self._live_namespace = live_namespace
        self._clock = clock or _utc_now

    def materialize_draft(
        self,
        candidate_or_id: str | ImprovementCandidate,
        namespace: str | None = None,
    ) -> CandidateMaterializationResult:
        candidate_id = (
            candidate_or_id
            if isinstance(candidate_or_id, str)
            else candidate_or_id.candidate_id
        )
        draft_namespace = namespace or self._draft_namespace
        record = self._get_or_add_candidate(candidate_or_id, candidate_id)
        candidate = record.candidate
        if not validate_candidate_prompt_safe(candidate):
            self._candidate_store.transition(
                candidate_id,
                CandidateStatus.REJECTED,
                reviewer="prompt_safety",
                notes="prompt-unsafe candidate rejected before materialization",
            )
            return CandidateMaterializationResult(
                candidate_id=candidate_id,
                status="rejected",
                target_module=candidate.target_module,
                draft_namespace=draft_namespace,
            )

        if record.status != CandidateStatus.APPROVED:
            return CandidateMaterializationResult(
                candidate_id=candidate_id,
                status="not_approved",
                target_module=candidate.target_module,
                draft_namespace=draft_namespace,
            )

        adapter = self._adapter_for(candidate.target_module)
        draft = self._draft_entry(candidate)
        draft_id = _save_review_only(adapter, draft, draft_namespace)
        self._candidate_store.transition(
            candidate_id,
            CandidateStatus.MATERIALIZED,
            reviewer="materializer",
            notes=f"review-only draft written to {draft_namespace}",
            metadata={"draft_id": draft_id, "draft_namespace": draft_namespace},
        )
        return CandidateMaterializationResult(
            candidate_id=candidate_id,
            status="materialized",
            target_module=candidate.target_module,
            draft_namespace=draft_namespace,
            metadata={"draft": draft, "draft_id": draft_id},
        )

    def rollback(
        self,
        candidate_id: str,
        reason: str,
        gate_report: CandidateRegressionReport | None = None,
    ) -> CandidateMaterializationResult:
        record = self._candidate_store.get(candidate_id)
        candidate = record.candidate
        if record.status != CandidateStatus.MATERIALIZED:
            return CandidateMaterializationResult(
                candidate_id=candidate_id,
                status="not_materialized",
                target_module=candidate.target_module,
                draft_namespace=self._draft_namespace,
            )
        adapter = self._adapter_for(candidate.target_module)
        draft_metadata = _draft_metadata(record)
        if draft_metadata is None:
            return CandidateMaterializationResult(
                candidate_id=candidate_id,
                status="missing_draft_metadata",
                target_module=candidate.target_module,
                draft_namespace=self._draft_namespace,
            )
        draft_id, draft_namespace = draft_metadata
        _quarantine(adapter, draft_id, reason, draft_namespace)
        self._candidate_store.transition(
            candidate_id,
            CandidateStatus.ROLLED_BACK,
            reviewer="regression_gate",
            notes=reason,
        )
        metadata: dict[str, Any] = {"reason": reason}
        if gate_report is not None:
            metadata["regression"] = gate_report.to_json_dict()
        return CandidateMaterializationResult(
            candidate_id=candidate_id,
            status="rolled_back",
            target_module=candidate.target_module,
            draft_namespace=draft_namespace,
            metadata=metadata,
        )

    def promote(
        self,
        candidate_id: str,
        regression_report: CandidateRegressionReport,
        *,
        live_namespace: str | None = None,
    ) -> CandidateMaterializationResult:
        record = self._candidate_store.get(candidate_id)
        candidate = record.candidate
        if record.status != CandidateStatus.MATERIALIZED:
            return CandidateMaterializationResult(
                candidate_id=candidate_id,
                status="not_materialized",
                target_module=candidate.target_module,
                draft_namespace=self._draft_namespace,
            )
        adapter = self._adapter_for(candidate.target_module)
        live_namespace = live_namespace or self._live_namespace
        if not regression_report.passed:
            return self.rollback(candidate_id, "regression_gate_failed", regression_report)
        draft_metadata = _draft_metadata(record)
        if draft_metadata is None:
            return CandidateMaterializationResult(
                candidate_id=candidate_id,
                status="missing_draft_metadata",
                target_module=candidate.target_module,
                draft_namespace=self._draft_namespace,
                live_namespace=live_namespace,
            )
        draft_id, draft_namespace = draft_metadata
        metadata = {
            "candidate_id": candidate.candidate_id,
            "target_module": candidate.target_module,
            "source_diagnosis_ids": list(candidate.source_diagnosis_ids),
            "regression": regression_report.to_json_dict(),
            "promoted_at": self._clock(),
        }
        _promote(adapter, draft_id, live_namespace, metadata)
        return CandidateMaterializationResult(
            candidate_id=candidate_id,
            status="promoted",
            target_module=candidate.target_module,
            draft_namespace=draft_namespace,
            live_namespace=live_namespace,
            metadata=metadata,
        )

    def _adapter_for(self, target_module: str) -> Any:
        if target_module == "rag":
            if self._rag_store is None:
                raise NotImplementedError("rag_store is required for RAG candidates")
            return self._rag_store
        if target_module == "reflection":
            if self._reflection_store is None:
                raise NotImplementedError(
                    "reflection_store is required for reflection candidates"
                )
            return self._reflection_store
        raise NotImplementedError(
            f"candidate materialization is not supported for {target_module!r}"
        )

    def _get_or_add_candidate(
        self,
        candidate_or_id: str | ImprovementCandidate,
        candidate_id: str,
    ) -> CandidateRecord:
        try:
            return self._candidate_store.get(candidate_id)
        except CandidateNotFound:
            if isinstance(candidate_or_id, str):
                raise
            return self._candidate_store.add(
                candidate_or_id,
                status=CandidateStatus.PENDING,
            )

    def _draft_entry(self, candidate: ImprovementCandidate) -> dict[str, Any]:
        entry = {
            "candidate_id": candidate.candidate_id,
            "target_module": candidate.target_module,
            "operation": candidate.operation,
            "source_diagnosis_ids": list(candidate.source_diagnosis_ids),
            "prompt_safe_payload": dict(candidate.prompt_safe_payload),
            "review_status": "review_only",
            "created_at": self._clock(),
        }
        if candidate.target_module == "reflection":
            entry["quality_status"] = "review_only"
        return entry


def _save_review_only(adapter: Any, entry: dict[str, Any], namespace: str) -> str:
    result = _call_save_review_only(adapter.save_review_only, entry, namespace)
    return str(result or entry["candidate_id"])


def _call_save_review_only(method: Any, entry: dict[str, Any], namespace: str) -> Any:
    signature = inspect.signature(method)
    params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(params) >= 2 and params[0].name in {"namespace", "draft_namespace"}:
        return method(namespace, entry)
    return method(entry, namespace)


def _draft_metadata(record: CandidateRecord) -> tuple[str, str] | None:
    draft_id = str(record.metadata.get("draft_id") or "")
    draft_namespace = str(record.metadata.get("draft_namespace") or "")
    if not draft_id or not draft_namespace:
        return None
    return draft_id, draft_namespace


def _promote(
    adapter: Any,
    draft_id: str,
    live_namespace: str,
    metadata: dict[str, Any],
) -> str:
    if hasattr(adapter, "promote"):
        return str(adapter.promote(draft_id, live_namespace, metadata))
    if hasattr(adapter, "promote_reviewed"):
        return str(adapter.promote_reviewed(draft_id, live_namespace, metadata))
    raise NotImplementedError("draft adapter must implement promote")


def _quarantine(
    adapter: Any,
    draft_id: str,
    reason: str,
    namespace: str,
) -> None:
    if hasattr(adapter, "quarantine"):
        adapter.quarantine(draft_id, reason, namespace=namespace)
        return
    if hasattr(adapter, "quarantine_review_only"):
        adapter.quarantine_review_only(namespace, draft_id, reason)
        return
    raise NotImplementedError("draft adapter must implement quarantine")
