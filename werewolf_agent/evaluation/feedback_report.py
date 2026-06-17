"""Compact JSON reports for feedback-loop evaluation outputs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from werewolf_agent.evaluation.ablation import AblationReport
from werewolf_agent.evaluation.full_game_ablation import FullGameAblationReport
from werewolf_agent.evaluation.feedback_metrics import (
    ModuleAttributionSummary,
    summarize_module_attribution,
)
from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    FailureDiagnosis,
    ImprovementCandidate,
)
from werewolf_agent.evaluation.regression_gate import CandidateRegressionReport


_PLAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PRIVATE_AUDIT_KEYS = {
    "actual_role",
    "ground_truth",
    "key_assignments",
    "target_faction",
    "target_role",
    "true_world_rank",
}
_PRIVATE_AUDIT_TOKENS = {
    "actual_role",
    "ground_truth",
    "key_assignments",
    "target_faction",
    "target_role",
    "true_world_rank",
}


@dataclass(frozen=True)
class FeedbackReport:
    report_id: str
    batch_id: str
    trace_count: int
    module_metrics: dict[str, ModuleAttributionSummary]
    diagnoses: list[FailureDiagnosis] = field(default_factory=list)
    candidates: list[ImprovementCandidate] = field(default_factory=list)
    ablation_reports: list[AblationReport] = field(default_factory=list)
    generated_at: str = ""
    schema_version: int = 1
    source_refs: list[str] = field(default_factory=list)
    full_game_ablation_reports: list[FullGameAblationReport] = field(default_factory=list)
    candidate_gate_reports: list[CandidateRegressionReport] = field(default_factory=list)

    def to_json_dict(self, *, include_private_audit: bool = False) -> dict[str, Any]:
        failure_clusters = _failure_clusters(
            self.diagnoses,
            self.candidates,
            include_private_audit=include_private_audit,
        )
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "batch_id": self.batch_id,
            "generated_at": self.generated_at,
            "trace_count": self.trace_count,
            "source_refs": _refs_to_dict(
                self.source_refs,
                include_private_audit=include_private_audit,
            ),
            "module_metrics": {
                module: _module_metric_to_dict(summary)
                for module, summary in self.module_metrics.items()
            },
            "diagnoses": [
                _diagnosis_to_dict(
                    diagnosis,
                    include_private_audit=include_private_audit,
                )
                for diagnosis in self.diagnoses
            ],
            "failure_clusters": failure_clusters,
            "candidates": [
                _candidate_to_dict(
                    candidate,
                    include_private_audit=include_private_audit,
                )
                for candidate in self.candidates
            ],
            "candidate_workflow": _candidate_workflow_summary(
                self.candidates,
                self.candidate_gate_reports,
                include_private_audit=include_private_audit,
            ),
            "regression_summary": _regression_summary(self.candidates),
            "ablations": [
                _ablation_to_dict(report)
                for report in self.ablation_reports
            ],
            "full_game_ablations": [
                _full_game_ablation_to_dict(report)
                for report in self.full_game_ablation_reports
            ],
        }

    def to_json(self, *, include_private_audit: bool = False) -> str:
        return json.dumps(
            self.to_json_dict(include_private_audit=include_private_audit),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def build_feedback_report(
    *,
    report_id: str,
    batch_id: str,
    traces: list[EvaluationTrace],
    diagnoses: list[FailureDiagnosis] | None = None,
    candidates: list[ImprovementCandidate] | None = None,
    ablation_reports: list[AblationReport] | None = None,
    full_game_ablation_reports: list[FullGameAblationReport] | None = None,
    candidate_gate_reports: list[CandidateRegressionReport] | None = None,
    generated_at: str = "",
) -> FeedbackReport:
    """Build a compact feedback report from local evaluation artifacts."""
    return FeedbackReport(
        report_id=report_id,
        batch_id=batch_id,
        trace_count=len(traces),
        module_metrics=summarize_module_attribution(traces),
        diagnoses=list(diagnoses or []),
        candidates=list(candidates or []),
        ablation_reports=list(ablation_reports or []),
        full_game_ablation_reports=list(full_game_ablation_reports or []),
        candidate_gate_reports=list(candidate_gate_reports or []),
        generated_at=generated_at,
        source_refs=_collect_source_refs(traces),
    )


def _collect_source_refs(traces: list[EvaluationTrace]) -> list[str]:
    refs: set[str] = set()
    for trace in traces:
        refs.add(f"trace:{trace.trace_id}")
        refs.update(trace.source_refs)
    return sorted(refs)


def _module_metric_to_dict(summary: ModuleAttributionSummary) -> dict[str, Any]:
    return {
        "module": summary.module,
        "exposure_count": summary.exposure_count,
        "supported_count": summary.supported_count,
        "unsupported_count": summary.unsupported_count,
        "prompt_visible_count": summary.prompt_visible_count,
        "cited_count": summary.cited_count,
        "aligned_count": summary.aligned_count,
        "harmful_count": summary.harmful_count,
        "prompt_visible_rate": summary.prompt_visible_rate,
        "citation_rate": summary.citation_rate,
        "alignment_rate": summary.alignment_rate,
        "harmful_rate": summary.harmful_rate,
    }


def _diagnosis_to_dict(
    diagnosis: FailureDiagnosis,
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    data = {
        "diagnosis_id": diagnosis.diagnosis_id,
        "category": diagnosis.category,
        "severity": diagnosis.severity,
        "primary_module": diagnosis.primary_module,
        "supporting_modules": list(diagnosis.supporting_modules),
    }
    if include_private_audit:
        data.update({
            "trace_id": diagnosis.trace_id,
            "evidence_refs": list(diagnosis.evidence_refs),
            "explanation": diagnosis.explanation,
        })
    return data


def _candidate_to_dict(
    candidate: ImprovementCandidate,
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    data = {
        "candidate_id": _public_safe_identifier(
            candidate.candidate_id,
            fallback="redacted_candidate",
            include_private_audit=include_private_audit,
        ),
        "source_diagnosis_ids": list(candidate.source_diagnosis_ids),
        "target_module": candidate.target_module,
        "operation": candidate.operation,
        "priority": candidate.priority,
        "prompt_safe_payload": _candidate_prompt_payload(
            candidate.prompt_safe_payload,
            include_private_audit=include_private_audit,
        ),
        "audit_evidence": _public_audit_evidence(candidate.audit_evidence),
        "regression_seed_set": list(candidate.regression_seed_set),
        "review_status": candidate.review_status,
    }
    if include_private_audit:
        data.update({
            "audit_evidence": _json_safe(candidate.audit_evidence),
            "moderator_notes": candidate.moderator_notes,
            "safety_notes": list(candidate.safety_notes),
        })
    return data


def _failure_clusters(
    diagnoses: list[FailureDiagnosis],
    candidates: list[ImprovementCandidate],
    *,
    include_private_audit: bool,
) -> list[dict[str, Any]]:
    candidate_ids_by_diagnosis: dict[str, list[str]] = {}
    for candidate in candidates:
        for diagnosis_id in candidate.source_diagnosis_ids:
            candidate_ids_by_diagnosis.setdefault(diagnosis_id, []).append(
                candidate.candidate_id
            )

    clusters: dict[tuple[str, str, str], dict[str, Any]] = {}
    for diagnosis in diagnoses:
        key = (diagnosis.primary_module, diagnosis.category, diagnosis.severity)
        cluster = clusters.setdefault(
            key,
            {
                "cluster_id": ":".join(_clean(part) for part in key),
                "category": diagnosis.category,
                "primary_module": diagnosis.primary_module,
                "severity": diagnosis.severity,
                "count": 0,
                "diagnosis_ids": [],
                "trace_refs": set(),
                "candidate_ids": set(),
            },
        )
        cluster["count"] += 1
        cluster["diagnosis_ids"].append(diagnosis.diagnosis_id)
        for ref in diagnosis.evidence_refs:
            if ref.startswith("trace:") and (
                include_private_audit or _is_public_safe_text(ref)
            ):
                cluster["trace_refs"].add(ref)
        for candidate_id in candidate_ids_by_diagnosis.get(
            diagnosis.diagnosis_id,
            [],
        ):
            cluster["candidate_ids"].add(candidate_id)

    result: list[dict[str, Any]] = []
    for cluster in clusters.values():
        result.append({
            "cluster_id": cluster["cluster_id"],
            "category": cluster["category"],
            "primary_module": cluster["primary_module"],
            "severity": cluster["severity"],
            "count": cluster["count"],
            "diagnosis_ids": sorted(cluster["diagnosis_ids"]),
            "trace_refs": sorted(cluster["trace_refs"]),
            "candidate_ids": sorted(cluster["candidate_ids"]),
        })
    return sorted(
        result,
        key=lambda item: (
            item["primary_module"],
            item["category"],
            item["severity"],
        ),
    )


def _regression_summary(candidates: list[ImprovementCandidate]) -> dict[str, Any]:
    seeds = sorted({
        seed
        for candidate in candidates
        for seed in candidate.regression_seed_set
    })
    candidate_count = sum(
        1 for candidate in candidates if candidate.regression_seed_set
    )
    return {
        "status": "pending" if seeds else "not_configured",
        "seed_count": len(seeds),
        "seed_set": seeds,
        "candidate_count": candidate_count,
        "unsupported_metrics": [],
    }


def _candidate_workflow_summary(
    candidates: list[ImprovementCandidate],
    gate_reports: list[CandidateRegressionReport],
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    counts = {
        "pending": 0,
        "approved": 0,
        "rejected": 0,
        "materialized": 0,
        "rolled_back": 0,
    }
    for candidate in candidates:
        status = str(candidate.review_status or "pending")
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return {
        **counts,
        "gate_results": [
            _gate_report_to_dict(
                report,
                include_private_audit=include_private_audit,
            )
            for report in gate_reports
        ],
    }


def _candidate_prompt_payload(
    payload: dict[str, Any],
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    if include_private_audit:
        return _json_safe(payload)
    safe = _public_safe_json(payload)
    return safe if isinstance(safe, dict) else {}


def _gate_report_to_dict(
    report: CandidateRegressionReport,
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    data = report.to_json_dict()
    if include_private_audit:
        return _json_safe(data)
    safe = _public_safe_json(data)
    return safe if isinstance(safe, dict) else {}


def _public_safe_identifier(
    value: str,
    *,
    fallback: str,
    include_private_audit: bool,
) -> str:
    if include_private_audit:
        return value
    return value if _is_public_safe_text(value) else fallback


def _refs_to_dict(refs: list[str], *, include_private_audit: bool) -> list[str]:
    if include_private_audit:
        return list(refs)
    return [ref for ref in refs if _is_public_safe_text(ref)]


def _ablation_to_dict(report: AblationReport) -> dict[str, Any]:
    return {
        "mode": report.mode,
        "removed_modules": list(report.removed_modules),
        "baseline_trace_count": report.baseline_trace_count,
        "ablated_trace_count": report.ablated_trace_count,
        "baseline_module_metrics": {
            module: _module_metric_to_dict(summary)
            for module, summary in report.baseline_module_metrics.items()
        },
        "ablated_module_metrics": {
            module: _module_metric_to_dict(summary)
            for module, summary in report.ablated_module_metrics.items()
        },
        "unsupported_metrics": dict(sorted(report.unsupported_metrics.items())),
    }


def _full_game_ablation_to_dict(report: FullGameAblationReport) -> dict[str, Any]:
    return {
        "batch_id": report.batch_id,
        "mode": report.mode,
        "agent_mode": report.agent_mode,
        "removed_modules": list(report.removed_modules),
        "pair_count": report.pair_count,
        "metric_deltas": {
            metric: _json_safe(delta)
            for metric, delta in sorted(report.metric_deltas.items())
        },
        "unsupported_metrics": dict(sorted(report.unsupported_metrics.items())),
    }


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _public_audit_evidence(value: dict[str, Any]) -> dict[str, Any]:
    safe = _public_safe_json(value)
    return safe if isinstance(safe, dict) else {}


def _public_safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _PRIVATE_AUDIT_KEYS:
                continue
            if not _is_public_safe_text(key_text):
                continue
            safe_item = _public_safe_json(item)
            if safe_item is not None:
                result[key_text] = safe_item
        return result
    if isinstance(value, list):
        result = [
            safe_item
            for item in value
            if (safe_item := _public_safe_json(item)) is not None
        ]
        return result
    if isinstance(value, tuple):
        return _public_safe_json(list(value))
    if isinstance(value, str):
        return value if _is_public_safe_text(value) else None
    return value


def _is_public_safe_text(value: str) -> bool:
    text = str(value)
    lowered = text.lower()
    if _PLAYER_ID_RE.search(text):
        return False
    return not any(token in lowered for token in _PRIVATE_AUDIT_TOKENS)


def _clean(value: str) -> str:
    return str(value).replace(":", "_").replace(" ", "_")
