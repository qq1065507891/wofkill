# -*- coding: utf-8 -*-
"""
反馈报告 JSON 序列化与公开视图脱敏 helper。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.evaluation.feedback_report_serialization import refs_to_dict
    >>> refs_to_dict(["trace:public"], include_private_audit=False)
    ['trace:public']
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

from werewolf_agent.evaluation.ablation import AblationReport
from werewolf_agent.evaluation.feedback_metrics import ModuleAttributionSummary
from werewolf_agent.evaluation.feedback_schemas import (
    FailureDiagnosis,
    ImprovementCandidate,
)
from werewolf_agent.evaluation.full_game_ablation import FullGameAblationReport
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


def module_metric_to_dict(summary: ModuleAttributionSummary) -> dict[str, Any]:
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
        # 2026-07-22 R9a: LLM prompt cache 字段 (R2/R6 写, R7 落 sink).
        # R9a 折入 module_metrics 统一模板, 让 LangSmith dashboard 0 改动
        # 看到 cache_hit_ratio. cache_creation/cache_read 默认 0 (兼容既有 entry).
        "cache_creation_tokens": summary.cache_creation_tokens,
        "cache_read_tokens": summary.cache_read_tokens,
        "cache_hit_ratio": summary.cache_hit_ratio,
    }


def diagnosis_to_dict(
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


def candidate_to_dict(
    candidate: ImprovementCandidate,
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    data = {
        "candidate_id": public_safe_identifier(
            candidate.candidate_id,
            fallback="redacted_candidate",
            include_private_audit=include_private_audit,
        ),
        "source_diagnosis_ids": list(candidate.source_diagnosis_ids),
        "target_module": candidate.target_module,
        "operation": candidate.operation,
        "priority": candidate.priority,
        "prompt_safe_payload": candidate_prompt_payload(
            candidate.prompt_safe_payload,
            include_private_audit=include_private_audit,
        ),
        "audit_evidence": public_audit_evidence(candidate.audit_evidence),
        "regression_seed_set": list(candidate.regression_seed_set),
        "review_status": candidate.review_status,
    }
    if include_private_audit:
        data.update({
            "audit_evidence": json_safe(candidate.audit_evidence),
            "moderator_notes": candidate.moderator_notes,
            "safety_notes": list(candidate.safety_notes),
        })
    return data


def failure_clusters(
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
                "cluster_id": ":".join(clean(part) for part in key),
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
                include_private_audit or is_public_safe_text(ref)
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


def regression_summary(candidates: list[ImprovementCandidate]) -> dict[str, Any]:
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


def candidate_workflow_summary(
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
            gate_report_to_dict(
                report,
                include_private_audit=include_private_audit,
            )
            for report in gate_reports
        ],
    }


def candidate_prompt_payload(
    payload: dict[str, Any],
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    if include_private_audit:
        return json_safe(payload)
    safe = public_safe_json(payload)
    return safe if isinstance(safe, dict) else {}


def gate_report_to_dict(
    report: CandidateRegressionReport,
    *,
    include_private_audit: bool,
) -> dict[str, Any]:
    data = report.to_json_dict()
    if include_private_audit:
        return json_safe(data)
    safe = public_safe_json(data)
    return safe if isinstance(safe, dict) else {}


def public_safe_identifier(
    value: str,
    *,
    fallback: str,
    include_private_audit: bool,
) -> str:
    if include_private_audit:
        return value
    return value if is_public_safe_text(value) else fallback


def refs_to_dict(refs: list[str], *, include_private_audit: bool) -> list[str]:
    if include_private_audit:
        return list(refs)
    return [ref for ref in refs if is_public_safe_text(ref)]


def ablation_to_dict(report: AblationReport) -> dict[str, Any]:
    return {
        "mode": report.mode,
        "removed_modules": list(report.removed_modules),
        "baseline_trace_count": report.baseline_trace_count,
        "ablated_trace_count": report.ablated_trace_count,
        "baseline_module_metrics": {
            module: module_metric_to_dict(summary)
            for module, summary in report.baseline_module_metrics.items()
        },
        "ablated_module_metrics": {
            module: module_metric_to_dict(summary)
            for module, summary in report.ablated_module_metrics.items()
        },
        "unsupported_metrics": dict(sorted(report.unsupported_metrics.items())),
    }


def full_game_ablation_to_dict(report: FullGameAblationReport) -> dict[str, Any]:
    return {
        "batch_id": report.batch_id,
        "mode": report.mode,
        "agent_mode": report.agent_mode,
        "removed_modules": list(report.removed_modules),
        "pair_count": report.pair_count,
        "metric_deltas": {
            metric: json_safe(delta)
            for metric, delta in sorted(report.metric_deltas.items())
        },
        "unsupported_metrics": dict(sorted(report.unsupported_metrics.items())),
    }


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def public_audit_evidence(value: dict[str, Any]) -> dict[str, Any]:
    safe = public_safe_json(value)
    return safe if isinstance(safe, dict) else {}


def public_safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _PRIVATE_AUDIT_KEYS:
                continue
            if not is_public_safe_text(key_text):
                continue
            safe_item = public_safe_json(item)
            if safe_item is not None:
                result[key_text] = safe_item
        return result
    if isinstance(value, list):
        result = [
            safe_item
            for item in value
            if (safe_item := public_safe_json(item)) is not None
        ]
        return result
    if isinstance(value, tuple):
        return public_safe_json(list(value))
    if isinstance(value, str):
        return value if is_public_safe_text(value) else None
    return value


def is_public_safe_text(value: str) -> bool:
    text = str(value)
    lowered = text.lower()
    if _PLAYER_ID_RE.search(text):
        return False
    return not any(token in lowered for token in _PRIVATE_AUDIT_TOKENS)


def clean(value: str) -> str:
    return str(value).replace(":", "_").replace(" ", "_")


__all__ = [
    "ablation_to_dict",
    "candidate_prompt_payload",
    "candidate_to_dict",
    "candidate_workflow_summary",
    "clean",
    "diagnosis_to_dict",
    "failure_clusters",
    "full_game_ablation_to_dict",
    "gate_report_to_dict",
    "is_public_safe_text",
    "json_safe",
    "module_metric_to_dict",
    "public_audit_evidence",
    "public_safe_identifier",
    "public_safe_json",
    "refs_to_dict",
    "regression_summary",
]
