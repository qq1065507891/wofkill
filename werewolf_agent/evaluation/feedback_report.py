# -*- coding: utf-8 -*-
"""
功能描述：为反馈回路评估输出生成紧凑JSON报告，并携带脱敏监控摘要。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-13
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.evaluation.ablation import AblationReport
from werewolf_agent.evaluation import feedback_report_serialization
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
from werewolf_agent.runtime.exposure_audit import summarize_persona_prompt_confirmation

_ablation_to_dict = feedback_report_serialization.ablation_to_dict
_candidate_prompt_payload = feedback_report_serialization.candidate_prompt_payload
_candidate_to_dict = feedback_report_serialization.candidate_to_dict
_candidate_workflow_summary = feedback_report_serialization.candidate_workflow_summary
_clean = feedback_report_serialization.clean
_diagnosis_to_dict = feedback_report_serialization.diagnosis_to_dict
_failure_clusters = feedback_report_serialization.failure_clusters
_full_game_ablation_to_dict = feedback_report_serialization.full_game_ablation_to_dict
_gate_report_to_dict = feedback_report_serialization.gate_report_to_dict
_is_public_safe_text = feedback_report_serialization.is_public_safe_text
_json_safe = feedback_report_serialization.json_safe
_module_metric_to_dict = feedback_report_serialization.module_metric_to_dict
_public_audit_evidence = feedback_report_serialization.public_audit_evidence
_public_safe_identifier = feedback_report_serialization.public_safe_identifier
_public_safe_json = feedback_report_serialization.public_safe_json
_refs_to_dict = feedback_report_serialization.refs_to_dict
_regression_summary = feedback_report_serialization.regression_summary


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
    monitoring_exposures: list[dict[str, Any]] = field(default_factory=list)
    persona_prompt_confirmation: dict[str, Any] = field(default_factory=dict)

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
            "monitoring_exposures": _public_safe_json(self.monitoring_exposures),
            "persona_prompt_confirmation": _public_safe_json(
                self.persona_prompt_confirmation
            ),
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
        monitoring_exposures=_monitoring_exposure_rows(traces),
        persona_prompt_confirmation=summarize_persona_prompt_confirmation(traces),
    )


def _collect_source_refs(traces: list[EvaluationTrace]) -> list[str]:
    refs: set[str] = set()
    for trace in traces:
        refs.add(f"trace:{trace.trace_id}")
        refs.update(trace.source_refs)
    return sorted(refs)


def _monitoring_exposure_rows(traces: list[EvaluationTrace]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        trace_hash = hashlib.sha256(trace.trace_id.encode("utf-8")).hexdigest()[:16]
        for exposure in trace.module_exposures:
            if exposure.module not in {
                "skill_tool_calls",
                "prompt_injections",
                "persona_prompt_confirmation",
            }:
                continue
            rows.append({
                "trace_hash": trace_hash,
                "module": exposure.module,
                "item_id": exposure.item_id,
                "score": exposure.score,
                "prompt_visible": exposure.prompt_visible,
                "metadata": _public_safe_json(exposure.metadata),
            })
    return rows
