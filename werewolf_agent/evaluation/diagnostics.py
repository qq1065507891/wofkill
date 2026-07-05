# -*- coding: utf-8 -*-
"""
功能描述：**：对反馈回路评估轨迹进行确定性失败诊断
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    FailureDiagnosis,
    ModuleExposure,
)
from werewolf_agent.evaluation.world_model_eval import WorldRankSample


def diagnose_trace_failures(
    traces: list[EvaluationTrace],
    *,
    world_rank_samples: list[WorldRankSample] | None = None,
    low_rank_threshold: int = 3,
) -> list[FailureDiagnosis]:
    """Classify trace failures into typed, reviewable diagnoses."""
    diagnoses: list[FailureDiagnosis] = []
    for trace in traces:
        diagnoses.extend(_decision_diagnoses(trace))
        diagnoses.extend(_module_exposure_diagnoses(trace))

    trace_ids = {trace.trace_id for trace in traces}
    for sample in world_rank_samples or []:
        if sample.trace_id not in trace_ids:
            continue
        if sample.support != "supported":
            continue
        if sample.true_world_rank > low_rank_threshold:
            diagnoses.append(FailureDiagnosis(
                diagnosis_id=_diagnosis_id(
                    sample.trace_id,
                    "low_true_world_rank",
                    str(sample.true_world_rank),
                ),
                trace_id=sample.trace_id,
                category="low_true_world_rank",
                severity="medium",
                primary_module="possible_worlds",
                evidence_refs=[f"world_rank:{sample.trace_id}"],
                explanation=(
                    f"True world ranked outside threshold: "
                    f"rank={sample.true_world_rank}, threshold={low_rank_threshold}."
                ),
            ))
    return diagnoses


def _decision_diagnoses(trace: EvaluationTrace) -> list[FailureDiagnosis]:
    diagnoses: list[FailureDiagnosis] = []
    decision = trace.decision
    outcome = trace.outcome
    if decision is None and outcome is None:
        return diagnoses
    action_type = decision.action_type if decision else ""
    target_id = decision.target_id if decision else None
    if outcome and outcome.legal is False:
        diagnoses.append(_diagnosis(
            trace,
            category="illegal_action",
            severity="high",
            primary_module="decision",
            explanation=f"Decision action '{action_type}' was marked illegal.",
        ))
    if outcome and outcome.leaked_hidden_info:
        diagnoses.append(_diagnosis(
            trace,
            category="hidden_info_leak",
            severity="high",
            primary_module="dialogue",
            explanation="Decision or dialogue leaked hidden information.",
        ))
    if _is_wrong_good_target(trace, action_type, target_id, outcome):
        diagnoses.append(_diagnosis(
            trace,
            category="wrong_target",
            severity="medium",
            primary_module="decision",
            explanation="Good-faction harmful target action hit a good-faction player.",
        ))
    return diagnoses


def _is_wrong_good_target(
    trace: EvaluationTrace,
    action_type: str,
    target_id: str | None,
    outcome: object,
) -> bool:
    if not target_id or outcome is None:
        return False
    target_faction = getattr(outcome, "target_faction", "")
    if trace.faction != "good" or target_faction != "good":
        return False
    return action_type in {
        "vote",
        "use_poison",
        "hunter_shot",
        "sheriff_vote",
    }


def _module_exposure_diagnoses(trace: EvaluationTrace) -> list[FailureDiagnosis]:
    diagnoses: list[FailureDiagnosis] = []
    for exposure in trace.module_exposures:
        if exposure.module == "rag" and _is_harmful(exposure):
            diagnoses.append(_exposure_diagnosis(
                trace,
                exposure,
                category="rag_harmful_transfer",
                primary_module="rag",
                severity="medium",
            ))
        elif exposure.module == "reflection" and _is_harmful(exposure):
            diagnoses.append(_exposure_diagnosis(
                trace,
                exposure,
                category="reflection_harmful_transfer",
                primary_module="reflection",
                severity="medium",
            ))
        elif exposure.module == "simulator" and _is_false_positive(exposure):
            diagnoses.append(_exposure_diagnosis(
                trace,
                exposure,
                category="simulator_false_positive",
                primary_module="simulator",
                severity="medium",
            ))
    return diagnoses


def _diagnosis(
    trace: EvaluationTrace,
    *,
    category: str,
    severity: str,
    primary_module: str,
    explanation: str,
) -> FailureDiagnosis:
    return FailureDiagnosis(
        diagnosis_id=_diagnosis_id(trace.trace_id, category),
        trace_id=trace.trace_id,
        category=category,
        severity=severity,
        primary_module=primary_module,
        evidence_refs=[f"trace:{trace.trace_id}"],
        explanation=explanation,
    )


def _exposure_diagnosis(
    trace: EvaluationTrace,
    exposure: ModuleExposure,
    *,
    category: str,
    primary_module: str,
    severity: str,
) -> FailureDiagnosis:
    return FailureDiagnosis(
        diagnosis_id=_diagnosis_id(trace.trace_id, category, exposure.item_id),
        trace_id=trace.trace_id,
        category=category,
        severity=severity,
        primary_module=primary_module,
        evidence_refs=[f"trace:{trace.trace_id}", f"exposure:{exposure.item_id}"],
        explanation=f"{primary_module} exposure '{exposure.item_id}' was flagged.",
    )


def _is_harmful(exposure: ModuleExposure) -> bool:
    return (
        exposure.metadata.get("harmful_transfer") is True
        or exposure.metadata.get("harmful") is True
    )


def _is_false_positive(exposure: ModuleExposure) -> bool:
    return (
        exposure.metadata.get("false_positive") is True
        or exposure.metadata.get("prediction_false_positive") is True
    )


def _diagnosis_id(trace_id: str, category: str, *parts: str) -> str:
    suffix = ":".join(_clean(part) for part in parts if part)
    base = f"{_clean(trace_id)}:{category}"
    return f"{base}:{suffix}" if suffix else base


def _clean(value: str) -> str:
    return str(value).replace(":", "_").replace(" ", "_")
