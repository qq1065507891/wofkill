# -*- coding: utf-8 -*-
"""
构造发言语义修复 V2 校验与运行时审计结果。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-19

使用示例:
    >>> validate_semantic_repair(context, source, final).accepted
    True
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from werewolf_agent.agents.schemas import (
    AgentContext,
    FallbackAction,
    PlayerAction,
    TaskType,
)
from werewolf_agent.evaluation.balance_public_claims import (
    PublicClaimAuditKey,
    classify_public_claims,
    public_claim_audit_key,
    public_claim_audit_keys,
)
from werewolf_agent.runtime.speech_quality import extract_speech_quality


_FALLBACK_KINDS = frozenset({
    "no_fallback",
    "generic_template",
    "target_specific",
    "verified_claim",
    "task_specific",
})
_REJECTION_REASON_ORDER = (
    "unsupported_public_claim",
    "speaker_attribution_changed",
    "negation_changed",
)
_REJECTION_MESSAGES = {
    "unsupported_public_claim": "输出包含缺少公开证据支持的事实声明",
    "speaker_attribution_changed": "已有声明的说话人归属发生变化",
    "negation_changed": "已有声明的否定关系发生变化",
}
_CORRECTION_HINTS = {
    "unsupported_public_claim": "删除或改写缺少公开证据支持的事实声明",
    "speaker_attribution_changed": "恢复公开记录中的说话人归属，或删除该声明",
    "negation_changed": "恢复公开记录中的否定关系，或删除该声明",
}


@dataclass(frozen=True)
class SemanticRepairValidationResult:
    """返回 V2 语义门结论、稳定原因码和安全审计字段。"""

    accepted: bool
    reason_codes: tuple[str, ...]
    audit: dict[str, Any]


def _public_speeches(context: AgentContext) -> list[tuple[str, str]]:
    return [
        (str(item.get("speaker") or ""), str(item.get("text") or ""))
        for item in context.public_claim_ledger
        if isinstance(item, dict)
    ]


def _target(action: PlayerAction | FallbackAction) -> str | None:
    if action.target_id:
        return action.target_id
    quality = extract_speech_quality(action.speech)
    candidates = quality["suspicion_targets"]
    return quality["vote_leaning"] or (candidates[0] if candidates else None)


def build_semantic_repair_audit(
    context: AgentContext,
    source: PlayerAction,
    final: PlayerAction | FallbackAction,
    *,
    success: bool,
    generic_template_used: bool = False,
) -> dict[str, Any]:
    """比较首次可修复失败与最终输出，生成不含原始发言的审计字段。"""
    public_speeches = _public_speeches(context)
    source_claims, verified_claims = public_claim_audit_keys(
        source.speech, public_speeches
    )
    final_claims, final_verified_claims = public_claim_audit_keys(
        final.speech, public_speeches
    )
    speaker_attribution_preserved = _speaker_attribution_preserved(
        source_claims, final_claims
    )
    negation_preserved = _negation_preserved(source_claims, final_claims)
    introduced_claims = {
        claim for claim in final_claims
        if claim not in source_claims
        and not _is_supported_attribution_completion(
            claim, source_claims, final_verified_claims
        )
    }
    unsupported_final_claims = final_claims - final_verified_claims
    reason_codes = _semantic_repair_reason_codes(
        unsupported_public_claim_count=len(unsupported_final_claims),
        speaker_attribution_preserved=speaker_attribution_preserved,
        negation_preserved=negation_preserved,
    )
    source_target = _target(source)
    retained_count = len(verified_claims & final_claims)
    final_target = _target(final)
    if success:
        fallback_kind = "no_fallback"
    elif retained_count > 0:
        fallback_kind = "verified_claim"
    elif generic_template_used:
        fallback_kind = "generic_template"
    elif source_target is not None and final_target == source_target:
        fallback_kind = "target_specific"
    else:
        fallback_kind = "task_specific"
    return {
        "semantic_gate_version": 2,
        "repairable": True,
        "success": success,
        "target_preserved": final_target == source_target,
        "speaker_attribution_preserved": speaker_attribution_preserved,
        "negation_preserved": negation_preserved,
        "introduced_claim_count": len(introduced_claims),
        "unsupported_public_claim_count": len(unsupported_final_claims),
        "verified_claim_count": len(verified_claims),
        "retained_verified_claim_count": retained_count,
        "rejection_reason_codes": list(reason_codes),
        "generic_template_used": generic_template_used,
        "fallback_kind": fallback_kind,
    }


def validate_semantic_repair(
    context: AgentContext,
    source: PlayerAction,
    final: PlayerAction,
) -> SemanticRepairValidationResult:
    """按 V2 规则校验修复结果，并返回所有适用的稳定原因码。"""
    audit = build_semantic_repair_audit(context, source, final, success=True)
    reason_codes = tuple(audit["rejection_reason_codes"])
    return SemanticRepairValidationResult(
        accepted=not reason_codes,
        reason_codes=reason_codes,
        audit=audit,
    )


def semantic_repair_rejection_message(reason_codes: Iterable[str]) -> str:
    """仅依据原因码生成不含发言、身份真相或供应商错误的固定说明。"""
    explanations = [
        _REJECTION_MESSAGES[code]
        for code in _ordered_reason_codes(reason_codes)
    ]
    return "语义修复未通过：" + "；".join(explanations) + "。"


def semantic_repair_correction_hint(reason_codes: Iterable[str]) -> str:
    """仅依据原因码生成可安全进入重试提示的固定纠正建议。"""
    hints = [
        _CORRECTION_HINTS[code]
        for code in _ordered_reason_codes(reason_codes)
    ]
    return "请" + "；并请".join(hints) + "。"


def semantic_repair_retains_verified_claim(
    context: AgentContext,
    source: PlayerAction,
    final: PlayerAction,
) -> bool:
    """兼容旧调用方，以布尔值返回 V2 语义校验结论。"""
    return validate_semantic_repair(context, source, final).accepted


def preserve_verified_claim_in_fallback(
    context: AgentContext,
    source: PlayerAction,
    fallback: FallbackAction,
) -> FallbackAction:
    """从源发言中仅复制已获公开支撑的最小 claim span 到确定性 fallback。"""
    public_speeches = _public_speeches(context)
    source_claims, verified_claims = public_claim_audit_keys(
        source.speech, public_speeches
    )
    verified_texts: list[str] = []
    retained_keys: set[PublicClaimAuditKey] = set()
    for claim in classify_public_claims(source.speech):
        key = public_claim_audit_key(claim)
        if key in verified_claims and key not in retained_keys:
            verified_texts.append(claim.text.strip().rstrip("。"))
            retained_keys.add(key)

    source_target = _target(source)
    speech = fallback.speech
    target_id = source_target if source_target in context.legal_targets else None
    fallback_claims, _ = public_claim_audit_keys(speech, public_speeches)
    if fallback_claims - source_claims:
        speech = ""
    if source_target and source_target in context.legal_targets:
        speech = (
            f"我继续关注{source_target}，这是倾向而非事实判断；"
            f"我当前倾向投{source_target}。"
        )
    elif not speech.strip():
        speech = _task_safe_semantic_fallback(context.task_type)
    if verified_texts:
        preserved = "。".join(verified_texts) + "。"
        speech = f"{speech.rstrip()} {preserved}".strip()
    return fallback.model_copy(update={"target_id": target_id, "speech": speech})


def _speaker_attribution_preserved(
    source_claims: set[PublicClaimAuditKey],
    final_claims: set[PublicClaimAuditKey],
) -> bool:
    """已有归因可删除但不可换人；空归因可由后续公开证据补齐。"""
    changed_source_claims = source_claims - final_claims
    for final_claim in final_claims - source_claims:
        matching_source = [
            source_claim for source_claim in changed_source_claims
            if source_claim.content_identity == final_claim.content_identity
        ]
        if not matching_source:
            continue
        known_speakers = {
            claim.speaker_attribution for claim in matching_source
            if claim.speaker_attribution
        }
        attribution_was_missing = any(
            not claim.speaker_attribution for claim in matching_source
        )
        if (
            final_claim.speaker_attribution
            and known_speakers
            and final_claim.speaker_attribution not in known_speakers
            and not attribution_was_missing
        ):
            return False
    return True


def _negation_preserved(
    source_claims: set[PublicClaimAuditKey],
    final_claims: set[PublicClaimAuditKey],
) -> bool:
    """同一内容的否定关系只能保留或删除，不可翻转。"""
    changed_source_claims = source_claims - final_claims
    for final_claim in final_claims - source_claims:
        matching_source = [
            source_claim for source_claim in changed_source_claims
            if source_claim.content_identity == final_claim.content_identity
        ]
        if matching_source and final_claim.negated not in {
            claim.negated for claim in matching_source
        }:
            return False
    return True


def _semantic_repair_reason_codes(
    *,
    unsupported_public_claim_count: int,
    speaker_attribution_preserved: bool,
    negation_preserved: bool,
) -> tuple[str, ...]:
    """按固定优先级返回全部适用且不重复的拒绝原因。"""
    reasons = {
        "unsupported_public_claim": unsupported_public_claim_count > 0,
        "speaker_attribution_changed": not speaker_attribution_preserved,
        "negation_changed": not negation_preserved,
    }
    return tuple(code for code in _REJECTION_REASON_ORDER if reasons[code])


def _ordered_reason_codes(reason_codes: Iterable[str]) -> tuple[str, ...]:
    """过滤未知原因码，并按 V2 固定顺序去重。"""
    requested = set(reason_codes)
    return tuple(code for code in _REJECTION_REASON_ORDER if code in requested)


def _is_supported_attribution_completion(
    final_claim: PublicClaimAuditKey,
    source_claims: set[PublicClaimAuditKey],
    final_verified_claims: set[PublicClaimAuditKey],
) -> bool:
    """只有公开账本支持的非空归因，才能补齐源声明缺失的归因。"""
    return (
        bool(final_claim.speaker_attribution)
        and final_claim in final_verified_claims
        and any(
            not source_claim.speaker_attribution
            and source_claim.attribution_agnostic_identity
            == final_claim.attribution_agnostic_identity
            for source_claim in source_claims
        )
    )


def _task_safe_semantic_fallback(task_type: TaskType) -> str:
    """没有可保留论点或目标时，只输出与当前任务相符的边界声明。"""
    task_specific = {
        TaskType.DEFENSE_SPEECH: "我只回应当前质疑，不补充未经公开验证的事实。",
        TaskType.SHERIFF_SPEECH: "本轮上警发言只表达立场，不补充未经验证的事实。",
        TaskType.PK_SPEECH: "本轮PK发言只回应当前争议，不补充未经验证的事实。",
        TaskType.LAST_WORDS: "本轮遗言只保留公开立场，不补充未经验证的事实。",
    }
    return task_specific.get(
        task_type,
        "本轮只保留当前立场，不补充未经公开验证的事实。",
    )


__all__ = [
    "_FALLBACK_KINDS",
    "SemanticRepairValidationResult",
    "build_semantic_repair_audit",
    "preserve_verified_claim_in_fallback",
    "semantic_repair_correction_hint",
    "semantic_repair_rejection_message",
    "semantic_repair_retains_verified_claim",
    "validate_semantic_repair",
]
