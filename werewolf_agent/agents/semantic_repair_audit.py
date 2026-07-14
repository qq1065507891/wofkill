# -*- coding: utf-8 -*-
"""
构造发言语义重试的运行时审计结果。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-14

使用示例:
    >>> build_semantic_repair_audit(context, source, final, success=True)
"""

from __future__ import annotations

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
        "repairable": True,
        "success": success,
        "target_preserved": final_target == source_target,
        "speaker_attribution_preserved": speaker_attribution_preserved,
        "negation_preserved": negation_preserved,
        "introduced_claim_count": len(introduced_claims),
        "verified_claim_count": len(verified_claims),
        "retained_verified_claim_count": retained_count,
        "generic_template_used": generic_template_used,
        "fallback_kind": fallback_kind,
    }


def semantic_repair_retains_verified_claim(
    context: AgentContext,
    source: PlayerAction,
    final: PlayerAction,
) -> bool:
    """成功修复必须保持目标、全部已验证论点，且不得新增事实 claim。"""
    audit = build_semantic_repair_audit(context, source, final, success=True)
    return (
        audit["target_preserved"] is True
        and audit["speaker_attribution_preserved"] is True
        and audit["negation_preserved"] is True
        and audit["introduced_claim_count"] == 0
        and audit["retained_verified_claim_count"] == audit["verified_claim_count"]
    )


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
    for final_claim in final_claims:
        matching_source = [
            source_claim for source_claim in source_claims
            if source_claim.content_identity == final_claim.content_identity
            and source_claim.negated == final_claim.negated
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
    """同一内容和兼容归因的否定关系只能保留或删除，不可翻转。"""
    for final_claim in final_claims:
        matching_source = [
            source_claim for source_claim in source_claims
            if source_claim.content_identity == final_claim.content_identity
            and (
                source_claim.speaker_attribution == final_claim.speaker_attribution
                or not source_claim.speaker_attribution
            )
        ]
        if matching_source and final_claim.negated not in {
            claim.negated for claim in matching_source
        }:
            return False
    return True


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
    "build_semantic_repair_audit",
    "preserve_verified_claim_in_fallback",
    "semantic_repair_retains_verified_claim",
]
