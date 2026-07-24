# -*- coding: utf-8 -*-
"""
构造发言语义修复 V2 校验与运行时审计结果。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-24

使用示例:
    >>> validate_semantic_repair(context, source, final).accepted
    True
"""

from __future__ import annotations

import re
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
    attributed_role_claim_supported,
    classify_public_claims,
    night_info_claim_supported,
    public_claim_audit_key,
    public_claim_audit_keys,
    public_claim_is_negated,
    role_claim_supported,
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
_HARMLESS_FIRST_PERSON_PREFIXES = ("其实", "不过")
_HARMLESS_FIRST_PERSON_PREFIX_PATTERN = "|".join(
    re.escape(prefix) for prefix in _HARMLESS_FIRST_PERSON_PREFIXES
)
_ROLE_NEGATION_FORMS = ("不是", "并非", "不为")
_ROLE_NEGATION_PATTERN = "|".join(
    re.escape(form) for form in _ROLE_NEGATION_FORMS
)
_FIRST_PERSON_DENIAL_PREFIXES = ("没有", "并未", "未曾", "从未", "不")
_FIRST_PERSON_DENIAL_PREFIX_PATTERN = "|".join(
    re.escape(prefix) for prefix in _FIRST_PERSON_DENIAL_PREFIXES
)
_FIRST_PERSON_DENIAL_VERBS = (
    "说",
    "声称",
    "表示",
    "宣称",
    "自认",
    "自称",
    "是",
    "知道",
    "获知",
    "掌握",
    "认为",
)
_FIRST_PERSON_DENIAL_VERB_PATTERN = "|".join(
    re.escape(verb) for verb in _FIRST_PERSON_DENIAL_VERBS
)
_FIRST_PERSON_EVIDENCE_REF = re.compile(
    rf"^(?:{_HARMLESS_FIRST_PERSON_PREFIX_PATTERN})?我"
)
_SELF_ROLE_EVIDENCE_REF = re.compile(
    rf"(?P<subject>我|自己)"
    rf"(?:(?!我|自己|p\d{{2}})[^，。；;]){{0,8}}?"
    rf"(?P<relation>{_ROLE_NEGATION_PATTERN}|是)"
    rf"(?:(?!p\d{{2}})[^，。；;]){{0,4}}"
)
_FIRST_PERSON_DENIAL_REF = re.compile(
    rf"^(?:{_HARMLESS_FIRST_PERSON_PREFIX_PATTERN})?我"
    rf"(?:{_FIRST_PERSON_DENIAL_PREFIX_PATTERN})"
    rf"(?:{_FIRST_PERSON_DENIAL_VERB_PATTERN})"
)
_QUOTED_NEGATED_ROLE_REF = re.compile(
    rf"^(?P<subject>p\d{{2}}|我)(?:{_ROLE_NEGATION_PATTERN})(?P<role>.+)$"
)
_QUOTED_SELF_DENIAL_ROLE_REF = re.compile(
    rf"^(?:{_HARMLESS_FIRST_PERSON_PREFIX_PATTERN})?我"
    rf"(?:{_FIRST_PERSON_DENIAL_PREFIX_PATTERN})"
    rf"(?:{_FIRST_PERSON_DENIAL_VERB_PATTERN})(?:我|自己)是(?P<role>.+)$"
)
_SOFT_ATTRIBUTION_DELIMITERS = frozenset(("，", ",", "：", ":"))
_BALANCED_QUOTE_PAIRS = {"“": "”", '"': '"'}
_REPORTING_ATTRIBUTION_FRAGMENT_REF = re.compile(
    r"^p\d{2}(?:(?!p\d{2})[^，。；;！？]){0,10}"
    r"(?:声称|说|表示|宣称)$"
)


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
    repair_failure_history: Iterable[str] = (),
) -> dict[str, Any]:
    """比较首次可修复失败与最终输出，生成不含原始发言的审计字段。"""
    public_speeches = _public_speeches(context)
    source_claims, verified_claims = public_claim_audit_keys(
        source.speech, public_speeches
    )
    final_claims, legacy_final_verified_claims = public_claim_audit_keys(
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
            claim, source_claims, legacy_final_verified_claims
        )
    }
    v2_verified_claims = _polarity_aware_verified_claims(
        final_claims,
        legacy_final_verified_claims,
        public_speeches,
    )
    unsupported_final_claims = final_claims - v2_verified_claims
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
        "repair_failure_history": [
            code
            for code in repair_failure_history
            if code in {"speech_quality", "semantic_claim_retention"}
        ],
    }


def validate_semantic_repair(
    context: AgentContext,
    source: PlayerAction,
    final: PlayerAction,
    *,
    repair_failure_history: Iterable[str] = (),
) -> SemanticRepairValidationResult:
    """按 V2 规则校验修复结果，并返回所有适用的稳定原因码。"""
    audit = build_semantic_repair_audit(
        context,
        source,
        final,
        success=True,
        repair_failure_history=repair_failure_history,
    )
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
    source_claims, legacy_verified_claims = public_claim_audit_keys(
        source.speech, public_speeches
    )
    verified_claims = _polarity_aware_verified_claims(
        source_claims,
        legacy_verified_claims,
        public_speeches,
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
    fallback_claims, fallback_legacy_verified_claims = public_claim_audit_keys(
        speech,
        public_speeches,
    )
    fallback_verified_claims = _polarity_aware_verified_claims(
        fallback_claims,
        fallback_legacy_verified_claims,
        public_speeches,
    )
    if (fallback_claims - source_claims) or (
        fallback_claims - fallback_verified_claims
    ):
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
    """已有非空归因不可删除或换人；空归因可由后续公开证据补齐。"""
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
            known_speakers
            and not final_claim.speaker_attribution
            and not attribution_was_missing
        ):
            return False
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
    """优先按同一说话人匹配，且不让归因变化掩盖可判定的否定翻转。"""
    changed_source_claims = source_claims - final_claims
    for final_claim in final_claims - source_claims:
        content_matches = [
            source_claim for source_claim in changed_source_claims
            if source_claim.content_identity == final_claim.content_identity
        ]
        matching_source = [
            source_claim for source_claim in content_matches
            if source_claim.speaker_attribution == final_claim.speaker_attribution
        ]
        if not matching_source and final_claim.speaker_attribution:
            matching_source = [
                source_claim for source_claim in content_matches
                if not source_claim.speaker_attribution
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


def ordered_semantic_repair_reason_codes(
    reason_codes: Iterable[str],
) -> tuple[str, ...]:
    """对外提供语义修复原因码的固定去重顺序。"""
    return _ordered_reason_codes(reason_codes)


def _polarity_aware_verified_claims(
    final_claims: set[PublicClaimAuditKey],
    legacy_verified_claims: set[PublicClaimAuditKey],
    public_speeches: list[tuple[str, str]],
) -> set[PublicClaimAuditKey]:
    """仅保留存在同极性公开证据键的 V2 已支持声明。"""
    return {
        claim for claim in final_claims
        if (
            claim in legacy_verified_claims
            if claim.claim_type == "system_fact"
            else _has_same_polarity_public_evidence(claim, public_speeches)
        )
    }


def _has_same_polarity_public_evidence(
    claim: PublicClaimAuditKey,
    public_speeches: list[tuple[str, str]],
) -> bool:
    """逐条验证公开来源的说话人、内容和否定极性。"""
    for public_speaker, text in public_speeches:
        speech = [(public_speaker, text)]
        direct_keys, normalized_keys = _classified_evidence_keys(
            claim,
            public_speaker,
            text,
        )
        if claim.claim_type == "player_claim" and claim in direct_keys:
            return True
        if claim.support_kind == "role":
            supported = role_claim_supported(
                claim.target,
                claim.role,
                speech,
                negated=claim.negated,
            )
        elif claim.support_kind == "role_assignment":
            supported = attributed_role_claim_supported(
                claim.speaker_attribution,
                claim.target,
                claim.role,
                speech,
                negated=claim.negated,
            )
        elif claim.support_kind == "night_info":
            supported = night_info_claim_supported(
                claim.target,
                speech,
                negated=claim.negated,
            )
        else:
            supported = False
        if supported and claim in normalized_keys:
            return True
    return False


def _classified_evidence_keys(
    claim: PublicClaimAuditKey,
    public_speaker: str,
    text: str,
) -> tuple[set[PublicClaimAuditKey], set[PublicClaimAuditKey]]:
    """按分句合并直接归因与规范化证据，避免跨句污染否定关系。"""
    direct_evidence_keys: set[PublicClaimAuditKey] = set()
    normalized_evidence_keys: set[PublicClaimAuditKey] = set()
    for clause in _public_evidence_clauses(text):
        direct_keys = {
            public_claim_audit_key(evidence)
            for evidence in classify_public_claims(clause)
        }
        matching_direct_keys = {
            evidence_key for evidence_key in direct_keys
            if evidence_key.content_identity == claim.content_identity
        }
        direct_evidence_keys.update(matching_direct_keys)
        if matching_direct_keys and any(
            evidence_key.speaker_attribution != public_speaker
            for evidence_key in matching_direct_keys
        ):
            continue
        if _is_normalizable_public_evidence(claim, clause):
            normalized_evidence_keys.add(
                PublicClaimAuditKey(
                    claim_type=claim.claim_type,
                    target=claim.target,
                    role=claim.role,
                    support_kind=claim.support_kind,
                    speaker_attribution=public_speaker,
                    negated=_public_evidence_is_negated(claim, clause),
                )
            )
    return direct_evidence_keys, normalized_evidence_keys


def _public_evidence_clauses(text: str) -> list[str]:
    """分离互不影响的公开陈述分句，保留各自的归因和否定范围。"""
    raw_clauses: list[tuple[str, str]] = []
    parts = re.split(r"([，,：:。；;！？!?])", text)
    for index in range(0, len(parts), 2):
        stripped_clause = parts[index].strip()
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        raw_clauses.append((stripped_clause, delimiter))

    clauses: list[str] = []
    index = 0
    while index < len(raw_clauses):
        clause, delimiter = raw_clauses[index]
        if not clause:
            index += 1
            continue
        if (
            index + 1 < len(raw_clauses)
            and delimiter in _SOFT_ATTRIBUTION_DELIMITERS
            and _REPORTING_ATTRIBUTION_FRAGMENT_REF.fullmatch(clause)
            and raw_clauses[index + 1][0]
        ):
            clauses.append(
                _normalized_reporting_clause(
                    clause,
                    raw_clauses[index + 1][0],
                )
            )
            index += 2
            continue
        clauses.append(clause)
        index += 1
    return clauses


def _normalized_reporting_clause(
    reporting_fragment: str,
    continuation: str,
) -> str:
    """只在已确认的转述后解包成对引号，保留转述者归因。"""
    original_clause = f"{reporting_fragment}{continuation}"
    if not _REPORTING_ATTRIBUTION_FRAGMENT_REF.fullmatch(reporting_fragment):
        return original_clause
    opening_quote = continuation[:1]
    if _BALANCED_QUOTE_PAIRS.get(opening_quote) != continuation[-1:]:
        return original_clause
    quoted_claim = continuation[1:-1].strip()
    reporter = reporting_fragment[:3]
    if negated_claim := _canonical_quoted_negated_claim(reporter, quoted_claim):
        return negated_claim
    if quoted_claim.startswith("我"):
        quoted_claim = f"自己{quoted_claim[1:]}"
    return f"{reporting_fragment}{quoted_claim or continuation}"


def _canonical_quoted_negated_claim(reporter: str, quoted_claim: str) -> str | None:
    """把受限引号内的角色否定转换为既有的否认声明形式。"""
    self_denial = _QUOTED_SELF_DENIAL_ROLE_REF.fullmatch(quoted_claim)
    if self_denial:
        return f"{reporter}并未声称自己是{self_denial.group('role')}"
    negated_role = _QUOTED_NEGATED_ROLE_REF.fullmatch(quoted_claim)
    if not negated_role:
        return None
    subject = negated_role.group("subject")
    if subject == "我":
        subject = "自己"
    return f"{reporter}并未声称{subject}是{negated_role.group('role')}"


def _is_normalizable_public_evidence(
    claim: PublicClaimAuditKey,
    text: str,
) -> bool:
    """仅把明确的一人称或直接目标身份判断补齐为账本说话人的证据。"""
    if (
        (
            _FIRST_PERSON_EVIDENCE_REF.search(text)
            or _self_role_evidence_is_bound(text)
        )
        and _clause_carries_claim(
        claim,
        text,
        )
    ):
        return True
    return bool(
        claim.support_kind == "role_assignment"
        and claim.role
        and re.search(
            rf"{re.escape(claim.target)}(?:(?!p\d{{2}})[^，。；;]){{0,8}}"
            rf"(?:是|{_ROLE_NEGATION_PATTERN})[^，。；;]{{0,4}}"
            rf"{re.escape(claim.role)}",
            text,
        )
    )


def _clause_carries_claim(claim: PublicClaimAuditKey, text: str) -> bool:
    """判断一人称分句是否实际包含候选声明的必要内容。"""
    if claim.support_kind == "role":
        return bool(claim.role and claim.role in text)
    if claim.support_kind == "role_assignment":
        return bool(claim.target in text and claim.role in text)
    if claim.support_kind == "night_info":
        return bool(
            re.search(r"(?:知道|获知|掌握)", text)
            and re.search(r"(?:狼刀|刀口|狼队刀|被刀)", text)
        )
    return False


def _public_evidence_is_negated(
    claim: PublicClaimAuditKey,
    text: str,
) -> bool:
    """识别一人称否认、直接身份判断及权威前缀中的否定极性。"""
    if _FIRST_PERSON_DENIAL_REF.search(text):
        return True
    self_role_match = _SELF_ROLE_EVIDENCE_REF.search(text)
    if self_role_match and not _self_role_evidence_is_bound(text):
        self_role_match = None
    if self_role_match:
        if self_role_match.group("relation") in _ROLE_NEGATION_FORMS:
            return True
        return public_claim_is_negated(text, self_role_match.start("subject"))
    target_start = text.find(claim.target)
    if target_start >= 0 and public_claim_is_negated(text, target_start):
        return True
    if claim.role and re.search(
        rf"{re.escape(claim.target)}(?:(?!p\d{{2}})[^，。；;]){{0,8}}"
        rf"(?:{_ROLE_NEGATION_PATTERN})[^，。；;]{{0,4}}"
        rf"{re.escape(claim.role)}",
        text,
    ):
        return True
    return bool(
        claim.support_kind == "night_info"
        and re.search(r"(?:不|并不|未|没有)(?:知道|获知|掌握)", text)
    )


def _self_role_evidence_is_bound(text: str) -> bool:
    """仅接受句首或受限第一人称前缀，避免把引用他人内容归给当前说话者。"""
    match = _SELF_ROLE_EVIDENCE_REF.search(text)
    if match is None:
        return False
    prefix = text[: match.start("subject")]
    if not prefix or prefix in _HARMLESS_FIRST_PERSON_PREFIXES:
        return True
    return bool(
        re.search(
            r"(?:不能|无法|不应)说$|(?:并不|不)认为$|"
            r"(?:并未|没有|未曾|否认)[^，。；;]{0,8}"
            r"(?:声称|说|表示|宣称|自认)$",
            prefix,
        )
    )


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
    "ordered_semantic_repair_reason_codes",
    "preserve_verified_claim_in_fallback",
    "semantic_repair_correction_hint",
    "semantic_repair_rejection_message",
    "semantic_repair_retains_verified_claim",
    "validate_semantic_repair",
]
