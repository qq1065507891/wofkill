# -*- coding: utf-8 -*-
"""
统计并分类公开发言中需要公开来源支撑的事实声明。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-27

使用示例:
    >>> from werewolf_agent.evaluation.balance_public_claims import (
    ...     unsupported_public_fact_claim_count,
    ... )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameEvent

_PUBLIC_ROLE_CLAIM_REF = re.compile(
    r"(p\d{2})(?:(?!p\d{2})[^，。；;]){0,10}"
    r"(?:已?自认|认了?|自称|声称自己(?:不是|并非|不为|是)|说自己(?:不是|并非|不为|是)|(?<!对)跳)"
    r"(狼人|预言家|女巫|猎人|白痴|村民|民)"
)
_PUBLIC_ATTRIBUTED_ROLE_ASSIGNMENT_REF = re.compile(
    r"(p\d{2})(?:(?!p\d{2})[^，。；;]){0,10}"
    r"(?:声称|说|表示|宣称)(p\d{2})(?:(?!p\d{2})[^，。；;]){0,8}(?:不是|并非|不为|是)"
    r"(狼人|预言家|女巫|猎人|白痴|村民|民)"
)
_PUBLIC_UNATTRIBUTED_ROLE_ASSIGNMENT_REF = re.compile(
    r"有人(?:(?!p\d{2})[^，。；;]){0,8}(?:声称|说|表示|宣称)"
    r"(p\d{2})(?:(?!p\d{2})[^，。；;]){0,8}(?:不是|并非|不为|是)"
    r"(狼人|预言家|女巫|猎人|白痴|村民|民)"
)
_PUBLIC_NIGHT_INFO_REF = re.compile(
    r"(p\d{2})[^，。；;]{0,14}(?:声称|说|表示|宣称)?[^，。；;]{0,8}"
    r"(?:知道|获知|掌握)[^，。；;]{0,10}(?:狼刀|刀口|狼队刀|被刀)"
)
_NIGHT_INFO_NEGATION_FORMS = (
    "不知道",
    "并不知道",
    "未获知",
    "没有获知",
    "未曾获知",
    "没获知",
)
_NIGHT_INFO_RELATION_PATTERN = "|".join(
    (*_NIGHT_INFO_NEGATION_FORMS, "知道", "获知", "掌握")
)
_SYSTEM_ROLE_FACT_REF = re.compile(
    r"(?:系统|主持人|法官)(?:(没有|并未|尚未|未))?(?:已经|已)?确认(p\d{2})是"
    r"(狼人|预言家|女巫|猎人|白痴|村民|民)"
)
_ASSERTIVE_NEGATION_PREFIXES = (
    re.compile(r"(?:不能|无法|不可)否认$"),
    re.compile(r"没有理由不信$"),
)
_NEGATING_PREFIXES = (
    re.compile(r"(?:不能|无法|不可|不应)说$"),
    re.compile(r"(?:并不|不)认为$"),
    re.compile(r"(?:并非|不是|不代表)$"),
)
_CURRENT_PLAYER_INFERENCE_REF = re.compile(
    r"(?:我认为|我怀疑|我推测)(p\d{2})[^，。；;]{0,12}(?:是狼人|更可疑|有问题|像狼)?"
)
_PUBLIC_ACTION_CLAIM_RE = re.compile(
    r"(?P<actor>我|p\d{2})?[^，。；;]{0,12}"
    r"(?P<action>已经开枪|开枪带走|首夜用解药救了|用解药救了)"
    r"\s*(?P<target>p\d{2})(?![A-Za-z0-9_])"
)
_COMPLETED_ACTION_SUPPORT_KINDS = frozenset({"hunter_shot", "witch_antidote"})


class PublicClaimType(str, Enum):
    """公开文本中需要不同证据规则的声明类型。"""

    SYSTEM_FACT = "system_fact"
    PLAYER_CLAIM = "player_claim"
    CURRENT_PLAYER_INFERENCE = "current_player_inference"


@dataclass(frozen=True)
class ClassifiedPublicClaim:
    """保留声明类型、原文和文本位置，供校验与修复共享。"""

    claim_type: PublicClaimType
    text: str
    start: int
    end: int
    target: str | None = None
    role: str | None = None
    support_kind: str | None = None
    speaker_attribution: str | None = None
    negated: bool = False
    day: int | None = None


@dataclass(frozen=True)
class PublicClaimAuditKey:
    """事实声明的稳定结构键，独立保留说话者归属与否定关系。"""

    claim_type: str
    target: str
    role: str
    support_kind: str
    speaker_attribution: str
    negated: bool

    @property
    def content_identity(self) -> tuple[str, str, str, str]:
        """返回不含说话者和否定关系的声明内容身份。"""
        return self.claim_type, self.target, self.role, self.support_kind

    @property
    def attribution_agnostic_identity(self) -> tuple[str, str, str, str, bool]:
        """返回用于判定合法补归因的声明身份。"""
        return (*self.content_identity, self.negated)


def classify_public_claims(
    text: str,
    *,
    speaker: str | None = None,
) -> list[ClassifiedPublicClaim]:
    """按语义来源分类，不把玩家归因或当前推断提升为系统事实。"""
    found: list[ClassifiedPublicClaim] = []
    for match in _PUBLIC_ROLE_CLAIM_REF.finditer(text):
        found.append(
            ClassifiedPublicClaim(
                PublicClaimType.PLAYER_CLAIM,
                match.group(0),
                match.start(),
                match.end(),
                target=match.group(1),
                role=match.group(2),
                support_kind="role",
                speaker_attribution=match.group(1),
                negated=_player_claim_is_negated(text, match.start(), match.end()),
            )
        )
    for match in _PUBLIC_ATTRIBUTED_ROLE_ASSIGNMENT_REF.finditer(text):
        found.append(
            ClassifiedPublicClaim(
                PublicClaimType.PLAYER_CLAIM,
                match.group(0),
                match.start(),
                match.end(),
                target=match.group(2),
                role=match.group(3),
                support_kind="role_assignment",
                speaker_attribution=match.group(1),
                negated=_player_claim_is_negated(text, match.start(), match.end()),
            )
        )
    for match in _PUBLIC_UNATTRIBUTED_ROLE_ASSIGNMENT_REF.finditer(text):
        found.append(
            ClassifiedPublicClaim(
                PublicClaimType.PLAYER_CLAIM,
                match.group(0),
                match.start(),
                match.end(),
                target=match.group(1),
                role=match.group(2),
                support_kind="role_assignment",
                negated=_player_claim_is_negated(text, match.start(), match.end()),
            )
        )
    for match in _PUBLIC_NIGHT_INFO_REF.finditer(text):
        end = match.end()
        if text.startswith("信息", end):
            end += len("信息")
        found.append(
            ClassifiedPublicClaim(
                PublicClaimType.PLAYER_CLAIM,
                text[match.start():end],
                match.start(),
                end,
                target=match.group(1),
                support_kind="night_info",
                speaker_attribution=match.group(1),
                negated=_player_claim_is_negated(text, match.start(), end),
            )
        )
    for match in _SYSTEM_ROLE_FACT_REF.finditer(text):
        found.append(
            ClassifiedPublicClaim(
                PublicClaimType.SYSTEM_FACT,
                match.group(0),
                match.start(),
                match.end(),
                target=match.group(2),
                role=match.group(3),
                speaker_attribution="system",
                negated=bool(match.group(1)) or public_claim_is_negated(
                    text, match.start()
                ),
            )
        )
    for match in _CURRENT_PLAYER_INFERENCE_REF.finditer(text):
        found.append(
            ClassifiedPublicClaim(
                PublicClaimType.CURRENT_PLAYER_INFERENCE,
                match.group(0),
                match.start(),
                match.end(),
                target=match.group(1),
            )
        )
    for match in _PUBLIC_ACTION_CLAIM_RE.finditer(text):
        if not _completed_action_match_is_valid(text, match):
            continue
        action_text = match.group("action")
        actor = match.group("actor")
        found.append(
            ClassifiedPublicClaim(
                PublicClaimType.PLAYER_CLAIM,
                match.group(0),
                match.start(),
                match.end(),
                target=match.group("target"),
                support_kind=(
                    "witch_antidote" if "解药" in action_text else "hunter_shot"
                ),
                speaker_attribution=speaker if actor == "我" else actor,
                negated=_completed_action_claim_is_negated(text, match),
                day=1 if "首夜" in match.group(0) else None,
            )
        )
    return _resolve_overlapping_claims(found)
_ROLE_MARKERS = {
    "狼人": ("我是狼人", "认狼", "自认狼人", "我们狼队"),
    "预言家": ("我是预言家", "我跳预言家", "认预言家", "跳预言家", "悍跳预言家"),
    "女巫": ("我是女巫", "我认女巫", "跳女巫"),
    "猎人": ("我是猎人", "我认猎人", "跳猎人"),
    "白痴": ("我是白痴", "我认白痴", "跳白痴"),
    "村民": ("我是村民", "我是民", "我认民"),
    "民": ("我是村民", "我是民", "我认民"),
}
_ROLE_NEGATION_FORMS = ("不是", "并非", "不为")
_ROLE_RELATION_PATTERN = "|".join(
    ("不是", "并非", "不为", "是")
)


def unsupported_public_fact_claim_count(game: dict[str, Any]) -> int:
    """统计单局中缺少公开发言支撑的事实引用数量。"""
    public_speeches: list[tuple[str, str]] = []
    count = 0
    for event in game.get("events", []):
        event_type = event.get("type")
        payload = event.get("payload") or {}
        if not isinstance(payload, Mapping):
            continue

        if event_type in {
            "speech",
            "sheriff_speech",
            "sheriff_pk_speech",
            "exile_last_words",
            "night_death_last_words",
        }:
            speaker = str(
                payload.get("speaker")
                or payload.get("player_id")
                or payload.get("candidate_id")
                or ""
            )
            text = str(payload.get("text") or payload.get("speech") or "")
            if text:
                count += unsupported_claims_in_text(text, public_speeches)
            if speaker and text:
                public_speeches.append((speaker, text))
            continue

        if event_type == "vote_resolved":
            for vote in payload.get("votes") or []:
                if isinstance(vote, Mapping):
                    count += unsupported_claims_in_text(
                        str(vote.get("reason") or ""),
                        public_speeches,
                    )
    return count


def unsupported_claims_in_text(
    text: str,
    public_speeches: list[tuple[str, str]],
) -> int:
    """统计一段文本里没有历史公开材料支持的角色或夜间信息引用。"""
    return sum(
        not _claim_is_supported(claim, public_speeches)
        for claim in classify_public_claims(text)
        if claim.claim_type != PublicClaimType.CURRENT_PLAYER_INFERENCE
    )


def public_claim_audit_keys(
    text: str,
    public_speeches: list[tuple[str, str]],
    *,
    speaker: str | None = None,
    public_evidence: Mapping[str, Any] | None = None,
) -> tuple[set[PublicClaimAuditKey], set[PublicClaimAuditKey]]:
    """返回事实 claim 的稳定键及其中有公开来源支撑的子集。"""
    claims = [
        claim for claim in classify_public_claims(text, speaker=speaker)
        if claim.claim_type != PublicClaimType.CURRENT_PLAYER_INFERENCE
    ]
    keyed = {
        public_claim_audit_key(claim): claim
        for claim in claims
    }
    verified = {
        key for key, claim in keyed.items()
        if (
            _completed_action_claim_is_supported(claim, public_evidence)
            if claim.support_kind in _COMPLETED_ACTION_SUPPORT_KINDS
            else _claim_is_supported(claim, public_speeches)
        )
    }
    return set(keyed), verified


def public_claim_audit_key(claim: ClassifiedPublicClaim) -> PublicClaimAuditKey:
    """把分类结果投影为不含原文的稳定审计键。"""
    return PublicClaimAuditKey(
        claim_type=claim.claim_type.value,
        target=claim.target or "",
        role=claim.role or "",
        support_kind=claim.support_kind or "",
        speaker_attribution=claim.speaker_attribution or "",
        negated=claim.negated,
    )


def sanitize_public_text(
    text: str,
    public_speeches: list[tuple[str, str]],
) -> tuple[str, int]:
    """在公开事件写入前屏蔽没有公开来源支撑的事实引用。"""
    unsupported = [
        claim
        for claim in classify_public_claims(text)
        if claim.claim_type != PublicClaimType.CURRENT_PLAYER_INFERENCE
        and not _claim_is_supported(claim, public_speeches)
    ]
    sanitized = text
    for claim in reversed(unsupported):
        target = claim.target or "该玩家"
        replacement = f"对{target}的身份声明暂不采信，需继续核验"
        sanitized = sanitized[:claim.start] + replacement + sanitized[claim.end:]
    return sanitized, len(unsupported)


def _resolve_overlapping_claims(
    claims: list[ClassifiedPublicClaim],
) -> list[ClassifiedPublicClaim]:
    """将分类器的重叠 span 解析为互不重叠的确定性结果。

    同一文本区间只能修复一次；优先保留覆盖信息更完整的长 span。
    """
    ordered = sorted(claims, key=lambda claim: (claim.start, -claim.end))
    resolved: list[ClassifiedPublicClaim] = []
    cluster: list[ClassifiedPublicClaim] = []
    cluster_end = -1
    for claim in ordered:
        if cluster and claim.start >= cluster_end:
            resolved.append(_most_specific_claim(cluster))
            cluster = []
            cluster_end = -1
        cluster.append(claim)
        cluster_end = max(cluster_end, claim.end)
    if cluster:
        resolved.append(_most_specific_claim(cluster))
    return sorted(resolved, key=lambda claim: (claim.start, claim.end))


def _most_specific_claim(
    claims: list[ClassifiedPublicClaim],
) -> ClassifiedPublicClaim:
    """选择最长、语义字段最完整的声明。"""
    return max(
        claims,
        key=lambda claim: (
            claim.end - claim.start,
            claim.claim_type == PublicClaimType.CURRENT_PLAYER_INFERENCE,
            bool(claim.support_kind),
            bool(claim.role),
            -claim.start,
        ),
    )


def _claim_is_supported(
    claim: ClassifiedPublicClaim,
    public_speeches: list[tuple[str, str]],
) -> bool:
    """依据分类结果校验公开支撑，避免清洗器维护第二套识别规则。"""
    if claim.claim_type == PublicClaimType.SYSTEM_FACT:
        return claim.negated
    if claim.claim_type == PublicClaimType.CURRENT_PLAYER_INFERENCE:
        return True
    if claim.support_kind == "role" and claim.target and claim.role:
        return role_claim_supported(
            claim.target,
            claim.role,
            public_speeches,
            negated=claim.negated,
        )
    if (
        claim.support_kind == "role_assignment"
        and claim.speaker_attribution
        and claim.target
        and claim.role
    ):
        return attributed_role_claim_supported(
            claim.speaker_attribution,
            claim.target,
            claim.role,
            public_speeches,
            negated=claim.negated,
        )
    if claim.support_kind == "night_info" and claim.target:
        return night_info_claim_supported(
            claim.target,
            public_speeches,
            negated=claim.negated,
        )
    return False


def public_claim_is_negated(text: str, claim_start: int) -> bool:
    """按同一分句的权威语法范围判断声明是否被否定。"""
    clause = re.split(r"[，。；;！？]", text[:claim_start])[-1].strip()
    if any(pattern.search(clause) for pattern in _ASSERTIVE_NEGATION_PREFIXES):
        return False
    return any(pattern.search(clause) for pattern in _NEGATING_PREFIXES)


def _player_claim_is_negated(text: str, claim_start: int, claim_end: int) -> bool:
    """识别玩家声明关系自身的否定，不把双重否定误判为否定。"""
    prefix = re.split(r"[，。；;！？]", text[:claim_start])[-1].strip()
    if any(pattern.search(prefix) for pattern in _ASSERTIVE_NEGATION_PREFIXES):
        return False
    relation = text[claim_start:claim_end]
    return public_claim_is_negated(text, claim_start) or bool(
        re.search(
            r"(?:并未|没有|未曾|否认)[^，。；;]{0,8}"
            r"(?:声称|说|表示|宣称|自认)",
            relation,
        )
        or re.search(
            r"(?:不是|并非|不为)[^，。；;]{0,4}"
            r"(?:狼人|预言家|女巫|猎人|白痴|村民|民)",
            relation,
        )
        or re.search(
            rf"(?:{'|'.join(_NIGHT_INFO_NEGATION_FORMS)})[^，。；;]{{0,10}}"
            r"(?:狼刀|刀口|狼队刀|被刀)",
            relation,
        )
    )


def _completed_action_claim_is_negated(text: str, match: re.Match[str]) -> bool:
    """识别行动谓语前的局部否定，不把否认执行当成已执行。"""
    action_prefix = text[match.start():match.start("action")]
    return public_claim_is_negated(text, match.start()) or bool(
        re.search(r"(?:没有|并未|未曾|从未|不曾|没)[^，。；;]{0,4}$", action_prefix)
    )


def _completed_action_match_is_valid(text: str, match: re.Match[str]) -> bool:
    """排除建议性措辞，并阻止 ASCII 单词粘连到玩家标识。"""
    actor = match.group("actor")
    if actor:
        actor_start = match.start("actor")
        if actor_start and re.match(r"[A-Za-z0-9_]", text[actor_start - 1]):
            return False
    action_prefix = text[match.start():match.start("action")]
    if re.search(r"(?:建议|计划|准备|打算|想要|考虑|提议)", action_prefix):
        return False
    if actor is None and re.search(r"[A-Za-z0-9_]p\d{2}", action_prefix):
        return False
    return True


def _completed_action_claim_is_supported(
    claim: ClassifiedPublicClaim,
    public_evidence: Mapping[str, Any] | None,
) -> bool:
    """仅接受 actor/action/target 完全一致的引擎确认行动。"""
    if (
        claim.negated
        or not claim.speaker_attribution
        or not claim.target
        or not isinstance(public_evidence, Mapping)
    ):
        return False
    claim_days = _completed_action_claim_days(claim, public_evidence)
    confirmed_actions = public_evidence.get("confirmed_actions", ())
    if not isinstance(confirmed_actions, (list, tuple)):
        return False
    return any(
        isinstance(action, Mapping)
        and action.get("actor") == claim.speaker_attribution
        and action.get("action") == claim.support_kind
        and action.get("target") == claim.target
        and any(
            _action_days_are_compatible(claim_day, action.get("day", 0))
            for claim_day in claim_days
        )
        for action in confirmed_actions
    )


def _completed_action_claim_days(
    claim: ClassifiedPublicClaim,
    public_evidence: Mapping[str, Any],
) -> tuple[Any, ...]:
    """从同一玩家声明绑定显式日次，但不把声明当成执行证据。"""
    if claim.day is not None:
        return (claim.day,)
    action_claims = public_evidence.get("action_claims", ())
    if not isinstance(action_claims, (list, tuple)):
        return (0,)
    days = tuple(
        action.get("day", 0)
        for action in action_claims
        if isinstance(action, Mapping)
        and action.get("speaker") == claim.speaker_attribution
        and action.get("action") == claim.support_kind
        and action.get("target") == claim.target
    )
    return days or (0,)


def _action_days_are_compatible(claim_day: Any, action_day: Any) -> bool:
    """与公开账本一致：显式日次必须相等，缺失或零值保留兼容。"""
    return not claim_day or not action_day or claim_day == action_day


def _public_event_parts(event: Any) -> tuple[Any, Mapping[str, Any]] | None:
    """提取公开事件的类型和 payload，兼容结构型旧事件。"""
    if isinstance(event, Mapping):
        event_type = event.get("type")
        payload = event.get("payload") or {}
        visibility = event.get("visibility")
    elif isinstance(event, GameEvent):
        if event_visibility(event) is not EventVisibility.PUBLIC:
            return None
        return event.type, event.payload or {}
    else:
        event_type = getattr(event, "type", None)
        payload = getattr(event, "payload", {}) or {}
        visibility = getattr(event, "visibility", None)

    if not isinstance(payload, Mapping):
        return None
    if visibility in (None, ""):
        visibility = payload.get("visibility", "public")
    if EventVisibility.from_legacy(visibility) is not EventVisibility.PUBLIC:
        return None
    return event_type, payload


def public_speech_history(events: list[Any]) -> list[tuple[str, str]]:
    """提取当前事件之前已经公开的发言，供发布前事实校验使用。"""
    history: list[tuple[str, str]] = []
    for event in events:
        parts = _public_event_parts(event)
        if parts is None:
            continue
        event_type, payload = parts
        if event_type not in {
            "speech",
            "sheriff_speech",
            "sheriff_pk_speech",
            "tie_pk_speech",
            "exile_last_words",
            "night_death_last_words",
        } or not isinstance(payload, Mapping):
            continue
        speaker = str(
            payload.get("speaker")
            or payload.get("player_id")
            or payload.get("candidate_id")
            or ""
        )
        text = str(payload.get("text") or payload.get("speech") or "")
        if speaker and text:
            history.append((speaker, text))
    return history


def role_claim_supported(
    player_id: str,
    role: str,
    public_speeches: list[tuple[str, str]],
    *,
    negated: bool = False,
) -> bool:
    """判断玩家公开发言是否已经支撑某个角色声明。"""
    markers = _ROLE_MARKERS.get(role, (role,))
    if not negated:
        for speaker, speech in public_speeches:
            if speaker != player_id:
                continue
            if _role_relation_supported_in_speech(
                player_id,
                role,
                speech,
                negated=True,
                self_claim=True,
            ):
                continue
            if _role_relation_supported_in_speech(
                player_id,
                role,
                speech,
                negated=False,
                self_claim=True,
            ) or any(marker in speech for marker in markers):
                return True
        return False
    return any(
        speaker == player_id
        and _role_relation_supported_in_speech(
            player_id,
            role,
            speech,
            negated=True,
            self_claim=True,
        )
        for speaker, speech in public_speeches
    )


def attributed_role_claim_supported(
    speaker: str,
    target: str,
    role: str,
    public_speeches: list[tuple[str, str]],
    *,
    negated: bool = False,
) -> bool:
    """判断指定玩家是否曾公开对目标作出该角色归属声明。"""
    return any(
        public_speaker == speaker
        and _role_relation_supported_in_speech(
            target,
            role,
            speech,
            negated=negated,
        )
        for public_speaker, speech in public_speeches
    )


def _role_relation_supported_in_speech(
    target: str,
    role: str,
    speech: str,
    *,
    negated: bool,
    self_claim: bool = False,
) -> bool:
    """匹配同一玩家与角色的明确肯定或否定关系，避免只按词面判断。"""
    target_ref = r"(?:我|自己)" if self_claim else re.escape(target)
    search_speech = speech
    if not self_claim:
        search_speech = _clause_containing_target(speech, target)
        if search_speech is None:
            return False
    if self_claim:
        self_match = re.search(
            rf"(?P<subject>我|自己)"
            rf"(?:(?!我|自己|p\d{{2}})[^，。；;]){{0,8}}?"
            rf"(?P<relation>{_ROLE_RELATION_PATTERN})"
            rf"(?:(?!p\d{{2}})[^，。；;]){{0,4}}{re.escape(role)}",
            search_speech,
        )
        if self_match is not None:
            relation_negated = (
                self_match.group("relation") in _ROLE_NEGATION_FORMS
            )
            prefix_negated = public_claim_is_negated(
                search_speech,
                self_match.start("subject"),
            )
            prefix_negated = prefix_negated or bool(
                re.search(
                    r"(?:并未|没有|未曾|否认)[^，。；;]{0,8}"
                    r"(?:声称|说|表示|宣称|自认)",
                    search_speech[: self_match.start("subject")],
                )
            )
            return (relation_negated or prefix_negated) is negated
    if not self_claim and re.search(
        rf"(?:我|自己)(?:并未|没有|未曾|否认)[^，。；;]{{0,8}}"
        rf"{re.escape(target)}(?:(?!p\d{{2}})[^，。；;]){{0,4}}"
        rf"{re.escape(role)}",
        search_speech,
    ):
        return negated
    if re.search(
        rf"{target_ref}(?:并未|没有|未曾|否认)[^，。；;]{{0,12}}"
        rf"(?:声称|说|表示|宣称|自认)?[^，。；;]{{0,4}}{re.escape(role)}",
        search_speech,
    ):
        return negated
    match = re.search(
        rf"{target_ref}(?P<between>(?:(?!p\d{{2}})[^，。；;]){{0,8}}?)"
        rf"(?P<relation>{_ROLE_RELATION_PATTERN})"
        rf"(?:(?!p\d{{2}})[^，。；;]){{0,4}}{re.escape(role)}",
        search_speech,
    )
    if match is None:
        return False
    relation_negated = match.group("relation") in _ROLE_NEGATION_FORMS
    if relation_negated:
        return negated
    target_start = search_speech.find(target)
    prefix_negated = target_start >= 0 and public_claim_is_negated(
        search_speech,
        target_start,
    )
    if prefix_negated:
        return negated
    return not negated


def _clause_containing_target(speech: str, target: str) -> str | None:
    """返回包含目标的单个标点分句，避免跨目标拼接角色证据。"""
    target_start = speech.find(target)
    if target_start < 0:
        return None
    clauses = re.split(r"[，。；;！？!?:：]", speech)
    offset = 0
    for clause in clauses:
        clause_end = offset + len(clause)
        if offset <= target_start < clause_end:
            return clause
        offset = clause_end + 1
    return None


def night_info_claim_supported(
    player_id: str,
    public_speeches: list[tuple[str, str]],
    *,
    negated: bool = False,
) -> bool:
    """判断玩家公开发言是否已经支撑夜间信息来源声明。"""
    for speaker, speech in public_speeches:
        if speaker != player_id:
            continue
        match = re.search(
            rf"{re.escape(player_id)}(?:(?!p\d{{2}})[^，。；;]){{0,14}}?"
            rf"(?P<relation>{_NIGHT_INFO_RELATION_PATTERN})"
            rf"(?:(?!p\d{{2}})[^，。；;]){{0,10}}"
            r"(?:狼刀|刀口|狼队刀|被刀)",
            speech,
        )
        if match is None:
            match = re.search(
                rf"(?:我|自己)(?:(?!p\d{{2}})[^，。；;]){{0,8}}?"
                rf"(?P<relation>{_NIGHT_INFO_RELATION_PATTERN})"
                rf"(?:(?!p\d{{2}})[^，。；;]){{0,10}}"
                r"(?:狼刀|刀口|狼队刀|被刀)",
                speech,
            )
        if match is not None:
            relation_negated = (
                match.group("relation") in _NIGHT_INFO_NEGATION_FORMS
            )
            if relation_negated is negated:
                return True
    return False
