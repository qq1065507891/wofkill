# -*- coding: utf-8 -*-
"""
统计公开发言中缺少公开来源支撑的事实声明。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.evaluation.balance_public_claims import (
    ...     unsupported_public_fact_claim_count,
    ... )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

_PUBLIC_ROLE_CLAIM_REF = re.compile(
    r"(p\d{2})[^，。；;]{0,10}(?:已?自认|认了?|自称|声称自己是|说自己是|跳)"
    r"(狼人|预言家|女巫|猎人|白痴|村民|民)"
)
_PUBLIC_NIGHT_INFO_REF = re.compile(
    r"(p\d{2}).{0,14}(?:声称|说|表示|宣称)?.{0,8}"
    r"(?:知道|获知|掌握).{0,10}(?:狼刀|刀口|狼队刀|被刀)"
)
_SYSTEM_ROLE_FACT_REF = re.compile(
    r"(?:系统|主持人|法官)(?:已经|已)?确认(p\d{2})是"
    r"(狼人|预言家|女巫|猎人|白痴|村民|民)"
)
_NEGATION_MARKERS = ("不认为", "并非", "没有", "未", "不能")
_CURRENT_PLAYER_INFERENCE_REF = re.compile(
    r"(?:我认为|我怀疑|我推测)(p\d{2})[^，。；;]{0,12}(?:是狼人|更可疑|有问题|像狼)?"
)


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


def classify_public_claims(text: str) -> list[ClassifiedPublicClaim]:
    """按语义来源分类，不把玩家归因或当前推断提升为系统事实。"""
    found: list[ClassifiedPublicClaim] = []
    for pattern, claim_type in (
        (_PUBLIC_ROLE_CLAIM_REF, PublicClaimType.PLAYER_CLAIM),
        (_PUBLIC_NIGHT_INFO_REF, PublicClaimType.PLAYER_CLAIM),
        (_SYSTEM_ROLE_FACT_REF, PublicClaimType.SYSTEM_FACT),
        (_CURRENT_PLAYER_INFERENCE_REF, PublicClaimType.CURRENT_PLAYER_INFERENCE),
    ):
        found.extend(
            ClassifiedPublicClaim(claim_type, match.group(0), match.start(), match.end())
            for match in pattern.finditer(text)
        )
    return sorted(found, key=lambda claim: (claim.start, claim.end))
_ROLE_MARKERS = {
    "狼人": ("我是狼人", "认狼", "自认狼人", "我们狼队"),
    "预言家": ("我是预言家", "我跳预言家", "认预言家", "跳预言家", "悍跳预言家"),
    "女巫": ("我是女巫", "我认女巫", "跳女巫"),
    "猎人": ("我是猎人", "我认猎人", "跳猎人"),
    "白痴": ("我是白痴", "我认白痴", "跳白痴"),
    "村民": ("我是村民", "我是民", "我认民"),
    "民": ("我是村民", "我是民", "我认民"),
}


def unsupported_public_fact_claim_count(game: dict[str, Any]) -> int:
    """统计单局中缺少公开发言支撑的事实引用数量。"""
    public_speeches: list[tuple[str, str]] = []
    count = 0
    for event in game.get("events", []):
        event_type = event.get("type")
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
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
                if isinstance(vote, dict):
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
    count = 0
    for match in _PUBLIC_ROLE_CLAIM_REF.finditer(text):
        player_id, role = match.group(1), match.group(2)
        if not role_claim_supported(player_id, role, public_speeches):
            count += 1
    for match in _PUBLIC_NIGHT_INFO_REF.finditer(text):
        player_id = match.group(1)
        if not night_info_claim_supported(player_id, public_speeches):
            count += 1
    for match in _SYSTEM_ROLE_FACT_REF.finditer(text):
        if not _match_is_negated(text, match):
            count += 1
    return count


def sanitize_public_text(
    text: str,
    public_speeches: list[tuple[str, str]],
) -> tuple[str, int]:
    """在公开事件写入前屏蔽没有公开来源支撑的事实引用。"""
    redacted = 0

    def replace_role(match: re.Match[str]) -> str:
        nonlocal redacted
        player_id, role = match.group(1), match.group(2)
        if role_claim_supported(player_id, role, public_speeches):
            return match.group(0)
        redacted += 1
        return "[未公开事实]"

    def replace_night(match: re.Match[str]) -> str:
        nonlocal redacted
        player_id = match.group(1)
        if night_info_claim_supported(player_id, public_speeches):
            return match.group(0)
        redacted += 1
        return "[未公开事实]"

    def replace_system_fact(match: re.Match[str]) -> str:
        nonlocal redacted
        if _match_is_negated(text, match):
            return match.group(0)
        redacted += 1
        return f"对{match.group(1)}的身份仅作公开推测"

    sanitized = _PUBLIC_ROLE_CLAIM_REF.sub(replace_role, text)
    sanitized = _PUBLIC_NIGHT_INFO_REF.sub(replace_night, sanitized)
    sanitized = _SYSTEM_ROLE_FACT_REF.sub(replace_system_fact, sanitized)
    return sanitized, redacted


def _match_is_negated(text: str, match: re.Match[str]) -> bool:
    """仅检查同一短分句中的否定词，避免否定范围跨句扩散。"""
    prefix = text[max(0, match.start() - 12):match.start()]
    clause = re.split(r"[，。；;]", prefix)[-1]
    return any(marker in clause for marker in _NEGATION_MARKERS)


def public_speech_history(events: list[Any]) -> list[tuple[str, str]]:
    """提取当前事件之前已经公开的发言，供发布前事实校验使用。"""
    history: list[tuple[str, str]] = []
    for event in events:
        event_type = getattr(event, "type", None)
        payload = getattr(event, "payload", {}) or {}
        if isinstance(event, dict):
            event_type = event.get("type")
            payload = event.get("payload") or {}
        if event_type not in {
            "speech",
            "sheriff_speech",
            "sheriff_pk_speech",
            "tie_pk_speech",
            "exile_last_words",
            "night_death_last_words",
        } or not isinstance(payload, dict):
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
) -> bool:
    """判断玩家公开发言是否已经支撑某个角色声明。"""
    markers = _ROLE_MARKERS.get(role, (role,))
    return any(
        speaker == player_id and any(marker in speech for marker in markers)
        for speaker, speech in public_speeches
    )


def night_info_claim_supported(
    player_id: str,
    public_speeches: list[tuple[str, str]],
) -> bool:
    """判断玩家公开发言是否已经支撑夜间信息来源声明。"""
    knowledge_markers = ("知道", "获知", "掌握")
    night_markers = ("狼刀", "刀口", "狼队刀", "被刀")
    return any(
        speaker == player_id
        and any(marker in speech for marker in knowledge_markers)
        and any(marker in speech for marker in night_markers)
        for speaker, speech in public_speeches
    )
