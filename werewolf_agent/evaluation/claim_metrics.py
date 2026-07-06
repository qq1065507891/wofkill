# -*- coding: utf-8 -*-
"""
评价指标中的角色声明事件抽取 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.evaluation.claim_metrics import _extract_claim_events
    >>> _extract_claim_events([{"type": "speech", "payload": {"text": "我是预言家", "speaker": "p01"}}])
"""

from __future__ import annotations

import re
from typing import Any


_CLAIM_ROLE_MAP = {
    "预言家": "seer",
    "女巫": "witch",
    "猎人": "hunter",
    "白痴": "idiot",
    "村民": "villager",
    "平民": "villager",
    "混血儿": "hybrid",
    "狼人": "werewolf",
}


def _extract_claim_events(event_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for event in event_log:
        event_type = event.get("type")
        payload = event.get("payload") or {}
        if event_type == "claim_role":
            claims.append(event)
            continue
        if event_type not in ("speech", "sheriff_speech", "pk_speech", "tie_pk_speech"):
            continue
        text = str(payload.get("text") or event.get("text") or "")
        speaker = str(payload.get("speaker") or event.get("player_id") or "")
        match = re.search(
            r"(?:我是|我跳|我认)\s*(预言家|女巫|猎人|白痴|村民|平民|混血儿|狼人)",
            text,
        )
        if not match or not speaker:
            continue
        claims.append({
            "type": "claim_role",
            "payload": {
                "player_id": speaker,
                "claimed_role": _CLAIM_ROLE_MAP[match.group(1)],
            },
        })
    return claims


__all__ = [
    "_CLAIM_ROLE_MAP",
    "_extract_claim_events",
]
