# -*- coding: utf-8 -*-
"""
校验法官 HITL 注入事件的安全边界。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.judge_hitl_guards import validate_event_type
    >>> validate_event_type("custom_note") is None
    True
"""

from __future__ import annotations

import json
from typing import Any


PROTECTED_TOP_KEYS = frozenset({
    "players",  # 玩家状态只能由规则流程修改
    "deaths",  # 死亡记录只能由规则结算产生
    "votes",  # 投票结果必须来自投票节点
    "phase",  # 阶段流转必须遵循图流程
    "winning_faction",  # 胜负结果必须来自规则引擎
    "hybrid_result",  # 混血儿结果必须来自胜负结算
})

PROTECTED_PLAYER_KEYS = frozenset({
    "role",
    "alive",
    "faction",
    "vote_enabled",
    "revealed_idiot",
    "badge_eligible",
})

_RESERVED_EVENT_TYPES = frozenset({
    "judge_hitl_interaction",
    "judge_broadcast",
})


def validate_event_type(event_type: str) -> str | None:
    """返回事件类型拒绝原因；通过时返回 None。"""
    if not event_type or len(event_type) > 64:
        return f"拒绝: 事件类型无效（空或过长: {len(event_type)}字符）"
    if event_type.startswith("_"):
        return "拒绝: 事件类型不能以下划线开头（保留给内部事件）"
    if not event_type.startswith("custom_"):
        return (
            f"拒绝: 事件类型必须以 'custom_' 开头（收到: '{event_type}'）。"
            "系统保留类型（如 vote_resolved / phase_changed / deaths 等）不可注入。"
        )
    if event_type in _RESERVED_EVENT_TYPES:
        return f"拒绝: '{event_type}' 是系统保留事件类型"
    return None


def parse_payload_tokens(tokens: list[str]) -> dict[str, Any]:
    """解析 key=value 参数，JSON dict/list 会保留结构供递归校验。"""
    payload: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        try:
            decoded = json.loads(raw_value)
        except (ValueError, TypeError):
            decoded = raw_value
        payload[key] = decoded
    return payload


def find_protected_key(value: Any) -> str | None:
    """递归查找 payload 中第一个受保护字段名。"""
    protected = {key.lower() for key in PROTECTED_TOP_KEYS}
    protected |= {key.lower() for key in PROTECTED_PLAYER_KEYS}

    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in protected:
                return key
            found = find_protected_key(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_protected_key(item)
            if found is not None:
                return found
    return None


def validate_payload(payload: dict[str, Any], *, max_bytes: int = 4096) -> str | None:
    """返回 payload 拒绝原因；通过时返回 None。"""
    bad_key = find_protected_key(payload)
    if bad_key is not None:
        return f"拒绝: '{bad_key}' 是受保护字段（递归检查），不能通过 inject_event 修改。"

    try:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (ValueError, TypeError) as exc:
        return f"拒绝: payload 无法序列化: {exc}"
    byte_size = len(serialized.encode("utf-8"))
    if byte_size > max_bytes:
        return f"拒绝: 注入 payload 超过 4KB 限制（{byte_size}字节）。"
    return None


__all__ = [
    "PROTECTED_PLAYER_KEYS",
    "PROTECTED_TOP_KEYS",
    "find_protected_key",
    "parse_payload_tokens",
    "validate_event_type",
    "validate_payload",
]
