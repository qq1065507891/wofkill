# -*- coding: utf-8 -*-
"""
提供私有记忆中的角色信息脱敏和站边目标解析。

作者: Mike
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.private_memory_safety import _sanitize_role_claims
    >>> _sanitize_role_claims("我是狼人")
"""

from __future__ import annotations

import re

from werewolf_agent.core.models import GameState

# P0-I4: 玩家 ID 模式用于移除跨局文本中的具体身份锚点。
# 注意不要使用 \b；ASCII 字母和中文之间不会产生预期的词边界。
_PLAYER_ID_RE = re.compile(r"[Pp]\d{1,2}")

# 内部角色 ID 到中文标签的映射，供站边目标解析和旧路径兼容导入使用。
_ROLE_LABEL_CN = {
    "villager": "村民",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "idiot": "白痴",
    "werewolf": "狼人",
    "hybrid": "混血儿",
}

_ROLE_SELF_DECLAIM_RE = re.compile(
    r"我(?:的)?(?:身份|是|扮演|底牌是|角色是|真身是|阵营是)"
    r"(?:一名|一个|那只)?"
    r"(狼人|预言家|女巫|猎人|白痴|混血儿|村民)"
)

# P0-M2: 捕获“X 是我的队友/同伴”这类私密队友泄露。
_TEAMMATE_DISCLOSURE_RE = re.compile(
    r"(?:的|是)?(?:队友|同伴|同伙|同党)"
)

# P0-M2: 捕获“我的阵营是 X”这类阵营泄露。
_FACTION_DISCLOSURE_RE = re.compile(
    r"我(?:的|方)?阵营(?:是|为|属于)?(好人|狼人|神职|平民|村民)"
)

# P0-M2: 捕获第一人称查验式角色泄露。
_FIRST_PERSON_CHECK_RE = re.compile(
    r"我(?:看穿|发现|看出|验出|查到|查到)(?:了)?\s*(?:[Pp]\d{1,2}|他|她|它)?\s*(?:是)?(狼人|预言家|女巫|猎人|白痴|混血儿|村民)"
)

# MEM-03: 否定站边标记会反转“X 是角色”的含义，不能回显角色标签。
_NEGATION_MARKERS = (
    "不是",
    "不信",
    "不站",
    "否认",
    "反",
    "否定",
    "不认为",
)


def _sanitize_role_claims(text: str) -> str:
    """移除会泄露私密身份、队友或阵营的信息。"""
    if not text:
        return text
    text = _ROLE_SELF_DECLAIM_RE.sub("[角色信息已省略]", text)
    text = _TEAMMATE_DISCLOSURE_RE.sub("[角色信息已省略]", text)
    text = _FACTION_DISCLOSURE_RE.sub("[角色信息已省略]", text)
    text = _FIRST_PERSON_CHECK_RE.sub("[角色信息已省略]", text)
    return text


def _resolve_stance_target(target: str, game_state: GameState | None) -> str:
    """将站边目标解析为安全的角色标签，避免跨局泄露具体玩家 ID。"""
    if not target:
        return "玩家"
    text = str(target).strip()
    if not text:
        return "玩家"
    for marker in _NEGATION_MARKERS:
        if marker in text:
            return "[否认]"

    if game_state is not None:
        player = game_state.players.get(text)
        if player is not None:
            return _ROLE_LABEL_CN.get(player.role, "玩家")

    embedded_ids = _PLAYER_ID_RE.findall(text)
    had_embedded_id = bool(embedded_ids)
    if had_embedded_id and game_state is not None:
        pid = embedded_ids[0].lower()
        player = game_state.players.get(pid) or game_state.players.get(embedded_ids[0])
        if player is not None:
            return _ROLE_LABEL_CN.get(player.role, "玩家")

    stripped = _PLAYER_ID_RE.sub("", text).strip() if had_embedded_id else text
    if not stripped:
        return "玩家"

    normalized = stripped.lower()
    if normalized in _ROLE_LABEL_CN:
        return _ROLE_LABEL_CN[normalized]
    if stripped in _ROLE_LABEL_CN.values():
        return stripped
    if not had_embedded_id:
        return stripped
    return stripped
