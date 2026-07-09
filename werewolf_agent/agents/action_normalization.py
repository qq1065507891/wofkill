# -*- coding: utf-8 -*-
"""
规范化 LLM action 数据和清理可选审计字段。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-09

使用示例:
    >>> from werewolf_agent.agents.action_normalization import normalize_action_data
    >>> normalize_action_data({"confidence": "0.8"})
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import FactionGoal, RiskFlag

logger = logging.getLogger("werewolf_agent.agents.output_parser")


def normalize_action_data(data: Any) -> Any:
    """Normalize provider quirks before schema validation."""
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    if isinstance(normalized.get("target_id"), str) and normalized["target_id"].strip().lower() in {
        "",
        "null",
        "none",
    }:
        normalized["target_id"] = None
    if "confidence" in normalized and isinstance(normalized["confidence"], str):
        try:
            normalized["confidence"] = float(normalized["confidence"].strip())
        except ValueError:
            pass
    return normalized


# P1-G3223805846-5: 常见 LLM 字段名 typo 归一化映射。LLM 经常写出
# 拼写错误的字段名（如 `not_vading_reason`、`targe_id`），直接在解析
# 入口做归一化，避免下游 Pydantic 校验把这些数据当 schema error
# 丢弃。映射表只覆盖反复出现过的 typo，不要试图做成模糊匹配。
_TYPO_ALIASES: dict[str, str] = {
    "not_vading_reason": "not_voting_reason",
    "not_vote_reason": "not_voting_reason",
    "candidate_compare": "candidate_comparison",
    "candidate_evidence_comparison": "candidate_comparison",
    "targe_id": "target_id",
    "targt_id": "target_id",
}


def _normalize_typos(data: dict[str, Any]) -> dict[str, Any]:
    """P1-G3223805846-5: 归一常见 LLM typo。返回新 dict，不修改原对象。

    仅在原 dict 同时缺少正确字段名时才替换，避免覆盖下游已经
    填好的合法值。返回新对象，调用方拿到 dict 即可安全复用。
    """
    if not isinstance(data, dict):
        return data
    result = dict(data)
    for typo, correct in _TYPO_ALIASES.items():
        if typo in result and correct not in result:
            result[correct] = result.pop(typo)
    return result


def clean_enum_value(value: Any, allowed: set[str]) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned in allowed else None


# D4-6 (P2): placeholder filter set for `clean_reason`. The previous
# hard-coded set was only 4 entries; real LLM output produces ~15
# distinct placeholder strings for the reason / private_reason /
# suspect_reason fields. Anything that survives the parser gets
# logged into the audit trail and surfaced in the dashboard,
# polluting downstream review. Set is split 8 Chinese + 4 English +
# 3 punctuation to make the regression test parametrization
# explicit.
_REASON_PLACEHOLDERS: frozenset[str] = frozenset({
    # Chinese (8)
    "未说明",
    "无",
    "未知",
    "不清楚",
    "暂无",
    "未填",
    "无理由",
    "没办法",
    # English (4)
    "none",
    "null",
    "N/A",
    "n/a",
    # Punctuation (3) ? what happens when the LLM gives up mid-thought
    "-",
    "?",
    "...",
})


def clean_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in _REASON_PLACEHOLDERS:
        # D4-6 (P2): surface the placeholder substitution to ops. A
        # silent filter would lose the signal that the LLM is
        # filling the reason field with garbage. We only log on
        # the filter path (not the empty-string path) so the
        # volume stays proportional to the LLM's actual failure
        # rate.
        if text and text in _REASON_PLACEHOLDERS:
            logger.warning(
                "clean_reason: filtered placeholder reason %r to ''",
                text,
            )
        return ""
    return text


def sanitize_optional_private_fields(data: Any) -> Any:
    """Drop malformed optional audit fields without invalidating core action."""
    if not isinstance(data, dict):
        return data
    private_intent = data.get("private_intent")
    if not isinstance(private_intent, dict):
        return data

    sanitized = dict(data)
    sanitized_intent = dict(private_intent)

    valid_goals = {goal.value for goal in FactionGoal}
    if sanitized_intent.get("faction_goal") not in valid_goals:
        true_role = str(sanitized_intent.get("true_role") or sanitized.get("action_type") or "")
        sanitized_intent["faction_goal"] = (
            FactionGoal.CONFUSE_GOOD.value
            if true_role == "werewolf"
            else FactionGoal.FIND_WOLVES.value
        )

    valid_flags = {flag.value for flag in RiskFlag}
    flags = sanitized_intent.get("risk_flags")
    if isinstance(flags, list):
        sanitized_intent["risk_flags"] = [
            flag for flag in flags
            if isinstance(flag, str) and flag in valid_flags
        ]
    else:
        sanitized_intent["risk_flags"] = []

    # P1-S7 (residual): claimed_view is documented as an identity-
    # perspective identifier (PrivateIntent schema), not a free-form
    # Chinese phrase. Game trace g_3528592081 showed real wolves
    # writing "我是好人，混水摸鱼" — a strategy note in natural
    # Chinese. Sanitize any non-enum value to a safe default so the
    # audit log / dashboard only sees clean identifiers. The valid set
    # is the union of: the canonical safe default, all role names, and
    # the seer-specific "seer" identifier. Anything else gets replaced.
    raw_claimed = sanitized_intent.get("claimed_view")
    if not isinstance(raw_claimed, str) or raw_claimed not in _VALID_CLAIMED_VIEW_VALUES:
        # Detect the LLM writing a Chinese natural-language claim —
        # if it contains Chinese characters and isn't in the valid set,
        # it's almost certainly the bad pattern from the game trace.
        sanitized_intent["claimed_view"] = _safe_default_claimed_view(
            sanitized_intent.get("true_role"),
        )

    sanitized["private_intent"] = sanitized_intent
    return sanitized


# P1-S7 (residual): enum-like identifiers acceptable as claimed_view.
# The safe default "good_player_without_night_info" is the standard
# good-side claim; role names are valid because a wolf can claim any
# role publicly (e.g., "villager", "witch"). "seer" is canonical for
# the seer's public claim; "good_player_without_night_info" is the
# generic catch-all.
_VALID_CLAIMED_VIEW_VALUES: frozenset[str] = frozenset({
    "good_player_without_night_info",
    "seer",
    "werewolf",
    "villager",
    "witch",
    "hunter",
    "idiot",
    "hybrid",
})


def _safe_default_claimed_view(true_role: Any) -> str:
    """Pick a safe default claimed_view based on the agent's true_role.

    - seer → "seer" (the only public claim that makes sense for seer)
    - everything else → "good_player_without_night_info" (the standard
      good-side cover, used by all non-wolf roles and by wolves
      pretending to be good)
    """
    if isinstance(true_role, str) and true_role == "seer":
        return "seer"
    return "good_player_without_night_info"
