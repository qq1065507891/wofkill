# -*- coding: utf-8 -*-
"""
功能描述：**：将用户人设包转换为 PersonaRouter 可用的运行时配置。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import re
from typing import Any


LEVEL_TO_FLOAT = {
    "low": 0.25,
    "medium": 0.55,
    "high": 0.85,
}


def adapt_persona_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Convert a user-facing pack into PersonaRouter-compatible data."""

    pack_id = str(pack.get("profile_pack_id") or pack.get("id") or "custom_pack")
    profiles: dict[str, dict[str, Any]] = {}
    assignments: dict[str, str] = {}
    for player in sorted(pack.get("players", []), key=lambda item: int(item.get("seat", 0))):
        seat = int(player["seat"])
        profile_id = f"{pack_id}_seat_{seat:02d}_{_slug(player.get('archetype', 'player'))}"
        profiles[profile_id] = _build_profile(player, profile_id)
        assignments[f"p{seat:02d}"] = profile_id

    return {
        "persona_profiles": profiles,
        "player_assignments": assignments,
        "diff_against_default": [],
        "adapter_version": 1,
    }


def _build_profile(player: dict[str, Any], profile_id: str) -> dict[str, Any]:
    logic_skill = _level(player.get("logic_focus"))
    cooperation = _level(player.get("cooperation"))
    aggression = _level(player.get("aggression"))
    emotionality = _level(player.get("emotionality"))
    speech_style = _slug(player.get("speech_style", "calm"))
    return {
        "display_name": str(player.get("name") or profile_id),
        "base": {
            "risk_tolerance": _level(player.get("risk_tolerance")),
            "deception_skill": _level(player.get("deception")),
            "logic_skill": logic_skill,
            "leadership": max(cooperation, aggression * 0.75),
            "emotion_control": 1.0 - emotionality,
            "learning_rate": _level(player.get("memory_focus")),
            "aggression": aggression,
        },
        "task_styles": {
            "speech": f"{speech_style}_speech",
            "vote": f"{speech_style}_vote",
            "night_action": f"{speech_style}_night",
            "deception": f"{speech_style}_deception",
            "sheriff_speech": f"{speech_style}_sheriff",
            "defense_speech": f"{speech_style}_defense",
            "last_words": f"{speech_style}_last_words",
            "reflection": f"{speech_style}_reflection",
        },
        "dynamic_policy": {
            "when_suspected": {
                "aggression_delta": 0.08 if aggression >= 0.55 else 0.03,
                "speech_length_delta": 0.08,
            },
            "when_teammate_exiled": {
                "risk_tolerance_delta": -0.08,
            },
            "when_trusted_by_good_players": {
                "leadership_delta": 0.05 if cooperation >= 0.55 else 0.02,
            },
        },
    }


def _level(value: Any) -> float:
    return LEVEL_TO_FLOAT.get(str(value), LEVEL_TO_FLOAT["medium"])


def _slug(value: Any) -> str:
    """C2 (post-review-v2): 保留 ASCII alnum + CJK 字符 + 日文假名。

    原始实现使用 ``str.isalnum()``，虽然对 CJK 有效，但意图不显式：
    一旦 Python 解释器/locale 改变（例如窄构建或 pypy 行为差异），
    行为就可能回退为把 CJK 替换为 ``_``。本实现显式列出允许的
    Unicode 区间，让行为可预测、可审计。
    """
    text = str(value or "").strip().lower()
    if not text:
        return "default"
    # \w 覆盖 ASCII 字母数字 + 下划线；再显式加上 CJK 统一表意符号
    # 和日文平假名/片假名。其它字符压成 ``-`` 以便 profile_id 可读。
    cleaned = re.sub(r"[^\w一-鿿぀-ヿ]+", "-", text)
    cleaned = cleaned.strip("-")
    return cleaned or "default"
