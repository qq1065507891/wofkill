# -*- coding: utf-8 -*-
"""
构造跨局记忆、反思卡片和认知矩阵提示。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.context_memory_hints import _profile_memory_hint
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

HINT_BUDGET = 8
REFLECTION_CARD_BUDGET = 3
_EXPLICIT_GOOD_ROLES = {"villager", "seer", "witch", "hunter", "idiot"}


def _profile_memory_hint(
    profile: Any,
    role_stats: dict[str, dict[str, int]],
    current_role: str,
) -> dict[str, Any]:
    """构造玩家画像记忆提示。"""

    def _rank(score: float) -> str:
        if score > 0.66:
            return "前 30%"
        if score > 0.33:
            return "中等"
        return "需要提升"

    stats = role_stats.get(current_role, {"count": 0, "wins": 0})
    win_rate_pct = (
        round(100 * stats["wins"] / stats["count"]) if stats["count"] > 0 else 0
    )
    logic_rank = _rank(float(getattr(profile, "logic", 0.5)))
    deception_rank = _rank(float(getattr(profile, "deception", 0.5)))
    leadership_rank = _rank(float(getattr(profile, "leadership", 0.5)))
    credibility_rank = _rank(float(getattr(profile, "credibility", 0.5)))

    def _confidence_label(games: int) -> str:
        if games == 0:
            return "无历史"
        if games < 3:
            return f"样本不足(仅{games}局)"
        if games < 10:
            return f"样本中等({games}局)"
        return f"样本充足({games}局)"

    return {
        "games_played": profile.games_played,
        "current_role": current_role,
        "current_role_games": stats["count"],
        "current_role_win_rate_pct": win_rate_pct,
        "win_rate_confidence": _confidence_label(stats["count"]),
        "logic_rank": logic_rank,
        "deception_rank": deception_rank,
        "leadership_rank": leadership_rank,
        "credibility_rank": credibility_rank,
    }


def _reflection_memory_hints(reflections: list[Any], current_role: str, current_faction: str) -> list[dict[str, Any]]:
    """按角色相关性、胜局和时间顺序筛选反思记忆提示。"""

    max_per_role = 2

    def _ref_score(reflection: Any) -> tuple[int, int, str, str]:
        priority = 0
        if reflection.role == current_role:
            priority = 2
        elif (reflection.role == "werewolf" and current_faction == "werewolf") or (
            reflection.role in _EXPLICIT_GOOD_ROLES and current_faction == "good"
        ):
            priority = 1
        won = 1 if getattr(reflection, "faction_won", False) else 0
        game_id = str(getattr(reflection, "game_id", "") or "")
        timestamp = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})", game_id)
        if timestamp is not None:
            yyyy, mm, dd = int(timestamp.group(1)), int(timestamp.group(2)), int(timestamp.group(3))
            neg_game_id = f"{(9999 - yyyy):04d}-{(12 - mm):02d}-{(31 - dd):02d}"
        else:
            neg_game_id = ""
        return (-priority, -won, neg_game_id, str(reflection.entry_id))

    role_counts: dict[str, int] = {}
    hints: list[dict[str, Any]] = []
    for ref in sorted(reflections, key=_ref_score):
        if len(hints) >= HINT_BUDGET:
            break
        role = getattr(ref, "role", "") or ""
        if role_counts.get(role, 0) >= max_per_role:
            continue
        role_counts[role] = role_counts.get(role, 0) + 1
        prompt_card = getattr(ref, "prompt_card", None)
        if prompt_card is not None:
            hints.append({
                "role": ref.role,
                "result": "胜" if ref.faction_won else "负",
                "theme": prompt_card.theme,
                "lesson": prompt_card.lesson,
                "trigger_signals": list(prompt_card.trigger_signals),
                "recommended_action": prompt_card.recommended_action,
                "misuse_risk": prompt_card.misuse_risk,
                "entry_id": ref.entry_id,
            })
        else:
            hints.append({
                "role": ref.role,
                "result": "胜" if ref.faction_won else "负",
                "text": ref.text,
                "situation": ref.situation,
            })
    return hints


def _evidence_id_ref(text: str) -> str:
    """把证据文本渲染成稳定短 ID，避免提示暴露原文。"""

    if not text:
        return "salience_items#empty"
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]
    return f"salience_items#{digest}"


def _cognition_matrix_hint(restored_memory: Any, player_id: str) -> dict[str, Any]:
    get_matrix = getattr(restored_memory, "get_matrix", None)
    if not callable(get_matrix):
        return {}
    matrix = get_matrix(player_id)
    if matrix is None or not hasattr(matrix, "all_entries"):
        return {}

    suspects: list[dict[str, Any]] = []
    trusted: list[dict[str, Any]] = []
    for entry in matrix.all_entries():
        def _claim_str(item: Any) -> str:
            claim = getattr(item, "claim", None)
            return str(claim) if claim is not None else str(item)

        key_evidence = [
            _evidence_id_ref(_claim_str(text))
            for text in list(getattr(entry, "key_evidence", []))[:3]
        ]
        open_questions = [
            _evidence_id_ref(text)
            for text in list(getattr(entry, "open_questions", []))[:3]
        ]
        item = {
            "player": entry.player_id,
            "faction_read": entry.faction_read,
            "trust": round(float(entry.trust), 2),
            "key_evidence": key_evidence,
            "open_questions": open_questions,
        }
        if entry.faction_read == "wolf_lean" or float(entry.trust) < 0.35:
            suspects.append(item)
        elif entry.faction_read == "good_lean" or float(entry.trust) > 0.65:
            trusted.append(item)

    hint: dict[str, Any] = {}
    if suspects:
        hint["suspects"] = sorted(suspects, key=lambda x: x["trust"])[:5]
    if trusted:
        hint["trusted"] = sorted(trusted, key=lambda x: -x["trust"])[:5]
    return hint
