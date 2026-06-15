"""Reflection Memory V2 migration helpers.

Pure helpers live here so migration behavior can be tested without a database.
The CLI in ``scripts/migrate_reflection_memory_v2.py`` wires these helpers to a
repository implementation.
"""

from __future__ import annotations

import copy
from typing import Any


_GENERIC_PHRASES = (
    "复盘失败对局，关注关键转折点的信息缺失",
    "关注关键转折点的信息缺失",
    "下局继续努力",
    "总结经验",
)


def dry_run_legacy_reflection(row: dict[str, Any]) -> dict[str, Any]:
    """Return the V2 migration decision for one legacy reflection row."""
    schema_version = int(row.get("schema_version") or 1)
    if schema_version == 2:
        status = str(row.get("quality_status") or "review_only")
        return {
            "entry_id": row.get("entry_id", ""),
            "old_schema_version": 2,
            "score": float(row.get("quality_score") or 0.0),
            "decision": status,
            "flags": list(row.get("quality_flags") or []),
            "reason": "Already V2",
        }

    text = str(row.get("text") or "").strip()
    role = str(row.get("role") or "").strip()
    flags: list[str] = []
    score = 0.0

    if len(text) >= 80:
        score += 0.20
    else:
        flags.append("short_text")
    if role and (role in text or _role_hint(role) in text):
        score += 0.15
    else:
        flags.append("not_role_specific")
    if any(token in text for token in ("先", "不要", "避免", "核验", "比较", "警徽流", "票型")):
        score += 0.25
    else:
        flags.append("missing_action")
    if any(token in text for token in ("错误", "误判", "保留", "做对", "优点")):
        score += 0.15
    else:
        flags.append("missing_pattern")
    if any(phrase in text for phrase in _GENERIC_PHRASES):
        score -= 0.25
        flags.append("generic_text")

    score = max(0.0, min(1.0, round(score, 2)))
    if score >= 0.70:
        decision = "approved"
        reason = "Legacy text has enough role-specific actionable signal"
    elif score >= 0.40:
        decision = "review_only"
        reason = "Legacy text has limited actionable signal"
    else:
        decision = "rejected"
        reason = "No concrete trigger or next-game action"

    return {
        "entry_id": row.get("entry_id", ""),
        "old_schema_version": 1,
        "score": score,
        "decision": decision,
        "flags": sorted(set(flags)),
        "reason": reason,
    }


def clean_snapshot_reflection_boundary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return a snapshot copy whose ``reflections`` field contains only IDs.

    Dirty legacy snapshot bodies are not retained. Approved V2 bodies are
    replaced by their ``entry_id``; review-only/rejected bodies are removed.
    Existing string IDs are preserved.
    """
    cleaned = copy.deepcopy(snapshot)
    refs = cleaned.get("reflections", [])
    if not isinstance(refs, list):
        cleaned["reflections"] = []
        return cleaned

    kept_ids: list[str] = []
    for item in refs:
        if isinstance(item, str):
            if item:
                kept_ids.append(item)
            continue
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("entry_id") or "").strip()
        if not entry_id:
            continue
        status = str(item.get("quality_status") or "")
        if item.get("schema_version") == 2 or status:
            if status == "approved":
                kept_ids.append(entry_id)
            continue
        # Legacy bodies are intentionally dropped: they have not passed V2
        # prompt-safety and should not be a second source of reflection truth.

    cleaned["reflections"] = kept_ids
    return cleaned


def _role_hint(role: str) -> str:
    return {
        "seer": "预言家",
        "werewolf": "狼人",
        "witch": "女巫",
        "hunter": "猎人",
        "idiot": "白痴",
        "villager": "村民",
        "hybrid": "混血儿",
    }.get(role, role)
