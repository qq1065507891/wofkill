"""Per-player private memory derived from that player's own cognition."""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState

MEMORY_EVENT_TYPES = {
    "speech",
    "sheriff_speech",
    "tie_pk_speech",
    "exile_last_words",
    "action_trace_audit",
}

PRIVATE_VISIBILITIES = {
    "werewolf_team_only",
    "moderator_only",
    "seer_private",
    "witch_private",
}

LOGIC_FLAW_MARKERS = (
    "逻辑漏洞",
    "漏洞",
    "矛盾",
    "前后不一",
    "不一致",
    "没解释",
    "沒有解释",
    "没有解释",
    "站边摇摆",
)

VALID_POINT_MARKERS = (
    "合理",
    "正确",
    "说得通",
    "成立",
    "可信",
    "对得上",
)


def build_private_memory(game_state: GameState, player_id: str) -> dict[str, list[dict[str, Any]]]:
    """Build memory visible only to ``player_id``.

    This intentionally uses only the player's own public statements and private
    audit traces. It does not create a shared omniscient summary of all speeches.
    """
    memory: dict[str, list[dict[str, Any]]] = {
        "logic_flaws": [],
        "valid_points": [],
        "stance_notes": [],
        "vote_thoughts": [],
    }
    for event in game_state.events:
        if event.type not in MEMORY_EVENT_TYPES:
            continue
        if event.type == "action_trace_audit":
            _add_private_vote_thought(memory, event, player_id)
            continue
        if event.payload.get("visibility") in PRIVATE_VISIBILITIES:
            continue
        _add_own_speech_notes(memory, event, player_id)
    return {key: value[-12:] for key, value in memory.items() if value}


def _add_private_vote_thought(
    memory: dict[str, list[dict[str, Any]]],
    event: GameEvent,
    player_id: str,
) -> None:
    actor = event.payload.get("player_id") or event.payload.get("agent_id")
    if actor != player_id:
        return
    thought = event.payload.get("private_vote_thought")
    if not isinstance(thought, dict):
        thought = _private_vote_thought_from_trace(event.payload.get("action_trace"))
    if not thought:
        return
    item = {
        "day": event.payload.get("day_number", 0),
        "target": thought.get("target"),
        "standing_with_seer": thought.get("standing_with_seer", ""),
        "suspect_reason": _clip(thought.get("suspect_reason", "")),
        "not_voting_reason": _clip(thought.get("not_voting_reason", "")),
        "private_reason": _clip(thought.get("private_reason", "")),
        "source_event": event.type,
    }
    memory["vote_thoughts"].append(item)
    if item["suspect_reason"]:
        memory["logic_flaws"].append({
            "day": item["day"],
            "speaker": player_id,
            "point": item["suspect_reason"],
            "source_event": event.type,
        })
    if item["not_voting_reason"]:
        memory["valid_points"].append({
            "day": item["day"],
            "speaker": player_id,
            "point": item["not_voting_reason"],
            "source_event": event.type,
        })
    if item["standing_with_seer"]:
        memory["stance_notes"].append({
            "day": item["day"],
            "speaker": player_id,
            "point": f"站边 {item['standing_with_seer']}",
            "source_event": event.type,
        })


def _private_vote_thought_from_trace(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    parsed = trace.get("parsed_action")
    if not isinstance(parsed, dict):
        return {}
    return {
        "target": parsed.get("target_id"),
        "standing_with_seer": parsed.get("standing_with_seer", ""),
        "suspect_reason": parsed.get("suspect_reason", ""),
        "not_voting_reason": parsed.get("not_voting_reason", ""),
        "private_reason": parsed.get("private_reason", ""),
    }


def _add_own_speech_notes(
    memory: dict[str, list[dict[str, Any]]],
    event: GameEvent,
    player_id: str = "",
) -> None:
    """提取发言中的逻辑漏洞、合理点、站边记录。"""
    speaker = event.payload.get("speaker", "")
    # 跳过私密频道发言（如狼队频道），只处理公开发言
    visibility = event.payload.get("visibility", "")
    if visibility == "werewolf_team_only" and player_id and speaker != player_id:
        return
    day = event.payload.get("day_number", 0)
    text = str(event.payload.get("text", ""))
    for sentence in _split_sentences(text):
        if any(marker in sentence for marker in LOGIC_FLAW_MARKERS):
            memory["logic_flaws"].append({
                "day": day,
                "speaker": speaker,
                "point": _clip(sentence),
                "source_event": event.type,
            })
        if any(marker in sentence for marker in VALID_POINT_MARKERS):
            memory["valid_points"].append({
                "day": day,
                "speaker": speaker,
                "point": _clip(sentence),
                "source_event": event.type,
            })
        if "站边" in sentence:
            memory["stance_notes"].append({
                "day": day,
                "speaker": speaker,
                "point": _clip(sentence),
                "source_event": event.type,
            })


def _split_sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[。！？!?；;\n]+", text)
        if part.strip()
    ]


def _clip(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    return text[:limit]
