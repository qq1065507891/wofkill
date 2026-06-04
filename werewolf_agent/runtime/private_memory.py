"""Per-player private memory derived from that player's own cognition."""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState

# P0-I4: player-ID pattern. We strip any p\d{1,2} token (e.g. p03, p11)
# from cross-game-facing strings to prevent concrete game identities from
# leaking across runs.
# Note: do NOT use \b — `\b` does not match between an ASCII letter and a
# CJK character, so a token like "p03的预言家" wouldn't be detected.
_PLAYER_ID_RE = re.compile(r"[Pp]\d{1,2}")

# Mapping from internal role id → Chinese role label. Used when a stance
# target resolves to a known player.
_ROLE_LABEL_CN = {
    "villager": "村民",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "idiot": "白痴",
    "werewolf": "狼人",
    "hybrid": "混血儿",
}

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
    "没解释",
    "沒有解释",
    "没有解释",
    "站边摇摆",
)
"""Markers that suggest a logical flaw in a speech (P1-M10 caveat below).

P1-M10: these are CRUDE SIGNALS, not authoritative verdicts. Many
non-logic speeches contain words like "漏洞" / "没解释" in a neutral
or self-referential way ("我没解释清楚" / "这里没有解释"). The LLM
must NOT treat a match as proof of a logical flaw. Cross-check with
the actual sentence content before tagging a speaker as flawed.
"""

VALID_POINT_MARKERS = (
    "合理",
    "正确",
    "说得通",
    "成立",
    "可信",
    "对得上",
)
"""Markers that suggest a valid point in a speech (P1-M10 caveat below).

P1-M10: these are CRUDE SIGNALS, not authoritative verdicts. Words
like "合理" / "可信" / "说得通" are commonly used in casual
endorsement ("听起来合理" / "你说的可信") without genuine logical
validation. The LLM must NOT treat a match as proof the point is
sound. Cross-check with the actual argument structure before
endorsing a speaker.
"""

# P1-M10: an explicit hint string that prompt renderers can append
# near any private-memory section. Carries the same caveat as the
# per-marker docstrings, in a form that survives code review even
# when the marker constants are skimmed.
_LLM_AWARE_HINT = (
    "【P1-M10 提示】私有记忆中的'逻辑漏洞'与'合理点'是基于关键词"
    "（如'漏洞'/'合理'/'可信'）的粗粒度信号，并非权威判定。LLM 在"
    "引用这些条目时，必须结合原句上下文判断，不可直接当作逻辑结论。"
)

_ROLE_SELF_DECLAIM_RE = re.compile(
    r"我(?:的)?(?:身份|是|扮演|底牌是|角色是|真身是|阵营是)"
    r"(?:一名|一个|那只)?"
    r"(狼人|预言家|女巫|猎人|白痴|混血儿|村民)"
)

# P0-M2: Catch "X 是我的队友/同伴" patterns (game trace g_3528592081 Action 56 leak)
_TEAMMATE_DISCLOSURE_RE = re.compile(
    r"(?:的|是)?(?:队友|同伴|同伙|同党)"
)

# P0-M2: Catch "我的阵营是X" patterns
_FACTION_DISCLOSURE_RE = re.compile(
    r"我(?:的|方)?阵营(?:是|为|属于)?(好人|狼人|神职|平民|村民)"
)

# P0-M2: Catch "我(发现|看穿|验出)X是(role)" — first-person seer-style leaks
_FIRST_PERSON_CHECK_RE = re.compile(
    r"我(?:看穿|发现|看出|验出|查到|查到)(?:了)?\s*(?:[Pp]\d{1,2}|他|她|它)?\s*(?:是)?(狼人|预言家|女巫|猎人|白痴|混血儿|村民)"
)


def _sanitize_role_claims(text: str) -> str:
    """Strip role/team-mate/faction claims that would leak private info.

    Patterns sanitized (P0-M2 expansion):
    - 我是/我扮演/我的真身/我的阵营 + role name
    - 队友/同伴/同伙/同党 (teammate disclosure)
    - 我看穿/我发现/我验出 + role
    - 我的阵营 + faction
    """
    if not text:
        return text
    text = _ROLE_SELF_DECLAIM_RE.sub("[角色信息已省略]", text)
    text = _TEAMMATE_DISCLOSURE_RE.sub("[角色信息已省略]", text)
    text = _FACTION_DISCLOSURE_RE.sub("[角色信息已省略]", text)
    text = _FIRST_PERSON_CHECK_RE.sub("[角色信息已省略]", text)
    return text


# P0-I4: Resolve a stance target to a role-based label, stripping
# concrete player IDs. This is the only safe way to carry
# ``standing_with_seer`` text from the in-game ``private_memory`` into
# the long-term ``ReflectionEntry`` — game-to-game identity leakage
# would otherwise reveal that "I stood with player 03 in game A and
# player 07 in game B" — i.e. a stable, identifiable target.
def _resolve_stance_target(target: str, game_state: GameState | None) -> str:
    """Return a role-based label for ``target``.

    Resolution order:
      1. If ``target`` is a known player id in ``game_state.players``,
         map their role to a Chinese label (e.g. ``预言家``).
      2. If ``target`` contains a player-id substring (e.g. ``p03的
         预言家``), look up the embedded id, and if it resolves to a
         role, return that role.
      3. Strip any remaining pIDs from the residual text. If the
         residual is empty or unresolvable, fall back to a neutral
         ``玩家`` placeholder. Concrete pIDs are never echoed back.
    """
    if not target:
        return "玩家"
    text = str(target).strip()
    if not text:
        return "玩家"
    # Try direct lookup first: target is exactly a player id.
    if game_state is not None:
        player = game_state.players.get(text)
        if player is not None:
            return _ROLE_LABEL_CN.get(player.role, "玩家")
    # Look for any embedded p\d{1,2} tokens. If exactly one is found
    # and it resolves to a known player, use that role.
    embedded_ids = _PLAYER_ID_RE.findall(text)
    had_embedded_id = bool(embedded_ids)
    if had_embedded_id and game_state is not None:
        # Use the first id (rare to have multiple).
        pid = embedded_ids[0].lower()
        player = game_state.players.get(pid) or game_state.players.get(embedded_ids[0])
        if player is not None:
            return _ROLE_LABEL_CN.get(player.role, "玩家")
    # Otherwise, strip any embedded p\d{1,2} tokens and re-resolve
    # whatever role hint remains.
    stripped = _PLAYER_ID_RE.sub("", text).strip() if had_embedded_id else text
    if not stripped:
        return "玩家"
    # If the residual is already a known role (Chinese or English),
    # normalize to Chinese label.
    normalized = stripped.lower()
    if normalized in _ROLE_LABEL_CN:
        return _ROLE_LABEL_CN[normalized]
    if stripped in _ROLE_LABEL_CN.values():
        return stripped
    # If we never found a pID, the input must be a role label or hint
    # already. Just return it as-is (still no IDs to leak).
    if not had_embedded_id:
        return stripped
    # Generic non-empty role hint: keep it (still no IDs).
    return stripped


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
            _add_private_vote_thought(memory, event, player_id, game_state)
            continue
        if event.payload.get("visibility") in PRIVATE_VISIBILITIES:
            continue
        _add_own_speech_notes(memory, event, player_id)
    return {key: value[-12:] for key, value in memory.items() if value}


def _add_private_vote_thought(
    memory: dict[str, list[dict[str, Any]]],
    event: GameEvent,
    player_id: str,
    game_state: GameState | None = None,
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
        "suspect_reason": _sanitize_role_claims(_clip(thought.get("suspect_reason", ""))),
        "not_voting_reason": _clip(thought.get("not_voting_reason", "")),
        "private_reason": _sanitize_role_claims(_clip(thought.get("private_reason", ""))),
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
        # P0-I4: replace the raw player id with a role-based label so
        # this stance note is safe to carry into cross-game reflection.
        label = _resolve_stance_target(item["standing_with_seer"], game_state)
        memory["stance_notes"].append({
            "day": item["day"],
            "speaker": player_id,
            "point": f"站边 {label}",
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
