# -*- coding: utf-8 -*-
"""
根据玩家自身认知生成私有记忆，并委托安全模块清洗身份信息。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.private_memory import build_private_memory
    >>> build_private_memory(...)
"""


from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.private_memory_safety import (
    _FACTION_DISCLOSURE_RE as _FACTION_DISCLOSURE_RE,
    _FIRST_PERSON_CHECK_RE as _FIRST_PERSON_CHECK_RE,
    _NEGATION_MARKERS,
    _PLAYER_ID_RE as _PLAYER_ID_RE,
    _ROLE_LABEL_CN as _ROLE_LABEL_CN,
    _ROLE_SELF_DECLAIM_RE as _ROLE_SELF_DECLAIM_RE,
    _TEAMMATE_DISCLOSURE_RE as _TEAMMATE_DISCLOSURE_RE,
    _resolve_stance_target,
    _sanitize_role_claims,
)

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
    # MEM-11: ``moderator_full`` is a debug / moderator-only view
    # that exposes every private fact. Player agents must never
    # see it; the renderer filters it out by including it here.
    "moderator_full",
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

# P1-M14: total token budget for private memory. When the rendered
# memory exceeds this, drop entries from the lowest-priority category
# first (see _PRIORITY_ORDER).
_MAX_PRIVATE_MEMORY_TOKENS = 2000

# P1-M14: priority-ordered categories. Index 0 is the HIGHEST priority
# (kept longest), index -1 is the LOWEST (dropped first). The renderer
# walks this list and drops entries from the rightmost end first.
# Order reflects the strategic value of each note type:
#   - vote_thoughts: direct action-relevant; survive longest
#   - stance_notes:   public-claim tracking; valuable for cross-day
#                     consistency checks
#   - logic_flaws:    subjective keyword signals (P1-M10 caveat)
#   - valid_points:   subjective keyword signals (P1-M10 caveat);
#                     dropped first when over budget
_PRIORITY_ORDER = (
    "vote_thoughts",
    "stance_notes",
    "logic_flaws",
    "valid_points",
)


def _estimate_entry_tokens(entry: dict[str, Any]) -> int:
    """Rough token estimate for a single memory entry.

    MEM-04: legacy ``len(serialized) // 2`` treated CJK as 0.5 tokens
    per character, but real BPE is closer to 1.5-2 tokens per CJK
    character (each Hanzi may split into 2-3 sub-tokens). Use a
    rough CJK+BPE-aware heuristic: count CJK ideographs separately
    (2 tokens each) and ASCII letters/digits separately (1 token
    per 4 chars). Other bytes (punctuation, whitespace) are
    negligible.

    AUDIT-2-05: the legacy estimator also misses two real BPE
    costs:

    1. CJK punctuation marks (。，！？、；：「」『』【】《》
       ——— …… —) and the full-width space (U+3000) live in
       Unicode blocks outside U+4E00..U+9FFF (they're in
       U+3000..U+303F, U+FF00..U+FFEF, U+2010..U+205F), so the
       cjk counter skips them. Real BPE emits ~1 token per
       punctuation mark — a string of 100 CJK commas tokens to
       ~100, not 0.

    2. The dict wrapper itself (JSON braces, quotes, colons,
       commas between fields) adds tokens the body estimator
       doesn't capture. A single-key entry ``{"text": ""}``
       tokenizes to ~5 BPE tokens even though both counters
       return 0 for the body.

    Fix: apply a 1.1× inflation factor to the running total to
    catch the systemic punctuation under-count, and add a +4
    token overhead for the dict wrapper itself.

    The exact ratio matters less than getting the relative ordering
    right; the per-category priority drop depends on accurate
    relative sizes, not on absolute accuracy.
    """
    import json as _json
    serialized = _json.dumps(entry, ensure_ascii=False, sort_keys=True)
    cjk_chars = sum(1 for ch in serialized if '一' <= ch <= '鿿')
    ascii_chars = sum(
        1 for ch in serialized
        if ch.isascii() and ch.isprintable() and not ch.isspace()
    )
    body_estimate = cjk_chars * 2 + ascii_chars // 4
    # AUDIT-2-05: 1.1× inflation catches CJK punctuation +
    # full-width space (not in the cjk ideograph range) and other
    # small token costs (emoji, control-char handling, etc.) that
    # real BPE accounts for. +4 covers the dict wrapper (JSON
    # braces, quotes, colons, the structural tokens around the
    # field set).
    inflated = int(body_estimate * 1.1) + 4
    return max(1, inflated)


def _truncate_by_priority(
    memory: dict[str, list[dict[str, Any]]],
    max_tokens: int = _MAX_PRIVATE_MEMORY_TOKENS,
) -> dict[str, list[dict[str, Any]]]:
    """Truncate a private_memory dict to fit within ``max_tokens``.

    P1-M14: drops from the lowest-priority category first. Within a
    category, drops from the OLDEST entries first (preserve the most
    recent thinking).

    Returns a NEW dict; the input is not mutated. Empty categories
    are still included (with empty lists) so the caller can rely on
    the schema.

    MEM-20: the P1-M10 caveat hint is force-appended to the returned
    dict when logic_flaws or valid_points survives the truncation.
    This keeps the function self-contained — callers no longer need
    a second pass to re-add the hint, and direct callers (unit tests,
    new renderers) can't accidentally drop it.
    """
    result: dict[str, list[dict[str, Any]]] = {
        category: list(memory.get(category, []))
        for category in _PRIORITY_ORDER
    }

    def _total_tokens() -> int:
        return sum(_estimate_entry_tokens(e) for entries in result.values() for e in entries)

    # Already fits: return as-is.
    if _total_tokens() <= max_tokens:
        # MEM-20: caveat must still be applied if keyword signals
        # are present (the original caller logic in
        # ``build_private_memory`` does the same; keeping it here
        # means the contract is local to one function).
        if result.get("logic_flaws") or result.get("valid_points"):
            result["_llm_aware_hint"] = _LLM_AWARE_HINT
        return result

    # Walk categories in REVERSE priority order (lowest priority first).
    # Within each category, drop the OLDEST entries first.
    for category in reversed(_PRIORITY_ORDER):
        if _total_tokens() <= max_tokens:
            break
        # Drop from the head (oldest) until either empty or within budget.
        while result[category] and _total_tokens() > max_tokens:
            result[category].pop(0)

    # MEM-20: after truncation, if any keyword-signal category
    # survives, surface the P1-M10 caveat so the LLM treats those
    # entries as crude signals, not authoritative verdicts.
    if result.get("logic_flaws") or result.get("valid_points"):
        result["_llm_aware_hint"] = _LLM_AWARE_HINT
    return result


def build_private_memory(
    game_state: GameState,
    player_id: str,
    include_action_trace_audit: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Build memory visible only to ``player_id``, plus the P1-M10
    caveat hint as a separate top-level return value.

    Returns:
        A tuple ``(memory, caveat)`` where:
          - ``memory`` is a dict of category-name → list-of-entries
            (only categories with non-empty content are included).
            No meta keys are mixed in.
          - ``caveat`` is a non-empty string with the P1-M10
            "keyword signals are crude" warning when logic_flaws
            or valid_points survive the token budget; ``""`` when
            there's nothing to caveat.

    This intentionally uses only the player's own public statements and private
    audit traces. It does not create a shared omniscient summary of all speeches.

    P1-M14: when the per-category total exceeds ``_MAX_PRIVATE_MEMORY_TOKENS``,
    drop from the lowest-priority category first. Within a category, keep the
    most recent entries.

    MEM-23: ``include_action_trace_audit`` lets callers opt out of the
    action_trace_audit event stream. Per-action events accumulate
    rapidly over a game's lifetime; the ``vote_thoughts`` derived
    from them only retain the newest entry (older ones are pure
    storage overhead). The default is True to preserve backwards
    compatibility with callers that have not been audited.

    MEM-NEW-8: returns a tuple ``(memory, caveat)`` instead of a
    dict that mixed category lists with a meta ``_llm_aware_hint``
    string key. The schema is now uniform (memory values are all
    list-typed) and the caveat flows through a typed channel.
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
            if not include_action_trace_audit:
                continue
            _add_private_vote_thought(memory, event, player_id, game_state)
            continue
        if event.payload.get("visibility") in PRIVATE_VISIBILITIES:
            continue
        _add_own_speech_notes(memory, event, player_id)
    # P1-M14: priority-ordered truncation replaces the previous
    # `[-12:]` per-category cap. The 12-entry cap is no longer needed
    # because the token budget enforces a more meaningful constraint.
    # MEM-NEW-8: _truncate_by_priority still mutates the dict to
    # add the caveat, but we extract it here so the returned tuple
    # keeps the memory dict schema-clean (no mixed-type values).
    truncated = _truncate_by_priority(memory, max_tokens=_MAX_PRIVATE_MEMORY_TOKENS)
    # Pull the caveat out of the truncated dict before filtering
    # empty categories. If absent, return an empty string.
    caveat = truncated.pop("_llm_aware_hint", "") or ""
    result = {key: value for key, value in truncated.items() if value}
    return result, caveat


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
    if player_id and speaker != player_id:
        return
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
                "point": _sanitize_role_claims(_clip(sentence)),
                "source_event": event.type,
            })
        if any(marker in sentence for marker in VALID_POINT_MARKERS):
            memory["valid_points"].append({
                "day": day,
                "speaker": speaker,
                "point": _sanitize_role_claims(_clip(sentence)),
                "source_event": event.type,
            })
        # MEM-16: a stance negation ('我不站边 p03', 'p03 不站边 预言家')
        # must NOT trigger a stance_notes entry — the speaker is
        # actively disclaiming alignment, not declaring it. Reuse
        # the negation marker list from MEM-03.
        if "站边" in sentence and not any(
            marker in sentence for marker in _NEGATION_MARKERS
        ):
            memory["stance_notes"].append({
                "day": day,
                "speaker": speaker,
                "point": _sanitize_role_claims(_clip(sentence)),
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
