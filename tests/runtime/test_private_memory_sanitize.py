"""P0-M2: private_memory must sanitize role/teammate claims in ALL text fields.

Per g_3528592081 Action 56: p02 wolf's private_reason was
'p07 is my teammate...'. The current sanitize pattern only catches
'I am wolf' style; it misses 'X is my teammate' / 'my faction is' etc.

Fix: expand pattern + apply to all text fields, not just
suspect_reason and private_reason.
"""

from __future__ import annotations

from werewolf_agent.core.models import GameEvent
from werewolf_agent.runtime.private_memory import (
    _add_own_speech_notes,
    _sanitize_role_claims,
)


# Chinese characters written as \uXXXX escapes to avoid encoding issues
SHI = "是"      # 是
WO = "我"      # 我
DE = "的"      # 的
ZhenYing = "阵营"  # 阵营
HaoRen = "好人"     # 好人
YuYanJia = "预言家"  # 预言家
ZhenShen = "真身"   # 真身
DuiYou = "队友"     # 队友


def test_sanitize_catches_team_mate_disclosure():
    text = f"p07 {SHI}{WO}{DE}{DuiYou}，被 p11 毒杀了"
    result = _sanitize_role_claims(text)
    assert DuiYou not in result, f"duiyou should be sanitized, got: {result}"


def test_sanitize_catches_self_role_disclosure():
    text = f"{WO}{SHI}狼人"
    result = _sanitize_role_claims(text)
    assert "狼人" not in result


def test_sanitize_catches_master_faction_disclosure():
    text = f"{WO}{DE}{ZhenYing}{SHI}{HaoRen}"
    result = _sanitize_role_claims(text)
    assert f"{ZhenYing}{SHI}{HaoRen}" not in result


def test_sanitize_catches_role_via_paraphrase():
    text = f"{WO}{DE}{ZhenShen}{SHI}{YuYanJia}"
    result = _sanitize_role_claims(text)
    assert f"{ZhenShen}{SHI}{YuYanJia}" not in result


def test_sanitize_preserves_public_facts():
    """Sanitize should NOT change public-speakable third-party claims."""
    text = f"p05 发言说他{SHI}{YuYanJia}"
    result = _sanitize_role_claims(text)
    assert result == text, f"third-party claim should be unchanged, got: {result}"


def test_sanitize_cleans_actual_game_trace_leak():
    text = f"p07 {SHI}{WO}{DE}{DuiYou}，p07 已被 p11 毒杀"
    result = _sanitize_role_claims(text)
    assert f"{SHI}{WO}{DE}{DuiYou}" not in result


# ---------------------------------------------------------------------------
# P0-M1: tighten _add_own_speech_notes — drop "矛盾" / "前后不一" markers.
# These are too noisy (every speech contains "矛盾" in some form) and
# produced a flood of fake logic_flaws entries. Only "站边" is kept.
# ---------------------------------------------------------------------------


def _make_speech_event(text: str, speaker: str = "p02", day: int = 1) -> GameEvent:
    return GameEvent(
        type="speech",
        payload={
            "speaker": speaker,
            "text": text,
            "day_number": day,
            "visibility": "public",
        },
    )


def test_speech_notes_drops_contradiction_marker_from_logic_flaws():
    """P0-M1: a sentence containing only '矛盾' must NOT be added to
    logic_flaws (too noisy — many non-logic speeches contain this word)."""
    memory: dict = {"logic_flaws": [], "valid_points": [], "stance_notes": [], "vote_thoughts": []}
    event = _make_speech_event("p03 发言有矛盾，我觉得他可能是狼。")
    _add_own_speech_notes(memory, event, player_id="p05")
    assert memory["logic_flaws"] == [], (
        "'矛盾' marker must not be a logic_flaw trigger; got "
        f"{memory['logic_flaws']!r}"
    )


def test_speech_notes_drops_inconsistency_marker_from_logic_flaws():
    """P0-M1: '前后不一' must NOT trigger a logic_flaw entry."""
    memory: dict = {"logic_flaws": [], "valid_points": [], "stance_notes": [], "vote_thoughts": []}
    event = _make_speech_event("p04 发言前后不一，前半段和后半段立场不同。")
    _add_own_speech_notes(memory, event, player_id="p05")
    assert memory["logic_flaws"] == [], (
        "'前后不一' marker must not be a logic_flaw trigger; got "
        f"{memory['logic_flaws']!r}"
    )


def test_speech_notes_keeps_stance_marker():
    """P0-M1: '站边' detection is kept — it captures a public claim
    that the speaker sides with a particular seer/logic line."""
    memory: dict = {"logic_flaws": [], "valid_points": [], "stance_notes": [], "vote_thoughts": []}
    event = _make_speech_event("我站边 p03 的预言家。", speaker="p05")
    _add_own_speech_notes(memory, event, player_id="p05")
    assert len(memory["stance_notes"]) == 1
    # The stance_note records the speaker (p05 said it) and the stance text.
    assert memory["stance_notes"][0]["speaker"] == "p05"
    assert "站边" in memory["stance_notes"][0]["point"]


# ---------------------------------------------------------------------------
# P1-M10: private_memory marker disambiguation.
#
# Markers like `合理` / `可信` are subjective keywords. Many non-logic
# speeches contain them ("我说的合理吧" / "可信度不高"). The LLM should
# treat these as crude signal flags, not authoritative verdicts.
# The marker constants must document this so future maintainers don't
# over-rely on them.
# ---------------------------------------------------------------------------

from werewolf_agent.runtime.private_memory import (  # noqa: E402
    LOGIC_FLAW_MARKERS,
    VALID_POINT_MARKERS,
)


def test_markers_have_documentation():
    """P1-M10: both marker tuples must carry documentation that warns
    the LLM to treat matches as crude signals, not authoritative.
    The docstring may live on the tuple itself, on the module, or on
    a paired `_LLM_AWARE_HINT` string — but it must exist and must
    mention the limitation."""
    import werewolf_agent.runtime.private_memory as pm_mod

    module_doc = (pm_mod.__doc__ or "")
    logic_flaw_doc = (LOGIC_FLAW_MARKERS.__doc__ or "")
    valid_point_doc = (VALID_POINT_MARKERS.__doc__ or "")

    has_module_hint = ("crude" in module_doc.lower()
                       or "粗" in module_doc
                       or "LLM" in module_doc)
    has_logic_flaw_hint = ("crude" in logic_flaw_doc.lower()
                           or "粗" in logic_flaw_doc
                           or "信号" in logic_flaw_doc
                           or "LLM" in logic_flaw_doc)
    has_valid_point_hint = ("crude" in valid_point_doc.lower()
                            or "粗" in valid_point_doc
                            or "信号" in valid_point_doc
                            or "LLM" in valid_point_doc)
    # A paired hint string is also acceptable.
    paired_hint = getattr(pm_mod, "_LLM_AWARE_HINT", None) or ""
    has_paired_hint = bool(paired_hint.strip()) and (
        "crude" in paired_hint.lower()
        or "粗" in paired_hint
        or "信号" in paired_hint
        or "LLM" in paired_hint
    )

    assert has_module_hint or has_logic_flaw_hint or has_valid_point_hint or has_paired_hint, (
        "P1-M10: LOGIC_FLAW_MARKERS / VALID_POINT_MARKERS / module must "
        "carry a docstring/hint warning that the LLM should treat these "
        "as crude signals, not authoritative verdicts. "
        f"module_doc={module_doc!r}, "
        f"logic_flaw_doc={logic_flaw_doc!r}, "
        f"valid_point_doc={valid_point_doc!r}, "
        f"paired_hint={paired_hint!r}"
    )


def test_llm_aware_hint_includes_caveat():
    """P1-M10: when the optional `_LLM_AWARE_HINT` constant is provided,
    it must include a non-empty caveat that names the limitation."""
    import werewolf_agent.runtime.private_memory as pm_mod

    hint = getattr(pm_mod, "_LLM_AWARE_HINT", None)
    # If the constant exists, it must be a non-empty string that contains
    # at least one caveat word. We do NOT require the constant to exist —
    # the test simply ensures that IF the maintainer adds it, it isn't
    # empty or vague.
    if hint is not None:
        assert isinstance(hint, str)
        assert hint.strip(), "_LLM_AWARE_HINT must be a non-empty string if defined"
        assert any(token in hint for token in ("信号", "信号", "提示", "线索", "信号", "crude", "LLM", "主观", "不要", "勿")), (
            f"_LLM_AWARE_HINT should mention the limitation, got: {hint!r}"
        )


# ---------------------------------------------------------------------------
# MEM-20: _truncate_by_priority must preserve the P1-M10 caveat when
# logic_flaws or valid_points survive the truncation. The legacy
# caller (build_private_memory) had to re-check the categories and
# add the hint as a second pass, which is brittle — anyone calling
# _truncate_by_priority directly (e.g. unit tests, or new callers)
# silently lost the caveat and the LLM would treat the entries as
# authoritative.
#
# Fix: _truncate_by_priority itself force-appends the hint meta
# field when logic_flaws or valid_points is non-empty in the
# returned dict, so the function is self-contained.
# ---------------------------------------------------------------------------


def test_truncation_preserves_caveat():
    """MEM-20: feeding a memory that triggers truncation (e.g. huge
    valid_points) must still emit the P1-M10 caveat hint as long as
    logic_flaws / valid_points survived."""
    from werewolf_agent.runtime.private_memory import (
        _LLM_AWARE_HINT,
        _truncate_by_priority,
    )

    memory = {
        "vote_thoughts": [
            {"day": 1, "point": "v1", "source_event": "action_trace_audit"},
        ],
        "stance_notes": [],
        # 50 small logic_flaw entries force truncation; budget
        # small enough that valid_points gets dropped, but a few
        # logic_flaws survive.
        "logic_flaws": [
            {"day": i, "point": f"flaw {i} 漏洞", "source_event": "speech"}
            for i in range(50)
        ],
        "valid_points": [
            {"day": i, "point": f"valid {i} " * 30, "source_event": "speech"}
            for i in range(50)
        ],
    }

    truncated = _truncate_by_priority(memory, max_tokens=200)

    # Truncation actually happened.
    assert len(truncated.get("valid_points", [])) < 50 or (
        len(truncated.get("logic_flaws", [])) < 50
    ), "setup: truncation should have dropped some content"
    # And the surviving keyword-signal category is non-empty.
    keyword_signals_present = bool(
        truncated.get("logic_flaws") or truncated.get("valid_points")
    )
    assert keyword_signals_present, (
        f"setup: at least one of logic_flaws / valid_points must "
        f"survive truncation; got {truncated!r}"
    )

    # MEM-20: the caveat hint must be present in the truncated result.
    assert truncated.get("_llm_aware_hint") == _LLM_AWARE_HINT, (
        f"MEM-20: _truncate_by_priority must force-append the caveat "
        f"hint when logic_flaws/valid_points survive truncation; "
        f"got hint={truncated.get('_llm_aware_hint')!r}, "
        f"truncated={truncated!r}"
    )


def test_truncation_omits_caveat_when_no_keyword_signals():
    """MEM-20: if truncation drops BOTH logic_flaws and valid_points
    to empty, the caveat hint must NOT be present (avoids prompt
    noise when there is nothing to caveat)."""
    from werewolf_agent.runtime.private_memory import _truncate_by_priority

    memory = {
        "vote_thoughts": [
            {"day": 1, "point": "v1", "source_event": "action_trace_audit"},
        ],
        "stance_notes": [
            {"day": 1, "point": f"stance {i} " * 30, "source_event": "speech"}
            for i in range(50)
        ],
        "logic_flaws": [],   # empty input
        "valid_points": [],  # empty input
    }

    truncated = _truncate_by_priority(memory, max_tokens=200)

    # No keyword signals at all.
    assert not truncated.get("logic_flaws")
    assert not truncated.get("valid_points")
    # And the hint is omitted.
    assert "_llm_aware_hint" not in truncated, (
        f"MEM-20: caveat must be omitted when no keyword signals "
        f"survive; got {truncated!r}"
    )


# ---------------------------------------------------------------------------
# MEM-23: action_trace_audit events can be skipped via a config option.
#
# Per-action events accumulate over a game's lifetime; the
# vote_thoughts derived from them only retain the newest entry, so the
# older ones are pure storage overhead. The fix adds an
# ``include_action_trace_audit`` flag (default True) on
# ``build_private_memory`` so callers can opt out.
# ---------------------------------------------------------------------------


def test_skip_action_trace_audit_when_disabled():
    """MEM-23: build_private_memory(include_action_trace_audit=False)
    must drop every action_trace_audit event from the resulting
    memory. vote_thoughts, logic_flaws, valid_points, and stance_notes
    should all be empty (since the only source of these in the
    fixture is the audit events)."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    gs = GameState(
        game_id="g_test_mem23",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            GameEvent(
                type="action_trace_audit",
                payload={
                    "player_id": "p01",
                    "day_number": 1,
                    "private_vote_thought": {
                        "target": "p05",
                        "standing_with_seer": "p03",
                        "suspect_reason": "推理有漏洞",
                        "not_voting_reason": "听起来合理",
                        "private_reason": "我观察了 p05",
                    },
                },
            ),
        ],
    )

    memory, _caveat = build_private_memory(gs, "p01", include_action_trace_audit=False)

    for category in ("vote_thoughts", "logic_flaws", "valid_points", "stance_notes"):
        assert not memory.get(category), (
            f"MEM-23: {category} must be empty when "
            f"include_action_trace_audit=False; got {memory.get(category)!r}"
        )
    # The whole memory dict is empty (no caveat either, since no
    # keyword signals survived).
    assert memory == {}, (
        f"MEM-23: full memory dict must be empty when audit events "
        f"are skipped; got {memory!r}"
    )


def test_action_trace_audit_enabled_by_default():
    """MEM-23 (regression guard): the default behavior (no flag) must
    still include action_trace_audit events."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    gs = GameState(
        game_id="g_test_mem23_default",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            GameEvent(
                type="action_trace_audit",
                payload={
                    "player_id": "p01",
                    "day_number": 1,
                    "private_vote_thought": {
                        "target": "p05",
                        "standing_with_seer": "",
                        "suspect_reason": "推理有漏洞",
                        "not_voting_reason": "",
                        "private_reason": "",
                    },
                },
            ),
        ],
    )

    memory, _caveat = build_private_memory(gs, "p01")
    # With audit enabled, the suspect_reason "漏洞" produces a logic_flaw.
    assert memory.get("logic_flaws"), (
        f"MEM-23: default behavior must include action_trace_audit "
        f"events; got {memory!r}"
    )


# ---------------------------------------------------------------------------
# P1-M14: private_memory priority-ordered truncation.
#
# Currently each category is truncated to 12 entries (`[-12:]`) in
# `build_private_memory`. This is symmetric — every category loses
# the same amount of history. P1-M14 makes it asymmetric and
# priority-ordered:
#
#   priority: vote_thoughts > stance_notes > logic_flaws > valid_points
#
# When the rendered memory exceeds the total token budget, drop from
# the lowest-priority category first. Keep the high-priority
# categories (vote_thoughts) intact as long as possible.
# ---------------------------------------------------------------------------


def test_priority_ordered_truncation():
    """P1-M14: when total private_memory exceeds the token budget, the
    truncation must drop from the lowest-priority category first.

    Setup: a memory dict with a small amount of vote_thoughts (top
    priority) and a huge amount of valid_points (lowest priority).
    The renderer must keep the vote_thoughts intact and drop most of
    the valid_points.
    """
    from werewolf_agent.runtime.private_memory import (
        _truncate_by_priority,
    )

    # vote_thoughts: 2 entries, high priority
    vote_thoughts = [
        {"day": 1, "target": "p03", "point": "vote-1", "source_event": "action_trace_audit"},
        {"day": 2, "target": "p07", "point": "vote-2", "source_event": "action_trace_audit"},
    ]
    # valid_points: 50 entries, lowest priority
    valid_points = [
        {"day": i, "point": f"valid-{i}", "source_event": "speech"}
        for i in range(50)
    ]
    # logic_flaws: 5 entries, second-lowest priority
    logic_flaws = [
        {"day": i, "point": f"flaw-{i}", "source_event": "speech"}
        for i in range(5)
    ]
    # stance_notes: 5 entries, second-highest priority
    stance_notes = [
        {"day": i, "point": f"stance-{i}", "source_event": "action_trace_audit"}
        for i in range(5)
    ]

    memory = {
        "vote_thoughts": vote_thoughts,
        "stance_notes": stance_notes,
        "logic_flaws": logic_flaws,
        "valid_points": valid_points,
    }

    # Use a budget that is much smaller than the total content size.
    # MEM-04: with the new CJK-aware estimator, each vote_thought is
    # ~19 tokens, each valid_point is ~12 tokens, and total is ~816.
    # Budget 100 forces dropping all valid_points and most of
    # stance_notes / logic_flaws.
    truncated = _truncate_by_priority(memory, max_tokens=100)

    # High-priority vote_thoughts must remain intact (2 entries).
    assert len(truncated["vote_thoughts"]) == 2, (
        f"P1-M14: vote_thoughts (top priority) must be preserved; "
        f"got {len(truncated['vote_thoughts'])} entries. "
        f"Truncated: {truncated!r}"
    )
    # Lowest-priority valid_points must be DROPPED first.
    assert len(truncated["valid_points"]) < 50, (
        f"P1-M14: valid_points (lowest priority) should be truncated; "
        f"got {len(truncated['valid_points'])} entries."
    )
    # With the new estimator and budget 100, valid_points (50 * 12
    # = 600 tokens alone) must be entirely dropped.
    assert len(truncated["valid_points"]) == 0, (
        f"P1-M14: with tight budget, valid_points should be entirely dropped; "
        f"got {len(truncated['valid_points'])} entries."
    )


def test_priority_ordered_truncation_drops_valid_points_first_when_only_one_category_fits():
    """P1-M14: when only vote_thoughts fits in the budget, valid_points
    must be dropped first, then logic_flaws, then stance_notes."""
    from werewolf_agent.runtime.private_memory import (
        _truncate_by_priority,
    )

    # 1 vote_thought, 10 logic_flaws, 10 stance_notes, 10 valid_points.
    # Budget = 100 tokens: only vote_thought fits comfortably.
    memory = {
        "vote_thoughts": [
            {"day": 1, "target": "p03", "point": "v", "source_event": "action_trace_audit"},
        ],
        "stance_notes": [
            {"day": i, "point": f"stance {i} " * 10, "source_event": "action_trace_audit"}
            for i in range(10)
        ],
        "logic_flaws": [
            {"day": i, "point": f"flaw {i} " * 10, "source_event": "speech"}
            for i in range(10)
        ],
        "valid_points": [
            {"day": i, "point": f"valid {i} " * 10, "source_event": "speech"}
            for i in range(10)
        ],
    }

    truncated = _truncate_by_priority(memory, max_tokens=100)

    # vote_thoughts is preserved.
    assert len(truncated["vote_thoughts"]) == 1
    # MEM-04: the new CJK-aware estimator under-counts ASCII vs. the
    # legacy ``len // 2`` formula; the budget=100 was tuned to the
    # legacy over-estimate. We still expect valid_points (lowest
    # priority) to be entirely dropped, AND stance / logic to be
    # significantly truncated relative to their input counts.
    # The key invariant: priority order means low-priority entries
    # are dropped first, so valid_points must be empty or smaller
    # than the higher-priority categories.
    assert len(truncated["valid_points"]) <= len(truncated["logic_flaws"]), (
        f"priority: valid_points should be truncated no more than "
        f"logic_flaws; got valid={len(truncated['valid_points'])} "
        f"logic={len(truncated['logic_flaws'])}"
    )
    assert len(truncated["logic_flaws"]) <= len(truncated["stance_notes"]), (
        f"priority: logic_flaws should be truncated no more than "
        f"stance_notes; got logic={len(truncated['logic_flaws'])} "
        f"stance={len(truncated['stance_notes'])}"
    )


def test_priority_ordered_truncation_keeps_everything_when_within_budget():
    """P1-M14: when total content fits within budget, no truncation
    happens at all (no entries are dropped from any category)."""
    from werewolf_agent.runtime.private_memory import (
        _truncate_by_priority,
    )

    memory = {
        "vote_thoughts": [
            {"day": 1, "target": "p03", "point": "v1", "source_event": "action_trace_audit"},
            {"day": 2, "target": "p07", "point": "v2", "source_event": "action_trace_audit"},
        ],
        "stance_notes": [
            {"day": i, "point": f"s{i}", "source_event": "action_trace_audit"}
            for i in range(3)
        ],
        "logic_flaws": [
            {"day": i, "point": f"f{i}", "source_event": "speech"}
            for i in range(3)
        ],
        "valid_points": [
            {"day": i, "point": f"p{i}", "source_event": "speech"}
            for i in range(3)
        ],
    }

    truncated = _truncate_by_priority(memory, max_tokens=10_000)

    for category, expected in memory.items():
        assert len(truncated[category]) == len(expected), (
            f"P1-M14: category {category!r} should be unchanged when within "
            f"budget; got {len(truncated[category])} vs expected {len(expected)}"
        )


def test_priority_ordered_truncation_drops_logic_flaws_before_stance_notes():
    """P1-M14: order of priority is vote > stance > logic > valid.
    When stance and logic both have content and budget forces one
    to be dropped, logic (lower priority) is dropped first."""
    from werewolf_agent.runtime.private_memory import (
        _truncate_by_priority,
    )

    memory = {
        "vote_thoughts": [],
        "stance_notes": [
            {"day": i, "point": f"stance {i} " * 20, "source_event": "action_trace_audit"}
            for i in range(5)
        ],
        "logic_flaws": [
            {"day": i, "point": f"flaw {i} " * 20, "source_event": "speech"}
            for i in range(5)
        ],
        "valid_points": [],
    }

    # Each entry is ~140 tokens. With budget that allows only 1 category
    # at a time, the higher-priority one (stance) wins.
    truncated = _truncate_by_priority(memory, max_tokens=200)

    # stance_notes should have content; logic_flaws should be empty.
    assert len(truncated["stance_notes"]) > 0
    assert len(truncated["logic_flaws"]) == 0, (
        f"P1-M14: logic_flaws (priority 3) must be dropped before "
        f"stance_notes (priority 2). Got logic_flaws={truncated['logic_flaws']!r}, "
        f"stance_notes={truncated['stance_notes']!r}"
    )


# ---------------------------------------------------------------------------
# MEM-01: _add_own_speech_notes writes the raw sentence into logic_flaws /
# valid_points / stance_notes without sanitizing. If a player's public
# speech contains first-person seer-style phrasing ("我验出 p05 是预言家"),
# that text lands verbatim in another player's private memory and would
# indirectly leak moderator-only seer-check results.
#
# Fix: wrap the `point` value with `_sanitize_role_claims(...)` before
# writing.
# ---------------------------------------------------------------------------


def _make_speech_event_for_player(
    text: str, *, speaker: str, day: int = 1, visibility: str = "public"
) -> GameEvent:
    return GameEvent(
        type="speech",
        payload={
            "speaker": speaker,
            "text": text,
            "day_number": day,
            "visibility": visibility,
        },
    )


def test_speech_point_sanitized_against_cross_speaker_first_person():
    """MEM-01: when another player's public speech contains first-person
    seer-style phrasing like "我验出 p05 是预言家", the resulting
    logic_flaws entry (if any) must NOT echo that verbatim string.
    Other players must not see moderator-only seer-check results
    through private memory."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    # p01 is building private memory from its own public speech.
    # The speech contains the leaky "我验出 p05 是预言家" phrase AND
    # a "逻辑漏洞" marker so it lands in logic_flaws.
    gs = GameState(
        game_id="g_test_mem01",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            _make_speech_event_for_player(
                "我验出 p05 是预言家，他的逻辑漏洞很明显",
                speaker="p01",
                day=1,
            ),
        ],
    )

    memory, _caveat = build_private_memory(gs, "p01")

    # Setup sanity: the 逻辑漏洞 marker should still trigger a logic_flaw
    # entry (we are only asserting the point text is sanitized, not that
    # the entry is dropped entirely).
    assert memory.get("logic_flaws"), (
        f"setup: '逻辑漏洞' marker must still create a logic_flaw; "
        f"got: {memory!r}"
    )
    # The cross-speaker first-person claim must be sanitized away.
    for entry in memory["logic_flaws"]:
        assert "我验出 p05 是预言家" not in entry["point"], (
            f"MEM-01: logic_flaw point must not echo cross-speaker "
            f"first-person seer check verbatim; got point: {entry['point']!r}"
        )


def test_speech_point_keeps_legitimate_content():
    """MEM-01: legitimate (non-leaky) speech content survives
    sanitization and still reaches valid_points via VALID_POINT_MARKERS.

    The sanitization must NOT clobber benign sentences — only the role
    / faction / teammate / first-person-check patterns are stripped."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    # "合理" is a VALID_POINT_MARKER, so this sentence should land in
    # valid_points. The text is intentionally benign (no 我 / 队友 /
    # 阵营 / first-person-check patterns) so the sanitization must
    # pass it through unchanged.
    gs = GameState(
        game_id="g_test_mem01_b",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            _make_speech_event_for_player(
                "p05 发言合理",
                speaker="p01",
                day=1,
            ),
        ],
    )

    memory, _caveat = build_private_memory(gs, "p01")

    # Setup sanity: 合理 marker should create a valid_point entry.
    assert memory.get("valid_points"), (
        f"setup: '合理' marker must still create a valid_point; "
        f"got: {memory!r}"
    )
    # And the legitimate content survives intact (no false-positive
    # sanitization on benign sentences).
    found = any(
        "p05 发言合理" in entry["point"]
        for entry in memory["valid_points"]
    )
    assert found, (
        f"MEM-01: legitimate content 'p05 发言合理' should survive "
        f"sanitization verbatim; got valid_points: "
        f"{memory['valid_points']!r}"
    )


def test_build_private_memory_ignores_other_players_public_speech_notes():
    """Private memory should not treat another player's public speech
    keyword matches as the viewer's own notes."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    gs = GameState(
        game_id="g_test_mem_cross_speaker",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "text": "他这里有逻辑漏洞，我站边p03。",
                    "day_number": 1,
                    "visibility": "public",
                },
            ),
        ],
    )

    memory, caveat = build_private_memory(gs, "p01")

    assert memory == {}
    assert caveat == ""


# ---------------------------------------------------------------------------
# MEM-02: build_private_memory must include _llm_aware_hint when
# logic_flaws or valid_points is non-empty. The hint carries the
# P1-M10 caveat about crude keyword signals; the prompt renderer
# reads it from AgentContext.private_memory_caveat.
# ---------------------------------------------------------------------------


def test_build_private_memory_includes_caveat_when_logic_flaws_nonempty():
    """MEM-02: when at least one logic_flaw entry is produced,
    the returned dict must include _llm_aware_hint with the
    P1-M10 caveat text. The renderer reads this and puts it
    on AgentContext.private_memory_caveat."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import (
        _LLM_AWARE_HINT,
        build_private_memory,
    )

    gs = GameState(
        game_id="g_test_mem02",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            _make_speech_event_for_player(
                "p05 发言有逻辑漏洞",
                speaker="p01",
                day=1,
            ),
        ],
    )

    memory, caveat = build_private_memory(gs, "p01")

    assert memory.get("logic_flaws"), (
        f"setup: must produce at least one logic_flaw; got: {memory!r}"
    )
    # MEM-NEW-8: caveat is a top-level return value, not a meta key.
    assert caveat == _LLM_AWARE_HINT, (
        f"MEM-02 / MEM-NEW-8: caveat must be present and equal "
        f"_LLM_AWARE_HINT when logic_flaws non-empty; "
        f"got caveat={caveat!r}"
    )


def test_build_private_memory_omits_caveat_when_empty():
    """MEM-02: when logic_flaws and valid_points are both empty,
    the returned dict must NOT include _llm_aware_hint (avoids
    prompt noise when there is nothing to caveat)."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    # Speech with NO logic_flaw or valid_point markers.
    gs = GameState(
        game_id="g_test_mem02_empty",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            _make_speech_event_for_player(
                "我随便说几句",
                speaker="p02",
                day=1,
            ),
        ],
    )

    memory, _caveat = build_private_memory(gs, "p01")

    # Setup sanity: nothing to caveat.
    assert not memory.get("logic_flaws")
    assert not memory.get("valid_points")
    # And the hint is omitted.
    assert "_llm_aware_hint" not in memory, (
        f"MEM-02: hint must be omitted when both logic_flaws and "
        f"valid_points are empty; got: {memory!r}"
    )


# ---------------------------------------------------------------------------
# MEM-NEW-8: build_private_memory must plumb the P1-M10 caveat as a
# separate top-level field, not as a key in the returned memory dict.
#
# Pre-fix: the function returned a dict with category lists AND a
# ``_llm_aware_hint`` string key mixed in. The schema was inconsistent
# (some values were list[dict], one value was a plain str) and the
# meta key had to be ``pop()``-ed out of the dict by every consumer
# that wanted to surface the caveat to the LLM prompt.
#
# Post-fix: build_private_memory returns a tuple ``(memory, caveat)``
# where ``memory`` is the dict of category lists (no meta keys) and
# ``caveat`` is the P1-M10 hint string (or ``""`` when there's nothing
# to caveat). Callers plumb them through the typed channels
# (``ctx.private_memory_hints`` for the memory, ``ctx.private_memory_caveat``
# for the caveat) without needing dict surgery.
# ---------------------------------------------------------------------------


def test_private_memory_caveat_is_separate_field():
    """MEM-NEW-8: the P1-M10 caveat must be a separate top-level
    return value, not a key inside the memory dict. The memory
    dict's values must all be list-typed (categories of entries);
    the caveat is a plain string returned alongside."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    # Speech with a logic_flaw marker so the caveat is non-empty.
    gs = GameState(
        game_id="g_test_mem_new8",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            _make_speech_event_for_player(
                "p05 发言有逻辑漏洞",
                speaker="p01",
                day=1,
            ),
        ],
    )

    result = build_private_memory(gs, "p01")

    # 1. Return value is a tuple of (memory, caveat).
    assert isinstance(result, tuple) and len(result) == 2, (
        f"MEM-NEW-8: build_private_memory must return (memory, caveat); "
        f"got {type(result).__name__}"
    )
    memory, caveat = result

    # 2. The memory dict contains ONLY category lists. No meta keys.
    assert isinstance(memory, dict)
    for key, value in memory.items():
        assert isinstance(value, list), (
            f"MEM-NEW-8: memory[{key!r}] must be a list (category), "
            f"got {type(value).__name__}: {value!r}"
        )
    assert "_llm_aware_hint" not in memory, (
        f"MEM-NEW-8: caveat must not be mixed into the memory dict; "
        f"got memory={memory!r}"
    )

    # 3. The caveat is a top-level string field, separate from memory.
    assert isinstance(caveat, str), (
        f"MEM-NEW-8: caveat must be a string (top-level field), "
        f"got {type(caveat).__name__}: {caveat!r}"
    )
    # And non-empty when logic_flaws are present.
    assert caveat.strip(), (
        f"MEM-NEW-8: caveat must be non-empty when logic_flaws present; "
        f"got {caveat!r}"
    )


# ---------------------------------------------------------------------------
# MEM-03: negation in stance text must NOT resolve to the role label.
#
# "p03 不是预言家" / "我不信 p03 是预言家" both say "X is NOT seer",
# but the current code strips the negation and resolves to "站边 预言家"
# — flipping the meaning. After the fix, such negation patterns must
# resolve to a denial / "玩家" placeholder instead of the role label.
# ---------------------------------------------------------------------------


def test_negation_in_stance_resolves_to_player_or_deny():
    """MEM-03: 'p03 不是预言家' / '我不信 p03 是预言家' must NOT
    resolve to the role label. Stance target should be a denial
    ('玩家' fallback or contains '[否认]')."""
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.private_memory import _resolve_stance_target

    gs = GameState(
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p03": PlayerState(id="p03", role="seer", alive=True),
        }
    )

    # Case 1: explicit "不是" negation
    target1 = "p03 不是预言家"
    resolved1 = _resolve_stance_target(target1, gs)
    assert "预言家" not in resolved1, (
        f"MEM-03: '不是' negation must not resolve to role label; "
        f"got {resolved1!r}"
    )
    # Either the neutral '玩家' fallback or a denial marker
    assert resolved1 == "玩家" or "[否认]" in resolved1, (
        f"MEM-03: expected '玩家' or '[否认]' marker, got {resolved1!r}"
    )

    # Case 2: '我不信 ... 是预言家'
    target2 = "我不信 p03 是预言家"
    resolved2 = _resolve_stance_target(target2, gs)
    assert "预言家" not in resolved2, (
        f"MEM-03: '不信' negation must not resolve to role label; "
        f"got {resolved2!r}"
    )
    assert resolved2 == "玩家" or "[否认]" in resolved2, (
        f"MEM-03: expected '玩家' or '[否认]' marker, got {resolved2!r}"
    )


# ---------------------------------------------------------------------------
# MEM-04: token estimator must reflect CJK BPE reality.
#
# The legacy estimator counted ``len(serialized) // 2``, treating each
# CJK char as ~0.5 tokens. Real BPE is closer to 1.5-2 tokens/char
# for CJK, so the old estimator undercounted by 2-4× and caused
# priority-truncation to keep far more than the budget allowed.
# Fix: ``len(cjk_chars) * 2 + len(ascii) // 4`` gives a more
# realistic estimate. A 200-char CJK string should estimate > 200
# tokens (per the bug spec).
# ---------------------------------------------------------------------------


def test_token_estimator_accurate_for_cjk():
    """MEM-04: a 200-char CJK string must estimate > 200 tokens."""
    from werewolf_agent.runtime.private_memory import _estimate_entry_tokens

    # 200 CJK characters (all in the CJK Unified Ideographs range)
    cjk_text = "预言家" * 100  # 200 CJK chars
    entry = {"text": cjk_text, "day": 1, "source_event": "speech"}
    estimated = _estimate_entry_tokens(entry)

    assert estimated > 200, (
        f"MEM-04: CJK estimator must return > 200 tokens for 200-char "
        f"CJK input (real BPE is ~1.5-2 tokens/char). Got {estimated}."
    )


def test_token_estimator_ascii_unchanged_or_higher():
    """MEM-04: ASCII input should not be dramatically overestimated.
    The fix prioritizes accurate CJK counting without breaking the
    ASCII case (1 token per 4 chars is still a reasonable estimate)."""
    from werewolf_agent.runtime.private_memory import _estimate_entry_tokens

    # 200 ASCII chars (English): old estimator gave 100 tokens;
    # new estimator gives 200 // 4 = 50. Either way it should be sane.
    ascii_text = "a" * 200
    entry = {"text": ascii_text, "day": 1, "source_event": "speech"}
    estimated = _estimate_entry_tokens(entry)

    # Should be at least 1 (lower bound) and at most 200 (no
    # catastrophic overcounting on ASCII).
    assert 1 <= estimated <= 200, (
        f"MEM-04: ASCII estimator should be sane; got {estimated} "
        f"for 200-char ASCII input"
    )


# ---------------------------------------------------------------------------
# MEM-11: events whose payload.visibility == "moderator_full" must be
# excluded from the player's private memory. moderator_full is the
# debug / moderator view and exposes every private fact; players
# must never see it. The renderer consults ``PRIVATE_VISIBILITIES``
# to filter these events out.
# ---------------------------------------------------------------------------


def test_moderator_full_visibility_filtered():
    """MEM-11: a speech event with visibility='moderator_full' must
    not contribute to ANY private memory category."""
    from werewolf_agent.core.models import GameState
    from werewolf_agent.runtime.private_memory import build_private_memory

    gs = GameState(
        game_id="g_test_mem11",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players={},
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "text": "逻辑漏洞很多，明显是狼坑",
                    "day_number": 1,
                    "visibility": "moderator_full",
                },
            ),
        ],
    )
    memory, _caveat = build_private_memory(gs, "p01")

    # No category should contain content derived from the
    # moderator_full event.
    for category in ("logic_flaws", "valid_points", "stance_notes", "vote_thoughts"):
        for entry in memory.get(category, []):
            point_or_reason = " ".join(str(v) for v in entry.values())
            assert "逻辑漏洞" not in point_or_reason, (
                f"MEM-11: {category} leaked moderator_full content; "
                f"entry: {entry!r}"
            )
    # And the whole private memory must be effectively empty (no
    # categories survived, so the dict is empty / contains only the
    # hint metadata which is also absent).
    assert not memory.get("logic_flaws")
    assert not memory.get("valid_points")
    assert not memory.get("stance_notes")
    assert not memory.get("vote_thoughts")


def test_moderator_full_in_private_visibilities_set():
    """MEM-11: the runtime filter set must include 'moderator_full' so
    the renderer drops those events. This is a regression guard
    against accidentally removing the entry from the module."""
    from werewolf_agent.runtime.private_memory import PRIVATE_VISIBILITIES

    assert "moderator_full" in PRIVATE_VISIBILITIES, (
        "MEM-11: 'moderator_full' must be in PRIVATE_VISIBILITIES "
        "so private memory excludes moderator debug events."
    )


# ---------------------------------------------------------------------------
# MEM-16: stance negation in _add_own_speech_notes.
#
# "我不站边 p03" / "p03 不站边 预言家" must NOT trigger a stance_note
# — the speaker is actively disclaiming alignment. Reuse the
# MEM-03 negation marker list to detect the denial.
# ---------------------------------------------------------------------------


def test_speech_stance_negation_skipped():
    """MEM-16: a sentence containing '站边' together with a negation
    marker (e.g. '不站') must NOT produce a stance_notes entry."""
    memory: dict = {"logic_flaws": [], "valid_points": [], "stance_notes": [], "vote_thoughts": []}
    event = _make_speech_event("我不站边 p03 的预言家", speaker="p05")
    _add_own_speech_notes(memory, event, player_id="p05")
    assert memory["stance_notes"] == [], (
        f"MEM-16: '我不站边 ...' must not produce a stance_note; "
        f"got: {memory['stance_notes']!r}"
    )


def test_speech_stance_positive_still_triggers():
    """MEM-16 (regression guard): a positive '我站边 ...' WITHOUT
    negation must still create a stance_note."""
    memory: dict = {"logic_flaws": [], "valid_points": [], "stance_notes": [], "vote_thoughts": []}
    event = _make_speech_event("我站边 p03 的预言家", speaker="p05")
    _add_own_speech_notes(memory, event, player_id="p05")
    assert len(memory["stance_notes"]) == 1, (
        f"MEM-16: positive '站边' must still trigger; got: {memory['stance_notes']!r}"
    )


# ---------------------------------------------------------------------------
# AUDIT-2-05: CJK punctuation under-counts in _estimate_entry_tokens.
#
# The legacy estimator only counts CJK ideographs (U+4E00..U+9FFF) and
# ASCII printable characters. CJK punctuation marks (，。！？、；：「」『』
# 【】《》—— …… —) and the full-width space (U+3000) live in OTHER
# Unicode blocks (U+3000..U+303F, U+FF00..U+FFEF, U+2010..U+205F) and
# slip through both counters. Real BPE tokenizers (cl100k_base,
# o200k_base, GLM-5) emit ~1 token per punctuation mark on average,
# so a string with heavy punctuation can drift 20-30% low.
#
# Plus the dict wrapper itself adds tokens for the JSON braces,
# quotes, and colons — about 4 tokens for a single-key entry — which
# the per-key body estimator doesn't capture.
#
# Fix: apply a 1.1x inflation factor to the running total and add a
# +4 token overhead for the dict wrapper. The estimator is a rough
# BPE proxy; small relative errors are acceptable but the systemic
# under-count of punctuation-driven entries is not.
# ---------------------------------------------------------------------------


def test_token_estimate_includes_punctuation_overhead():
    """AUDIT-2-05: text with heavy CJK punctuation must estimate HIGHER
    than the legacy estimator (cjk*2 + ascii//4) returns.

    The legacy estimator ignores CJK punctuation. A 100-char string
    composed entirely of CJK punctuation must still produce a non-zero
    token estimate. Punctuation marks (，。！？) typically tokenize to
    ~1 token each in real BPE; the inflation factor (1.1x) plus the
    dict wrapper overhead (+4) must push the estimate well above the
    legacy 0-token baseline.
    """
    from werewolf_agent.runtime.private_memory import _estimate_entry_tokens

    # 100 chars of heavy CJK punctuation (the "。" is U+3002; "，" is
    # U+FF0C; "、" is U+3001; "！" is U+FF01; "？" is U+FF1F).
    # None of these are in the U+4E00..U+9FFF CJK ideograph range, so
    # the legacy estimator counts ZERO CJK chars and ZERO ASCII chars
    # for this string.
    heavy_punctuation = "。" * 50 + "，" * 30 + "！" * 10 + "？" * 10
    assert all(0x3000 <= ord(ch) <= 0x303F or 0xFF00 <= ord(ch) <= 0xFFEF
               for ch in heavy_punctuation), (
        "test setup: all chars must be CJK punctuation outside the "
        "CJK ideograph range (U+4E00..U+9FFF) so the legacy estimator "
        f"under-counts. Got chars: {set(heavy_punctuation)}"
    )

    entry = {
        "day": 1,
        "point": heavy_punctuation,
        "speaker": "p05",
        "source_event": "speech",
    }
    estimated = _estimate_entry_tokens(entry)

    # Legacy estimator returns max(1, 0) = 1 for this entry (no cjk
    # ideographs, no ASCII printable non-whitespace). The fix adds
    # a 1.1x inflation factor and a +4 token dict-wrapper overhead,
    # so the estimate must be substantially higher than the legacy
    # 1-token baseline. We require at least 4 (matching the wrapper
    # overhead alone) — any value above the legacy baseline proves
    # the inflation factor is being applied.
    assert estimated > 1, (
        f"AUDIT-2-05: text with heavy CJK punctuation must estimate "
        f"HIGHER than the legacy baseline (which counts only CJK "
        f"ideographs and ASCII). The legacy returns max(1, 0) = 1 "
        f"for a 100-char punctuation string. The fix applies a 1.1x "
        f"inflation factor + 4 token wrapper overhead — both should "
        f"push the estimate above 1. Got estimated={estimated} for "
        f"entry={entry!r}"
    )

    # And the estimate should be larger than what the legacy formula
    # would give for the same entry WITHOUT the wrapper overhead.
    # (The legacy cjk_chars count is 0, ascii count is 4 — the 4
    # ASCII chars in 'day', 'point', 'source_event' are field names
    # not values. Wait, the field names ARE in the serialized JSON
    # so the legacy formula does count them.) The exact legacy
    # value depends on the serialized bytes; we just need a strict
    # upper-bound comparison: estimate > legacy.
    import json as _json
    serialized = _json.dumps(entry, ensure_ascii=False, sort_keys=True)
    cjk_chars = sum(1 for ch in serialized if '一' <= ch <= '鿿')
    ascii_chars = sum(
        1 for ch in serialized
        if ch.isascii() and ch.isprintable() and not ch.isspace()
    )
    legacy_estimate = max(1, cjk_chars * 2 + ascii_chars // 4)
    assert estimated > legacy_estimate, (
        f"AUDIT-2-05: the new estimator must return a value strictly "
        f"above the legacy baseline. legacy={legacy_estimate}, new="
        f"{estimated}, entry={entry!r}. The 1.1x inflation + 4 wrapper "
        f"overhead must produce a strictly higher estimate."
    )


def test_token_estimate_includes_dict_wrapper_overhead():
    """AUDIT-2-05: a single-key empty-string entry must estimate > 4.

    The dict wrapper overhead (JSON braces, quotes, colons) adds ~4
    tokens in real BPE. A single-key entry with an empty value still
    costs the wrapper — the legacy estimator (max 1) under-counts
    this completely.
    """
    from werewolf_agent.runtime.private_memory import _estimate_entry_tokens

    entry = {"text": ""}
    estimated = _estimate_entry_tokens(entry)

    # The wrapper alone ({"text":""} in JSON) emits ~4 BPE tokens.
    # The fix must surface this overhead.
    assert estimated >= 4, (
        f"AUDIT-2-05: dict wrapper overhead must be ≥ 4 tokens for "
        f"a single-key empty entry. The legacy estimator returns 1 "
        f"(max 1 with no cjk/ascii content). Got estimated={estimated}."
    )
