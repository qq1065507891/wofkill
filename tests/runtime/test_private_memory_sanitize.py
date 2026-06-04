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
    # 50 valid_points (each ~30 chars + overhead) = ~1300+ tokens.
    # 5 stance_notes = ~165. 5 logic_flaws = ~125. 2 vote = ~74.
    # Total ~1664. Budget 400 forces dropping all valid_points and
    # some logic_flaws / stance_notes too.
    truncated = _truncate_by_priority(memory, max_tokens=400)

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
    # In fact, with budget 400, valid_points should be empty.
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
    # All 3 lower-priority categories are empty (budget too small).
    assert len(truncated["stance_notes"]) == 0
    assert len(truncated["logic_flaws"]) == 0
    assert len(truncated["valid_points"]) == 0


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
