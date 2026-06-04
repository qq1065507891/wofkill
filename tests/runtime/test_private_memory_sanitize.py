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
