"""Tests for werewolf_agent.runtime.context reflection and profile memory hints."""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.memory.schemas import PlayerProfile, ReflectionEntry
from werewolf_agent.runtime.context import _profile_memory_hint, _reflection_memory_hints


def _make_reflection(
    *, game_id: str, role: str, player_id: str = "p01", text: str = "t", faction_won: bool = False
) -> ReflectionEntry:
    return ReflectionEntry(
        entry_id=f"reflection_{game_id}_{player_id}",
        game_id=game_id,
        player_id=player_id,
        role=role,
        faction_won=faction_won,
        text=text,
        situation=text,
    )


def test_reflection_hints_orders_newer_game_id_first_within_same_priority() -> None:
    """Within same priority bucket, newer game_id should sort first."""
    refs = [
        _make_reflection(game_id="2024-12-01", role="seer", text="old"),
        _make_reflection(game_id="2024-12-02", role="seer", text="middle"),
        _make_reflection(game_id="2024-12-03", role="seer", text="new"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    assert [h["text"] for h in hints] == ["new", "middle", "old"]


def test_reflection_hints_tie_broken_by_game_id_descending() -> None:
    """Same role + faction priority; ties broken by game_id descending."""
    refs = [
        _make_reflection(game_id="2024-11-15", role="seer", text="older"),
        _make_reflection(game_id="2024-12-20", role="seer", text="newer"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    assert [h["text"] for h in hints] == ["newer", "older"]


def test_reflection_hints_same_game_id_orders_by_entry_id_stable() -> None:
    """Within same game_id, sort is stable via entry_id (ascending)."""
    refs = [
        _make_reflection(
            game_id="2024-12-01", role="seer", player_id="p02", text="p02"
        ),
        _make_reflection(
            game_id="2024-12-01", role="seer", player_id="p01", text="p01"
        ),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    # entry_id for p01 < entry_id for p02, so p01 first
    assert [h["text"] for h in hints] == ["p01", "p02"]


def test_reflection_hints_higher_priority_wins_over_newer_game() -> None:
    """Same role > same faction > other; higher priority dominates game_id recency."""
    refs = [
        # priority 0 (other), but very new game
        _make_reflection(game_id="2024-12-30", role="hunter", text="new-other"),
        # priority 2 (same role), but older game
        _make_reflection(game_id="2024-12-01", role="seer", text="old-same"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    # Priority 2 (same role) must beat priority 0 (other), even when other is newer
    assert [h["text"] for h in hints] == ["old-same", "new-other"]


# ---------------------------------------------------------------------------
# P0-M4: profile rank description (no raw ability floats, current-role only)
# ---------------------------------------------------------------------------

def _make_profile(
    *,
    games_played: int = 5,
    logic: float = 0.5,
    deception: float = 0.5,
    credibility: float = 0.5,
) -> Any:
    """Build a PlayerProfile-like object for testing the rank-description render.

    Real PlayerProfile is a dataclass with the same attribute names.
    """
    return PlayerProfile(
        player_id="p01",
        games_played=games_played,
        logic=logic,
        deception=deception,
        credibility=credibility,
    )


def test_profile_hint_logic_high_uses_rank_description_not_raw_float() -> None:
    """logic=0.8 must render as a rank description, not as 'logic: 0.8'."""
    profile = _make_profile(logic=0.8)
    role_stats: dict[str, dict[str, int]] = {"werewolf": {"count": 1, "wins": 1}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    # The new rank description must appear
    assert "前 30%" in hint["summary"] or hint["logic_rank"] == "前 30%"
    # Raw float 0.8 must NOT appear in any field
    assert "0.8" not in str(hint)
    # The "logic" key, if present, should be the rank string, not the float
    if "logic" in hint:
        assert hint["logic"] == "前 30%"


def test_profile_hint_logic_mid_uses_medium_rank() -> None:
    """logic=0.5 must render as '中等', not as the raw float."""
    profile = _make_profile(logic=0.5)
    role_stats: dict[str, dict[str, int]] = {"villager": {"count": 1, "wins": 0}}

    hint = _profile_memory_hint(profile, role_stats, current_role="villager")

    assert hint["logic_rank"] == "中等"
    assert "0.5" not in str(hint)


def test_profile_hint_logic_low_uses_needs_improvement_rank() -> None:
    """logic=0.2 must render as '需要提升', not as the raw float."""
    profile = _make_profile(logic=0.2)
    role_stats: dict[str, dict[str, int]] = {"villager": {"count": 1, "wins": 0}}

    hint = _profile_memory_hint(profile, role_stats, current_role="villager")

    assert hint["logic_rank"] == "需要提升"
    assert "0.2" not in str(hint)


def test_profile_hint_only_exposes_current_role_win_rate() -> None:
    """With multiple roles played, only the current role's win rate is exposed.

    Setup: werewolf 3/4=75%, villager 2/6=33%. current_role='werewolf'.
    The hint must surface 75 (werewolf) and must NOT surface 33 (villager's
    raw win-rate count) in a way that leaks the other role's stats.
    """
    profile = _make_profile(games_played=10)
    role_stats = {
        "werewolf": {"count": 4, "wins": 3},   # 75% win rate
        "villager": {"count": 6, "wins": 2},   # 33% win rate
    }

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    # Current-role win rate IS exposed
    assert hint["current_role"] == "werewolf"
    assert hint["current_role_games"] == 4
    assert hint["current_role_win_rate_pct"] == 75

    # Other role's win rate is NOT exposed as a separate field
    assert "villager" not in hint
    assert "roles" not in hint or hint["roles"] == [{"role": "werewolf", "games": 4, "wins": 3}]


def test_profile_hint_no_raw_float_patterns_in_top_level_fields() -> None:
    """No top-level field should contain raw float like '0.' or '0.6' patterns.

    Summary may contain floats for legacy compat, but other fields (e.g.,
    *_rank, current_role_win_rate_pct) should be integer or string.
    """
    profile = _make_profile(games_played=10, logic=0.7, deception=0.6, credibility=0.5)
    role_stats = {
        "werewolf": {"count": 5, "wins": 3},
        "villager": {"count": 5, "wins": 1},
    }

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    # Check top-level fields except 'summary' and 'games_played' for raw floats
    raw_float_pattern = re.compile(r"\b0\.\d+\b")
    for key, value in hint.items():
        if key in ("summary", "games_played"):
            continue
        rendered = str(value)
        assert not raw_float_pattern.search(rendered), (
            f"Field {key}={rendered!r} contains raw float pattern; "
            "P0-M4 requires rank description or integer-only values"
        )


def test_profile_hint_does_not_mention_learning_rate_or_risk_preference() -> None:
    """learning_rate and risk_preference are review/judge-only; never in live prompt.

    NOTE: superseded by P0-M5 (test_all_six_profile_dims_in_prompt_below).
    This test is kept as a backwards-compat check that the dim KEYS are
    not exposed as raw keys; M5's neutral phrasing goes through
    'summary' and dedicated rank fields.
    """
    profile = _make_profile()
    role_stats: dict[str, dict[str, int]] = {"werewolf": {"count": 1, "wins": 1}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    # The snake_case key names must NOT appear in the hint (M4 decision,
    # carried forward). M5 renders these as neutral Chinese descriptions
    # in the summary string, not as raw keys.
    assert "learning_rate" not in hint
    assert "risk_preference" not in hint
    # The summary may still describe these dimensions in Chinese
    # (M5 added this); we just don't want the raw keys.


def test_profile_hint_keeps_games_played_and_summary() -> None:
    """Backwards-compat: games_played and a human-readable summary remain."""
    profile = _make_profile(games_played=12, logic=0.7, deception=0.4, credibility=0.8)
    role_stats = {"werewolf": {"count": 5, "wins": 3}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    assert hint["games_played"] == 12
    assert isinstance(hint["summary"], str)
    assert "12" in hint["summary"]  # games_played surfaces in summary
    # Summary should reference at least one rank description
    assert any(rank in hint["summary"] for rank in ("前 30%", "中等", "需要提升"))


def test_profile_hint_handles_missing_current_role_stats() -> None:
    """If current_role has no historical stats (new player for this role),
    win rate is 0 and games count is 0. No crash."""
    profile = _make_profile(games_played=3)
    role_stats = {
        "villager": {"count": 3, "wins": 1},
        # no 'werewolf' entry
    }

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    assert hint["current_role"] == "werewolf"
    assert hint["current_role_games"] == 0
    assert hint["current_role_win_rate_pct"] == 0


# ---------------------------------------------------------------------------
# P0-M5: all 6 profile dims rendered (with neutral phrasing for
# learning_rate / risk_preference so they don't demoralize the LLM).
# ---------------------------------------------------------------------------


def test_all_six_profile_dims_in_prompt() -> None:
    """P0-M5: _profile_memory_hint must render all 6 dimensions, with
    leadership/learning_rate/risk_preference using neutral phrasing
    so the LLM doesn't read the dim name and assume the worst.

    Schema fields: logic, deception, leadership, credibility,
                   learning_rate, risk_preference
    The hint must surface rank for all 6 (either as a *_rank key or
    in the summary), with raw floats only in `_rank_<dim>` style or
    in a top-level integer-style field. The text phrasing for
    learning_rate and risk_preference must be neutral
    ('处于中等' / '处于较高' style), never '你学得慢' or similar.
    """
    profile = PlayerProfile(
        player_id="p01",
        games_played=5,
        logic=0.7,
        deception=0.4,
        leadership=0.5,
        credibility=0.6,
        learning_rate=0.3,
        risk_preference=0.5,
    )
    role_stats = {"werewolf": {"count": 3, "wins": 1}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    # Each of the 6 dims must surface a rank somewhere in the hint.
    # We accept either a dedicated rank key or inclusion in summary.
    summary = hint.get("summary", "")

    # Required: all 6 dims represented in summary text
    for dim_cn in ("逻辑", "欺骗", "领导", "可信度", "学习", "风险"):
        assert dim_cn in summary, (
            f"Dimension {dim_cn!r} must be mentioned in summary; "
            f"got summary={summary!r}"
        )

    # Neutral phrasing for learning_rate and risk_preference:
    # the phrasing must NOT be demoralizing. Accept "处于中等", "处于较高",
    # "处于较低", or just the rank token ("中等" / "前 30%" / "需要提升").
    # Reject raw "你学得慢" / "你很慢" / "0.3" patterns.
    summary_lower = summary
    assert "你学得慢" not in summary_lower, (
        "learning_rate must not be phrased as '你学得慢'; use neutral "
        f"'你的学习速度处于...'. Got: {summary!r}"
    )
    assert "0.3" not in str(hint), (
        f"Raw learning_rate float must not appear in hint. Got: {hint!r}"
    )
    assert "0.5" not in summary, (
        "Raw risk_preference float must not appear in summary. "
        f"Got summary={summary!r}"
    )

    # Schema must still keep all 6 fields (constraint: don't trim the
    # PlayerProfile dataclass).
    assert hasattr(profile, "logic")
    assert hasattr(profile, "deception")
    assert hasattr(profile, "leadership")
    assert hasattr(profile, "credibility")
    assert hasattr(profile, "learning_rate")
    assert hasattr(profile, "risk_preference")


def test_all_six_profile_dims_learning_rate_low_uses_neutral_phrasing() -> None:
    """P0-M5: a low learning_rate (≤ 0.33) renders as '你的学习速度处于偏低'
    (or similar neutral), NOT as a raw '需要提升' rank used for the
    main 4 dims (which is too judgmental for an internal trait)."""
    profile = PlayerProfile(
        player_id="p01",
        games_played=10,
        logic=0.7,
        deception=0.7,
        leadership=0.7,
        credibility=0.7,
        learning_rate=0.2,  # very low
        risk_preference=0.5,
    )
    role_stats = {"werewolf": {"count": 5, "wins": 3}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")
    summary = hint.get("summary", "")

    # The learning-rate portion must use a neutral descriptor.
    # Acceptable patterns (any of):
    #   "学习速度处于偏低"
    #   "学习速度处于较低"
    #   "学习能力处于中等" (if we treat 0.2 as 'low-but-not-flagged')
    # We must NOT have "学习" followed immediately by "需要提升"
    # because that is the main-dim rank wording (sounds like a critique).
    if "学习" in summary:
        # Find the substring around '学习' to check phrasing
        idx = summary.find("学习")
        window = summary[idx:idx + 30]
        assert "处于" in window, (
            f"learning_rate phrasing should use neutral '处于 X' style; "
            f"got window: {window!r} in summary: {summary!r}"
        )
