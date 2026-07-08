"""Tests for werewolf_agent.runtime.context reflection and profile memory hints."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.memory.schemas import PlayerProfile, ReflectionEntry
from tests.memory.test_reflection_v2 import _v2_entry
from werewolf_agent.runtime.context import (
    _cognition_matrix_hint,
    _inject_seed_rag_hints,
    _normalize_legal_actions_to_tags,
    _profile_memory_hint,
    _rag_phase_for_task,
    _reflection_memory_hints,
    build_agent_context,
)


def test_context_role_directives_are_split_from_context_facade() -> None:
    from werewolf_agent.runtime import context, context_role_directives

    assert (
        context._apply_role_strategy_context
        is context_role_directives.apply_role_strategy_context
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-8: _analysis_exempt_skills uses correct enum value
# ---------------------------------------------------------------------------


def test_analysis_exempt_uses_correct_enum_value() -> None:
    """NEW-R4-P2-8: the `_analysis_exempt_skills` set in
    `context.py` was built with a hard-coded tuple
    `("last_words", "review_correction", "review_correct")`.
    But the actual enum value is `"review_correct"` (the Python
    name is `REVIEW_CORRECTION`, but the enum value is
    `"review_correct"`). The string `"review_correction"` never
    matches any enum, so the check is partially dead.

    Post-fix: the tuple only contains real enum values
    (`"last_words"` and `"review_correct"`). The dead string is
    removed.

    This test scans the context module source to confirm there
    is NO literal string `"review_correction"` referenced in
    the exempt set construction.
    """
    from pathlib import Path as _Path
    from werewolf_agent.runtime import context as _ctx

    src = _Path(_ctx.__file__).read_text(encoding="utf-8")
    # The dead string `review_correction` (the underscored form,
    # not the actual enum value `review_correct`) must NOT appear
    # as a bare string literal in the context module source.
    assert '"review_correction"' not in src, (
        "NEW-R4-P2-8: context.py must not reference the dead "
        "string 'review_correction' (the actual enum value is "
        "'review_correct' for SkillName.REVIEW_CORRECTION). "
        "Remove the dead string from the exempt tuple."
    )
    # And the real enum value MUST appear (positive control).
    assert '"review_correct"' in src, (
        "NEW-R4-P2-8: context.py must reference 'review_correct' "
        "(the real enum value) in the exempt set construction."
    )


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
    """Within same priority bucket, newer game_id should sort first.
    P1-M12 caps at 2 hints per role, so the 2 newest are kept and
    the 3rd is dropped."""
    refs = [
        _make_reflection(game_id="2024-12-01", role="seer", text="old"),
        _make_reflection(game_id="2024-12-02", role="seer", text="middle"),
        _make_reflection(game_id="2024-12-03", role="seer", text="new"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    # 3 inputs → 2 outputs (cap=2 per role), in newest-first order.
    assert [h["text"] for h in hints] == ["new", "middle"]


def test_hybrid_reflection_is_not_generic_good_faction_history() -> None:
    refs = [
        _make_reflection(
            game_id="2025-01-01",
            role="hybrid",
            text="hybrid-history",
        ),
        _make_reflection(
            game_id="2025-01-01",
            role="werewolf",
            text="wolf-history",
        ),
    ]
    refs[0].entry_id = "z_hybrid"
    refs[1].entry_id = "a_wolf"

    hints = _reflection_memory_hints(
        refs,
        current_role="seer",
        current_faction="good",
    )

    assert [hint["text"] for hint in hints[:2]] == [
        "wolf-history",
        "hybrid-history",
    ]


def test_build_agent_context_extracts_public_seer_credibility_lines() -> None:
    from werewolf_agent.runtime.graph import _new_engine

    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    gs = GameState(
        game_id="seer_credibility_context",
        phase="day",
        day_number=1,
        players=players,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p01",
                    "day_number": 1,
                    "text": "p01 claims seer",
                    "claims": [{"type": "role", "value": "seer"}],
                },
            ),
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "day_number": 1,
                    "text": "p02 counterclaims seer",
                    "claims": [{"type": "role", "value": "seer"}],
                },
            ),
        ],
    )

    ctx = build_agent_context(
        _new_engine(),
        gs,
        "p03",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
    )

    lines = ctx.seer_credibility["seer_lines"]
    assert {line["claimant"] for line in lines} == {"p01", "p02"}


def test_build_agent_context_ignores_third_party_seer_recaps() -> None:
    from werewolf_agent.runtime.graph import _new_engine

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="seer", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
        "p09": PlayerState(id="p09", role="werewolf", alive=True),
        "p11": PlayerState(id="p11", role="villager", alive=True),
    }
    gs = GameState(
        game_id="seer_credibility_recap_context",
        phase="day",
        day_number=2,
        players=players,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "day_number": 1,
                    "text": "我是预言家，昨晚查验p01是好人。",
                },
            ),
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p09",
                    "day_number": 1,
                    "text": "我是预言家，昨晚查验p05是狼人。",
                },
            ),
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p11",
                    "day_number": 2,
                    "text": "p02报p01金水，p09报p05查杀，我会继续对比两个预言家。",
                },
            ),
        ],
    )

    ctx = build_agent_context(
        _new_engine(),
        gs,
        "p03",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
    )

    lines = ctx.seer_credibility["seer_lines"]
    assert {line["claimant"] for line in lines} == {"p02", "p09"}


def test_reflection_hints_tie_broken_by_game_id_descending() -> None:
    """Same role + faction priority; ties broken by game_id descending."""
    refs = [
        _make_reflection(game_id="2024-11-15", role="seer", text="older"),
        _make_reflection(game_id="2024-12-20", role="seer", text="newer"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    assert [h["text"] for h in hints] == ["newer", "older"]


def test_reflection_hints_sort_robust_to_game_id_format() -> None:
    """P2-9: the chr-invert sort trick (which inverts each char code)
    was fragile to game_id format variations.  New parser-based sort
    uses an explicit YYYY-MM-DD regex + arithmetic invert, which
    correctly ranks the same date across separator styles.

    Pre-fix, ``g_2024-12-20`` and ``g2024-12-20`` could rank differently
    because the underscore vs no-underscore changed the char-code
    inversion at that position.  New implementation must rank the
    same.
    """
    refs = [
        _make_reflection(game_id="g_2024-12-20", role="seer", text="underscore"),
        _make_reflection(game_id="g2024-12-20", role="seer", text="no_underscore"),
        _make_reflection(game_id="g_2024-11-15", role="seer", text="old_underscore"),
    ]
    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")
    # The 2 same-date entries tie on date; tie-broken by entry_id stable
    # sort. The 2024-12-20 entries come first (newer than 2024-11-15).
    texts = [h["text"] for h in hints]
    assert "old_underscore" not in texts, (
        "P2-9: older game (2024-11-15) must drop when 2 same-role cap fires"
    )
    assert set(texts) == {"underscore", "no_underscore"}, (
        f"P2-9: 2 same-date entries must both surface, regardless of "
        f"separator; got {texts!r}"
    )


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


def test_v2_reflection_hints_render_prompt_card_only() -> None:
    ref = _v2_entry(
        quality_status="approved",
        quality_score=0.86,
        source={
            "llm_self_review": "raw source should not render",
            "auto_review_summary": "actual_role=werewolf should not render",
            "merged_by": "reflection_synthesizer_v2",
        },
    )

    hints = _reflection_memory_hints([ref], current_role="seer", current_faction="good")

    assert len(hints) == 1
    hint = hints[0]
    assert hint["theme"] == ref.prompt_card.theme
    assert hint["lesson"] == ref.prompt_card.lesson
    assert hint["trigger_signals"] == ref.prompt_card.trigger_signals
    assert hint["recommended_action"] == ref.prompt_card.recommended_action
    assert hint["misuse_risk"] == ref.prompt_card.misuse_risk
    assert "source" not in hint
    assert "quality_score" not in hint
    assert "mistake_patterns" not in hint


def test_build_agent_context_injects_reflections_without_profile() -> None:
    from werewolf_agent.memory.store import MemoryStore
    from werewolf_agent.runtime.graph import _new_engine

    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="werewolf", alive=True),
        "p04": PlayerState(id="p04", role="villager", alive=True),
    }
    gs = GameState(
        game_id="v2_no_profile",
        phase="day",
        day_number=1,
        night_number=1,
        players=players,
    )
    memory = MemoryStore()
    memory.reflections.store_v2(
        _v2_entry(quality_status="approved", quality_score=0.86)
    )

    ctx = build_agent_context(
        _new_engine(),
        gs,
        "p01",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        restored_memory=memory,
    )

    assert ctx.profile_memory_hint == {}
    assert ctx.reflection_memory_hints
    assert ctx.reflection_memory_hints[0]["theme"] == "对跳局先核验警徽流"
    assert ctx.error_pattern_hint["top_mistakes"] == [("vote_mistake", 1)]


def test_build_agent_context_does_not_inject_legacy_reflection_text() -> None:
    from werewolf_agent.memory.store import MemoryStore
    from werewolf_agent.runtime.graph import _new_engine

    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="werewolf", alive=True),
    }
    gs = GameState(
        game_id="legacy_reflection_no_live_prompt",
        phase="day",
        day_number=1,
        night_number=1,
        players=players,
    )
    memory = MemoryStore()
    memory.store_reflection(ReflectionEntry(
        entry_id="legacy_ref_1",
        game_id="old_game",
        player_id="p01",
        role="seer",
        faction_won=False,
        text="legacy raw reflection must not reach live prompt",
        tags=["seer"],
    ))

    ctx = build_agent_context(
        _new_engine(),
        gs,
        "p01",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        restored_memory=memory,
    )

    assert ctx.reflection_memory_hints == []
    assert "legacy raw reflection" not in str(ctx.error_pattern_hint)


def test_build_agent_context_queries_v2_reflections_with_card_budget() -> None:
    from werewolf_agent.runtime.graph import _new_engine

    captured: dict[str, int] = {}

    class _ReflectionMemory:
        def query_live(self, query):
            captured["max_results"] = query.max_results
            return []

    class _RestoredMemory:
        reflections = _ReflectionMemory()

        def get_profile(self, pid):
            return None

    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
    }
    gs = GameState(
        game_id="v2_card_budget",
        phase="day",
        day_number=1,
        night_number=1,
        players=players,
    )

    build_agent_context(
        _new_engine(),
        gs,
        "p01",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        restored_memory=_RestoredMemory(),
    )

    assert captured["max_results"] == 3


# ---------------------------------------------------------------------------
# P0-M4: profile rank description (no raw ability floats, current-role only)
# ---------------------------------------------------------------------------

def test_death_cause_eval_phase_gated() -> None:
    """D-15: the death_cause_evaluation directive is only injected for
    speech / vote / sheriff slots, never for night_action.

    Pre-fix the evaluator ran on every context build, which meant
    night-action prompts (witch decision, seer check, wolf kill)
    carried death-cause guidance the model wasn't asking for and
    that bloated the prompt unnecessarily.
    """
    from werewolf_agent.runtime.graph import _new_engine

    # Minimal 4-player game to keep the test fast.
    players = {
        "p01": PlayerState(id="p01", role="werewolf"),
        "p02": PlayerState(id="p02", role="werewolf"),
        "p03": PlayerState(id="p03", role="villager"),
        "p04": PlayerState(id="p04", role="seer"),
    }
    events = [
        GameEvent(type="speech", payload={
            "speaker": "p01", "text": "我女巫，我毒了p02", "day_number": 2,
        }),
    ]
    gs = GameState(
        game_id="phase_gate_test",
        players=players,
        phase="night",
        night_number=2,
        events=events,
        poison_used=True,
    )
    engine = _new_engine()

    # Day-phase slots SHOULD receive the directive.
    for task in (
        TaskType.SPEECH, TaskType.PK_SPEECH, TaskType.SHERIFF_SPEECH,
        TaskType.VOTE, TaskType.DEFENSE_SPEECH,
    ):
        ctx = build_agent_context(
            engine, gs, "p04", task,
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p01", "p02", "p03"],
        )
        assert "death_cause_evaluation" in ctx.strategy_directive, (
            f"task {task} should receive death_cause_evaluation"
        )

    # Night-action slots MUST NOT receive the directive.
    for task in (TaskType.NIGHT_ACTION, TaskType.HUNTER_SHOT, TaskType.WOLF_DISCUSSION):
        ctx = build_agent_context(
            engine, gs, "p04", task,
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p01", "p02", "p03"],
        )
        assert "death_cause_evaluation" not in ctx.strategy_directive, (
            f"task {task} must NOT receive death_cause_evaluation (D-15 phase gate)"
        )


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

    # P2-M15: the canonical rendering is the structured *_rank field,
    # not the legacy 'summary' string (which was dropped to avoid
    # duplication with the structured dims).
    assert hint["logic_rank"] == "前 30%"
    # Raw float 0.8 must NOT appear in any field
    assert "0.8" not in str(hint)


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


def test_profile_hint_keeps_games_played_and_ranks() -> None:
    """Backwards-compat: games_played remains; P2-M15 dropped summary but *_rank keys are present."""
    profile = _make_profile(games_played=12, logic=0.7, deception=0.4, credibility=0.8)
    role_stats = {"werewolf": {"count": 5, "wins": 3}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    assert hint["games_played"] == 12
    # P2-M15: summary field is gone; structured *_rank fields take its place.
    assert "summary" not in hint
    assert hint["logic_rank"] == "前 30%"



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
# P1-M11: profile hint must show ONLY the current role's win rate.
# This is a stricter regression test for the M4 contract: when a profile
# has 3 roles with different win rates, the hint surfaces only the
# current-role stats. Other roles' wins/games are NEVER present in the
# hint (top-level fields, summary text, or anywhere).
# ---------------------------------------------------------------------------


def test_profile_hint_only_current_role_winrate_strict() -> None:
    """P1-M11: when a profile has 3 roles with different win rates, the
    hint must surface only the current role's stats. Other roles' wins
    and games must NOT appear anywhere in the hint dict (top-level
    fields, summary text, nested values)."""
    profile = _make_profile(games_played=20)
    # Set up 3 roles with very different win rates so leakage would
    # be obvious:
    #   werewolf:  8/10 = 80% (current)
    #   seer:      3/5  = 60%
    #   villager:  1/5  = 20%
    role_stats = {
        "werewolf": {"count": 10, "wins": 8},
        "seer": {"count": 5, "wins": 3},
        "villager": {"count": 5, "wins": 1},
    }

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    # Current-role stats ARE exposed.
    assert hint["current_role"] == "werewolf"
    assert hint["current_role_games"] == 10
    assert hint["current_role_win_rate_pct"] == 80

    # Other roles' names must NOT appear anywhere in the hint.
    # We check every string field of the hint recursively.
    def _gather_strings(obj: Any) -> list[str]:
        out: list[str] = []
        if isinstance(obj, dict):
            for v in obj.values():
                out.extend(_gather_strings(v))
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                out.extend(_gather_strings(v))
        elif isinstance(obj, str):
            out.append(obj)
        return out

    all_strings = _gather_strings(hint)
    full_blob = " | ".join(all_strings)

    assert "seer" not in full_blob, (
        f"P1-M11: 'seer' must not appear in the hint; other roles "
        f"must be filtered. Hint strings: {all_strings!r}"
    )
    # 'villager' is a sub-token of 'werewolf' AND may legitimately
    # appear in the Chinese 村民 wording — we cannot rely on the
    # ASCII token. But the count 5 (seer's games) and 1 (villager's
    # wins) and 3 (seer's wins) are unique leakage signals.
    assert " 5" not in full_blob or " 50" in full_blob or "前 5" in full_blob, (
        f"P1-M11: other roles' counts (5 = seer games, 1 = villager "
        f"wins, 3 = seer wins) must not appear in the hint. "
        f"Got: {full_blob!r}"
    )
    # The two 5s (seer count, villager count) are the leak risk.
    # Check that no field surfaces these counts.
    for key, value in hint.items():
        if key in ("summary",):
            # The summary embeds current_role_games (10) and
            # current_role_win_rate_pct (80); neither matches 5.
            continue
        assert value != 5, (
            f"P1-M11: field {key!r} leaked another role's games count "
            f"(5). Hint: {hint!r}"
        )
        assert value != 3, (
            f"P1-M11: field {key!r} leaked another role's wins count "
            f"(3). Hint: {hint!r}"
        )
        assert value != 1, (
            f"P1-M11: field {key!r} leaked another role's wins count "
            f"(1). Hint: {hint!r}"
        )


# ---------------------------------------------------------------------------
# P0-M5: all 6 profile dims rendered (with neutral phrasing for
# learning_rate / risk_preference so they don't demoralize the LLM).
# ---------------------------------------------------------------------------


def test_all_six_profile_dims_in_prompt() -> None:
    """P0-M5: _profile_memory_hint must render all 6 dimensions, with
    leadership/learning_rate/risk_preference using neutral phrasing
    so the LLM doesn'''t read the dim name and assume the worst.

    Schema fields: logic, deception, leadership, credibilidad,
                   learning_rate, risk_preference
    P2-M15: structured *_rank fields are the single source of truth
    (no summary text). All 6 dims must be in the hint as *_rank keys.
    The text phrasing for learning_rate and risk_preference must be
    neutral (中等 / 较高 style), never 你学得慢 or similar.
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

    # Phase 2 P2-7: inner traits (learning_rate / risk_preference) are
    # review/judge-only.  They must NOT appear in the player-facing
    # hint dict.  Pre-fix they leaked as rank keys.
    expected_rank_keys = {
        "logic_rank", "deception_rank", "leadership_rank", "credibility_rank",
    }
    for key in expected_rank_keys:
        assert key in hint, f'Missing structured rank key: {key!r}. Hint: {hint!r}'
        assert hint[key] in {"前 30%", "中等", "需要提升"}, (
            f'Unexpected rank for {key!r}: {hint[key]!r}'
        )

    # P2-7: learning_rate_rank / risk_preference_rank must NOT exist
    assert "learning_rate_rank" not in hint, (
        "P2-7: learning_rate_rank is review/judge-only, must not appear in player hint"
    )
    assert "risk_preference_rank" not in hint, (
        "P2-7: risk_preference_rank is review/judge-only, must not appear in player hint"
    )

    # Raw floats must not appear
    assert "0.3" not in str(hint), f'Raw learning_rate float leaked: {hint!r}'
    assert "0.5" not in str(hint), f'Raw risk_preference float leaked: {hint!r}'


# ---------------------------------------------------------------------------
# Phase 2 P2-8: profile hint adds win_rate_confidence label
# ---------------------------------------------------------------------------


def test_profile_hint_win_rate_confidence_for_zero_games() -> None:
    """P2-8: when current_role has 0 historical games, the confidence
    label must be ``无历史`` so the LLM treats 0% win rate as
    uninformative, not as definitive 'always lose'."""
    profile = _make_profile(games_played=0)
    role_stats: dict[str, dict[str, int]] = {}  # no entries at all

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    assert "win_rate_confidence" in hint, (
        "P2-8: hint must include win_rate_confidence label"
    )
    assert hint["win_rate_confidence"] == "无历史", (
        f"P2-8: 0 games should yield 无历史; got {hint['win_rate_confidence']!r}"
    )


def test_profile_hint_win_rate_confidence_for_low_n_games() -> None:
    """P2-8: 1-2 games sample size is too small to trust — label
    ``样本不足(仅N局)`` so the LLM weights it lightly."""
    profile = _make_profile(games_played=2)
    role_stats = {"werewolf": {"count": 2, "wins": 1}}  # 50% from 2 games

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    assert hint["win_rate_confidence"] == "样本不足(仅2局)", (
        f"P2-8: 2 games should yield 样本不足(仅2局); "
        f"got {hint['win_rate_confidence']!r}"
    )
    # 50% from 2 games is the same as 67% from 3 games numerically;
    # the LLM must NOT see them as equivalently trustworthy.
    assert "current_role_win_rate_pct" in hint  # raw pct still present, just contextualized


def test_profile_hint_win_rate_confidence_for_medium_n_games() -> None:
    """P2-8: 3-9 games is the medium-confidence tier."""
    profile = _make_profile(games_played=5)
    role_stats = {"werewolf": {"count": 5, "wins": 3}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    assert hint["win_rate_confidence"] == "样本中等(5局)", (
        f"P2-8: 5 games should yield 样本中等(5局); "
        f"got {hint['win_rate_confidence']!r}"
    )


def test_profile_hint_win_rate_confidence_for_high_n_games() -> None:
    """P2-8: ≥10 games is the high-confidence tier."""
    profile = _make_profile(games_played=20)
    role_stats = {"werewolf": {"count": 15, "wins": 10}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    assert hint["win_rate_confidence"] == "样本充足(15局)", (
        f"P2-8: 15 games should yield 样本充足(15局); "
        f"got {hint['win_rate_confidence']!r}"
    )



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


# ---------------------------------------------------------------------------
# P2-M15: profile hint drops duplicate `summary` field.
#
# P0-M4 (commit 08c733e) added rank descriptions for
# logic/deception/credibility. P0-M5 (commit 5d9b267) added the
# 4th/5th/6th dim ranks. Both `summary` AND the structured
# *_rank fields now carry the same rank text — duplication
# bloats the prompt. Drop `summary`; the structured fields are
# the canonical rendering.
# ---------------------------------------------------------------------------


def test_profile_hint_no_summary_duplication() -> None:
    """P2-M15: _profile_memory_hint must NOT include a 'summary' key
    once the 6-dim structured rank fields are the canonical rendering.
    The structured fields (current_role_*, logic_rank, etc.) are the
    single source of truth; no string duplicate of the same data."""
    profile = PlayerProfile(
        player_id="p01",
        games_played=12,
        logic=0.7,
        deception=0.4,
        leadership=0.5,
        credibility=0.8,
        learning_rate=0.3,
        risk_preference=0.5,
    )
    role_stats = {"werewolf": {"count": 5, "wins": 3}}

    hint = _profile_memory_hint(profile, role_stats, current_role="werewolf")

    # The 'summary' key is dropped to avoid duplication with the
    # 6-dim structured *_rank fields. Downstream prompt renderers
    # that previously read hint["summary"] must now read the
    # structured fields.
    assert "summary" not in hint, (
        f"P2-M15: 'summary' must be dropped (duplicated by structured "
        f"*_rank fields). Hint keys: {sorted(hint.keys())!r}"
    )
    # The 6 structured rank fields are still present.
    for key in (
        "logic_rank",
        "deception_rank",
        "leadership_rank",
        "credibility_rank",
        "games_played",
        "current_role",
    ):
        assert key in hint, (
            f"P2-M15: structured field {key!r} must still be in hint; "
            f"got keys: {sorted(hint.keys())!r}"
        )
    # And the rank values are the canonical rank strings (not raw
    # floats), proving the data didn't disappear — it just moved
    # from the summary string into structured fields.
    assert hint["logic_rank"] in ("前 30%", "中等", "需要提升")
    assert hint["current_role_win_rate_pct"] == 60  # 3/5 = 60%


# ---------------------------------------------------------------------------
# P0-M9: _cognition_matrix_hint renders evidence/questions as ID refs.
# ---------------------------------------------------------------------------


def _make_fake_matrix_for_context_test(entries: list[dict[str, Any]]) -> SimpleNamespace:
    """Build a fake restored_memory with a get_matrix that returns a
    CognitionMatrix populated with the given entries."""

    from werewolf_agent.memory.cognition_matrix import CognitionMatrix

    matrix = CognitionMatrix("p01")
    matrix.initialize(["p01", "p02", "p03", "p04"])
    for e in entries:
        entry = matrix.get(e["player_id"])
        if entry is None:
            continue
        entry.faction_read = e.get("faction_read", "unknown")
        entry.trust = e.get("trust", 0.5)
        entry.key_evidence = list(e.get("key_evidence", []))
        entry.open_questions = list(e.get("open_questions", []))

    return SimpleNamespace(get_matrix=lambda _pid: matrix)


def test_cognition_matrix_no_text_evidence_in_context() -> None:
    """P0-M9: key_evidence text never reaches the rendered hint."""
    private_evidence = "p02 私下说 p07 是狼人"
    private_question = "p02 是否在倒钩"
    store = _make_fake_matrix_for_context_test([
        {
            "player_id": "p02",
            "faction_read": "wolf_lean",
            "trust": 0.2,
            "key_evidence": [private_evidence],
            "open_questions": [private_question],
        },
    ])

    hint = _cognition_matrix_hint(store, "p01")
    assert "suspects" in hint
    suspect = hint["suspects"][0]
    # The raw text must not appear in the rendered hint
    serialized = repr(hint)
    assert private_evidence not in serialized
    assert private_question not in serialized
    # And the lists are non-empty id refs
    assert suspect["key_evidence"], "key_evidence refs should not be empty"
    assert suspect["open_questions"], "open_questions refs should not be empty"
    for ref in suspect["key_evidence"] + suspect["open_questions"]:
        assert ref.startswith("salience_items#")


# ---------------------------------------------------------------------------
# P1-G6: RAG injection skipped for REFLECTION and JUDGE_* task types
# ---------------------------------------------------------------------------


class _FakeRAGService:
    """Records every retrieve_live_hints call. Used to assert that
    _inject_seed_rag_hints skips non-player task types entirely."""

    def __init__(self, hits=None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._hits = hits or []

    def retrieve_live_hints(self, query, *, game_id: str = "", player_id: str = ""):
        self.calls.append(
            {
                "role": query.role,
                "phase": query.phase,
                "situation": query.situation,
                "game_id": game_id,
                "player_id": player_id,
            }
        )
        return list(self._hits)

    def hits_to_prompt_lines(self, hits, max_items: int = 3):
        return [
            {"title": h.title, "summary": h.summary, "key_decisions": h.key_decisions}
            for h in hits[:max_items]
        ]


def _make_ctx(task_type, own_role: str = "seer", phase: str = "night"):
    from werewolf_agent.agents.schemas import AgentContext, TaskType

    return AgentContext(
        agent_id="p01",
        task_type=task_type,
        phase=phase,
        own_role=own_role,
    )


def test_rag_injection_skipped_for_reflection() -> None:
    """P1-G6: REFLECTION task is a post-game review of the agent's own
    play; it does not benefit from RAG strategy hints and must skip
    retrieval entirely (saves the cost of an unnecessary embed/rerank).
    """
    from werewolf_agent.agents.schemas import TaskType

    fake = _FakeRAGService()
    ctx = _make_ctx(TaskType.REFLECTION)
    out = _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    assert fake.calls == [], (
        f"P1-G6: REFLECTION must not call retrieve_live_hints; got {fake.calls!r}"
    )
    # Context returned unchanged.
    assert out.rag_hints == ctx.rag_hints


def test_rag_injection_skipped_for_judge_phase() -> None:
    """P1-G6: JUDGE_PHASE is a moderator persona; no player strategy
    hints, so retrieval is skipped."""
    from werewolf_agent.agents.schemas import TaskType

    fake = _FakeRAGService()
    ctx = _make_ctx(TaskType.JUDGE_PHASE)
    out = _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    assert fake.calls == []
    assert out.rag_hints == ctx.rag_hints


def test_rag_injection_skipped_for_all_judge_tasks() -> None:
    """P1-G6: every JUDGE_* task type skips RAG retrieval."""
    from werewolf_agent.agents.schemas import TaskType

    judge_tasks = [
        TaskType.JUDGE_PHASE,
        TaskType.JUDGE_DEATH,
        TaskType.JUDGE_VOTE_CALLING,
        TaskType.JUDGE_VOTE_TALLY,
        TaskType.JUDGE_SKILL_GUIDE,
        TaskType.JUDGE_SHERIFF,
        TaskType.JUDGE_EXILE,
    ]
    for task in judge_tasks:
        fake = _FakeRAGService()
        ctx = _make_ctx(task)
        _inject_seed_rag_hints(
            ctx,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            rag_service=fake,
            game_id="g_test",
        )
        assert fake.calls == [], (
            f"P1-G6: {task.value!r} must not call retrieve_live_hints; "
            f"got {fake.calls!r}"
        )


def test_rag_injection_still_runs_for_player_speech() -> None:
    """Sanity: P1-G6 only skips REFLECTION and JUDGE_*. Player tasks
    (e.g. SPEECH) still call retrieve_live_hints."""
    from werewolf_agent.agents.schemas import TaskType

    fake = _FakeRAGService()
    ctx = _make_ctx(TaskType.SPEECH)
    _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    assert len(fake.calls) == 1
    assert fake.calls[0]["role"] == "seer"


# ---------------------------------------------------------------------------
# R3: previous slim rag_hit items must be cleared between turns.
#
# The filter at context.py:212-219 keeps items where ``type != "rag_hit"``.
# Before the fix, the slim renderer omitted ``type``, so the filter
# was a no-op and old slim items accumulated. With ``type="rag_hit"``
# on every slim line, the filter actually drops them.
# ---------------------------------------------------------------------------


class _FakeHit:
    """Minimal RAGHit-like object used for the R3 inject test."""

    def __init__(self, title: str, summary: str, key_decisions: list[str]) -> None:
        self.title = title
        self.summary = summary
        self.key_decisions = key_decisions


class _FakeRAGServiceRendering:
    """Fake service whose hits_to_prompt_lines mirrors the real slim
    renderer's output shape (with ``type='rag_hit'``)."""

    def __init__(self, hits: list[_FakeHit]) -> None:
        self._hits = hits
        self.call_count = 0

    def retrieve_live_hints(self, query, *, game_id: str = "", player_id: str = ""):
        self.call_count += 1
        return [
            _FakeHit(
                title=f"hit-{self.call_count}",
                summary=f"summary {self.call_count}",
                key_decisions=[f"decision-{self.call_count}"],
            )
        ]

    def hits_to_prompt_lines(self, hits, max_items: int = 3):
        from werewolf_agent.rag.prompt_renderer import render_hit_for_prompt
        return [render_hit_for_prompt(h) for h in hits[:max_items]]


def test_inject_seed_rag_hints_clears_previous_hits() -> None:
    """R3: calling _inject_seed_rag_hints twice on the same ctx must
    leave only the second call's hits in ctx.rag_hints. With the slim
    line discriminator (``type='rag_hit'``), the filter at context.py
    actually clears previous slim items instead of keeping them
    around forever.
    """
    from werewolf_agent.agents.schemas import TaskType
    from werewolf_agent.runtime.context import _inject_seed_rag_hints

    fake = _FakeRAGServiceRendering(hits=[])
    ctx = _make_ctx(TaskType.SPEECH)

    # First injection: ctx.rag_hints has 1 slim item.
    ctx1 = _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    first_titles = [item.get("title") for item in ctx1.rag_hints]
    assert "hit-1" in first_titles, (
        f"first injection should add hit-1; got {first_titles!r}"
    )

    # Second injection: previous "hit-1" must be dropped; only the
    # new "hit-2" should remain.
    ctx2 = _inject_seed_rag_hints(
        ctx1,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    second_titles = [item.get("title") for item in ctx2.rag_hints]
    assert "hit-1" not in second_titles, (
        f"R3: previous slim items must be cleared; got titles {second_titles!r}"
    )
    assert "hit-2" in second_titles, (
        f"R3: new injection should be present; got titles {second_titles!r}"
    )
    # And the slim items returned are tagged with type=rag_hit.
    rag_items = [item for item in ctx2.rag_hints if item.get("type") == "rag_hit"]
    assert rag_items, f"R3: at least one item must be tagged type='rag_hit'; got {ctx2.rag_hints!r}"


# ---------------------------------------------------------------------------
# P2-G11: RAG failure handling distinguishes expected vs anomaly.
#
# Two cases:
#   (a) rag_service is None — expected path (no RAG configured), no
#       log noise, no anomaly count.
#   (b) rag_service.retrieve_live_hints() raises — anomaly path,
#       warn-level log, rag_anomaly_count increments on AgentContext.
# ---------------------------------------------------------------------------


class _RaisingRAGService:
    """RAG service whose retrieve_live_hints always raises an unexpected
    exception. Used to assert anomaly handling (P2-G11)."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def retrieve_live_hints(self, query, *, game_id: str = "", player_id: str = ""):
        self.calls.append(query)
        raise RuntimeError("simulated RAG service crash")

    def hits_to_prompt_lines(self, hits, max_items: int = 3):
        return []


def test_rag_failure_distinguishes_expected_vs_anomaly() -> None:
    """P2-G11: rag_service=None is expected (silent), rag_service.raise
    is an anomaly (counts up on AgentContext.rag_anomaly_count)."""
    import logging

    from werewolf_agent.agents.schemas import TaskType

    # --- Case (a): rag_service is None → expected, no log noise ---
    ctx_none = _make_ctx(TaskType.SPEECH)
    out_none = _inject_seed_rag_hints(
        ctx_none,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=None,
        game_id="g_test",
    )
    # Returned context has no rag_hints mutated and no anomaly count.
    assert out_none.rag_hints == ctx_none.rag_hints
    # rag_anomaly_count defaults to 0 on the returned context.
    assert getattr(out_none, "rag_anomaly_count", 0) == 0

    # --- Case (b): rag_service raises → anomaly, count increments ---
    raising = _RaisingRAGService()
    ctx_anom = _make_ctx(TaskType.SPEECH)
    # Capture logger output to assert warn (not debug) for anomaly.
    cap_records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            cap_records.append(record)

    handler = _ListHandler(level=logging.DEBUG)
    logger = logging.getLogger("werewolf_agent.runtime.context")
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        out_anom = _inject_seed_rag_hints(
            ctx_anom,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            rag_service=raising,
            game_id="g_test",
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    # rag_service was called (anomaly is detected only when service
    # is actually invoked).
    assert len(raising.calls) == 1
    # The returned context still has empty rag_hints.
    assert out_anom.rag_hints == ctx_anom.rag_hints
    # rag_anomaly_count is incremented.
    assert getattr(out_anom, "rag_anomaly_count", 0) == 1
    # Anomaly was logged at WARN (not DEBUG).
    warn_records = [r for r in cap_records if r.levelno >= logging.WARNING]
    assert any("RAG" in r.getMessage() for r in warn_records), (
        f"P2-G11: RAG anomaly must log at WARN; got records: "
        f"{[r.getMessage() for r in cap_records]!r}"
    )
    # And no debug-only "Seed RAG injection failed" message (the
    # previous silent path) at WARN level.
    assert not any(
        "Seed RAG injection failed" in r.getMessage() and r.levelno < logging.WARNING
        for r in cap_records
    ), "P2-G11: anomaly path must not silently debug-log"


# ---------------------------------------------------------------------------
# G-R4-07: situation must not contain a ``phase=`` key that collides with
# the query's own ``phase`` field. The query's phase is the task phase
# (speech / night_action / wolf_discussion); the situation is supposed
# to carry the *game* phase (day / night) under a separate, unambiguous
# key. The previous ``phase=day`` substring in the situation blob was
# indistinguishable from a task-phase token at retriever-tokenize time.
# ---------------------------------------------------------------------------


def test_rag_situation_no_duplicate_phase() -> None:
    """G-R4-07: the situation blob and the RAGQuery must not carry the
    same ``phase=`` key with different semantics. The situation must
    use a separate key (e.g. ``game_phase=day``) for the game phase,
    leaving ``query.phase`` as the sole task-phase token.
    """
    from werewolf_agent.agents.schemas import (
        ActionType,
        AgentContext,
        TaskType,
    )

    class _RecordingService:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def retrieve_live_hints(self, query, *, game_id: str = "", player_id: str = ""):
            self.calls.append(query)
            return []

        def hits_to_prompt_lines(self, hits, max_items: int = 3):
            return []

    fake = _RecordingService()
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="seer",
        legal_actions=[ActionType.VOTE, ActionType.SPEECH],
    )
    _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    situation = fake.calls[0].situation
    # The situation must NOT contain a bare ``phase=`` token that
    # collides with the query's task-phase key. Tokenize the same
    # way the retriever would and confirm no such key exists.
    keys = {chunk.split("=", 1)[0] for chunk in situation.split() if "=" in chunk}
    assert "phase" not in keys, (
        f"G-R4-07: situation still carries ``phase=`` key that collides "
        f"with query.phase; situation={situation!r}"
    )
    # The game phase is preserved under a dedicated, unambiguous key.
    assert "game_phase" in keys, (
        f"G-R4-07: situation must carry ``game_phase=`` for the game "
        f"phase; situation={situation!r}"
    )
    # And the value side must still report day.
    assert "game_phase=day" in situation, (
        f"G-R4-07: game_phase=day must be present; got {situation!r}"
    )


# ---------------------------------------------------------------------------
# G-R4-08: LAST_WORDS must be in the RAG-skip set. The task type falls
# through ``_rag_phase_for_task`` to the raw game phase (``day``/``night``)
# which never matches any seed entry's ``phase`` value (seeds are tagged
# ``speech``/``night_action``/``night_discussion``/etc.). Retrieval runs
# for nothing, burning an embed/rerank call on a task type where the
# strategy hints are also of limited use (last-words are an end-of-life
# speech, not a decision point).
# ---------------------------------------------------------------------------


def test_last_words_rag_skipped() -> None:
    """G-R4-08: LAST_WORDS is a deathbed speech — strategy hints don't
    apply and the raw phase token (day/night) never matches any seed
    entry's phase, so retrieval should be skipped entirely."""
    from werewolf_agent.agents.schemas import TaskType

    fake = _FakeRAGService()
    ctx = _make_ctx(TaskType.LAST_WORDS, own_role="villager", phase="day")
    out = _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    # RAG service must NOT be called for LAST_WORDS.
    assert fake.calls == [], (
        f"G-R4-08: LAST_WORDS must skip RAG retrieval; got calls {fake.calls!r}"
    )
    # Context returned unchanged.
    assert out.rag_hints == ctx.rag_hints


# ---------------------------------------------------------------------------
# G-R4-14: legal_actions must be normalized to RAG tags before the
# situation is built. The previous format
# ``actions=['wolf_kill', 'sheriff_vote']`` had a Python list repr that
# never matched any seed entry's tag set (seeds use shape like
# ``[werewolf, deep_hook, deception]``). The retriever's tag-overlap
# scoring therefore never had a chance to surface a wolf_kill case for
# a wolf_kill query.
#
# Fix: maintain a ``legal_action → tag`` mapping table. Tags use the
# same shape as the seed entries so the retriever's tag-overlap
# scoring picks up the legal-action signal.
# ---------------------------------------------------------------------------


def test_rag_situation_actions_normalized_to_tags() -> None:
    """G-R4-14: the situation's ``actions=`` value must be a
    space-joined string of normalized tags, NOT a Python list repr
    of raw ``ActionType.value`` strings.
    """
    from werewolf_agent.agents.schemas import (
        ActionType,
        AgentContext,
        TaskType,
    )

    class _RecordingService:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def retrieve_live_hints(self, query, *, game_id: str = "", player_id: str = ""):
            self.calls.append(query)
            return []

        def hits_to_prompt_lines(self, hits, max_items: int = 3):
            return []

    fake = _RecordingService()
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.WOLF_DISCUSSION,
        phase="night",
        own_role="werewolf",
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL, ActionType.SPEECH],
    )
    _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    situation = fake.calls[0].situation
    # The action value list-repr (e.g. ``['wolf_kill']``) must NOT
    # appear in the situation. Tag-overlap scoring cannot recover
    # from a Python list repr.
    assert "['wolf_kill" not in situation, (
        f"G-R4-14: situation still carries raw action list repr; "
        f"got situation={situation!r}"
    )
    # The normalized tag substring must be present. The exact tag
    # shape is implementation-defined (it lives in a mapping table);
    # the contract is that the action value ``wolf_kill`` is mapped
    # to a tag token that the seed entries would actually use.
    actions_part = situation.split("actions=", 1)[1] if "actions=" in situation else ""
    tokens = actions_part.split()
    # The first three tokens should include the werewolf-side tags.
    # At least one of the legal actions must contribute a token
    # that's recognizable as a RAG tag (matches the seed entry tag
    # shape — e.g. ``werewolf`` or ``wolf_kill``).
    recognized_tags = {"werewolf", "wolf_kill", "wolf_no_kill", "speech",
                       "witch_save", "witch_poison", "seer_check", "hunter_shot",
                       "sheriff_vote", "sheriff_register", "hybrid_master"}
    matched = [t.strip("[],'\"") for t in tokens if t.strip("[],'\"") in recognized_tags]
    assert matched, (
        f"G-R4-14: actions part of situation should contain normalized "
        f"tags; got tokens={tokens!r}, expected at least one of "
        f"{recognized_tags!r}. Situation: {situation!r}"
    )
    # And the raw 'wolf_kill' string from the previous code path must
    # NOT survive as a standalone list element token in the situation.
    # (It might still appear inside a tag mapping, but it must not be
    # inside a ``[`` / ``]`` pair with quote chars around it.)
    assert "['wolf_kill']" not in situation
    assert "['wolf_kill'," not in situation


def test_vote_task_uses_vote_rag_phase() -> None:
    assert _rag_phase_for_task(TaskType.VOTE, "day") == "vote"


def test_self_destruct_uses_werewolf_rag_tags() -> None:
    tags = set(
        _normalize_legal_actions_to_tags(
            [ActionType.SELF_DESTRUCT],
        ).split()
    )
    assert {"werewolf", "self_destruct"}.issubset(tags)
    assert "idiot" not in tags
    assert "idiot_reveal" not in tags


# P1-M12: reflection hint diversity.
#
# `_reflection_memory_hints` previously took the top 5 reflections with
# the same priority — they could all be from the same role and tag.
# Add a diversity filter so a single role can contribute at most
# MAX_PER_ROLE hints, then per-tag diversity, so the prompt sees
# reflections from multiple perspectives.
# ---------------------------------------------------------------------------


def test_reflection_hints_capped_per_role() -> None:
    """P1-M12: at most 2 reflections per role are surfaced, even when
    the top 5 priority entries are all from the same role."""
    # Build 6 reflections all from the same role (seer), all same
    # priority (priority 2 — same role as current). They have
    # different game_ids so they would all be in the top 5 before
    # diversity filtering.
    refs = [
        _make_reflection(game_id="2024-12-01", role="seer", text="seer-1"),
        _make_reflection(game_id="2024-12-02", role="seer", text="seer-2"),
        _make_reflection(game_id="2024-12-03", role="seer", text="seer-3"),
        _make_reflection(game_id="2024-12-04", role="seer", text="seer-4"),
        _make_reflection(game_id="2024-12-05", role="seer", text="seer-5"),
        _make_reflection(game_id="2024-12-06", role="seer", text="seer-6"),
        # Plus 2 villager reflections at the same priority (priority 1,
        # same faction). Without diversity, the top 5 could be all
        # seer; with diversity, the cap forces at least one villager in.
        _make_reflection(game_id="2024-12-10", role="villager", text="villager-1"),
        _make_reflection(game_id="2024-12-11", role="villager", text="villager-2"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    # Total returned must not exceed the existing 5-hint cap.
    assert len(hints) <= 5, f"hint cap violated: got {len(hints)} hints"

    # No single role may contribute more than 2 hints.
    role_counts: dict[str, int] = {}
    for h in hints:
        role_counts[h["role"]] = role_counts.get(h["role"], 0) + 1
    for role, count in role_counts.items():
        assert count <= 2, (
            f"P1-M12: role {role!r} contributed {count} hints; "
            f"max 2 allowed. Role counts: {role_counts!r}"
        )

    # The 2 newest seer reflections should be present (priority 2
    # beats priority 1, and within priority 2 newest game wins).
    seer_texts = {h["text"] for h in hints if h["role"] == "seer"}
    assert "seer-6" in seer_texts, (
        f"newest seer reflection (seer-6) should be present; got: {seer_texts!r}"
    )
    assert "seer-5" in seer_texts, (
        f"second-newest seer reflection (seer-5) should be present; got: {seer_texts!r}"
    )
    # At most 2 seer hints in the output
    assert len(seer_texts) <= 2, (
        f"At most 2 seer hints allowed; got: {seer_texts!r}"
    )


def test_reflection_hints_diversity_when_no_other_role_available() -> None:
    """P1-M12: when all candidate reflections are from a single role,
    the cap is still respected (no more than 2 from that role)."""
    refs = [
        _make_reflection(game_id="2024-12-01", role="seer", text="a"),
        _make_reflection(game_id="2024-12-02", role="seer", text="b"),
        _make_reflection(game_id="2024-12-03", role="seer", text="c"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    role_counts: dict[str, int] = {}
    for h in hints:
        role_counts[h["role"]] = role_counts.get(h["role"], 0) + 1
    # Cap at 2 even when no other role is available.
    assert role_counts.get("seer", 0) <= 2, (
        f"P1-M12: even with no other role, seer cap must hold; "
        f"got: {role_counts!r}"
    )


def test_reflection_hints_diversity_preserves_priority_order() -> None:
    """P1-M12: diversity is a filter ON TOP of priority sorting, not a
    replacement. The newest top-priority reflections still beat older
    lower-priority reflections, capped per role.

    Setup: 5 seer candidates (priority 2, all same role) + 1 villager
    candidate (priority 1, different role). The cap of 2-per-role
    should kick in: only 2 seer are kept, then 1 villager fills the
    third slot. The villager (priority 1) beats the third seer
    (priority 2, but at-cap) — that is, we do NOT swap in a third
    seer just because priority 2 is higher than priority 1; we
    respect the diversity cap.
    """
    refs = [
        # priority 2 (same role as current) — 5 of these
        _make_reflection(game_id="2024-12-10", role="seer", text="seer-1"),
        _make_reflection(game_id="2024-12-11", role="seer", text="seer-2"),
        _make_reflection(game_id="2024-12-12", role="seer", text="seer-3"),
        _make_reflection(game_id="2024-12-13", role="seer", text="seer-4"),
        _make_reflection(game_id="2024-12-14", role="seer", text="seer-5"),
        # priority 1 (same faction, different role) — 1 of these
        _make_reflection(game_id="2024-12-20", role="villager", text="villager-1"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    # At most 2 seer hints in the output.
    seer_count = sum(1 for h in hints if h["role"] == "seer")
    assert seer_count <= 2, (
        f"P1-M12: seer cap of 2 violated; got {seer_count} seer hints. "
        f"Hints: {hints!r}"
    )

    # Villager is present (priority 1 + different role fills the slot).
    roles_present = [h["role"] for h in hints]
    assert "villager" in roles_present, (
        f"Villager should fill the third slot after the 2 seer. "
        f"Got: {roles_present!r}"
    )

    # The 2 newest seer reflections should be present.
    seer_texts = {h["text"] for h in hints if h["role"] == "seer"}
    assert seer_texts == {"seer-4", "seer-5"}, (
        f"Two newest seer (seer-4, seer-5) should be selected over older "
        f"ones. Got: {seer_texts!r}"
    )


# ---------------------------------------------------------------------------
# P1-M13: belief_state.my_suspects / my_trusted must exclude dead players.
#
# The belief_state is initialized with all 12 player IDs (alive AND
# dead). Without an explicit filter, dead players leak into the agent
# prompt as candidates for suspicion / trust — which is confusing at
# best, since the player can no longer act on that info.
# ---------------------------------------------------------------------------


def test_belief_state_excludes_dead_players() -> None:
    """P1-M13: belief_dict["my_suspects"] and belief_dict["my_trusted"]
    must NOT contain any player whose gs.players[pid].alive is False.

    Test setup: 3 players in a 3-player game.
    - p01 (self, alive, role seer)
    - p02 (alive, role wolf) — should appear in my_suspects
    - p03 (DEAD, role villager) — must NOT appear in either list

    The filter is at the output stage of build_agent_context: when
    iterating belief_state.beliefs, dead players are skipped before
    being added to suspect_list or trust_list.
    """
    from werewolf_agent.core.models import GameEvent, GameState, PlayerState
    from werewolf_agent.runtime.context import build_agent_context
    from werewolf_agent.agents.schemas import TaskType
    from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset

    # Build a minimal GameState with 3 players. p03 is dead from the
    # start; p02 is alive and a known wolf (high-suspicion).
    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=False),
    }
    gs = GameState(
        game_id="g_test_p1m13",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=2,
        night_number=2,
        phase="day",
        players=players,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "text": "我站边 p01 的预言家。",
                    "day_number": 1,
                    "visibility": "public",
                },
            ),
            GameEvent(
                type="seer_check",
                payload={
                    "target_id": "p02",
                    "alignment": "wolf",
                    "night_number": 1,
                },
            ),
        ],
    )

    # Use the real RuleEngine. We just need build_agent_context to
    # populate belief_dict from the visible facts.
    ruleset = Ruleset(raw={
        "player_count": 3,
        "roles": {
            "werewolf": {"count": 1},
            "villager": {"count": 1},
            "seer": {"count": 1},
        },
    })
    engine = RuleEngine(ruleset=ruleset)
    ctx = build_agent_context(
        engine=engine,
        gs=gs,
        player_id="p01",
        task_type=TaskType.SPEECH,
    )

    suspect_players = {entry["player"] for entry in ctx.belief_state["my_suspects"]}
    trusted_players = {entry["player"] for entry in ctx.belief_state["my_trusted"]}

    # p03 (dead) must not appear in either list.
    assert "p03" not in suspect_players, (
        f"P1-M13: dead player p03 leaked into my_suspects: {suspect_players!r}"
    )
    assert "p03" not in trusted_players, (
        f"P1-M13: dead player p03 leaked into my_trusted: {trusted_players!r}"
    )
    # p01 (self) must not appear in either list.
    assert "p01" not in suspect_players
    assert "p01" not in trusted_players


# ---------------------------------------------------------------------------
# MEM-02: _LLM_AWARE_HINT must reach the LLM prompt.
#
# The P1-M10 fix defined `_LLM_AWARE_HINT` as a constant but never wired
# it to the prompt renderer. MEM-02 plumbs it from
# `build_private_memory` -> `build_agent_context` ->
# `AgentContext.private_memory_caveat` -> `prompt_builder._build_private_memory_hints`.
#
# This end-to-end test verifies the full chain: a GameState with public
# speeches containing 逻辑漏洞 markers produces a logic_flaw entry, the
# build_private_memory helper injects the hint, build_agent_context puts
# it on the context, and the prompt builder surfaces it as a line in
# the final user prompt.
# ---------------------------------------------------------------------------


def _make_mem02_player(id: str, role: str, alive: bool = True) -> Any:
    from werewolf_agent.core.models import PlayerState

    return PlayerState(id=id, role=role, alive=alive)


def test_private_memory_caveat_reaches_prompt() -> None:
    """MEM-02: the viewer's own public speech containing '逻辑漏洞'
    must produce a logic_flaw entry, which in turn causes build_private_memory to
    emit `_llm_aware_hint`. build_agent_context plumbs that onto
    AgentContext.private_memory_caveat, and the prompt builder must
    surface it as a line in the rendered user prompt."""
    from werewolf_agent.core.models import GameEvent, GameState
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.schemas import RetryInfo, TaskType
    from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
    from werewolf_agent.runtime.context import build_agent_context
    from werewolf_agent.runtime.private_memory import _LLM_AWARE_HINT

    # Build a minimal GameState where p01 publicly claims p03 has a
    # logic flaw. Own public speech may become private memory; other
    # players' public speeches must not.
    players = {
        "p01": _make_mem02_player("p01", "seer"),
        "p02": _make_mem02_player("p02", "villager"),
        "p03": _make_mem02_player("p03", "werewolf"),
    }
    gs = GameState(
        game_id="g_test_mem02_e2e",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players=players,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p01",
                    "text": "p03 发言有明显的逻辑漏洞，没解释清楚",
                    "day_number": 1,
                    "visibility": "public",
                },
            ),
        ],
    )

    ruleset = Ruleset(raw={
        "player_count": 3,
        "roles": {
            "werewolf": {"count": 1},
            "villager": {"count": 1},
            "seer": {"count": 1},
        },
    })
    engine = RuleEngine(ruleset=ruleset)

    ctx = build_agent_context(
        engine=engine,
        gs=gs,
        player_id="p01",
        task_type=TaskType.SPEECH,
    )

    # The hint should now live on AgentContext.private_memory_caveat.
    assert ctx.private_memory_caveat == _LLM_AWARE_HINT, (
        f"MEM-02: build_agent_context must plumb the P1-M10 caveat onto "
        f"AgentContext.private_memory_caveat; got: "
        f"{ctx.private_memory_caveat!r}"
    )
    # And the memory dict that downstream renderers consume must NOT
    # contain the meta key (it's a renderer signal, not memory).
    assert "_llm_aware_hint" not in ctx.private_memory_hints, (
        f"MEM-02: meta key must be popped from private_memory_hints "
        f"before reaching renderers; got: {ctx.private_memory_hints!r}"
    )
    # And the same for the visible state.
    visible_mem = ctx.visible_world_state.get("private_memory", {}) or {}
    assert "_llm_aware_hint" not in visible_mem, (
        f"MEM-02: meta key must be popped from visible.private_memory; "
        f"got: {visible_mem!r}"
    )

    # End-to-end: the prompt builder must include the caveat in the
    # rendered user prompt. We build the full prompt to make sure the
    # caveat actually surfaces in the LLM-facing text.
    builder = PlayerPromptBuilder(ctx)
    user_prompt = builder.build_user_prompt(RetryInfo())

    assert _LLM_AWARE_HINT in user_prompt, (
        f"MEM-02: _LLM_AWARE_HINT must appear in the rendered user prompt. "
        f"Expected substring: {_LLM_AWARE_HINT!r}. "
        f"Got user prompt (excerpt around '私有记忆'): "
        f"{user_prompt[max(0, user_prompt.find('私有记忆') - 50): user_prompt.find('私有记忆') + 500]!r}"
    )


def test_private_memory_caveat_omitted_when_logic_flaws_empty() -> None:
    """MEM-02: when public speeches contain NO logic_flaw / valid_point
    markers, the AgentContext.private_memory_caveat must be empty and
    the prompt must NOT contain the caveat. (Avoids noise when there
    is nothing to caveat.)"""
    from werewolf_agent.core.models import GameEvent, GameState
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.schemas import RetryInfo, TaskType
    from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
    from werewolf_agent.runtime.context import build_agent_context
    from werewolf_agent.runtime.private_memory import _LLM_AWARE_HINT

    players = {
        "p01": _make_mem02_player("p01", "seer"),
        "p02": _make_mem02_player("p02", "villager"),
    }
    gs = GameState(
        game_id="g_test_mem02_empty_e2e",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        day_number=1,
        night_number=1,
        phase="day",
        players=players,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "text": "我随便聊几句",
                    "day_number": 1,
                    "visibility": "public",
                },
            ),
        ],
    )

    ruleset = Ruleset(raw={
        "player_count": 2,
        "roles": {
            "villager": {"count": 1},
            "seer": {"count": 1},
        },
    })
    engine = RuleEngine(ruleset=ruleset)

    ctx = build_agent_context(
        engine=engine,
        gs=gs,
        player_id="p01",
        task_type=TaskType.SPEECH,
    )

    # No caveat when there is nothing to caveat.
    assert ctx.private_memory_caveat == "", (
        f"MEM-02: caveat must be empty when logic_flaws / valid_points "
        f"are both empty; got: {ctx.private_memory_caveat!r}"
    )
    # And the rendered prompt does not contain it.
    builder = PlayerPromptBuilder(ctx)
    user_prompt = builder.build_user_prompt(RetryInfo())
    assert _LLM_AWARE_HINT not in user_prompt, (
        f"MEM-02: caveat must NOT appear in the rendered user prompt "
        f"when there is nothing to caveat. Got excerpt: "
        f"{user_prompt[:500]!r}"
    )


# ---------------------------------------------------------------------------
# S-04: _inject_skill_output's second return slot is populated.
# ---------------------------------------------------------------------------

def test_skill_analyses_field_populated() -> None:
    """S-04: `_inject_skill_output` must return a populated dict in the
    second slot so `AgentContext.skill_analyses` is non-empty.

    Pre-fix: the function returned `(strategy_directive, {})` — the
    second slot was always empty, so `AgentContext.skill_analyses` was
    always `{}` and `_build_skill_analysis_hints` rendered nothing.

    Post-fix: for each skill that fires, the second slot carries an
    entry `{skill_name: prompt_injectable}` (or empty string when the
    skill had nothing to say). The field is at least present.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output

    # Build a 6-player GameState with at least one villager so the
    # common-faction skills (push_vote, find_power, hide_identity, ...)
    # are dispatchable.
    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="seer", alive=True),
        "p04": PlayerState(id="p04", role="villager", alive=True),
        "p05": PlayerState(id="p05", role="witch", alive=True),
        "p06": PlayerState(id="p06", role="hunter", alive=True),
    }
    gs = GameState(
        ruleset_id="test",
        game_id="skill_analyses_test",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )

    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.cognition.world_state import build_world_state

    world_state = build_world_state(gs)
    belief = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    belief = BeliefUpdater().update(belief, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)

    _strategy_directive, analyses = _inject_skill_output(
        {}, gs, "p01", world_state, belief, alerts, "speech",
    )

    # S-04: the analyses dict must be populated — at minimum, it must
    # be a dict (not None) and contain an entry for at least one skill
    # that fired. Pre-fix this dict is always `{}`.
    assert isinstance(analyses, dict), (
        f"S-04: _inject_skill_output must return a dict as the second "
        f"slot; got {type(analyses).__name__}"
    )
    # And end-to-end: AgentContext.skill_analyses must round-trip
    # through build_agent_context.
    from werewolf_agent.agents.schemas import ActionType, TaskType
    from werewolf_agent.engine.rule_engine import RuleEngine

    engine = RuleEngine.from_yaml(
        Path(__file__).resolve().parents[2] / "config" / "rulesets"
        / "pre_witch_hunter_idiot_mixed.yaml"
    )
    context = build_agent_context(
        engine, gs, "p01", TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
    )
    # Pre-fix: context.skill_analyses == {}
    # Post-fix: must be non-empty (at least one skill that fired).
    assert context.skill_analyses, (
        f"S-04: AgentContext.skill_analyses must be non-empty when "
        f"skills fire; got {context.skill_analyses!r}"
    )


# ---------------------------------------------------------------------------
# S-05: _inject_skill_output's 7th param is `task_type`, not `phase`.
# ---------------------------------------------------------------------------

def test_inject_skill_output_receives_task_type(monkeypatch) -> None:
    """S-05: the 7th parameter of `_inject_skill_output` is `task_type`,
    not `phase`. The production call site passes `task_type.value` to
    it; inside the function, that value must be forwarded to
    `dispatch_for_role`'s `task_type` keyword (so the P0-K2
    `applies_to_task_types` filter actually fires).

    Pre-fix: the parameter was named `phase` (misnamed) and the call
    passed `task_type.value` to it. The function forwarded the value
    as the `phase` arg of `dispatch_for_role` and the local `task_type`
    was "" — so the precise task-type filter never fired.

    Post-fix: the parameter is renamed to `task_type`. We mock
    `dispatch_for_role` and assert it was called with the right
    `task_type` keyword.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime import context as context_mod

    # Mock SkillRegistry.dispatch_for_role to capture the kwargs.
    captured: dict[str, Any] = {}

    class _FakeRegistry:
        def dispatch_for_role(
            self, role, phase, skill_input, task_type="", gs=None
        ):
            captured["role"] = role
            captured["phase"] = phase
            captured["task_type"] = task_type
            captured["gs"] = gs
            return []

    monkeypatch.setattr(context_mod, "SkillRegistry", _FakeRegistry)

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
    }
    gs = GameState(
        ruleset_id="test",
        game_id="s05_test",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )

    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.cognition.world_state import build_world_state

    world_state = build_world_state(gs)
    belief = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    belief = BeliefUpdater().update(belief, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)

    # Call with the renamed parameter as a keyword argument. This is
    # the new contract: `task_type="speech"` is the 7th positional /
    # `task_type` keyword.
    _directive, _ = context_mod._inject_skill_output(
        {}, gs, "p01", world_state, belief, alerts,
        task_type="speech",
    )

    # S-05: dispatch_for_role must receive the task_type value as its
    # `task_type` keyword. Pre-fix this was "" (the local var was never
    # set; the value was bound to the misnamed `phase` param instead).
    assert captured.get("task_type") == "speech", (
        f"S-05: dispatch_for_role must receive task_type='speech'; "
        f"got task_type={captured.get('task_type')!r}. The 7th param "
        f"of _inject_skill_output is misnamed — it should be task_type, "
        f"not phase."
    )


# ---------------------------------------------------------------------------
# S-19: filter skill output entries that reference illegal targets.
# ---------------------------------------------------------------------------

def test_skill_output_filters_illegal_targets():
    """S-19: a skill output that recommends an illegal (dead / out of
    legal_targets) player must be dropped from skill_tactical_advice.

    Pre-fix: the post-step in _inject_skill_output didn't filter
    illegal-target advice.  A push_vote output naming p01 (dead) would
    pass through to the LLM, where it would confuse the action.

    Post-fix: structured advice entries whose `advice` text mentions
    a player_id NOT in legal_targets are dropped.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState,
    )
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine

    # Game state with one dead player (p05).  We monkeypatch
    # push_vote's handler to emit advice naming p05 (an illegal target).
    from werewolf_agent.skills.schemas import SkillName
    from werewolf_agent.skills.werewolf_skills import (
        SKILL_DEFINITIONS, register_handler, get_handler,
    )
    from werewolf_agent.skills.schemas import SkillInput, SkillOutput

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    # p05 is dead.
    players["p05"] = PlayerState(id="p05", role="villager", alive=False)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    ws = StructuredWorldState()
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    alerts = ContradictionEngine().detect(ws.facts, gs.day_number)

    # Monkeypatch push_vote handler to emit illegal-target advice.
    def _illegal_handler(inp, skill):
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["投p05"],
            confidence=0.6,
            reasoning="illegal target test",
            prompt_injectable="归票建议：投票 p05（illegal）",
        )
    register_handler(SkillName.PUSH_VOTE)(_illegal_handler)

    try:
        # legal_targets excludes p05 (dead).
        legal = [f"p{i:02d}" for i in range(1, 13) if i != 5 and f"p{i:02d}" != "p01"]
        directive, _ = _inject_skill_output(
            {}, gs, "p01", ws, bs, alerts, "speech",
            legal_targets=legal,
        )
        advice = directive.get("skill_tactical_advice", [])
        # Every push_vote entry must NOT mention p05 (illegal target).
        for entry in advice:
            if isinstance(entry, dict) and entry.get("skill") == "push_vote":
                assert "p05" not in entry.get("advice", ""), (
                    f"S-19: push_vote advice must not reference illegal "
                    f"target p05; got: {entry!r}"
                )
    finally:
        # Restore the real push_vote handler.
        from werewolf_agent.skills.werewolf_skills import push_vote_handler
        register_handler(SkillName.PUSH_VOTE)(push_vote_handler)


def test_skill_output_s19_widened_regex_catches_chinese_player_ids():
    """Phase 2 P2-10: the S-19 illegal-target post-step must catch
    Chinese-numbered player references that the old ``p\\d{2}`` regex
    missed.

    Pre-fix: advice that said "投10号玩家" or "玩家 03" slipped
    through S-19 because ``p\\d{2}`` only matched the bare pNN form.
    New regexes:
      - ``\\b[pP]\\d+\\b``            (uppercase P or single-digit)
      - ``(\\d+)\\s*号\\s*玩家?``     (Chinese "10号玩家")
      - ``玩家\\s*(\\d+)``            ("玩家 03")
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output
    from werewolf_agent.cognition.world_state import StructuredWorldState
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.skills.schemas import SkillName, SkillOutput
    from werewolf_agent.skills.werewolf_skills import register_handler

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test", game_id="g", phase="speech",
        day_number=1, night_number=1, players=players,
    )
    ws = StructuredWorldState()
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    alerts = ContradictionEngine().detect(ws.facts, gs.day_number)

    # Three different illegal-target styles to verify the widened regex.
    def _chinese_handler(inp, skill):
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["投10号玩家"],
            confidence=0.6,
            reasoning="chinese variant test",
            prompt_injectable="归票建议：投票 10号玩家（illegal）",
        )
    register_handler(SkillName.PUSH_VOTE)(_chinese_handler)

    try:
        legal = [f"p{i:02d}" for i in range(1, 13) if i != 10 and f"p{i:02d}" != "p01"]
        directive, _ = _inject_skill_output(
            {}, gs, "p01", ws, bs, alerts, "speech",
            legal_targets=legal,
        )
        advice = directive.get("skill_tactical_advice", [])
        # Every push_vote entry must NOT reference p10 in any variant
        for entry in advice:
            if isinstance(entry, dict) and entry.get("skill") == "push_vote":
                assert "10号" not in entry.get("advice", ""), (
                    f"P2-10: push_vote advice must not reference illegal "
                    f"target via Chinese '10号玩家' variant; got: {entry!r}"
                )
    finally:
        from werewolf_agent.skills.werewolf_skills import push_vote_handler
        register_handler(SkillName.PUSH_VOTE)(push_vote_handler)


# ---------------------------------------------------------------------------
# S-07: skill_tactical_advice is a structured list of dicts.
# ---------------------------------------------------------------------------

def test_skill_tactical_advice_is_structured():
    """S-07: strategy_directive["skill_tactical_advice"] must be a
    structured list of {skill, advice, confidence} dicts — not an
    opaque string.  Sibling directive keys (must_address_alerts,
    role_alerts) are already structured lists.  The prompt builder
    formats the list into a renderable block.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState,
    )
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    ws = StructuredWorldState()
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    alerts = ContradictionEngine().detect(ws.facts, gs.day_number)

    directive, _ = _inject_skill_output(
        {}, gs, "p01", ws, bs, alerts, "speech",
    )
    advice = directive.get("skill_tactical_advice", None)
    assert isinstance(advice, list), (
        f"S-07: skill_tactical_advice must be a list (structured), got "
        f"{type(advice).__name__}: {advice!r}"
    )
    if advice:  # if any skills fired, the entries are dicts
        for entry in advice:
            assert isinstance(entry, dict), (
                f"S-07: each advice entry must be a dict; got {type(entry).__name__}: {entry!r}"
            )
            assert "skill" in entry and "advice" in entry, (
                f"S-07: advice entry must have 'skill' and 'advice' keys; "
                f"got: {list(entry.keys())!r}"
            )
            assert "confidence" in entry, (
                f"S-07: advice entry must have 'confidence' key; "
                f"got: {list(entry.keys())!r}"
            )


def test_skill_tactical_advice_entries_use_frame_schema():
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output
    from werewolf_agent.cognition.world_state import StructuredWorldState
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    ws = StructuredWorldState()
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    alerts = ContradictionEngine().detect(ws.facts, gs.day_number)

    directive, _ = _inject_skill_output(
        {}, gs, "p01", ws, bs, alerts, "speech",
    )

    advice = directive.get("skill_tactical_advice", [])
    assert advice
    for entry in advice:
        assert set(entry) >= {
            "skill",
            "situation_signature",
            "recommended_use",
            "risk_alerts",
            "counter_signals",
            "forbidden_use",
            "confidence",
            "relevance",
        }
        assert isinstance(entry["risk_alerts"], list)
        assert isinstance(entry["counter_signals"], list)
        assert entry["recommended_use"]
        assert entry["forbidden_use"]


def test_skill_tactical_advice_gates_to_top_three_by_relevance(monkeypatch):
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime import context as context_mod
    from werewolf_agent.skills.schemas import SkillOutput

    class _FakeRegistry:
        def dispatch_for_role(self, role, phase, skill_input, task_type="", gs=None):
            return [
                SkillOutput(
                    skill_name=f"skill_{idx}",
                    confidence=0.2 + idx / 10,
                    prompt_injectable=f"技能建议 {idx}",
                )
                for idx in range(5)
            ]

    monkeypatch.setattr(context_mod, "SkillRegistry", _FakeRegistry)

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 7)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="skill_gate_test",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.cognition.world_state import build_world_state

    world_state = build_world_state(gs)
    belief = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)

    directive, _ = context_mod._inject_skill_output(
        {}, gs, "p01", world_state, belief, alerts, "speech",
    )

    advice = directive.get("skill_tactical_advice", [])
    assert len(advice) == 3
    assert [entry["skill"] for entry in advice] == [
        "skill_4",
        "skill_3",
        "skill_2",
    ]


# ---------------------------------------------------------------------------
# S-16: single-source wolf-role skip — context.py does NOT skip;
# the handler does.
# ---------------------------------------------------------------------------

def test_wolf_skip_in_handler_only():
    """S-16: the wolf-role skip (bold_claim for non-fake_seer wolves,
    deep_hook for fake_seer/pusher, swing_vote for hooker) is
    authoritative in the handler. context.py must not re-implement
    the skip — that risks drift between the two copies.

    Pin the contract: when a hooker wolf is in speech phase with
    a teammate as fake_seer, the bold_claim handler emits a
    role-neutral "已有队友占据预言家身份" prompt (S-14 phrasing).
    context.py must record that prompt in analyses — it must NOT
    silently filter bold_claim out at the context layer.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState,
    )
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="werewolf", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    ws = StructuredWorldState()
    ws.append(StructuredFact(
        fact_type="claimed_role", source_player="p05", value="seer",
        day=1,
    ))
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    alerts = ContradictionEngine().detect(ws.facts, gs.day_number)

    # Call the real _inject_skill_output with a wolf_team_plan that
    # makes p01 a "hooker" — the bold_claim handler must emit its
    # "已有队友占据预言家身份" prompt, and context.py must record it
    # without further filtering.
    directive, analyses = _inject_skill_output(
        {}, gs, "p01", ws, bs, alerts, "speech",
        wolf_team_plan={"fake_seer": "p05", "hooker": "p01"},
    )
    # The handler produces the S-14 role-neutral skip prompt. context.py
    # must NOT filter it out (S-16: handler is authoritative).
    assert "bold_claim" in analyses, (
        "S-16: context.py must not skip bold_claim on its own — the "
        "handler's filtered output should pass through. analyses keys: "
        f"{list(analyses.keys())!r}"
    )
    assert "已有队友占据预言家身份" in analyses["bold_claim"], (
        "S-16: handler's role-neutral skip prompt should pass through "
        f"context.py unchanged. Got: {analyses['bold_claim']!r}"
    )
    # Source-code contract: context.py must not contain a `wolf_role`
    # filter block. Read the module source and assert.
    from werewolf_agent.runtime import context as _ctx_mod
    import inspect
    src = inspect.getsource(_ctx_mod._inject_skill_output)
    assert 'wolf_role and wolf_role != "fake_seer"' not in src, (
        "S-16: context.py must not contain the bold_claim wolf-role "
        "skip — that's the handler's job. Found it in _inject_skill_output."
    )
    assert "wolf_role and wolf_role in (\"fake_seer\", \"pusher\")" not in src, (
        "S-16: context.py must not contain the deep_hook wolf-role "
        "skip — that's the handler's job. Found it in _inject_skill_output."
    )
    assert 'wolf_role == "hooker"' not in src, (
        "S-16: context.py must not contain the swing_vote hooker "
        "skip — that's the handler's job. Found it in _inject_skill_output."
    )


# ---------------------------------------------------------------------------
# Reflection V2 live prompt boundary: legacy V1 reflection bodies are audit /
# migration inputs only. They must not be injected into player prompts even
# when a profile exists.
# ---------------------------------------------------------------------------


def test_build_agent_context_ignores_legacy_reflections_even_with_profile() -> None:
    from werewolf_agent.agents.schemas import TaskType
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
    from werewolf_agent.runtime.context import build_agent_context

    # Build a GameState where p01 has a profile and legacy V1 reflection rows.
    # V2 live-learning must not fall back to these raw reflection bodies.
    players = {
        "p01": PlayerState(id="p01", role="hybrid", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    gs = GameState(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        game_id="g_test_mem_new3",
        day_number=1,
        night_number=1,
        phase="day",
        players=players,
        hybrid_master_faction="werewolf",
    )

    werewolf_ref = ReflectionEntry(
        entry_id="z_wolf",
        game_id="2025-01-01",
        player_id="p10",
        role="werewolf",
        faction_won=False,
        text="wolf-perspective-reflection",
    )
    seer_ref = ReflectionEntry(
        entry_id="a_seer",
        game_id="2025-01-01",
        player_id="p11",
        role="seer",
        faction_won=False,
        text="seer-perspective-reflection",
    )
    profile = PlayerProfile(player_id="p01", games_played=2)

    class _FakeRestoredMemory:
        def get_profile(self, pid):
            return profile if pid == "p01" else None

        def reflections_by_player(self, pid):
            if pid == "p01":
                return [werewolf_ref, seer_ref]
            return []

    ruleset = Ruleset(raw={
        "player_count": 3,
        "roles": {
            "werewolf": {"count": 1},
            "villager": {"count": 1},
            "hybrid": {"count": 1},
        },
    })
    engine = RuleEngine(ruleset=ruleset)
    ctx = build_agent_context(
        engine=engine,
        gs=gs,
        player_id="p01",
        task_type=TaskType.SPEECH,
        restored_memory=_FakeRestoredMemory(),
    )

    assert ctx.profile_memory_hint
    assert ctx.reflection_memory_hints == []
    assert "wolf-perspective-reflection" not in str(ctx.error_pattern_hint)
    assert "seer-perspective-reflection" not in str(ctx.error_pattern_hint)


def test_dead_hybrid_master_context_does_not_reveal_master_faction() -> None:
    from werewolf_agent.agents.schemas import ActionType, TaskType
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import build_agent_context
    from werewolf_agent.runtime.graph import _new_engine

    gs = GameState(
        game_id="hybrid_dead_master_visibility",
        phase="day",
        day_number=2,
        players={
            "p01": PlayerState(id="p01", role="hybrid", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
            "p03": PlayerState(id="p03", role="villager", alive=True),
        },
        hybrid_master_id="p02",
        hybrid_master_faction="werewolf",
    )

    context = build_agent_context(
        _new_engine(),
        gs,
        "p01",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
    )

    directive = str(context.strategy_directive)
    assert "狼人阵营" not in directive
    assert "主人p02已死亡" in directive
    assert "仍不知道主人的阵营" in directive


def test_all_configured_persona_speech_styles_have_runtime_hints() -> None:
    import yaml

    from werewolf_agent.runtime.context import _SPEECH_STYLE_HINTS

    profiles = yaml.safe_load(
        Path("config/personas/jingcheng_style_prototypes.yaml").read_text(
            encoding="utf-8"
        )
    )["persona_profiles"]
    configured_styles = {
        profile.get("base", {}).get("speech_style", "")
        for profile in profiles.values()
    }

    assert configured_styles <= set(_SPEECH_STYLE_HINTS)

# NEW-S02-A: dispatch_for_role receives gs so hybrid wolf-master dispatch works.
# ---------------------------------------------------------------------------


def test_inject_skill_output_passes_gs_to_dispatch(monkeypatch) -> None:
    """NEW-S02-A: `_inject_skill_output` must call `dispatch_for_role`
    with `gs=gs`. The S-02 unit test only passed `gs=gs` directly to
    the registry call; the production call site at context.py:520-522
    was missing the keyword, so a hybrid with master=werewolf in
    production couldn't dispatch WOLF-faction skills (faction_for_role
    fell back to GOOD without `gs`).
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime import context as context_mod

    captured: dict[str, Any] = {}

    class _FakeRegistry:
        def dispatch_for_role(
            self, role, phase, skill_input, task_type="", gs=None
        ):
            captured["role"] = role
            captured["phase"] = phase
            captured["task_type"] = task_type
            captured["gs"] = gs
            return []

    monkeypatch.setattr(context_mod, "SkillRegistry", _FakeRegistry)

    # 6-player game with a hybrid whose master is a werewolf. Without
    # gs passed to dispatch_for_role, the registry's `faction_for_role`
    # falls back to GOOD and WOLF-faction skills are unreachable.
    players = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="hybrid", alive=True),
        "p04": PlayerState(id="p04", role="villager", alive=True),
        "p05": PlayerState(id="p05", role="villager", alive=True),
        "p06": PlayerState(id="p06", role="seer", alive=True),
    }
    gs = GameState(
        ruleset_id="test",
        game_id="new_s02a_test",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
        hybrid_master_id="p01",
        hybrid_master_faction="werewolf",
    )

    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.cognition.world_state import build_world_state

    world_state = build_world_state(gs)
    belief = BeliefUpdater().initialize(list(gs.players.keys()), "p03")
    belief = BeliefUpdater().update(belief, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)

    context_mod._inject_skill_output(
        {}, gs, "p03", world_state, belief, alerts, "speech",
    )

    # NEW-S02-A: dispatch_for_role must receive gs as a keyword. Pre-fix
    # the production call site at context.py:520-522 was missing the
    # gs= keyword — so hybrid wolf-master dispatch broke.
    assert captured.get("gs") is gs, (
        f"NEW-S02-A: dispatch_for_role must be called with gs=gs; "
        f"got gs={captured.get('gs')!r}. The production call site at "
        f"context.py:520-522 is missing the gs= keyword — so hybrid "
        f"wolf-master dispatch breaks in production."
    )


# ---------------------------------------------------------------------------
# NEW-S16-A: dead code (wolf_role computation) removed from context.py.
# ---------------------------------------------------------------------------


def test_wolf_role_computation_removed() -> None:
    """NEW-S16-A: the `wolf_role = None` block was dead code — the
    wolf-role skip moved into the handler (S-16). The variable was
    computed but never read. Assert the source no longer contains the
    dead block.
    """
    from werewolf_agent.runtime import context as context_mod
    import inspect
    import re as _re
    src = inspect.getsource(context_mod._inject_skill_output)
    # Strip line comments so explanatory comments don't false-positive.
    code_lines = [
        ln for ln in src.splitlines()
        if ln.lstrip().startswith("#") is False
    ]
    code = "\n".join(code_lines)
    assert "wolf_role = None" not in code, (
        f"NEW-S16-A: dead code `wolf_role = None` must be removed. "
        f"Found in _inject_skill_output."
    )
    # Also assert the for-loop scanning wolf_team_plan is gone.
    assert 'for role_key in ("fake_seer", "pusher", "hooker", "deep_cover")' not in code, (
        f"NEW-S16-A: dead wolf-team-role scanning loop must be removed. "
        f"Found in _inject_skill_output."
    )


# ---------------------------------------------------------------------------
# NEW-S19-A: illegal-target filter does not drop last_words advice.
# ---------------------------------------------------------------------------


def test_skill_illegal_target_filter_skips_last_words() -> None:
    """NEW-S19-A: the S-19 illegal-target post-step must not drop
    advice from skills whose `applicable_phases` includes
    `last_words` (or `review`). These skills routinely mention dead
    players by id (`p05的遗言：...`), and the S-19 regex `p\\d{2}`
    would drop the entire entry if p05 isn't in `legal_targets` (which
    for last_words is `alive_others` — p05 is dead, so the filter
    flags it as illegal and the advice disappears).

    Fix: skip the S-19 filter for skills whose applicable_phases
    includes `last_words` (or `review`). Build a separate
    `legal_targets_for_analysis` set that includes dead players so
    the analysis still works.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output
    from werewolf_agent.cognition.world_state import (
        StructuredWorldState,
    )
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.skills.schemas import SkillName
    from werewolf_agent.skills.werewolf_skills import (
        register_handler,
    )
    from werewolf_agent.skills.schemas import SkillInput, SkillOutput

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    # p05 is dead — last_words will mention p05 by id, which S-19 would
    # flag as illegal (p05 not in legal_targets).
    players["p05"] = PlayerState(id="p05", role="villager", alive=False)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    ws = StructuredWorldState()
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    alerts = ContradictionEngine().detect(ws.facts, gs.day_number)

    # Monkeypatch last_words_analysis to emit p05 explicitly.
    def _last_words_handler(inp, skill):
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["遗言分析"],
            confidence=0.6,
            reasoning="last words mentions dead p05",
            prompt_injectable=(
                "遗言分析（1人死亡）：\n"
                "p05的遗言：身份声明：seer。"
            ),
        )

    register_handler(SkillName.LAST_WORDS_ANALYSIS)(_last_words_handler)
    try:
        # legal_targets = alive_others (p05 is dead, NOT in legal_targets).
        legal = [
            f"p{i:02d}" for i in range(1, 13)
            if i != 5 and f"p{i:02d}" != "p01"
        ]
        directive, _ = _inject_skill_output(
            {}, gs, "p01", ws, bs, alerts, "speech",
            legal_targets=legal,
        )
        advice = directive.get("skill_tactical_advice", [])
        last_words_entries = [
            e for e in advice
            if isinstance(e, dict) and e.get("skill") == "last_words"
        ]
        # NEW-S19-A: the last_words advice must NOT be dropped even
        # though it mentions p05 (a dead player).
        assert last_words_entries, (
            f"NEW-S19-A: last_words advice must not be dropped by "
            f"S-19 illegal-target filter when it mentions a dead "
            f"player. Advice entries: {advice!r}"
        )
        assert any("p05" in e.get("advice", "") for e in last_words_entries), (
            f"NEW-S19-A: at least one last_words advice entry must "
            f"contain p05. Advice: {last_words_entries!r}"
        )
    finally:
        # Restore the real handler.
        from werewolf_agent.skills.werewolf_skills import last_words_handler
        register_handler(SkillName.LAST_WORDS_ANALYSIS)(last_words_handler)


# ---------------------------------------------------------------------------
# NEW-S04-B: dedupe uses object identity, not full-prompt string compare.
# ---------------------------------------------------------------------------


def test_skill_seen_dedupe_uses_id_not_prompt(monkeypatch) -> None:
    """NEW-S04-B: the `seen` dedupe set in `_inject_skill_output`
    must key on object identity (`id(o)`) or `(skill_name, prompt[:50])`
    — NOT on the full `prompt_injectable` string. S-06's length cap
    produces identical `...（已省略）` truncations for two different
    skills whose original prompts differ; the string-based dedupe
    would hide the second skill.

    We mock `dispatch_for_role` to return two outputs with identical
    truncated prompts but different skill_names. Both must be
    included in the structured `skill_tactical_advice` output.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime import context as context_mod
    from werewolf_agent.skills.schemas import SkillOutput

    truncated = "前文...（已省略）"  # S-06 marker; same for both skills

    class _FakeRegistry:
        def dispatch_for_role(self, role, phase, skill_input, task_type="", gs=None):
            return [
                SkillOutput(
                    skill_name="push_vote",
                    speech_structure=["rally"],
                    confidence=0.7,
                    reasoning="truncated",
                    prompt_injectable=truncated,
                ),
                SkillOutput(
                    skill_name="find_power",
                    speech_structure=["analyze"],
                    confidence=0.6,
                    reasoning="truncated",
                    prompt_injectable=truncated,
                ),
            ]

    monkeypatch.setattr(context_mod, "SkillRegistry", _FakeRegistry)

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 7)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="dedup_id_test",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.cognition.world_state import build_world_state

    world_state = build_world_state(gs)
    belief = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    belief = BeliefUpdater().update(belief, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)

    directive, _ = context_mod._inject_skill_output(
        {}, gs, "p01", world_state, belief, alerts, "speech",
    )
    advice = directive.get("skill_tactical_advice", [])
    skill_names = {e.get("skill") for e in advice if isinstance(e, dict)}

    # NEW-S04-B: both push_vote and find_power must be in the output.
    assert {"push_vote", "find_power"}.issubset(skill_names), (
        f"NEW-S04-B: dedupe must keep both push_vote and find_power "
        f"even when their prompt_injectable strings are identical "
        f"(S-06 truncation). Got skills: {skill_names!r}"
    )


# ---------------------------------------------------------------------------
# NEW-S04-A: skill_analysis_hints is dropped; single render path is
#            strategy_directive.skill_tactical_advice.
# ---------------------------------------------------------------------------


def test_skill_analysis_hints_dedup() -> None:
    """NEW-S04-A: AgentContext.skill_analysis_hints must be empty
    after build_agent_context. The single source of truth for the
    prompt builder is `strategy_directive.skill_tactical_advice`.
    Previously, the same dict was passed to BOTH `skill_analyses`
    and `skill_analysis_hints` — the prompt builder rendered
    skill_analysis_hints as a JSON block AND strategy_directive
    rendered skill_tactical_advice, doubling the token budget.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime.context import _inject_skill_output
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.cognition.world_state import build_world_state

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 7)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="dedup_test",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    world_state = build_world_state(gs)
    belief = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    belief = BeliefUpdater().update(belief, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)

    directive, _analyses = _inject_skill_output(
        {}, gs, "p01", world_state, belief, alerts, "speech",
    )

    # NEW-S04-A: skill_analyses is the only opaque-dict slot. The
    # structured `skill_tactical_advice` lives in strategy_directive.
    # The prompt builder renders only the structured path. The
    # opaque `skill_analyses` is not duplicated to skill_analysis_hints.
    assert "skill_analyses" not in directive, (
        f"NEW-S04-A: skill_analyses must not be a key on the "
        f"strategy_directive dict; the structured "
        f"skill_tactical_advice is the single source of truth. "
        f"Got: {list(directive.keys())!r}"
    )
    # skill_tactical_advice is the structured render path.
    advice = directive.get("skill_tactical_advice", None)
    if advice:
        assert isinstance(advice, list), (
            f"NEW-S04-A: skill_tactical_advice must be a list of dicts. "
            f"Got: {type(advice).__name__}: {advice!r}"
        )


def test_skill_analysis_hints_field_empty_after_build_agent_context() -> None:
    """NEW-S04-A: end-to-end: build_agent_context must NOT populate
    `ctx.skill_analysis_hints`. The dual render path is gone.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime import context as context_mod
    from werewolf_agent.agents.schemas import ActionType, TaskType
    from werewolf_agent.engine.rule_engine import RuleEngine

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 7)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="end_to_end_dedup",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    engine = RuleEngine.from_yaml(
        Path(__file__).resolve().parents[2] / "config" / "rulesets"
        / "pre_witch_hunter_idiot_mixed.yaml"
    )
    ctx = context_mod.build_agent_context(
        engine, gs, "p01", TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
    )
    # NEW-S04-A: skill_analysis_hints is empty (the dual-render
    # duplication is gone). The skill advice is rendered from
    # strategy_directive.skill_tactical_advice only.
    assert not ctx.skill_analysis_hints, (
        f"NEW-S04-A: skill_analysis_hints must be empty after "
        f"build_agent_context. Got: {ctx.skill_analysis_hints!r}"
    )


def test_build_agent_context_prefers_live_cognition_manager_belief_summary() -> None:
    """Phase 1: live in-game cognition beats the recompute fallback."""
    from werewolf_agent.runtime.graph import _new_engine

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
    }
    gs = GameState(
        game_id="live_cognition_context",
        phase="day",
        day_number=1,
        night_number=1,
        players=players,
    )
    expected = {
        "my_suspects": [
            {
                "player": "p02",
                "faction_lean": "wolf_lean",
                "trust": 0.22,
                "top_role_guess": "werewolf",
                "top_role_prob": 0.78,
            }
        ],
        "my_trusted": [],
    }

    class _Manager:
        def prompt_belief_summary(self, viewer_id, game_state):
            assert viewer_id == "p01"
            assert game_state is gs
            return expected

    ctx = build_agent_context(
        _new_engine(),
        gs,
        "p01",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        cognition_state_manager=_Manager(),
    )

    assert ctx.belief_state == expected


def test_build_agent_context_populates_possible_worlds_from_belief_summary() -> None:
    """Phase 3: context carries bounded prompt-safe possible worlds."""
    from werewolf_agent.runtime.cognition_state import CognitionStateManager
    from werewolf_agent.runtime.graph import _new_engine

    roles = [
        "seer",
        "witch",
        "hunter",
        "idiot",
        "hybrid",
        "werewolf",
        "werewolf",
        "werewolf",
        "villager",
        "villager",
        "villager",
        "villager",
    ]
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role=role, alive=True)
        for i, role in enumerate(roles, start=1)
    }
    gs = GameState(
        game_id="possible_world_context",
        phase="day",
        day_number=1,
        night_number=1,
        players=players,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p01",
                    "text": "我是预言家，查验 p06 是狼人",
                    "day_number": 1,
                },
            )
        ],
    )
    manager = CognitionStateManager()
    manager.initialize(gs)
    manager.update_from_events(gs)

    ctx = build_agent_context(
        _new_engine(),
        gs,
        "p09",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        cognition_state_manager=manager,
    )

    assert ctx.possible_worlds["type"] == "possible_worlds"
    assert ctx.possible_worlds["top_worlds"]
    assert "roles" not in ctx.possible_worlds["top_worlds"][0]
    assert "warning" in ctx.possible_worlds


def test_build_agent_context_populates_simulation_predictions() -> None:
    """Phase 5: context carries bounded prompt-safe simulator predictions."""
    from werewolf_agent.runtime.cognition_state import CognitionStateManager
    from werewolf_agent.runtime.graph import _new_engine

    roles = [
        "seer",
        "witch",
        "hunter",
        "idiot",
        "hybrid",
        "werewolf",
        "werewolf",
        "werewolf",
        "villager",
        "villager",
        "villager",
        "villager",
    ]
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role=role, alive=True)
        for i, role in enumerate(roles, start=1)
    }
    gs = GameState(
        game_id="simulation_context",
        phase="day",
        day_number=2,
        night_number=1,
        players=players,
        events=[
            GameEvent(
                type="vote_resolved",
                payload={
                    "votes": [
                        {"voter": "p01", "target": "p06"},
                        {"voter": "p02", "target": "p06"},
                    ],
                    "day_number": 1,
                },
            )
        ],
    )
    manager = CognitionStateManager()
    manager.initialize(gs)
    manager.update_from_events(gs)

    ctx = build_agent_context(
        _new_engine(),
        gs,
        "p09",
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        cognition_state_manager=manager,
    )

    assert ctx.simulation_predictions["type"] == "simulation"
    assert ctx.simulation_predictions["warning"] == "Prediction, not fact."
    assert len(ctx.simulation_predictions["predictions"]) <= 2
    assert "roles" not in str(ctx.simulation_predictions)


def test_agent_reflection_passes_memory_context_managers() -> None:
    import inspect
    from werewolf_agent.runtime.agent_adapter import _agent_reflection

    source = inspect.getsource(_agent_reflection)

    assert 'restored_memory=state.get("restored_memory")' in source
    assert 'cognition_state_manager=state.get("cognition_state_manager")' in source


def test_summarize_positions_passes_memory_context_managers() -> None:
    import inspect
    from werewolf_agent.runtime.nodes.summary import summarize_positions

    source = inspect.getsource(summarize_positions)

    assert 'restored_memory=state.get("restored_memory")' in source
    assert 'cognition_state_manager=state.get("cognition_state_manager")' in source


# ---------------------------------------------------------------------------
# NEW-S19-C: illegal-target filter applies to the surviving render path.
# ---------------------------------------------------------------------------


def test_skill_illegal_target_filter_applies_to_all_render_paths() -> None:
    """NEW-S19-C: the S-19 illegal-target filter applies to the
    surviving render path (strategy_directive.skill_tactical_advice).
    The duplicate skill_analysis_hints render path is gone (NEW-S04-A),
    so the unfiltered leak is no longer possible. We assert:
      (1) AgentContext.skill_analysis_hints is empty (NEW-S04-A).
      (2) When an advice entry names an illegal target, it is
          filtered out of skill_tactical_advice (S-19).
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.runtime import context as context_mod
    from werewolf_agent.agents.schemas import ActionType, TaskType
    from werewolf_agent.engine.rule_engine import RuleEngine
    from werewolf_agent.skills.schemas import SkillName
    from werewolf_agent.skills.werewolf_skills import register_handler
    from werewolf_agent.skills.schemas import SkillOutput

    # Force a push_vote advice entry that names p05 (dead → illegal).
    def _illegal_handler(inp, skill):
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["投p05"],
            confidence=0.6,
            reasoning="illegal target test",
            prompt_injectable="归票建议：投票 p05（illegal）",
        )

    register_handler(SkillName.PUSH_VOTE)(_illegal_handler)
    try:
        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 13)
        }
        players["p05"] = PlayerState(id="p05", role="villager", alive=False)
        gs = GameState(
            ruleset_id="test",
            game_id="g",
            phase="speech",
            day_number=1,
            night_number=1,
            players=players,
        )
        engine = RuleEngine.from_yaml(
            Path(__file__).resolve().parents[2] / "config" / "rulesets"
            / "pre_witch_hunter_idiot_mixed.yaml"
        )
        # legal_targets excludes p05 (dead).
        legal = [
            f"p{i:02d}" for i in range(1, 13)
            if i != 5 and f"p{i:02d}" != "p01"
        ]
        ctx = context_mod.build_agent_context(
            engine, gs, "p01", TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
            legal_targets=legal,
        )
        # NEW-S04-A: skill_analysis_hints is empty (no dual render).
        assert not ctx.skill_analysis_hints, (
            f"NEW-S04-A/NEW-S19-C: skill_analysis_hints must be "
            f"empty. Got: {ctx.skill_analysis_hints!r}"
        )
        # NEW-S19-C: the surviving render path (skill_tactical_advice)
        # must filter the illegal target. We assert the advice does
        # not name p05 in any push_vote entry.
        advice = ctx.strategy_directive.get("skill_tactical_advice", [])
        for entry in advice:
            if isinstance(entry, dict) and entry.get("skill") == "push_vote":
                assert "p05" not in entry.get("advice", ""), (
                    f"NEW-S19-C: skill_tactical_advice push_vote entry "
                    f"must filter p05 (illegal). Got: {entry!r}"
                )
    finally:
        from werewolf_agent.skills.werewolf_skills import push_vote_handler
        register_handler(SkillName.PUSH_VOTE)(push_vote_handler)


# ---------------------------------------------------------------------------
# PR2: REFLECTION context 的 visible_world_state 必须是赛后摘要(回顾视角),
# 而非 live 局面。SPEECH/VOTE 保持原 visible 逻辑(回归)。
# ---------------------------------------------------------------------------

def _finished_gs_for_reflection() -> GameState:
    """赛后 GameState:p02/p05 死亡,p01(本玩家)存活到结束,狼人胜。"""
    from werewolf_agent.core.models import Death
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    players["p01"] = PlayerState(id="p01", role="seer", alive=True)
    players["p02"] = PlayerState(id="p02", role="werewolf", alive=False)
    players["p05"] = PlayerState(id="p05", role="villager", alive=False)
    deaths = [
        Death(player_id="p05", reason="exile", timing="day",
              resolution_batch="d1"),
        Death(player_id="p02", reason="wolf_kill", timing="night",
              resolution_batch="n2"),
    ]
    events = [
        GameEvent(type="seer_check", payload={
            "target_id": "p03", "alignment": "werewolf", "night_number": 1}),
        GameEvent(type="vote_resolved", payload={
            "day_number": 1,
            "votes": [{"voter": "p01", "target": "p05"}],
            "exiled": "p05"}),
        GameEvent(type="judge_broadcast", payload={"phase": "death_announce"}),
    ]
    return GameState(
        game_id="reflection_post_game",
        phase="finished",
        day_number=2,
        night_number=2,
        players=players,
        deaths=deaths,
        events=events,
        winning_faction="werewolf",
    )


def test_reflection_visible_is_post_game_summary_not_live():
    from werewolf_agent.runtime.graph import _new_engine

    gs = _finished_gs_for_reflection()
    ctx = build_agent_context(
        _new_engine(), gs, "p01", TaskType.REFLECTION,
        legal_actions=[ActionType.SPEECH],
    )
    visible = ctx.visible_world_state

    # 回顾视角:含胜负、自己存活结论、死亡顺序、自己行动时间线
    assert visible.get("game_phase") == "post_game"
    assert visible.get("winning_faction") == "werewolf"
    assert visible.get("viewer_survived") is True
    deaths = visible.get("deaths")
    assert isinstance(deaths, list) and deaths, "赛后摘要必须含死亡顺序"
    death_ids = [d.get("player_id") for d in deaths]
    assert "p05" in death_ids and "p02" in death_ids
    my_actions = visible.get("my_action_timeline")
    assert isinstance(my_actions, list) and my_actions, "必须含自己的行动时间线"

    # 进行时 live 字段必须不存在(回顾视角,非进行时)
    for live_key in ("alive_players", "phase", "day", "night", "phase_label"):
        assert live_key not in visible, (
            f"REFLECTION visible 不得含 live 字段 {live_key!r}: {visible!r}"
        )


def test_reflection_visible_excludes_other_private_state():
    """赛后摘要 visibility 安全:不含他人私身份/狼队信息。

    边界:winning_faction 是公开结果;viewer 自己的 seer_check
    结果是自己的私有信息——两者合法出现在摘要里。被排除的是
    他人私身份(wolf_teammates / 他人 role / 别人的 check_results)。
    """
    from werewolf_agent.runtime.graph import _new_engine

    gs = _finished_gs_for_reflection()
    ctx = build_agent_context(
        _new_engine(), gs, "p01", TaskType.REFLECTION,
        legal_actions=[ActionType.SPEECH],
    )
    visible = ctx.visible_world_state
    # 狼队信息(只有 werewolf 自己在 live 视角能看到)不得出现
    assert "wolf_teammates" not in visible
    assert "wolf_team_plan" not in visible
    # 死亡顺序只含公开结果(player_id/reason/timing),不含 role 字段
    for d in visible.get("deaths", []):
        assert "role" not in d, f"死亡记录不得暴露身份: {d!r}"
    # viewer 自己的 seer_check 是合法私有信息;但摘要不应混入 live
    # role-specific 字段(antidote_available / master_id 等)。
    for leak_key in ("antidote_available", "poison_available", "master_id"):
        assert leak_key not in visible, (
            f"赛后摘要不得含 live role-private 字段 {leak_key!r}"
        )


def test_speech_visible_is_unchanged_live_state():
    """回归:SPEECH context 的 visible 仍是 live 局面。"""
    from werewolf_agent.runtime.graph import _new_engine

    gs = _finished_gs_for_reflection()
    ctx = build_agent_context(
        _new_engine(), gs, "p01", TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
    )
    visible = ctx.visible_world_state
    # live 字段仍在
    assert "alive_players" in visible
    assert "phase" in visible
    # 赛后摘要专属字段不在 SPEECH context
    assert "game_phase" not in visible
    assert "my_action_timeline" not in visible


def test_reflection_visible_non_seer_viewer_no_seer_check_leak():
    """Critical: 非 seer viewer 不得拿到他人 seer_check 阵营信息。

    seer_check 事件无 seer_id(rule_engine H-5 故意省略),所以
    _extract_viewer_action_timeline 必须镜像 live 路径的 role 门控:
    仅 viewer.role=='seer' 才收 check。否则村民/狼人会拿到别人
    验出的真实阵营,泄漏被验玩家身份。
    """
    from werewolf_agent.core.models import Death
    from werewolf_agent.runtime.graph import _new_engine

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    # viewer = p01 村民(非 seer);p07 是真预言家(他人在验人)
    players["p01"] = PlayerState(id="p01", role="villager", alive=True)
    players["p07"] = PlayerState(id="p07", role="seer", alive=True)
    players["p03"] = PlayerState(id="p03", role="werewolf", alive=True)
    # p07 验 p03=werewolf —— 这是 p07 的私有信息,p01 不该看到
    events = [
        GameEvent(type="seer_check", payload={
            "target_id": "p03", "alignment": "werewolf", "night_number": 1}),
        GameEvent(type="vote_resolved", payload={
            "day_number": 1,
            "votes": [{"voter": "p01", "target": "p02"}],
            "exiled": None}),
        GameEvent(type="judge_broadcast", payload={"phase": "death_announce"}),
    ]
    gs = GameState(
        game_id="reflection_villager_leak_guard",
        phase="finished",
        day_number=1,
        night_number=1,
        players=players,
        events=events,
        winning_faction="werewolf",
    )
    ctx = build_agent_context(
        _new_engine(), gs, "p01", TaskType.REFLECTION,
        legal_actions=[ActionType.SPEECH],
    )
    timeline = ctx.visible_world_state.get("my_action_timeline", [])

    # viewer 自己的 vote 合法保留
    assert any(item.get("kind") == "vote" for item in timeline)
    # 不得含任何 seer_check 条目(那是 p07 的私有验人)
    seer_items = [item for item in timeline if item.get("kind") == "seer_check"]
    assert seer_items == [], (
        f"非 seer viewer 泄漏他人 seer_check: {seer_items!r}"
    )
    # 不得泄漏 p03 的真实阵营
    blob = str(ctx.visible_world_state).lower()
    # winning_faction=werewolf 合法出现;但 p03 的 alignment 不该经 seer_check 泄漏。
    # 直接断言 timeline 里无任何 alignment/target 字段
    for item in timeline:
        assert "alignment" not in item, f"泄漏 alignment: {item!r}"
        if item.get("kind") != "vote":
            assert "target" not in item, f"泄漏 target: {item!r}"
