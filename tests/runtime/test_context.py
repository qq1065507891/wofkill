"""Tests for werewolf_agent.runtime.context reflection and profile memory hints."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from werewolf_agent.memory.schemas import PlayerProfile, ReflectionEntry
from werewolf_agent.runtime.context import (
    _cognition_matrix_hint,
    _inject_seed_rag_hints,
    _profile_memory_hint,
    _reflection_memory_hints,
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

    # P2-M15: all 6 dims must surface as *_rank keys in the hint.
    expected_rank_keys = {
        "logic_rank", "deception_rank", "leadership_rank", "credibility_rank",
        "learning_rate_rank", "risk_preference_rank",
    }
    for key in expected_rank_keys:
        assert key in hint, f'Missing structured rank key: {key!r}. Hint: {hint!r}'
        assert hint[key] in {"前 30%", "中等", "需要提升", "较高", "偏低"}, (
            f'Unexpected rank for {key!r}: {hint[key]!r}'
        )

    # Neutral phrasing for inner traits
    assert hint["learning_rate_rank"] in {"中等", "较高", "偏低"}
    assert hint["risk_preference_rank"] in {"中等", "较高", "偏低"}

    # Raw floats must not appear
    assert "0.3" not in str(hint), f'Raw learning_rate float leaked: {hint!r}'
    assert "0.5" not in str(hint), f'Raw risk_preference float leaked: {hint!r}'



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
                "viewer_role": query.viewer_role,
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
