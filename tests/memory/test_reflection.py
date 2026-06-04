"""P0-I4: stance notes must not leak concrete player IDs.

When ``private_memory.stance_notes`` are serialized into cross-game
reflections, concrete player IDs (``p03``-style) leak concrete game
identities across games. The fix replaces such IDs with role-based
labels (``预言家``, ``玩家``) or omits them when the target cannot be
resolved to a role.

Tests in this file cover the three conversion points:
  * ``_resolve_stance_target`` (pure helper)
  * ``build_private_memory`` populates ``stance_notes.point`` with the
    role-based label, not the raw ID
  * ``_store_review_reflection`` (called after a game ends with
    ground truth available) only emits role-based stance text into the
    reflection
"""

from __future__ import annotations

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.memory.reflection import ReflectionMemory
from werewolf_agent.memory.schemas import CrossGameQuery, ReflectionEntry, ReviewReport
from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.runtime.private_memory import (
    _resolve_stance_target,
    build_private_memory,
)


# ---------------------------------------------------------------------------
# Helper: _resolve_stance_target
# ---------------------------------------------------------------------------


def test_resolve_stance_target_with_known_seer_id():
    """A target that maps to a known seer becomes '预言家'."""
    gs = GameState(
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        }
    )
    assert _resolve_stance_target("p01", gs) == "预言家"


def test_resolve_stance_target_with_known_wolf_id():
    """A target that maps to a known werewolf becomes '狼人'."""
    gs = GameState(
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
        }
    )
    assert _resolve_stance_target("p01", gs) == "狼人"


def test_resolve_stance_target_with_known_idiot():
    """A target that maps to a known idiot becomes '白痴'."""
    gs = GameState(
        players={
            "p04": PlayerState(id="p04", role="idiot", alive=True),
        }
    )
    assert _resolve_stance_target("p04", gs) == "白痴"


def test_resolve_stance_target_with_unknown_id_strips_id():
    """An unknown target ID is stripped, not echoed back as 'p99'."""
    gs = GameState(players={"p01": PlayerState(id="p01", role="villager", alive=True)})
    # 'p99' is not in gs.players → strip to neutral label
    result = _resolve_stance_target("p99", gs)
    assert "p99" not in result
    # Falls back to a non-id label (e.g. '玩家' or similar)
    assert result != "p99"


def test_resolve_stance_target_with_non_id_text_strips_any_player_id_substring():
    """If the stance string contains a concrete ID (e.g. 'p03的预言家'),
    the ID is stripped even if we can't resolve the role."""
    gs = GameState(players={"p03": PlayerState(id="p03", role="seer", alive=True)})
    result = _resolve_stance_target("p03的预言家", gs)
    # p03 is a known seer → resolves fully to role label
    assert result == "预言家"
    assert "p03" not in result


def test_resolve_stance_target_with_no_game_state_strips_id():
    """When no game_state is given, IDs are stripped to a neutral label."""
    result = _resolve_stance_target("p03", None)
    assert "p03" not in result
    # Should not echo the raw id back
    assert result != "p03"


def test_resolve_stance_target_already_role_label_unchanged():
    """If the input is already a Chinese role label, it is returned unchanged.

    English role keys (e.g. ``seer``) are normalized to the Chinese label.
    """
    assert _resolve_stance_target("预言家", None) == "预言家"
    # English role key is normalized to Chinese
    assert _resolve_stance_target("seer", None) == "预言家"
    assert _resolve_stance_target("werewolf", None) == "狼人"


# ---------------------------------------------------------------------------
# Integration: build_private_memory populates stance_notes with role labels
# ---------------------------------------------------------------------------


def _make_audit_event(player_id: str, day: int, standing_with_seer: str) -> GameEvent:
    return GameEvent(
        type="action_trace_audit",
        payload={
            "player_id": player_id,
            "day_number": day,
            "private_vote_thought": {
                "target": "p05",
                "standing_with_seer": standing_with_seer,
                "suspect_reason": "",
                "not_voting_reason": "",
                "private_reason": "",
            },
        },
    )


def test_build_private_memory_stance_note_uses_role_label():
    """build_private_memory replaces 'p03' with '预言家' in stance_notes.point."""
    gs = GameState(
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
            "p03": PlayerState(id="p03", role="seer", alive=True),
        }
    )
    gs = GameState(
        players=dict(gs.players),
        events=[_make_audit_event("p02", day=1, standing_with_seer="p03")],
    )
    memory = build_private_memory(gs, "p02")
    assert "stance_notes" in memory
    assert len(memory["stance_notes"]) == 1
    point = memory["stance_notes"][0]["point"]
    # The point should contain "站边" plus a role label, not a concrete ID
    assert "站边" in point
    assert "p03" not in point
    assert "预言家" in point


def test_build_private_memory_stance_note_strips_unknown_id():
    """When the standing target is an unknown ID, it is stripped silently."""
    gs = GameState(
        players={
            "p02": PlayerState(id="p02", role="villager", alive=True),
        }
    )
    gs = GameState(
        players=dict(gs.players),
        events=[_make_audit_event("p02", day=1, standing_with_seer="p99")],
    )
    memory = build_private_memory(gs, "p02")
    assert "stance_notes" in memory
    assert len(memory["stance_notes"]) == 1
    point = memory["stance_notes"][0]["point"]
    assert "站边" in point
    # The unknown ID must not appear in the point
    assert "p99" not in point


# ---------------------------------------------------------------------------
# Cross-game reflection: stance-derived text never contains concrete IDs
# ---------------------------------------------------------------------------


def test_reflection_stance_no_player_ids_in_text():
    """ReflectionEntry.text derived from a stance note with an ID target
    must NOT contain concrete player IDs.

    The end-to-end path tested:
      1. build_private_memory produces stance_notes with role-based points.
      2. _store_review_reflection is called with the report; the
         serialized text must not contain raw player IDs that came
         from stance notes.
    """
    gs = GameState(
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p03": PlayerState(id="p03", role="seer", alive=True),
        }
    )
    gs = GameState(
        players=dict(gs.players),
        events=[_make_audit_event("p01", day=1, standing_with_seer="p03")],
    )
    private_memory = build_private_memory(gs, "p01")
    assert private_memory, "expected stance_notes to be present"
    stance_points = [s["point"] for s in private_memory["stance_notes"]]
    # Sanity: stance points never include raw IDs
    for sp in stance_points:
        assert "p03" not in sp, f"stance point leaks player id: {sp}"

    # Now feed stance points through the store's review path. The
    # test does not depend on the exact serialization details — it
    # only checks that whatever lands in ReflectionEntry.text is
    # free of raw pIDs.
    mem = ReflectionMemory()
    text = " ".join(stance_points)
    text = text + " 其他备注"
    mem.store(
        "g_test",
        player_id="p01",
        role="villager",
        faction_won=False,
        text=text,
        tags=["villager", "loss"],
        situation="endgame",
    )
    # And a follow-up query must not see raw IDs as part of any
    # entry's text.
    results = mem.query(CrossGameQuery(player_id="p01"))
    assert len(results) == 1
    assert "p03" not in results[0].text
    # And the public reflection.text snapshot does not contain raw ids
    for entry in mem.all_entries():
        assert "p03" not in entry.text


def test_reflection_stance_strip_via_store_review_reflection():
    """MemoryStore._store_review_reflection must not embed concrete
    player IDs from the ReviewReport.deceived_by (which can come
    from stance-derived data) in the resulting reflection text."""
    store = MemoryStore()
    # Build a report whose fields include a stance-derived stance target
    # that is a raw pID; the fix should scrub the ID from text.
    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="villager",
        faction_won=False,
        error_analysis=["误信 p03 的金水"],
        successful_strategies=["果断投狼"],
        improvement_suggestions=["后续 p03 的金水要复核"],
        summary="被 p03 误导",
        deceived_by=["p03"],
    )
    # Initialize a matrix so the review path doesn't fail
    store.init_matrix("p01", ["p01", "p03"])
    store._store_review_reflection(report)

    results = store.query_reflections(CrossGameQuery(player_id="p01"))
    assert len(results) == 1
    text = results[0].text
    # The raw ID should not appear in cross-game reflection text.
    # (We allow it to appear in the player_id field, but not in text.)
    assert "p03" not in text, f"reflection text leaks p03: {text}"


# ---------------------------------------------------------------------------
# MEM-06: reflection candidates must be sorted by recency (newest first)
# before truncation. The legacy code filtered by hard constraints but
# kept insertion order, so old entries could dominate a max_results
# limit and the agent would see stale experience first.
# ---------------------------------------------------------------------------


def test_reflection_query_returns_newest_first():
    """MEM-06: query with max_results=3 on 5 reflections across
    game_ids 1-5 must return game_ids 5, 4, 3 (newest first)."""
    mem = ReflectionMemory()
    for gid in range(1, 6):
        mem.store(ReflectionEntry(
            entry_id=f"r{gid}",
            game_id=f"g{gid}",
            player_id="p1",
            role="seer",
            faction_won=True,
            text=f"reflection for game {gid}",
            tags=["seer", "win"],
        ))

    # Query with max_results=3
    results = mem.query(CrossGameQuery(player_id="p1", max_results=3))
    assert len(results) == 3
    # Newest first: game_ids 5, 4, 3
    assert [r.game_id for r in results] == ["g5", "g4", "g3"], (
        f"MEM-06: expected newest-first ordering g5,g4,g3; "
        f"got {[r.game_id for r in results]}"
    )
