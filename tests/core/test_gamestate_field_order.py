"""Tests for the NEW-P2-8 GameState field order fix.

The fields hybrid_result, paused, sheriff_interrupt_count,
sheriff_tie_count, sheriff_pk_candidates were declared *after*
__post_init__ in the dataclass. Python's dataclass machinery only sees
fields that appear *before* the first method, so those five were
silently ignored by __init__ — kwarg construction would have failed.
"""

from __future__ import annotations

from dataclasses import fields

from werewolf_agent.core.models import GameState


def test_post_post_init_fields_are_real_fields():
    """NEW-P2-8: the fields declared after __post_init__ (paused,
    hybrid_result, sheriff_interrupt_count, sheriff_tie_count,
    sheriff_pk_candidates) must be in __dataclass_fields__ and
    usable as kwargs to GameState(...).
    """
    field_names = {f.name for f in fields(GameState)}
    for name in (
        "paused",
        "hybrid_result",
        "sheriff_interrupt_count",
        "sheriff_tie_count",
        "sheriff_pk_candidates",
    ):
        assert name in field_names, (
            f"NEW-P2-8 not fixed: field {name!r} missing from "
            f"__dataclass_fields__ because it was declared after "
            f"__post_init__"
        )


def test_construct_gamestate_with_post_init_kwargs():
    """All five late-declared fields must be assignable via kwargs."""
    state = GameState(
        game_id="g_test",
        paused=True,
        hybrid_result="bound_to_wolves",
        sheriff_interrupt_count=2,
        sheriff_tie_count=1,
        sheriff_pk_candidates=["p01", "p02"],
    )
    assert state.paused is True
    assert state.hybrid_result == "bound_to_wolves"
    assert state.sheriff_interrupt_count == 2
    assert state.sheriff_tie_count == 1
    assert state.sheriff_pk_candidates == ["p01", "p02"]


# =====================================================================
# E3 (post-review-v2): PlayerState 应含 faction 字段，默认从 role 推导
# =====================================================================

class TestPlayerStateFactionField:
    """E3 (post-review-v2): PlayerState 应含 faction 字段，默认从 role 推导。"""

    def test_werewolf_faction(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="werewolf", alive=True)
        assert p.faction == "werewolf"

    def test_villager_faction(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="villager", alive=True)
        assert p.faction == "good"

    def test_seer_faction(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="seer", alive=True)
        assert p.faction == "good"

    def test_witch_faction(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="witch", alive=True)
        assert p.faction == "good"

    def test_hunter_faction(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="hunter", alive=True)
        assert p.faction == "good"

    def test_idiot_faction(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="idiot", alive=True)
        assert p.faction == "good"

    def test_hybrid_faction_none(self):
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="hybrid", alive=True)
        # hybrid 在未选 master 时 faction 是 None
        assert p.faction is None

    def test_faction_explicit_override(self):
        """E3: 显式传入 faction 应被尊重。"""
        from werewolf_agent.core.models import PlayerState
        p = PlayerState(id="p01", role="hybrid", alive=True, faction="werewolf")
        assert p.faction == "werewolf"

    def test_faction_field_in_dataclass(self):
        """E3: faction 字段应在 __dataclass_fields__ 中。"""
        from werewolf_agent.core.models import PlayerState
        from dataclasses import fields
        field_names = {f.name for f in fields(PlayerState)}
        assert "faction" in field_names, (
            f"PlayerState missing faction field. Fields: {field_names}"
        )
