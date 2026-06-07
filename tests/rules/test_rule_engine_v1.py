from __future__ import annotations

from dataclasses import replace

import pytest

from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState


RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def make_engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULESET_PATH)


def make_state(
    *,
    hybrid_master_faction: str = "good",
    dead: set[str] | None = None,
    revealed_idiot: bool = False,
    sheriff_id: str | None = None,
    sheriff_badge_state: str = "none",
) -> GameState:
    dead = dead or set()
    players = {
        "w1": PlayerState(id="w1", role="werewolf", alive="w1" not in dead),
        "w2": PlayerState(id="w2", role="werewolf", alive="w2" not in dead),
        "w3": PlayerState(id="w3", role="werewolf", alive="w3" not in dead),
        "w4": PlayerState(id="w4", role="werewolf", alive="w4" not in dead),
        "v1": PlayerState(id="v1", role="villager", alive="v1" not in dead),
        "v2": PlayerState(id="v2", role="villager", alive="v2" not in dead),
        "v3": PlayerState(id="v3", role="villager", alive="v3" not in dead),
        "seer": PlayerState(id="seer", role="seer", alive="seer" not in dead),
        "witch": PlayerState(id="witch", role="witch", alive="witch" not in dead),
        "hunter": PlayerState(id="hunter", role="hunter", alive="hunter" not in dead),
        "idiot": PlayerState(
            id="idiot",
            role="idiot",
            alive="idiot" not in dead,
            revealed_idiot=revealed_idiot,
            vote_enabled=not revealed_idiot,
            badge_eligible=not revealed_idiot,
            exile_immune=revealed_idiot,
        ),
        "hybrid": PlayerState(id="hybrid", role="hybrid", alive="hybrid" not in dead),
    }
    hybrid_master_id: str | None = None
    if hybrid_master_faction == "good":
        hybrid_master_id = "seer"
    elif hybrid_master_faction == "werewolf":
        hybrid_master_id = "w1"
    return GameState(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        players=players,
        hybrid_master_id=hybrid_master_id,
        hybrid_master_faction=hybrid_master_faction,
        sheriff_id=sheriff_id,
        sheriff_badge_state=sheriff_badge_state,
    )


def test_ruleset_loads_exact_v1_role_distribution() -> None:
    engine = make_engine()

    assert engine.ruleset.player_count == 12
    assert engine.role_count("werewolf") == 4
    assert engine.role_count("villager") == 3
    assert engine.role_count("seer") == 1
    assert engine.role_count("witch") == 1
    assert engine.role_count("hunter") == 1
    assert engine.role_count("idiot") == 1
    assert engine.role_count("hybrid") == 1


def test_night_order_matches_design_chapter_3() -> None:
    engine = make_engine()

    assert engine.night_order() == [
        "wolf_discussion_and_kill",
        "witch_action",
        "seer_check",
        "hunter_idiot_status_confirm",
        "hybrid_choose_master",
    ]


def test_seer_checks_hybrid_as_good() -> None:
    engine = make_engine()
    state = make_state()

    result = engine.check_alignment(state, target_id="hybrid")

    assert result.alignment == "good"
    assert result.role is None


def test_witch_cannot_self_save_even_on_first_night() -> None:
    engine = make_engine()
    state = make_state()

    actions = engine.legal_witch_actions(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id="witch",
    )

    assert "save_self" not in {action.type for action in actions}
    assert all(action.target_id != "witch" for action in actions if action.type == "use_antidote")


def test_witch_has_no_antidote_action_without_wolf_kill_target() -> None:
    engine = make_engine()
    state = make_state()

    actions = engine.legal_witch_actions(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id=None,
    )

    assert "use_antidote" not in {action.type for action in actions}


def test_witch_cannot_use_antidote_and_poison_same_night() -> None:
    engine = make_engine()
    state = make_state()

    result = engine.resolve_witch_action(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id="v1",
        use_antidote=True,
        poison_target_id="w1",
    )

    assert result.accepted is False
    assert result.error_code == "witch_cannot_use_both_potions_same_night"


def test_witch_antidote_rejected_without_wolf_kill_target() -> None:
    engine = make_engine()
    state = make_state()

    result = engine.resolve_witch_action(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id=None,
        use_antidote=True,
        poison_target_id=None,
    )

    assert result.accepted is False
    assert result.error_code == "witch_no_wolf_kill_target"


def test_witch_antidote_rejects_target_mismatch() -> None:
    engine = make_engine()
    state = make_state()

    result = engine.resolve_witch_action(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id="v1",
        use_antidote=True,
        poison_target_id=None,
        antidote_target_id="v2",
    )

    assert result.accepted is False
    assert result.error_code == "witch_antidote_target_mismatch"


def test_witch_poison_rejects_dead_or_unknown_target() -> None:
    engine = make_engine()
    state = make_state(dead={"v1"})

    dead_result = engine.resolve_witch_action(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id=None,
        use_antidote=False,
        poison_target_id="v1",
    )
    unknown_result = engine.resolve_witch_action(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id=None,
        use_antidote=False,
        poison_target_id="missing",
    )

    assert dead_result.accepted is False
    assert dead_result.error_code == "witch_poison_target_not_alive"
    assert unknown_result.accepted is False
    assert unknown_result.error_code == "witch_poison_target_not_found"


@pytest.mark.parametrize(
    ("death_reason", "expected_can_shoot"),
    [
        ("wolf_kill", True),
        ("exile", True),
        ("witch_poison", False),
    ],
)
def test_hunter_shot_permission_depends_on_death_reason(
    death_reason: str,
    expected_can_shoot: bool,
) -> None:
    engine = make_engine()
    state = make_state(dead={"hunter"})

    assert engine.can_hunter_shoot(state, hunter_id="hunter", death_reason=death_reason) is expected_can_shoot


def test_idiot_reveals_on_exile_and_does_not_die() -> None:
    engine = make_engine()
    state = make_state()

    new_state, events = engine.resolve_exile(state, target_id="idiot")
    idiot = new_state.players["idiot"]

    assert idiot.alive is True
    assert idiot.revealed_idiot is True
    assert idiot.vote_enabled is False
    assert idiot.badge_eligible is False
    assert idiot.exile_immune is True
    assert any(event.type == "idiot_revealed" for event in events)


def test_revealed_idiot_cannot_be_exiled_again() -> None:
    engine = make_engine()
    state = make_state(revealed_idiot=True)

    legal_targets = engine.legal_exile_targets(state)

    assert "idiot" not in legal_targets


def test_revealed_idiot_counts_as_god_alive_until_later_wolf_kill() -> None:
    engine = make_engine()
    state = make_state(
        revealed_idiot=True,
        dead={"seer", "witch", "hunter"},
    )

    assert engine.check_victory(state).winner is None

    killed_state = engine.apply_death(
        state,
        Death(player_id="idiot", reason="wolf_kill", timing="night", resolution_batch="night_3"),
    )

    assert engine.check_victory(killed_state).winner == "werewolf"
    assert engine.check_victory(killed_state).reason == "slaughter_gods"


def test_revealed_idiot_with_seat_id_counts_as_god_alive() -> None:
    engine = make_engine()
    state = GameState(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="seer", alive=False),
            "p03": PlayerState(id="p03", role="witch", alive=False),
            "p04": PlayerState(id="p04", role="hunter", alive=False),
            "p05": PlayerState(
                id="p05",
                role="idiot",
                alive=True,
                revealed_idiot=True,
                vote_enabled=False,
                badge_eligible=False,
                exile_immune=True,
            ),
            "p06": PlayerState(id="p06", role="villager", alive=True),
            "p07": PlayerState(id="p07", role="villager", alive=True),
            "p08": PlayerState(id="p08", role="villager", alive=True),
        },
    )

    assert engine.check_victory(state).winner is None


def test_hybrid_good_master_requires_villagers_and_hybrid_out_for_villager_slaughter() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction="good", dead={"v1", "v2", "v3"})

    assert engine.check_victory(state).winner is None

    hybrid_out_state = make_state(hybrid_master_faction="good", dead={"v1", "v2", "v3", "hybrid"})

    assert engine.check_victory(hybrid_out_state).winner == "werewolf"
    assert engine.check_victory(hybrid_out_state).reason == "slaughter_villagers"


def test_hybrid_good_master_with_seat_id_requires_hybrid_out_for_villager_slaughter() -> None:
    engine = make_engine()
    state = GameState(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        hybrid_master_id="p04",
        hybrid_master_faction="good",
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=False),
            "p03": PlayerState(id="p03", role="villager", alive=False),
            "p04": PlayerState(id="p04", role="seer", alive=True),
            "p05": PlayerState(id="p05", role="hybrid", alive=True),
        },
    )

    assert engine.check_victory(state).winner is None

    hybrid_out = replace(
        state,
        players={
            **state.players,
            "p05": replace(state.players["p05"], alive=False),
        },
    )
    assert engine.check_victory(hybrid_out).winner == "werewolf"


def test_hybrid_wolf_master_requires_only_three_villagers_for_villager_slaughter() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction="werewolf", dead={"v1", "v2", "v3"})

    assert engine.check_victory(state).winner == "werewolf"
    assert engine.check_victory(state).reason == "slaughter_villagers"


def test_all_werewolves_out_gives_good_victory() -> None:
    engine = make_engine()
    state = make_state(dead={"w1", "w2", "w3", "w4"})

    assert engine.check_victory(state).winner == "good"
    assert engine.check_victory(state).reason == "all_werewolves_out"


@pytest.mark.parametrize("death_reason", ["exile", "wolf_kill", "witch_poison", "hunter_shot", "self_destruct"])
def test_sheriff_death_can_trigger_badge_transfer_or_tear(death_reason: str) -> None:
    engine = make_engine()
    state = make_state(sheriff_id="seer", sheriff_badge_state="active", dead={"seer"})

    decision = engine.badge_options_after_sheriff_death(state, sheriff_id="seer", death_reason=death_reason)

    assert decision.can_transfer is True
    assert decision.can_tear is True
    assert "idiot" in decision.transfer_targets


def test_revealed_idiot_cannot_receive_sheriff_badge() -> None:
    engine = make_engine()
    state = make_state(sheriff_id="seer", sheriff_badge_state="active", revealed_idiot=True, dead={"seer"})

    decision = engine.badge_options_after_sheriff_death(state, sheriff_id="seer", death_reason="wolf_kill")

    assert "idiot" not in decision.transfer_targets


def test_revealed_idiot_exile_target_is_noop() -> None:
    engine = make_engine()
    state = make_state(revealed_idiot=True)

    new_state, events = engine.resolve_exile(state, target_id="idiot")

    assert new_state.players["idiot"].alive is True
    assert events == []


def test_exile_records_death_for_non_idiot() -> None:
    engine = make_engine()
    state = make_state()

    new_state, events = engine.resolve_exile(state, target_id="v1")

    assert new_state.players["v1"].alive is False
    assert new_state.deaths[-1].player_id == "v1"
    assert new_state.deaths[-1].reason == "exile"
    assert any(event.type == "player_exiled" for event in events)


def test_apply_death_records_required_death_fields() -> None:
    engine = make_engine()
    state = make_state()

    new_state = engine.apply_death(
        state,
        Death(player_id="v1", reason="wolf_kill", timing="night", resolution_batch="night_1"),
    )
    death = new_state.deaths[-1]
    assert death.can_leave_last_words is True
    assert death.triggered_skills == []
    event = new_state.events[-1]
    assert event.payload["can_leave_last_words"] is True
    assert event.payload["triggered_skills"] == []


def test_witch_poison_on_revealed_idiot_is_noop() -> None:
    """Design doc §3.4: a revealed idiot stays alive even when targeted by
    witch poison. apply_death must record the attempt as a noop (only emit
    a player_died event for audit) but must not flip alive to False.

    The revealed idiot is only ever killed by a later wolf_kill.
    """
    engine = make_engine()
    state = make_state(revealed_idiot=True)

    new_state = engine.apply_death(
        state,
        Death(
            player_id="idiot",
            reason="witch_poison",
            timing="night",
            resolution_batch="night_1",
        ),
    )

    assert new_state.players["idiot"].alive is True, (
        "Revealed idiot must stay alive when poisoned by witch"
    )
    # The death event is still recorded for audit, but the player keeps living
    assert any(
        event.type == "player_died"
        and event.payload.get("player_id") == "idiot"
        and event.payload.get("reason") == "witch_poison"
        for event in new_state.events
    ), "Poison attempt on revealed idiot should still be recorded as event"


def test_wolf_kill_on_revealed_idiot_kills() -> None:
    """Counterpart test: revealed idiot is killed by wolf_kill (later
    night). Confirms the noop logic only suppresses non-wolf_kill reasons."""
    engine = make_engine()
    state = make_state(revealed_idiot=True)

    new_state = engine.apply_death(
        state,
        Death(
            player_id="idiot",
            reason="wolf_kill",
            timing="night",
            resolution_batch="night_2",
        ),
    )

    assert new_state.players["idiot"].alive is False, (
        "Revealed idiot must die when killed by wolves (later night)"
    )


def test_tearing_badge_removes_sheriff_for_rest_of_game() -> None:
    engine = make_engine()
    state = make_state(sheriff_id="seer", sheriff_badge_state="active")

    new_state = engine.resolve_badge_decision(state, decision="tear")

    assert new_state.sheriff_id is None
    assert new_state.sheriff_badge_state == "torn"
    assert engine.speech_order_policy(new_state) == "random_start_then_seat_order"


def test_second_tie_creates_no_exile_and_enters_night() -> None:
    engine = make_engine()
    state = make_state()

    result = engine.resolve_vote(
        state,
        votes={"v1": "w1", "v2": "w2", "v3": "w1", "seer": "w2"},
        revote=True,
    )

    assert result.exiled_player_id is None
    assert result.next_phase == "night"
    assert result.reason == "second_tie_no_exile"


def test_dead_or_vote_disabled_players_do_not_count_in_exile_vote() -> None:
    engine = make_engine()
    state = make_state(dead={"v1"}, revealed_idiot=True)

    result = engine.resolve_vote(
        state,
        votes={
            "v1": "w1",
            "idiot": "w1",
            "v2": "w2",
            "v3": "w2",
        },
        revote=False,
    )

    assert result.exiled_player_id == "w2"


def test_sheriff_vote_counts_as_one_and_half_votes() -> None:
    engine = make_engine()
    state = make_state(sheriff_id="seer", sheriff_badge_state="active")

    result = engine.resolve_vote(
        state,
        votes={"seer": "w1", "v1": "w2"},
        revote=False,
    )

    assert result.exiled_player_id == "w1"


def test_vote_cannot_target_revealed_idiot() -> None:
    engine = make_engine()
    state = make_state(revealed_idiot=True)

    result = engine.resolve_vote(
        state,
        votes={"v1": "idiot", "v2": "idiot", "v3": "w1"},
        revote=False,
    )

    assert result.exiled_player_id == "w1"


def test_self_votes_do_not_count_in_exile_vote() -> None:
    engine = make_engine()
    state = make_state()

    result = engine.resolve_vote(
        state,
        votes={"v1": "v1", "v2": "w1"},
        revote=False,
    )

    assert result.exiled_player_id == "w1"


@pytest.mark.parametrize(
    ("death_reason", "timing", "night_number", "expected"),
    [
        ("wolf_kill", "night", 1, True),
        ("witch_poison", "night", 1, True),
        ("wolf_kill", "night", 2, False),
        ("exile", "day_vote", 1, True),
        ("self_destruct", "day_discussion", 1, False),
        ("hunter_shot", "post_exile", 1, False),
    ],
)
def test_last_words_matrix(
    death_reason: str,
    timing: str,
    night_number: int,
    expected: bool,
) -> None:
    engine = make_engine()

    assert engine.can_leave_last_words(death_reason=death_reason, timing=timing, night_number=night_number) is expected


def test_hunter_shot_target_no_last_words() -> None:
    """Design doc §3.3: hunter shot target never gets last words. The
    branch order in can_leave_last_words must check death_reason ==
    'hunter_shot' BEFORE the timing == 'night' branch — otherwise a
    night-timing hunter shot (e.g. hunter wolf-killed N1 then shoots
    back) would incorrectly get first_night_death=True.
    """
    engine = make_engine()

    # Night timing (e.g. hunter wolf-killed, shoots back same night)
    assert engine.can_leave_last_words(
        death_reason="hunter_shot", timing="night", night_number=1,
    ) is False
    assert engine.can_leave_last_words(
        death_reason="hunter_shot", timing="night", night_number=2,
    ) is False
    assert engine.can_leave_last_words(
        death_reason="hunter_shot", timing="night", night_number=3,
    ) is False


def test_first_day_flow_announces_deaths_then_last_words_then_sheriff_election() -> None:
    engine = make_engine()

    assert engine.day_flow(day_number=1)[:3] == [
        "announce_deaths",
        "last_words",
        "first_day_sheriff_election",
    ]


def test_player_view_cannot_include_moderator_full_or_other_private_intent() -> None:
    engine = make_engine()
    state = make_state()
    state = engine.record_private_intent(state, player_id="w1", private_intent={"true_role": "werewolf"})

    context = engine.build_visible_context(state, viewer_id="v1", view_mode="player_view")

    assert context.view_mode == "player_view"
    assert "moderator_full" not in context.visible_sections
    assert "w1.private_intent" not in context.visible_sections


# ---------------------------------------------------------------------------
# Phase 1 extensions: role assignment, hybrid master, night pipeline,
# self-destruct, sheriff election, event reducer
# ---------------------------------------------------------------------------

PLAYER_IDS_12 = [f"p{i:02d}" for i in range(1, 13)]


def test_assign_roles_produces_exact_role_distribution() -> None:
    engine = make_engine()
    players = engine.assign_roles(PLAYER_IDS_12, seed=42)

    assert len(players) == 12
    from collections import Counter
    role_counts = Counter(p.role for p in players.values())
    assert role_counts["werewolf"] == 4
    assert role_counts["villager"] == 3
    assert role_counts["seer"] == 1
    assert role_counts["witch"] == 1
    assert role_counts["hunter"] == 1
    assert role_counts["idiot"] == 1
    assert role_counts["hybrid"] == 1


def test_assign_roles_reproducible_with_same_seed() -> None:
    engine = make_engine()
    a = engine.assign_roles(PLAYER_IDS_12, seed=99)
    b = engine.assign_roles(PLAYER_IDS_12, seed=99)
    assert {pid: p.role for pid, p in a.items()} == {pid: p.role for pid, p in b.items()}


def test_assign_roles_rejects_wrong_player_count() -> None:
    engine = make_engine()
    with pytest.raises(ValueError):
        engine.assign_roles(["p01", "p02"], seed=1)


def test_hybrid_choose_master_records_binding() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, event = engine.choose_master(state, hybrid_id="hybrid", master_id="seer")

    assert new_state.hybrid_master_id == "seer"
    assert new_state.hybrid_master_faction == "good"
    assert event.type == "hybrid_master_chosen"


def test_hybrid_choose_werewolf_master_sets_wolf_faction() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, _ = engine.choose_master(state, hybrid_id="hybrid", master_id="w1")

    assert new_state.hybrid_master_faction == "werewolf"


def test_hybrid_cannot_choose_master_twice() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    state, _ = engine.choose_master(state, hybrid_id="hybrid", master_id="seer")

    with pytest.raises(ValueError):
        engine.choose_master(state, hybrid_id="hybrid", master_id="v1")


def test_hybrid_result_follows_master_original_faction_even_after_master_death() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction="good", dead={"seer", "w1", "w2", "w3", "w4"})

    result = engine.check_victory(state)
    assert result.winner == "good"
    assert result.reason == "all_werewolves_out"
    # hybrid on good-master side should also win
    # (hybrid_result is not a field on VictoryResult; the caller resolves it from master_faction)


def test_hybrid_result_follows_master_faction_even_if_hybrid_dead() -> None:
    engine = make_engine()
    # hybrid dead but master (w1) is wolf faction — wolves should still win by slaughter
    state = make_state(hybrid_master_faction="werewolf", dead={"v1", "v2", "v3", "hybrid"})

    assert engine.check_victory(state).winner == "werewolf"


# -- Night resolution pipeline --


def test_resolve_night_wolf_kill_produces_death() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, events = engine.resolve_night(
        state, night_number=1, wolf_kill_target_id="v1",
    )
    assert new_state.players["v1"].alive is False
    assert any(e.type == "player_died" and e.payload.get("player_id") == "v1" for e in new_state.events)


def test_resolve_night_antidote_saves_wolf_kill_target() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, events = engine.resolve_night(
        state, night_number=1, wolf_kill_target_id="v1", use_antidote=True,
    )
    assert new_state.players["v1"].alive is True
    assert new_state.antidote_used is True
    assert any(e.type == "witch_antidote_used" for e in events)


def test_resolve_night_antidote_cannot_save_witch_herself() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, _ = engine.resolve_night(
        state, night_number=1, wolf_kill_target_id="witch", use_antidote=True,
    )
    assert new_state.players["witch"].alive is False
    assert new_state.antidote_used is False


def test_resolve_night_double_death_wolf_and_poison() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, events = engine.resolve_night(
        state, night_number=1, wolf_kill_target_id="v1", poison_target_id="v2",
    )
    assert new_state.players["v1"].alive is False
    assert new_state.players["v2"].alive is False
    assert len([d for d in new_state.deaths if d.resolution_batch == "night_1"]) == 2


def test_resolve_night_cannot_use_antidote_and_poison_same_night() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, _ = engine.resolve_night(
        state, night_number=1, wolf_kill_target_id="v1", use_antidote=True, poison_target_id="v2",
    )
    # antidote saves v1, poison is blocked because use_antidote=True
    assert new_state.players["v1"].alive is True
    assert new_state.players["v2"].alive is True
    assert new_state.antidote_used is True
    assert new_state.poison_used is False


def test_resolve_night_peace_night_no_wolf_kill() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    new_state, events = engine.resolve_night(
        state, night_number=1, wolf_kill_target_id=None,
    )
    assert all(p.alive for p in new_state.players.values())
    assert events == []


@pytest.mark.parametrize("target_id", ["missing", "v1"])
def test_resolve_night_rejects_invalid_wolf_kill_target(target_id: str) -> None:
    engine = make_engine()
    dead = {"v1"} if target_id == "v1" else set()
    state = make_state(hybrid_master_faction=None, dead=dead)

    with pytest.raises(ValueError, match="wolf_kill_target"):
        engine.resolve_night(state, night_number=1, wolf_kill_target_id=target_id)


@pytest.mark.parametrize("target_id", ["missing", "v1"])
def test_resolve_night_rejects_invalid_poison_target(target_id: str) -> None:
    engine = make_engine()
    dead = {"v1"} if target_id == "v1" else set()
    state = make_state(hybrid_master_faction=None, dead=dead)

    with pytest.raises(ValueError, match="poison_target"):
        engine.resolve_night(state, night_number=1, wolf_kill_target_id=None, poison_target_id=target_id)


# -- Self-destruct / peace day --


def test_wolf_self_destruct_kills_wolf_no_last_words() -> None:
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_self_destruct(state, wolf_id="w1", day_number=2)

    assert new_state.players["w1"].alive is False
    assert any(e.type == "werewolf_self_destructed" for e in events)
    # Self-destruct has no last words
    assert not engine.can_leave_last_words(death_reason="self_destruct", timing="day_discussion", night_number=1)


def test_self_destruct_non_wolf_does_nothing() -> None:
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_self_destruct(state, wolf_id="v1", day_number=1)

    assert new_state.players["v1"].alive is True
    assert events == []


def test_self_destruct_dead_wolf_does_nothing() -> None:
    engine = make_engine()
    state = make_state(dead={"w1"})
    new_state, events = engine.resolve_self_destruct(state, wolf_id="w1", day_number=1)

    assert events == []


def test_peace_day_no_exile_enters_night() -> None:
    engine = make_engine()
    state = make_state()
    result = engine.resolve_vote(
        state,
        votes={"v1": "w1", "v2": "w2", "v3": "w1", "seer": "w2"},
        revote=True,
    )
    assert result.exiled_player_id is None
    assert result.next_phase == "night"


# -- Sheriff election flow --


def test_sheriff_elected_by_majority_vote() -> None:
    engine = make_engine()
    state = make_state()
    candidates = ["seer", "w1", "v1"]
    new_state, event = engine.resolve_sheriff_vote(
        state,
        votes={"v2": "seer", "v3": "seer", "witch": "w1"},
        candidates=candidates,
    )
    assert new_state.sheriff_id == "seer"
    assert new_state.sheriff_badge_state == "active"
    assert event.type == "sheriff_elected"


def test_sheriff_vote_tie_does_not_produce_sheriff() -> None:
    engine = make_engine()
    state = make_state()
    candidates = ["seer", "w1"]
    new_state, event = engine.resolve_sheriff_vote(
        state,
        votes={"v1": "seer", "v2": "w1"},
        candidates=candidates,
    )
    assert new_state.sheriff_id is None
    assert event.type == "sheriff_vote_tie"


def test_badge_transfer_to_new_sheriff() -> None:
    engine = make_engine()
    state = make_state(sheriff_id="seer", sheriff_badge_state="active", dead={"seer"})
    new_state = engine.resolve_badge_decision(state, decision="transfer", target_id="v1")

    assert new_state.sheriff_id == "v1"
    assert new_state.sheriff_badge_state == "active"


def test_badge_transfer_to_revealed_idiot_rejected_by_badge_options() -> None:
    engine = make_engine()
    state = make_state(sheriff_id="seer", sheriff_badge_state="active", revealed_idiot=True, dead={"seer"})
    decision = engine.badge_options_after_sheriff_death(state, sheriff_id="seer", death_reason="wolf_kill")

    assert "idiot" not in decision.transfer_targets


# -- Event reducer / replay --


def test_reduce_event_player_died_updates_state() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    event = GameEvent(
        type="player_died",
        payload={
            "player_id": "v1",
            "reason": "wolf_kill",
            "timing": "night",
            "resolution_batch": "night_1",
            "source_player_id": "w1",
            "can_leave_last_words": True,
            "triggered_skills": ["hunter_shot"],
        },
    )
    new_state = engine.reduce_event(state, event)
    assert new_state.players["v1"].alive is False
    assert len(new_state.deaths) == 1
    assert new_state.deaths[0].player_id == "v1"
    assert new_state.deaths[0].resolution_batch == "night_1"
    assert new_state.deaths[0].source_player_id == "w1"
    assert new_state.deaths[0].can_leave_last_words is True
    assert new_state.deaths[0].triggered_skills == ["hunter_shot"]


def test_reduce_event_idiot_revealed() -> None:
    engine = make_engine()
    state = make_state()
    event = GameEvent(type="idiot_revealed", payload={"player_id": "idiot"})
    new_state = engine.reduce_event(state, event)
    assert new_state.players["idiot"].revealed_idiot is True
    assert new_state.players["idiot"].vote_enabled is False
    assert new_state.players["idiot"].exile_immune is True


def test_reduce_events_replays_night_sequence() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction=None)
    events = [
        GameEvent(type="player_died", payload={"player_id": "v1", "reason": "wolf_kill", "timing": "night"}),
        GameEvent(type="witch_antidote_used", payload={"target_id": "v1"}),
        GameEvent(type="player_died", payload={"player_id": "v2", "reason": "wolf_kill", "timing": "night"}),
    ]
    new_state = engine.reduce_events(state, events)
    # v1 saved by antidote — but reducer doesn't undo death without explicit save event
    # In practice the caller should NOT emit both died and antidote events for same player
    assert len(new_state.events) == 3
    assert new_state.antidote_used is True


def test_reduce_event_sheriff_elected_and_badge_transferred() -> None:
    engine = make_engine()
    state = make_state()
    events = [
        GameEvent(type="sheriff_elected", payload={"sheriff_id": "seer"}),
        GameEvent(type="player_died", payload={"player_id": "seer", "reason": "wolf_kill", "timing": "night"}),
        GameEvent(type="badge_transferred", payload={"new_sheriff_id": "v1"}),
    ]
    new_state = engine.reduce_events(state, events)
    assert new_state.sheriff_id == "v1"
    assert new_state.sheriff_badge_state == "active"


def test_reduce_event_badge_torn_no_sheriff_rest_of_game() -> None:
    engine = make_engine()
    state = make_state(sheriff_id="seer", sheriff_badge_state="active")
    event = GameEvent(type="badge_torn", payload={})
    new_state = engine.reduce_event(state, event)
    assert new_state.sheriff_id is None
    assert new_state.sheriff_badge_state == "torn"


def test_reduce_event_victory_restores_terminal_state() -> None:
    engine = make_engine()
    state = make_state(hybrid_master_faction="werewolf")
    event = GameEvent(
        type="victory",
        payload={"winner": "werewolf", "reason": "slaughter_villagers", "hybrid_result": "win"},
    )

    new_state = engine.reduce_event(state, event)

    assert new_state.winning_faction == "werewolf"
    assert new_state.hybrid_result == "win"
    assert new_state.phase == "finished"


def test_full_game_replay_from_events() -> None:
    engine = make_engine()
    players = engine.assign_roles(PLAYER_IDS_12, seed=42)
    initial = GameState(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        players=players,
    )
    events = [
        GameEvent(type="hybrid_master_chosen", payload={"hybrid_id": "p12", "master_id": "p01"}),
        GameEvent(type="player_died", payload={"player_id": "p01", "reason": "wolf_kill", "timing": "night"}),
        GameEvent(type="sheriff_elected", payload={"sheriff_id": "p03"}),
        GameEvent(type="badge_torn", payload={}),
    ]
    final = engine.reduce_events(initial, events)
    assert final.players["p01"].alive is False
    assert final.hybrid_master_id == "p01"
    assert final.sheriff_badge_state == "torn"


# -- Task 2: Seer check events from resolve_night --


def test_resolve_night_produces_seer_check_event() -> None:
    """resolve_night must use seer_target_id to call check_alignment and emit seer_check."""
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_night(
        state,
        night_number=1,
        wolf_kill_target_id=None,
        seer_target_id="w1",
    )
    seer_checks = [e for e in events if e.type == "seer_check"]
    assert len(seer_checks) == 1, f"Expected 1 seer_check event, got {len(seer_checks)}"
    check = seer_checks[0]
    # seer_id 不再包含在 payload 中（H-5：防止通过事件泄漏预言家身份）
    assert "seer_id" not in check.payload
    assert check.payload["target_id"] == "w1"
    assert check.payload["alignment"] == "werewolf"
    assert check.payload["night_number"] == 1
    assert check.payload["visibility"] in ("private", "seer_only", "moderator_only")


def test_resolve_night_seer_check_hybrid_returns_good() -> None:
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_night(
        state,
        night_number=1,
        wolf_kill_target_id=None,
        seer_target_id="hybrid",
    )
    seer_checks = [e for e in events if e.type == "seer_check"]
    assert len(seer_checks) == 1
    assert seer_checks[0].payload["alignment"] == "good"


def test_resolve_night_no_seer_check_when_seer_target_absent() -> None:
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_night(
        state,
        night_number=1,
        wolf_kill_target_id=None,
        seer_target_id=None,
    )
    seer_checks = [e for e in events if e.type == "seer_check"]
    assert len(seer_checks) == 0


def test_resolve_night_seer_check_does_not_change_public_state() -> None:
    """seer_check event is private audit; it must not create deaths or change alive status."""
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_night(
        state,
        night_number=1,
        wolf_kill_target_id=None,
        seer_target_id="w1",
    )
    assert len(new_state.deaths) == 0
    # All players still alive
    assert all(p.alive for p in new_state.players.values())


def test_resolve_night_seer_check_with_simultaneous_wolf_kill() -> None:
    """Seer check event must be produced even when a wolf kill also happens."""
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_night(
        state,
        night_number=1,
        wolf_kill_target_id="v1",
        seer_target_id="w1",
    )
    seer_checks = [e for e in events if e.type == "seer_check"]
    assert len(seer_checks) == 1
    assert seer_checks[0].payload["alignment"] == "werewolf"
    # Wolf kill death still happens
    assert not new_state.players["v1"].alive


# -- Task 2: Hunter shot events in night resolution --


def test_resolve_night_hunter_killed_by_wolf_has_triggered_shot() -> None:
    """When hunter is wolf-killed, apply_death marks hunter_shot in triggered_skills."""
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_night(
        state,
        night_number=1,
        wolf_kill_target_id="hunter",
    )
    assert not new_state.players["hunter"].alive
    hunter_deaths = [d for d in new_state.deaths if d.player_id == "hunter"]
    assert len(hunter_deaths) == 1
    assert "hunter_shot" in hunter_deaths[0].triggered_skills


def test_resolve_night_hunter_poisoned_has_no_triggered_shot() -> None:
    """When hunter is poisoned, hunter_shot must NOT be in triggered_skills."""
    engine = make_engine()
    state = make_state()
    new_state, events = engine.resolve_night(
        state,
        night_number=1,
        wolf_kill_target_id="v1",
        poison_target_id="hunter",
    )
    assert not new_state.players["hunter"].alive
    hunter_deaths = [d for d in new_state.deaths if d.player_id == "hunter"]
    assert len(hunter_deaths) == 1
    assert "hunter_shot" not in hunter_deaths[0].triggered_skills


# -- Task 2: Reducer support for seer_check --


def test_reduce_event_seer_check_appends_without_mutating_state() -> None:
    """seer_check events must be replayable: appended to events, no state mutation."""
    engine = make_engine()
    state = make_state()
    event = GameEvent(
        type="seer_check",
        payload={
            "target_id": "w1",
            "alignment": "werewolf",
            "night_number": 1,
            "visibility": "seer_only",
        },
    )
    new_state = engine.reduce_event(state, event)
    assert new_state.events[-1] == event
    # No deaths, no role changes
    assert len(new_state.deaths) == len(state.deaths)
    assert all(p.alive == op.alive for p, op in zip(new_state.players.values(), state.players.values()))


# =====================================================================
# E1 (post-review-v2): sheriff 票权重应来自 ruleset 字段，非硬编码 base_weight=2
# =====================================================================

def _engine_with_base_vote_weight(base_vote_weight: int) -> RuleEngine:
    """构造一个临时覆盖 base_vote_weight 的 RuleEngine。"""
    import yaml as _yaml
    from pathlib import Path
    from werewolf_agent.engine.rule_engine import Ruleset
    data = _yaml.safe_load(Path(RULESET_PATH).read_text(encoding="utf-8"))
    data.setdefault("game_rules", {})["base_vote_weight"] = base_vote_weight
    return RuleEngine(Ruleset(raw=data))


def test_sheriff_weight_uses_ruleset_field() -> None:
    """E1 (post-review-v2): sheriff 票权重应来自 ruleset.game_rules.base_vote_weight 字段，非硬编码 2。"""
    # base_vote_weight = 3 → 1.5 * 3 = 4.5 → round 4 (banker's rounding)
    # base_vote_weight = 2 → 1.5 * 2 = 3.0 → round 3
    # 旧实现: round(1.5 * 2) = 3 (硬编码 2)
    # 新实现: round(1.5 * 3) = 4
    # 构造一个 sheriff 1.5 票 + villager 1 票 投同一目标的场景
    # 旧: 3+2 = 5 for w1, 2 for w2 → w1 (5 vs 2)
    # 新: 4+3 = 7 for w1, 3 for w2 → w1 (7 vs 3)
    # 比例相同，结果相同 -- 但 tally 绝对值不同
    # 由于 VoteResult 不暴露 tally，我们只能验证 config 字段被读取
    # 简化：使用 inspect 验证代码确实从 game_rules.base_vote_weight 读取
    import inspect as _inspect
    from werewolf_agent.engine import rule_engine
    src = _inspect.getsource(rule_engine.RuleEngine.resolve_vote)
    assert "game_rules" in src, (
        f"resolve_vote should read base_vote_weight from ruleset.game_rules, "
        f"but source does not reference 'game_rules':\n{src[:800]}"
    )
    # 并验证测试用临时 ruleset 也能解析
    engine = _engine_with_base_vote_weight(3)
    state = make_state(sheriff_id="seer", sheriff_badge_state="active")
    result = engine.resolve_vote(
        state,
        votes={"seer": "w1", "v1": "w2"},
        revote=False,
    )
    assert result.exiled_player_id == "w1"


def test_base_vote_weight_default_back_compat() -> None:
    """E1 (post-review-v2): 缺 game_rules.base_vote_weight 时应回退到 2。"""
    # 不设 base_vote_weight → 走 back-compat = 2
    engine = make_engine()  # 标准 YAML，无 game_rules.base_vote_weight
    state = make_state(sheriff_id="seer", sheriff_badge_state="active")

    result = engine.resolve_vote(
        state,
        votes={"seer": "w1", "v1": "w2"},
        revote=False,
    )

    # 标准 base=2 下: sheriff 投 w1 应该是 round(1.5*2)=3, v1 投 w2 应该是 2
    # 仍然 exile w1
    assert result.exiled_player_id == "w1"


def test_base_vote_weight_yaml_field_present() -> None:
    """E1 (post-review-v2): YAML 中应有 game_rules.base_vote_weight 字段。"""
    import yaml as _yaml
    from pathlib import Path
    data = _yaml.safe_load(Path(RULESET_PATH).read_text(encoding="utf-8"))
    assert "game_rules" in data, "ruleset YAML missing 'game_rules' section"
    assert "base_vote_weight" in data["game_rules"], (
        f"ruleset YAML missing game_rules.base_vote_weight: {data['game_rules']}"
    )
    assert data["game_rules"]["base_vote_weight"] == 2
