"""Tests for the cognitive coprocessor pipeline.

Covers:
- Structured world state extraction
- Visibility policy hard boundaries
- Belief updater deterministic updates
- Contradiction engine detection
- Visibility leak detection
"""

import pytest

from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.cognition.belief import BeliefState, BeliefUpdater
from werewolf_agent.cognition.contradiction import ContradictionEngine
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import (
    StructuredFact,
    StructuredWorldState,
    build_world_state,
    extract_facts,
)
from werewolf_agent.core.models import (
    Death,
    GameEvent,
    GameState,
    PlayerState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    events: list[GameEvent] | None = None,
    day: int = 1,
    night: int = 1,
    antidote_used: bool = False,
    poison_used: bool = False,
) -> GameState:
    players = {
        "p01": PlayerState(id="p01", role="werewolf"),
        "p02": PlayerState(id="p02", role="werewolf"),
        "p03": PlayerState(id="p03", role="werewolf"),
        "p04": PlayerState(id="p04", role="werewolf"),
        "p05": PlayerState(id="p05", role="villager"),
        "p06": PlayerState(id="p06", role="villager"),
        "p07": PlayerState(id="p07", role="villager"),
        "p08": PlayerState(id="p08", role="seer"),
        "p09": PlayerState(id="p09", role="witch"),
        "p10": PlayerState(id="p10", role="hunter"),
        "p11": PlayerState(id="p11", role="idiot"),
        "p12": PlayerState(id="p12", role="hybrid"),
    }
    return GameState(
        players=players,
        day_number=day,
        night_number=night,
        events=events or [],
        antidote_used=antidote_used,
        poison_used=poison_used,
    )


def _make_state_with_death(
    dead_id: str = "p05",
    reason: str = "wolf_kill",
) -> GameState:
    state = _make_state()
    dead_player = PlayerState(id=dead_id, role="villager", alive=False)
    players = {**state.players, dead_id: dead_player}
    death = Death(
        player_id=dead_id, reason=reason,
        timing="night", resolution_batch="night_1",
    )
    event = GameEvent(
        type="player_died",
        payload={"player_id": dead_id, "reason": reason, "timing": "night"},
    )
    return GameState(
        players=players,
        day_number=1, night_number=1,
        deaths=[death], events=[event],
    )


# ===================================================================
# TestStructuredWorldState
# ===================================================================

class TestStructuredWorldState:

    def test_extract_player_died(self):
        state = _make_state()
        event = GameEvent(
            type="player_died",
            payload={"player_id": "p05", "reason": "wolf_kill", "timing": "night"},
        )
        facts = extract_facts(event, state)
        assert len(facts) == 1
        assert facts[0].fact_type == "player_died"
        assert facts[0].target_player == "p05"
        assert facts[0].value == "wolf_kill"

    def test_extract_idiot_revealed(self):
        state = _make_state()
        event = GameEvent(type="idiot_revealed", payload={"player_id": "p11"})
        facts = extract_facts(event, state)
        assert facts[0].fact_type == "idiot_revealed"
        assert facts[0].target_player == "p11"

    def test_extract_self_destruct(self):
        state = _make_state()
        event = GameEvent(
            type="werewolf_self_destructed",
            payload={"player_id": "p01", "day_number": 2},
        )
        facts = extract_facts(event, state)
        assert facts[0].fact_type == "self_destruct"
        assert facts[0].source_player == "p01"
        assert facts[0].day == 2

    def test_extract_hybrid_master_chosen(self):
        state = _make_state()
        event = GameEvent(
            type="hybrid_master_chosen",
            payload={"hybrid_id": "p12", "master_id": "p05"},
        )
        facts = extract_facts(event, state)
        assert facts[0].fact_type == "hybrid_master_chosen"
        assert facts[0].source_player == "p12"
        assert facts[0].target_player == "p05"

    def test_extract_sheriff_elected(self):
        state = _make_state()
        event = GameEvent(type="sheriff_elected", payload={"sheriff_id": "p08"})
        facts = extract_facts(event, state)
        assert facts[0].fact_type == "sheriff_elected"
        assert facts[0].target_player == "p08"

    def test_extract_sheriff_registered(self):
        state = _make_state()
        event = GameEvent(
            type="sheriff_registered",
            payload={"candidates": ["p08", "p01", "p05"]},
        )
        facts = extract_facts(event, state)
        assert len(facts) == 3
        assert all(f.fact_type == "sheriff_registered" for f in facts)
        assert {f.source_player for f in facts} == {"p08", "p01", "p05"}

    def test_extract_speech_with_claims(self):
        state = _make_state()
        event = GameEvent(
            type="speech",
            payload={
                "speaker": "p08",
                "text": "我是预言家，查杀p01",
                "phase": "speech",
                "day_number": 1,
                "claims": [
                    {"type": "role", "target": "p08", "value": "seer"},
                    {"type": "suspect", "target": "p01", "value": "wolf"},
                ],
            },
        )
        facts = extract_facts(event, state)
        assert len(facts) == 3  # speech + 2 claims
        assert facts[0].fact_type == "speech"
        assert facts[1].fact_type == "claimed_role"
        assert facts[2].fact_type == "claimed_suspect"

    def test_extract_vote(self):
        state = _make_state()
        event = GameEvent(
            type="vote",
            payload={"voter": "p05", "target": "p01", "day_number": 1},
        )
        facts = extract_facts(event, state)
        assert facts[0].fact_type == "vote"
        assert facts[0].source_player == "p05"
        assert facts[0].target_player == "p01"

    def test_extract_witch_potions(self):
        state = _make_state()
        e1 = GameEvent(type="witch_antidote_used", payload={"target_id": "p05"})
        e2 = GameEvent(type="witch_poison_used", payload={"target_id": "p06"})
        f1 = extract_facts(e1, state)
        f2 = extract_facts(e2, state)
        assert f1[0].fact_type == "witch_antidote_used"
        assert f2[0].fact_type == "witch_poison_used"

    def test_extract_wolf_kill_selected_separates_wolf_and_witch_visibility(self):
        state = _make_state()
        event = GameEvent(
            type="wolf_kill_selected",
            payload={"night_number": 1, "target_id": "p05"},
        )

        facts = extract_facts(event, state)

        assert {fact.fact_type for fact in facts} == {"wolf_kill_selected", "witch_kill_target"}
        assert all(fact.target_player == "p05" for fact in facts)

    @pytest.mark.parametrize("event_type", ["wolf_no_kill_declared", "wolf_no_kill_timeout"])
    def test_extract_wolf_no_kill_events_as_wolf_team_private(self, event_type):
        state = _make_state()
        event = GameEvent(type=event_type, payload={"night_number": 1, "reason": "pressure"})

        facts = extract_facts(event, state)

        assert len(facts) == 1
        assert facts[0].fact_type == event_type
        assert facts[0].night == 1

    def test_build_world_state(self):
        events = [
            GameEvent(type="player_died", payload={"player_id": "p05", "reason": "wolf_kill", "timing": "night"}),
            GameEvent(type="idiot_revealed", payload={"player_id": "p11"}),
            GameEvent(type="sheriff_elected", payload={"sheriff_id": "p08"}),
        ]
        state = _make_state(events=events)
        ws = build_world_state(state)
        assert len(ws.facts) == 3
        assert ws.facts[0].fact_type == "player_died"
        assert ws.facts[1].fact_type == "idiot_revealed"
        assert ws.facts[2].fact_type == "sheriff_elected"

    def test_facts_of_type(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="speech", source_player="p05", value="test"))
        ws.append(StructuredFact(fact_type="vote", source_player="p05", target_player="p01"))
        ws.append(StructuredFact(fact_type="speech", source_player="p06", value="test2"))
        assert len(ws.facts_of_type("speech")) == 2
        assert len(ws.facts_of_type("vote")) == 1

    def test_facts_about(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="vote", source_player="p05", target_player="p01"))
        ws.append(StructuredFact(fact_type="speech", source_player="p06", target_player="p05"))
        about_p05 = ws.facts_about("p05")
        assert len(about_p05) == 2  # p05 voted, p06 talked about p05

    def test_unknown_event_type(self):
        state = _make_state()
        event = GameEvent(type="custom_event_type", payload={"foo": "bar"})
        facts = extract_facts(event, state)
        assert facts[0].fact_type == "custom_event_type"


# ===================================================================
# TestVisibilityPolicy
# ===================================================================

class TestVisibilityPolicy:

    def _ws_with_all_fact_types(self) -> StructuredWorldState:
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="player_died", target_player="p05", value="wolf_kill"))
        ws.append(StructuredFact(fact_type="speech", source_player="p05", value="test"))
        ws.append(StructuredFact(fact_type="vote", source_player="p06", target_player="p01"))
        ws.append(StructuredFact(fact_type="seer_check", source_player="p08", target_player="p01", value="werewolf"))
        ws.append(StructuredFact(fact_type="witch_antidote_used", target_player="p05", value="antidote_saved"))
        ws.append(StructuredFact(fact_type="witch_poison_used", target_player="p06", value="poison_killed"))
        ws.append(StructuredFact(fact_type="hybrid_master_chosen", source_player="p12", target_player="p05"))
        ws.append(StructuredFact(fact_type="wolf_discussion", value="discussing"))
        return ws

    def test_villager_sees_only_public(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "p05", "villager")
        visible_types = {f.fact_type for f in visible}
        assert "player_died" in visible_types
        assert "speech" in visible_types
        assert "vote" in visible_types
        # Private facts not visible
        assert "seer_check" not in visible_types
        assert "witch_antidote_used" not in visible_types
        assert "hybrid_master_chosen" not in visible_types
        assert "wolf_discussion" not in visible_types

    def test_seer_sees_own_checks(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "p08", "seer")
        visible_types = {f.fact_type for f in visible}
        assert "seer_check" in visible_types
        assert "player_died" in visible_types
        # Cannot see witch or hybrid private
        assert "witch_antidote_used" not in visible_types
        assert "hybrid_master_chosen" not in visible_types

    def test_witch_sees_own_potions(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "p09", "witch")
        visible_types = {f.fact_type for f in visible}
        assert "witch_antidote_used" in visible_types
        assert "witch_poison_used" in visible_types
        # Cannot see seer or wolf
        assert "seer_check" not in visible_types
        assert "wolf_discussion" not in visible_types

    def test_werewolf_sees_wolf_team(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "p01", "werewolf")
        visible_types = {f.fact_type for f in visible}
        assert "wolf_discussion" in visible_types
        assert "player_died" in visible_types
        # Cannot see seer checks or witch potions
        assert "seer_check" not in visible_types
        assert "witch_antidote_used" not in visible_types

    def test_hybrid_sees_own_master(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "p12", "hybrid")
        visible_types = {f.fact_type for f in visible}
        assert "hybrid_master_chosen" in visible_types
        # Cannot see seer or wolf
        assert "seer_check" not in visible_types
        assert "wolf_discussion" not in visible_types

    def test_owner_private_facts_require_matching_source_player(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()

        other_seer_visible = policy.filter_visible_facts(ws, "p10", "seer")
        other_hybrid_visible = policy.filter_visible_facts(ws, "p11", "hybrid")

        assert "seer_check" not in {f.fact_type for f in other_seer_visible}
        assert "hybrid_master_chosen" not in {
            f.fact_type for f in other_hybrid_visible
        }

    def test_visibility_report_has_audit_trail(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()
        report = policy.compute_visibility(ws, "p05", "villager")
        assert len(report.fact_labels) == len(ws.facts)
        for label in report.fact_labels:
            assert label.audit_reason  # Every label has an audit reason

    def test_forbidden_fact_type_never_visible(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="hidden_identity", value="p01 is werewolf"))
        ws.append(StructuredFact(fact_type="other_private_intent", value="secret"))
        ws.append(StructuredFact(fact_type="player_died", value="public"))
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "p05", "villager")
        visible_types = {f.fact_type for f in visible}
        assert "hidden_identity" not in visible_types
        assert "other_private_intent" not in visible_types
        assert "player_died" in visible_types

    def test_unmapped_fact_defaults_moderator_only(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="new_private_event", value="secret payload"))
        policy = VisibilityPolicy()

        visible = policy.filter_visible_facts(ws, "p05", "villager")
        label = policy.compute_fact_visibility(ws.facts[0], 0)

        assert visible == []
        assert label.visibility == "moderator_only"

    def test_wolf_no_kill_events_visible_only_to_wolf_team(self):
        events = [
            GameEvent(type="wolf_no_kill_declared", payload={"night_number": 1, "reason": "pressure"}),
            GameEvent(type="wolf_no_kill_timeout", payload={"night_number": 2}),
        ]
        ws = build_world_state(_make_state(events=events))
        policy = VisibilityPolicy()

        villager_types = {f.fact_type for f in policy.filter_visible_facts(ws, "p05", "villager")}
        wolf_types = {f.fact_type for f in policy.filter_visible_facts(ws, "p01", "werewolf")}

        assert "wolf_no_kill_declared" not in villager_types
        assert "wolf_no_kill_timeout" not in villager_types
        assert "wolf_no_kill_declared" in wolf_types
        assert "wolf_no_kill_timeout" in wolf_types

    def test_wolf_kill_selected_visibility_splits_wolves_and_witch(self):
        ws = build_world_state(_make_state(events=[
            GameEvent(type="wolf_kill_selected", payload={"night_number": 1, "target_id": "p05"}),
        ]))
        policy = VisibilityPolicy()

        villager_types = {f.fact_type for f in policy.filter_visible_facts(ws, "p06", "villager")}
        witch_types = {f.fact_type for f in policy.filter_visible_facts(ws, "p09", "witch")}
        wolf_types = {f.fact_type for f in policy.filter_visible_facts(ws, "p01", "werewolf")}

        assert "wolf_kill_selected" not in villager_types
        assert "witch_kill_target" not in villager_types
        assert "witch_kill_target" in witch_types
        assert "wolf_kill_selected" not in witch_types
        assert "wolf_kill_selected" in wolf_types

    def test_no_leak_check_passes(self):
        ws = self._ws_with_all_fact_types()
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "p05", "villager")
        passed, leaks = policy.check_no_leaks(ws, "p05", "villager", visible)
        assert passed
        assert len(leaks) == 0

    def test_no_leak_check_detects_leak(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="seer_check", source_player="p08", target_player="p01"))
        policy = VisibilityPolicy()
        # Deliberately include a seer_check fact for a villager
        leaked_facts = [ws.facts[0]]
        passed, leaks = policy.check_no_leaks(ws, "p05", "villager", leaked_facts)
        assert not passed
        assert len(leaks) == 1
        assert "seer_check" in leaks[0]

    def test_fact_visibility_map_only_contains_produced_fact_types(self):
        """Regression for COG-3-07: the visibility map must not list fact types
        that are never produced by world_state extractors.

        Pre-fix, "wolf_kill_target" and "idiot_reveal_status" were in the map
        but no extractor ever produced them. If a future producer starts
        emitting those types, they'd be silently routed (and could leak), so
        we enforce the "no stale entries" invariant with this test.
        """
        from werewolf_agent.cognition.visibility import _FACT_VISIBILITY_MAP
        # Drive every extractor with one event each, plus a speech that
        # exercises all claim-fact subtypes and a wolf_kill_selected that
        # produces both wolf_kill_selected + witch_kill_target.
        events = [
            GameEvent(type="player_died", payload={"player_id": "p05", "reason": "wolf_kill"}),
            GameEvent(type="idiot_revealed", payload={"player_id": "p11"}),
            GameEvent(type="player_exiled", payload={"player_id": "p05"}),
            GameEvent(type="werewolf_self_destructed", payload={"player_id": "p01"}),
            GameEvent(type="hybrid_master_chosen", payload={"hybrid_id": "p12", "master_id": "p05"}),
            GameEvent(type="sheriff_elected", payload={"sheriff_id": "p08"}),
            GameEvent(type="sheriff_registered", payload={"candidates": ["p08", "p01"]}),
            GameEvent(type="sheriff_withdraw", payload={"withdrew": ["p01"]}),
            GameEvent(type="sheriff_vote_tie", payload={"day_number": 1}),
            GameEvent(type="sheriff_vote_tie_first", payload={"day_number": 1, "candidates": ["p05"]}),
            GameEvent(type="badge_transferred", payload={"from": "p08", "to": "p05"}),
            GameEvent(type="badge_torn", payload={"sheriff_id": "p08"}),
            GameEvent(type="witch_antidote_used", payload={"target_id": "p05"}),
            GameEvent(type="witch_poison_used", payload={"target_id": "p06"}),
            GameEvent(type="wolf_kill_selected", payload={"target_id": "p05", "night_number": 1}),
            GameEvent(type="wolf_no_kill_declared", payload={"night_number": 1, "reason": "x"}),
            GameEvent(type="wolf_no_kill_timeout", payload={"night_number": 2}),
            GameEvent(type="wolf_discussion", payload={"text": "x"}),
            # speech that triggers every claim subtype
            GameEvent(type="speech", payload={
                "speaker": "p08", "day_number": 1, "phase": "speech",
                "text": (
                    "我是预言家 查杀p01 p02是狼人 p03是金水 "
                    "给p04发金水 警徽流p05 p07 怀疑p06是狼人"
                ),
            }),
            GameEvent(type="sheriff_speech", payload={"speaker": "p08", "text": "x", "day_number": 1}),
            GameEvent(type="vote", payload={"voter": "p05", "target": "p01", "day_number": 1}),
            GameEvent(type="seer_check", payload={"target_id": "p01", "alignment": "werewolf", "night_number": 1}),
            GameEvent(type="sheriff_no_election", payload={}),
        ]
        ws = build_world_state(_make_state(events=events))
        produced = {f.fact_type for f in ws.facts}
        # Map keys must be a subset of produced fact types
        stale = set(_FACT_VISIBILITY_MAP.keys()) - produced
        assert not stale, (
            f"_FACT_VISIBILITY_MAP has stale entries not produced by any "
            f"world_state extractor: {sorted(stale)}. Remove them so the "
            f"fail-closed default (moderator_only) applies."
        )


# ===================================================================
# TestBeliefUpdater
# ===================================================================

class TestBeliefUpdater:

    def test_initialize_creates_uniform_beliefs(self):
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03", "p04"], "p01")
        assert "p01" not in state.beliefs  # Self not included
        assert "p02" in state.beliefs
        for belief in state.beliefs.values():
            probs = belief.role_probabilities
            assert len(probs) == 7
            for p in probs.values():
                assert abs(p - 1.0 / 7) < 0.01

    def test_death_removes_player(self):
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03"], "p01")
        fact = StructuredFact(fact_type="player_died", target_player="p02")
        state = updater.update(state, [fact], 1)
        assert "p02" not in state.beliefs

    def test_self_destruct_confirms_wolf(self):
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03"], "p01")
        fact = StructuredFact(fact_type="self_destruct", source_player="p02", target_player="p02")
        state = updater.update(state, [fact], 1)
        assert "p02" not in state.beliefs  # Removed (confirmed wolf)

    def test_idiot_reveal_updates_belief(self):
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03"], "p01")
        fact = StructuredFact(fact_type="idiot_revealed", target_player="p02")
        state = updater.update(state, [fact], 1)
        assert state.beliefs["p02"].role_probabilities["idiot"] == 1.0
        assert state.beliefs["p02"].faction_lean == "good_lean"

    def test_idiot_reveal_preserves_all_role_keys(self):
        """Idiot reveal must keep all 7 role keys; only the idiot slot becomes 1.0.

        Pre-fix, _apply_idiot_reveal set role_probabilities={"idiot": 1.0},
        dropping the other 6 role keys. Any consumer that iterated over the
        dict (e.g. top_role_guess, distribution-aware prompts) would break
        because the invariant "all roles present" was violated.
        """
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03"], "p01")
        fact = StructuredFact(fact_type="idiot_revealed", target_player="p02")
        state = updater.update(state, [fact], 1)
        prob = state.beliefs["p02"].role_probabilities
        # All role keys must still be present
        expected_roles = {"villager", "seer", "witch", "hunter", "idiot", "werewolf", "hybrid"}
        assert set(prob.keys()) == expected_roles
        # Idiot is 1.0, all others are 0.0
        assert prob["idiot"] == 1.0
        for r in expected_roles - {"idiot"}:
            assert prob[r] == 0.0

    def test_role_claim_shifts_probabilities(self):
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03"], "p01")
        fact = StructuredFact(
            fact_type="claimed_role",
            source_player="p02",
            value="seer",
            day=1,
        )
        state = updater.update(state, [fact], 1)
        assert state.beliefs["p02"].role_probabilities["seer"] > 1.0 / 7

    def test_beliefs_normalize_after_update(self):
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03"], "p01")
        fact = StructuredFact(
            fact_type="claimed_role",
            source_player="p02",
            value="seer",
            day=1,
        )
        state = updater.update(state, [fact], 1)
        total = sum(state.beliefs["p02"].role_probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_top_role_guess(self):
        belief = BeliefUpdater().initialize(["p01", "p02"], "p01").beliefs["p02"]
        role, conf = belief.top_role_guess()
        assert role in ("villager", "seer", "witch", "hunter", "idiot", "werewolf", "hybrid")
        assert conf > 0

    def test_seer_claim_updates_role_probabilities(self):
        """Seer-claim (查杀) must boost the target's werewolf probability.

        Pre-fix, _apply_seer_claim only mutated faction_lean and trust, so
        top_role_guess() still returned near-uniform (the dominant role was
        whichever the random uniform gave a slight edge to). After the fix,
        a 查杀 claim should make werewolf the top-role guess for the target.
        """
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03"], "p01")
        initial_wolf_prob = state.beliefs["p03"].role_probabilities["werewolf"]
        # Simulate a seer-claim fact: p02 (claiming seer) says p03 is wolf
        fact = StructuredFact(
            fact_type="seer_check_claim",
            source_player="p02",
            target_player="p03",
            value="wolf",
            day=1,
        )
        state = updater.update(state, [fact], 1)
        boosted = state.beliefs["p03"].role_probabilities["werewolf"]
        assert boosted > initial_wolf_prob + 0.1  # substantial boost
        # Renormalized
        total = sum(state.beliefs["p03"].role_probabilities.values())
        assert abs(total - 1.0) < 0.01
        # top_role_guess now points to werewolf
        top_role, top_conf = state.beliefs["p03"].top_role_guess()
        assert top_role == "werewolf"

    # --- P0 belief-public-vote-signals: 投票行为应基于公开锚点更新 voter trust ---

    def test_vote_for_publicly_checked_wolf_increases_trust(self):
        """投票给被公开查杀的目标 → voter trust 上升。"""
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03", "p05", "p08"], "p05")
        facts = [
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p01", value="wolf", day=1),
            StructuredFact(fact_type="vote", source_player="p03",
                           target_player="p01", day=1),
        ]
        before = state.beliefs["p03"].trust
        state = updater.update(state, facts, 1)
        assert state.beliefs["p03"].trust > before

    def test_vote_for_gold_water_decreases_trust(self):
        """投票给金水目标 → voter trust 下降。"""
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03", "p05", "p07", "p08"], "p05")
        facts = [
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p07", value="good", day=1),
            StructuredFact(fact_type="vote", source_player="p03",
                           target_player="p07", day=1),
        ]
        before = state.beliefs["p03"].trust
        state = updater.update(state, facts, 1)
        assert state.beliefs["p03"].trust < before

    def test_vote_for_seer_claimant_decreases_trust(self):
        """投票给跳预言家的人 → voter trust 下降（冲预言家偏狼）。"""
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03", "p05", "p08"], "p05")
        facts = [
            StructuredFact(fact_type="claimed_role", source_player="p08",
                           value="seer", day=1),
            StructuredFact(fact_type="vote", source_player="p03",
                           target_player="p08", day=1),
        ]
        before = state.beliefs["p03"].trust
        state = updater.update(state, facts, 1)
        assert state.beliefs["p03"].trust < before

    def test_vote_for_public_suspect_increases_trust(self):
        """投票给被公开怀疑的目标 → voter trust 上升（弱）。"""
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03", "p05", "p06"], "p05")
        facts = [
            StructuredFact(fact_type="claimed_suspect", source_player="p06",
                           target_player="p01", value="wolf", day=1),
            StructuredFact(fact_type="vote", source_player="p03",
                           target_player="p01", day=1),
        ]
        before = state.beliefs["p03"].trust
        state = updater.update(state, facts, 1)
        assert state.beliefs["p03"].trust > before

    def test_vote_without_anchor_is_neutral(self):
        """无任何公开锚点时投票不改变 trust（防过度更新）。"""
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p05"], "p05")
        facts = [
            StructuredFact(fact_type="vote", source_player="p02",
                           target_player="p01", day=1),
        ]
        before = state.beliefs["p02"].trust
        state = updater.update(state, facts, 1)
        assert state.beliefs["p02"].trust == before

    def test_vote_signal_bounded(self):
        """多锚点叠加后 trust 仍落在 [0, 1]。"""
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p03", "p05", "p08"], "p05")
        facts = [
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p01", value="wolf", day=1),
            StructuredFact(fact_type="claimed_suspect", source_player="p08",
                           target_player="p01", value="wolf", day=1),
            StructuredFact(fact_type="vote", source_player="p02",
                           target_player="p01", day=1),
        ]
        state = updater.update(state, facts, 1)
        trust = state.beliefs["p02"].trust
        assert 0.0 <= trust <= 1.0

    # --- P1 belief-counterclaim-dampening: 对跳时削弱 seer 声明威力 ---

    def test_seer_check_wolf_boost_reduced_under_counterclaim(self):
        """对跳（两人跳预言家）时，查杀声明的 werewolf boost 应低于无对跳。"""
        updater = BeliefUpdater()

        def wolf_prob_after(facts):
            st = updater.initialize(["p01", "p02", "p05", "p08"], "p05")
            st = updater.update(st, facts, 1)
            return st.beliefs["p02"].role_probabilities["werewolf"]

        no_counter = [
            StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p02", value="wolf", day=1),
        ]
        counter = [
            StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
            StructuredFact(fact_type="claimed_role", source_player="p01", value="seer", day=1),
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p02", value="wolf", day=1),
        ]
        assert wolf_prob_after(counter) < wolf_prob_after(no_counter)

    def test_later_seer_claimant_gets_lower_seer_prob(self):
        """对跳时后 claim 者（p01）seer boost 低于先 claim 者（p08）。

        credibility 按时序：p08 claim 时是单 claimant（uncontested），
        p01 claim 时已存在对跳（contested，受 multi penalty）。
        """
        updater = BeliefUpdater()
        st = updater.initialize(["p01", "p05", "p08"], "p05")
        st = updater.update(st, [
            StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
            StructuredFact(fact_type="claimed_role", source_player="p01", value="seer", day=1),
        ], 1)
        assert st.beliefs["p01"].role_probabilities["seer"] < st.beliefs["p08"].role_probabilities["seer"]

    def test_counterclaim_gold_water_no_hard_good_lean(self):
        """对跳金水声明不设 good_lean（spec: contested gold 只升 trust，不硬站边）。

        单预言家（supported）金水仍设 good_lean。
        """
        updater = BeliefUpdater()

        def gold_faction(facts):
            st = updater.initialize(["p01", "p02", "p05", "p07", "p08"], "p05")
            st = updater.update(st, facts, 1)
            return st.beliefs["p07"].faction_lean

        no_counter = [
            StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p07", value="good", day=1),
        ]
        counter = [
            StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
            StructuredFact(fact_type="claimed_role", source_player="p01", value="seer", day=1),
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p07", value="good", day=1),
        ]
        assert gold_faction(no_counter) == "good_lean"
        assert gold_faction(counter) != "good_lean"

    def test_no_counterclaim_keeps_full_seer_boost(self):
        """无对跳（单预言家）查杀仍满 boost、top_role=werewolf（回归）。"""
        updater = BeliefUpdater()
        state = updater.initialize(["p01", "p02", "p05", "p08"], "p05")
        facts = [
            StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
            StructuredFact(fact_type="seer_check_claim", source_player="p08",
                           target_player="p02", value="wolf", day=1),
        ]
        state = updater.update(state, facts, 1)
        top_role, _ = state.beliefs["p02"].top_role_guess()
        assert top_role == "werewolf"


# ===================================================================
# TestContradictionEngine
# ===================================================================

class TestContradictionEngine:

    def test_stance_reversal_detected(self):
        facts = [
            StructuredFact(fact_type="claimed_suspect", source_player="p05", target_player="p01", value="good", day=1),
            StructuredFact(fact_type="claimed_suspect", source_player="p05", target_player="p01", value="wolf", day=2),
        ]
        engine = ContradictionEngine()
        alerts = engine.detect(facts, current_day=2)
        reversals = [a for a in alerts if a.alert_type == "stance_reversal"]
        assert len(reversals) == 1
        assert reversals[0].player_id == "p05"

    def test_no_reversal_same_day(self):
        facts = [
            StructuredFact(fact_type="claimed_suspect", source_player="p05", target_player="p01", value="good", day=1),
            StructuredFact(fact_type="claimed_suspect", source_player="p05", target_player="p01", value="wolf", day=1),
        ]
        engine = ContradictionEngine()
        alerts = engine.detect(facts, current_day=1)
        reversals = [a for a in alerts if a.alert_type == "stance_reversal"]
        assert len(reversals) == 0  # Same day = not a reversal

    def test_vote_conflict_detected(self):
        facts = [
            StructuredFact(fact_type="claimed_suspect", source_player="p05", target_player="p01", value="wolf", day=1),
            StructuredFact(fact_type="vote", source_player="p05", target_player="p02", day=1),
        ]
        engine = ContradictionEngine()
        alerts = engine.detect(facts, current_day=1)
        vote_conflicts = [a for a in alerts if a.alert_type == "vote_conflict"]
        assert len(vote_conflicts) == 1

    def test_claim_conflict_two_seers(self):
        facts = [
            StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
            StructuredFact(fact_type="claimed_role", source_player="p01", value="seer", day=1),
        ]
        engine = ContradictionEngine()
        alerts = engine.detect(facts, current_day=1)
        claim_conflicts = [a for a in alerts if a.alert_type == "claim_conflict"]
        assert len(claim_conflicts) == 1
        assert "seer" in claim_conflicts[0].description.lower()

    def test_two_villager_claims_fit_role_capacity(self):
        facts = [
            StructuredFact(fact_type="claimed_role", source_player="p09", value="villager", day=1),
            StructuredFact(fact_type="claimed_role", source_player="p12", value="villager", day=1),
        ]

        alerts = ContradictionEngine().detect(facts, current_day=1)

        assert not [a for a in alerts if a.alert_type == "claim_conflict"]

    def test_villager_claims_over_configured_capacity_conflict(self):
        facts = [
            StructuredFact(fact_type="claimed_role", source_player=f"p0{i}", value="villager", day=1)
            for i in range(1, 4)
        ]

        alerts = ContradictionEngine(
            role_capacities={"villager": 2, "seer": 1},
        ).detect(facts, current_day=1)

        claim_conflicts = [a for a in alerts if a.alert_type == "claim_conflict"]
        assert len(claim_conflicts) == 1
        assert claim_conflicts[0].evidence[0]["capacity"] == 2

    def test_speech_claims_are_extracted_into_structured_facts(self):
        from werewolf_agent.core.models import GameEvent, GameState
        from werewolf_agent.cognition.world_state import extract_facts

        event = GameEvent(
            type="speech",
            payload={
                "speaker": "p08",
                "day_number": 1,
                "text": "我是预言家，昨晚查验p01是狼人，今天我会投p01。",
            },
        )

        facts = extract_facts(event, GameState())

        assert any(f.fact_type == "claimed_role" and f.value == "seer" for f in facts)
        # "查验p01是狼人" 被 seer_check_claim 精确捕获，不再生成重复的 claimed_suspect
        assert any(
            f.fact_type == "seer_check_claim" and f.target_player == "p01" and f.value == "wolf"
            for f in facts
        )

    def test_wolf_perspective_phrase_does_not_create_self_claim(self):
        from werewolf_agent.core.models import GameEvent, GameState
        from werewolf_agent.cognition.world_state import extract_facts

        facts = extract_facts(
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "day_number": 1,
                    "text": "\u6211\u8fd9\u72fc\u961f\u89c6\u89d2\u770bp08\u50cf\u9884\u8a00\u5bb6\u3002",
                },
            ),
            GameState(),
        )

        assert not any(
            f.fact_type == "claimed_role" and f.value == "werewolf"
            for f in facts
        )

    def test_explicit_wolf_self_claim_still_creates_role_claim(self):
        from werewolf_agent.core.models import GameEvent, GameState
        from werewolf_agent.cognition.world_state import extract_facts

        facts = extract_facts(
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p02",
                    "day_number": 1,
                    "text": "\u6211\u662f\u72fc\u4eba\u3002",
                },
            ),
            GameState(),
        )

        assert any(
            f.fact_type == "claimed_role" and f.value == "werewolf"
            for f in facts
        )

    def test_no_false_positives(self):
        facts = [
            StructuredFact(fact_type="vote", source_player="p05", target_player="p01", day=1),
            StructuredFact(fact_type="vote", source_player="p06", target_player="p01", day=1),
            StructuredFact(fact_type="speech", source_player="p05", value="I think p01 is wolf", day=1),
        ]
        engine = ContradictionEngine()
        alerts = engine.detect(facts, current_day=1)
        assert len(alerts) == 0


# ===================================================================
# Task 5: Seer Claim Contract And Counterclaim Memory
# ===================================================================

class TestSeerClaimContractExtraction:
    """Extract structured seer claim contracts from speech."""

    def test_extracts_seer_claim_contract(self):
        """Speech '我是预言家，昨晚验 p01 查杀，警徽流 p05 p07' extracts full contract."""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text
        claims = _infer_claims_from_text(speaker="p03", text="我是预言家，昨晚验p01查杀，警徽流p05 p07", day=1)
        # Should extract: claimed_role=seer, seer_check_claim target=p01 wolf, badge_flow_claim
        claim_types = [c.fact_type for c in claims]
        # Must have a role claim
        assert "claimed_role" in claim_types
        # Seer check result must be captured
        seer_checks = [c for c in claims if c.fact_type == "seer_check_claim"]
        assert len(seer_checks) >= 1
        check = seer_checks[0]
        assert check.target_player == "p01"
        assert check.value in ("wolf", "查杀")

    def test_extracts_badge_flow(self):
        """警徽流 should be extracted as a separate fact."""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text
        claims = _infer_claims_from_text(speaker="p03", text="我是预言家，警徽流p05 p07", day=1)
        badge_facts = [c for c in claims if c.fact_type == "badge_flow_claim"]
        assert len(badge_facts) >= 1
        # The badge flow order should be in metadata
        assert badge_facts[0].metadata.get("badge_flow_order") == ["p05", "p07"]

    def test_gold_claim_extracted(self):
        """金水 (gold claim) should be extracted."""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text
        claims = _infer_claims_from_text(speaker="p03", text="我是预言家，p05是金水", day=1)
        gold_claims = [c for c in claims if c.fact_type == "seer_check_claim" and c.value == "good"]
        assert len(gold_claims) >= 1
        assert gold_claims[0].target_player == "p05"

    def test_third_party_seer_recap_does_not_create_speaker_seer_line(self):
        """Retelling another player's checks must not become the speaker's seer claim."""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        claims = _infer_claims_from_text(
            speaker="p11",
            text="p02报p01金水，p09报p05查杀，我会对比两个预言家。",
            day=2,
        )

        assert not [
            c for c in claims
            if c.fact_type == "seer_check_claim" and c.source_player == "p11"
        ]

    def test_third_party_fake_seer_label_does_not_claim_role_for_speaker(self):
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        claims = _infer_claims_from_text(
            speaker="p11",
            text="p09悍跳预言家，p01金水这条线我不完全信。",
            day=2,
        )

        assert not [
            c for c in claims
            if c.fact_type == "claimed_role" and c.value == "seer"
        ]

    def test_long_third_party_seer_recap_does_not_create_speaker_check(self):
        """长转述前缀中的查验结果不能归因给当前发言者。"""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        claims = _infer_claims_from_text(
            speaker="p06",
            text=(
                "p01陈思远，你跳预言家说验了p02金水，但你警徽流里含了我p06。"
                "我想问，你首夜验人的逻辑是什么？"
            ),
            day=1,
        )

        assert not [
            c for c in claims
            if c.fact_type == "seer_check_claim"
            and c.source_player == "p06"
            and c.target_player == "p02"
        ]
        assert not [
            c for c in claims
            if c.fact_type == "claimed_suspect"
            and c.source_player == "p06"
            and c.target_player == "p02"
        ]
        assert not [
            c for c in claims
            if c.fact_type == "badge_flow_claim" and c.source_player == "p06"
        ]
        assert not [
            c for c in claims
            if c.fact_type == "claimed_good"
            and c.source_player == "p06"
            and c.target_player == "p02"
        ]

    def test_cross_clause_third_party_seer_recap_does_not_create_speaker_check(self):
        """第三方编号与查验结果跨逗号时仍不能归因给当前发言者。"""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        for text in (
            "我是预言家，p01说，昨晚验了p02金水。",
            "我是预言家，p01说，昨晚验了p02查杀。",
            "我是预言家，p02报p01查杀，我认为不可信。",
        ):
            claims = _infer_claims_from_text(speaker="p06", text=text, day=1)

            assert not [
                c for c in claims
                if c.fact_type == "seer_check_claim"
                and c.source_player == "p06"
            ]
            assert not [
                c for c in claims
                if c.fact_type == "claimed_suspect"
                and c.source_player == "p06"
            ]
            assert not [
                c for c in claims
                if c.fact_type == "claimed_good" and c.source_player == "p06"
            ]

    def test_mixed_third_party_and_self_check_keeps_self_claim(self):
        """前置第三方查杀不能屏蔽同段后续自己的金水查验。"""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        claims = _infer_claims_from_text(
            speaker="p06",
            text="我是预言家，p01报p02查杀，我验了p03金水。",
            day=1,
        )

        assert any(
            c.fact_type == "seer_check_claim"
            and c.source_player == "p06"
            and c.target_player == "p03"
            and c.value == "good"
            for c in claims
        )
        assert not any(
            c.fact_type == "seer_check_claim"
            and c.source_player == "p06"
            and c.target_player == "p02"
            for c in claims
        )

    def test_multiline_badge_flow_preserves_order(self):
        """多行警徽流应提取首个目标并保留完整顺序。"""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        claims = _infer_claims_from_text(
            speaker="p03",
            text="警徽流：\n- N2我计划验p02赵猛\n- N3我计划验p11郑铭",
            day=1,
        )

        badge_facts = [c for c in claims if c.fact_type == "badge_flow_claim"]
        assert len(badge_facts) == 1
        assert badge_facts[0].target_player == "p02"
        assert badge_facts[0].metadata["badge_flow_order"] == ["p02", "p11"]

    def test_badge_flow_accepts_chinese_separators(self):
        """紧凑警徽流支持中文逗号和顿号并保留顺序。"""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        cases = (
            ("警徽流：p05，p07", ["p05", "p07"]),
            ("警徽流：p05、p07", ["p05", "p07"]),
            ("警徽流：先p05后p07再p09", ["p05", "p07", "p09"]),
        )
        for text, expected_order in cases:
            claims = _infer_claims_from_text(speaker="p03", text=text, day=1)
            badge_facts = [c for c in claims if c.fact_type == "badge_flow_claim"]
            assert len(badge_facts) == 1
            assert badge_facts[0].metadata["badge_flow_order"] == expected_order

    def test_real_log_seer_result_and_ordered_badge_flow(self):
        """真实日志句式应提取好人查验和先验/后验警徽流。"""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        claims = _infer_claims_from_text(
            speaker="p03",
            text=(
                "我是预言家，昨晚我验了p02，结果是好人。"
                "我的警徽流：先验p08，后验p06。"
            ),
            day=1,
        )

        good_checks = [
            c for c in claims
            if c.fact_type == "seer_check_claim"
            and c.target_player == "p02"
            and c.value == "good"
        ]
        assert len(good_checks) == 1

        badge_facts = [c for c in claims if c.fact_type == "badge_flow_claim"]
        assert len(badge_facts) == 1
        assert badge_facts[0].target_player == "p08"
        assert badge_facts[0].metadata["badge_flow_order"] == ["p08", "p06"]

    def test_badge_flow_requires_player_token_boundaries(self):
        """紧凑警徽流拒绝前缀 token，并支持 ASCII/中文分隔后的目标。"""
        from werewolf_agent.cognition.world_state import _infer_claims_from_text

        for text in ("警徽流：p020 p11", "警徽流：p02x p11"):
            claims = _infer_claims_from_text(speaker="p03", text=text, day=1)
            assert not [c for c in claims if c.fact_type == "badge_flow_claim"]

        valid_cases = (
            ("警徽流：p05， p07", ["p05", "p07"]),
            ("警徽流：p05, p07", ["p05", "p07"]),
        )
        for text, expected_order in valid_cases:
            claims = _infer_claims_from_text(speaker="p03", text=text, day=1)
            badge_facts = [c for c in claims if c.fact_type == "badge_flow_claim"]
            assert len(badge_facts) == 1
            assert badge_facts[0].metadata["badge_flow_order"] == expected_order


class TestSeerClaimCommitment:
    """Seer claim commitments persist and detect later contradictions."""

    def test_seer_claim_commitment_detects_later_contradiction(self):
        """If p01 claimed seer, later saying '等预言家跳出来' triggers contradiction."""
        from werewolf_agent.cognition.world_state import StructuredWorldState, StructuredFact, _infer_claims_from_text

        # Build world state with p01's seer claim
        ws = StructuredWorldState()
        claim_facts = _infer_claims_from_text(speaker="p01", text="我是预言家", day=1)
        for f in claim_facts:
            ws.append(f)

        # Now p01 says "等预言家跳出来" which contradicts claiming seer
        later_facts = _infer_claims_from_text(speaker="p01", text="等预言家跳出来我再发言", day=2)
        for f in later_facts:
            ws.append(f)

        # Also add the raw speech fact so the contradiction engine can inspect text
        ws.append(StructuredFact(
            fact_type="speech",
            source_player="p01",
            value="等预言家跳出来我再发言",
            day=2,
            metadata={"text": "等预言家跳出来我再发言"},
        ))

        # Contradiction engine should detect this
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        engine = ContradictionEngine()
        alerts = engine.detect(ws.facts, current_day=2)

        # Should find a claim-related contradiction for p01
        p01_alerts = [a for a in alerts if a.player_id == "p01"]
        assert len(p01_alerts) >= 1


class TestCounterclaimDetection:
    """Multiple players claiming seer creates counterclaim alert."""

    def test_two_seer_claimants_creates_counterclaim(self):
        """p01 and p05 both claim seer → claim_conflict alert."""
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        from werewolf_agent.cognition.world_state import StructuredWorldState, StructuredFact

        ws = StructuredWorldState()
        ws.append(StructuredFact(
            fact_type="claimed_role", source_player="p01", target_player="p01",
            day=1, night=0, phase="sheriff_speech", value="seer", metadata={},
        ))
        ws.append(StructuredFact(
            fact_type="claimed_role", source_player="p05", target_player="p05",
            day=1, night=0, phase="sheriff_speech", value="seer", metadata={},
        ))

        engine = ContradictionEngine()
        alerts = engine.detect(ws.facts, current_day=1)
        claim_conflicts = [a for a in alerts if a.alert_type == "claim_conflict"]
        assert len(claim_conflicts) >= 1
        assert claim_conflicts[0].priority == "high"


# === Task 12: Contradiction Alerts Must Be Answered ===

class TestContradictionContextPriority:
    """High-priority contradiction alerts reach next player context."""

    def test_high_priority_contradiction_reaches_next_player_context(self):
        """Self-exposure, claim conflict alerts are high priority and visible."""
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        from werewolf_agent.cognition.world_state import StructuredWorldState, StructuredFact

        ws = StructuredWorldState()
        # p01 claims seer
        ws.append(StructuredFact(
            fact_type="claimed_role", source_player="p01", target_player="p01",
            day=1, night=0, phase="sheriff_speech", value="seer", metadata={},
        ))
        # p05 also claims seer (counterclaim)
        ws.append(StructuredFact(
            fact_type="claimed_role", source_player="p05", target_player="p05",
            day=1, night=0, phase="sheriff_speech", value="seer", metadata={},
        ))

        engine = ContradictionEngine()
        alerts = engine.detect(ws.facts, current_day=1)

        # Should have high-priority claim_conflict
        high_priority = [a for a in alerts if a.priority == "high"]
        assert len(high_priority) >= 1

    def test_contradiction_alerts_available_in_context(self):
        """AgentContext.contradiction_alerts field is populated."""
        from werewolf_agent.agents.schemas import AgentContext, TaskType
        ctx = AgentContext(
            agent_id="p02",
            task_type=TaskType.SPEECH,
            contradiction_alerts=[
                {"alert_type": "claim_conflict", "priority": "high", "players": ["p01", "p05"]},
            ],
        )
        assert len(ctx.contradiction_alerts) == 1
        assert ctx.contradiction_alerts[0]["priority"] == "high"


class TestMustAddressAlerts:
    """Context field must_address_alerts contains top alerts for response."""

    def test_must_address_alerts_built_from_contradictions(self):
        """Build must_address_alerts from contradiction engine output."""
        from werewolf_agent.cognition.contradiction import ContradictionAlert

        alerts = [
            ContradictionAlert(
                player_id="p01", alert_type="claim_conflict", priority="high",
                description="p01 and p05 both claim seer",
                evidence=({"role": "seer", "claimers": ["p01", "p05"]},),
                day_range=(1, 1),
            ),
        ]

        # Build must_address from high priority alerts
        must_address = _build_must_address_alerts(alerts, viewer_id="p02")
        assert len(must_address) >= 1
        assert must_address[0]["alert_type"] == "claim_conflict"
        assert "required_response" in must_address[0]


def _build_must_address_alerts(alerts, viewer_id=None):
    """Helper to build must_address_alerts from contradiction alerts."""
    result = []
    for alert in alerts:
        if alert.priority == "high":
            entry = {
                "alert_type": alert.alert_type,
                "players": [p for p in alert.player_id.split(",") if p],
                "description": alert.description,
                "required_response": ["question", "side_with", "park"],
            }
            result.append(entry)
    return result


# =====================================================================
# E2 (post-review-v2): _extract_seer_check 不应全表扫 state.players 找 seer
# =====================================================================

class TestExtractorSeerCheckSignature:
    """E2 (post-review-v2): _extract_seer_check 签名应接受 seer_id 参数，不做全表扫。"""

    def test_seer_check_extractor_does_not_loop_state_players(self):
        from werewolf_agent.cognition import world_state
        import inspect
        # _extract_seer_check 源码不应再做 `for p in state.players` 全表扫
        fn = getattr(world_state, "_extract_seer_check", None)
        assert fn is not None, "world_state._extract_seer_check must exist"
        fn_src = inspect.getsource(fn)
        # 旧实现: next(p for pid, p in state.players.items() if p.role == "seer")
        # 新实现: 应直接接 seer_id 参数
        assert "state.players" not in fn_src, (
            f"_extract_seer_check still loops state.players (全表扫):\n{fn_src[:500]}"
        )
        assert "for.*p in state.players" not in fn_src or "seer_id" in fn_src, (
            f"_extract_seer_check still scans state.players for seer:\n{fn_src[:500]}"
        )

    def test_seer_check_extractor_accepts_seer_id_param(self):
        from werewolf_agent.cognition import world_state
        import inspect
        fn = getattr(world_state, "_extract_seer_check", None)
        assert fn is not None
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        # 应包含 seer_id 参数
        assert "seer_id" in params, (
            f"_extract_seer_check should accept seer_id as parameter, got: {params}"
        )

    def test_seer_check_extractor_uses_seer_id(self):
        """E2 (post-review-v2): _extract_seer_check 实际调用时使用传入的 seer_id。"""
        from werewolf_agent.cognition.world_state import _extract_seer_check
        event = GameEvent(
            type="seer_check",
            payload={"target_id": "p01", "alignment": "werewolf", "night_number": 1},
        )
        # 直接调用，传入 seer_id="p08"
        facts = _extract_seer_check(event, seer_id="p08")
        assert len(facts) == 1
        assert facts[0].source_player == "p08", (
            f"_extract_seer_check should use injected seer_id, got: {facts[0].source_player}"
        )
        assert facts[0].target_player == "p01"
        assert facts[0].value == "werewolf"

    def test_extract_facts_seer_check_via_dispatch(self):
        """E2 (post-review-v2): extract_facts 调度 seer_check 时也用注入的 seer_id。"""
        from werewolf_agent.cognition.world_state import extract_facts
        state = _make_state()
        event = GameEvent(
            type="seer_check",
            payload={"target_id": "p01", "alignment": "werewolf", "night_number": 1},
        )
        facts = extract_facts(event, state)
        # _make_state 中 p08 是 seer
        assert facts[0].source_player == "p08", (
            f"extract_facts did not resolve seer_id to p08: {facts[0].source_player}"
        )

    def test_extract_facts_seer_check_keeps_source_after_seer_death(self):
        """Historical seer_check facts still belong to the single seer
        after that seer dies later in the game."""
        from dataclasses import replace
        from werewolf_agent.cognition.world_state import extract_facts

        state = _make_state()
        state = replace(
            state,
            players={
                **state.players,
                "p08": replace(state.players["p08"], alive=False),
            },
        )
        event = GameEvent(
            type="seer_check",
            payload={"target_id": "p01", "alignment": "werewolf", "night_number": 1},
        )

        facts = extract_facts(event, state)

        assert facts[0].source_player == "p08"
