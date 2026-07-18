# -*- coding: utf-8 -*-
"""
验证技能机会链的权威事件与角色私有投影。

作者: Project contributors
创建日期: 2026-07-18
"""

from __future__ import annotations

from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import build_world_state
from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.skill_opportunity_events import (
    build_private_skill_event,
    build_public_skill_resolution,
)


def _state_with_events(events):
    return GameState(
        players={
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "seer": PlayerState(id="seer", role="seer"),
            "villager": PlayerState(id="villager", role="villager"),
        },
        events=list(events),
    )


def test_private_self_destruct_chain_has_authority_and_actor_projection() -> None:
    events = build_private_skill_event(
        "self_destruct_opportunity",
        actor_id="wolf",
        day_number=2,
        available_actions=["self_destruct", "continue"],
    )

    assert len(events) == 2
    authoritative, actor_view = events
    assert event_visibility(authoritative) is EventVisibility.MODERATOR_ONLY
    assert event_visibility(actor_view) is EventVisibility.ACTOR_PRIVATE
    assert actor_view.payload["visibility_actor_id"] == "wolf"
    assert authoritative.payload["actor_id"] == "wolf"
    assert authoritative.payload["available_actions"] == ["self_destruct", "continue"]

    state = _state_with_events(events)
    policy = VisibilityPolicy()
    wolf_facts = policy.filter_visible_facts(build_world_state(state), "wolf", "werewolf")
    villager_facts = policy.filter_visible_facts(
        build_world_state(state), "villager", "villager"
    )
    moderator_facts = policy.filter_visible_facts(
        build_world_state(state), "moderator", "moderator"
    )
    assert [fact.fact_type for fact in wolf_facts] == [
        "self_destruct_opportunity_actor_view"
    ]
    assert villager_facts == []
    assert [fact.fact_type for fact in moderator_facts] == [
        "self_destruct_opportunity"
    ]


def test_public_skill_resolution_excludes_private_reason_and_true_role() -> None:
    from werewolf_agent.runtime.exposure_audit import (
        is_safe_public_skill_resolution_payload,
    )
    event = build_public_skill_resolution(
        "hunter_shot_resolved",
        actor_id="hunter",
        target_id="wolf",
        public_result="target_died",
        private_reason="the model supplied a private reason",
        target_true_role="werewolf",
    )

    assert event_visibility(event) is EventVisibility.PUBLIC
    assert event.payload == {
        "actor_id": "hunter",
        "target_id": "wolf",
        "public_result": "target_died",
    }
    assert is_safe_public_skill_resolution_payload(event.payload)
    assert not is_safe_public_skill_resolution_payload({
        **event.payload,
        "private_reason": "sensitive",
    })


def test_seer_chain_is_never_public_and_actor_projection_is_owner_scoped() -> None:
    events = [
        *build_private_skill_event(
            "seer_check_opportunity",
            actor_id="seer",
            night_number=2,
            legal_targets=["wolf", "villager"],
        ),
        *build_private_skill_event(
            "seer_check_selected",
            actor_id="seer",
            night_number=2,
            target_id="wolf",
        ),
        *build_private_skill_event(
            "seer_check_resolved",
            actor_id="seer",
            night_number=2,
            target_id="wolf",
            alignment="werewolf",
        ),
    ]

    assert all(event_visibility(event) is not EventVisibility.PUBLIC for event in events)
    state = _state_with_events(events)
    policy = VisibilityPolicy()
    seer_facts = policy.filter_visible_facts(build_world_state(state), "seer", "seer")
    wolf_facts = policy.filter_visible_facts(build_world_state(state), "wolf", "werewolf")
    assert [fact.fact_type for fact in seer_facts] == [
        "seer_check_opportunity_actor_view",
        "seer_check_selected_actor_view",
        "seer_check_resolved_actor_view",
    ]
    assert wolf_facts == []


def test_self_destruct_is_recorded_only_for_an_available_wolf_choice() -> None:
    from werewolf_agent.agents.schemas import ActionType, PlayerAction, RetryInfo
    from werewolf_agent.runtime.graph import _new_engine, free_discussion
    from werewolf_agent.runtime.nodes.skills import resolve_self_destruct_node

    class SelfDestructAgent:
        def act(self, _context):
            return PlayerAction(action_type=ActionType.SELF_DESTRUCT), RetryInfo()

    class Registry:
        def get_agent(self, player_id):
            return SelfDestructAgent() if player_id == "wolf" else None

    wolf_state = GameState(
        game_id="self_destruct_opportunity",
        phase="day",
        day_number=1,
        players={
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "villager": PlayerState(id="villager", role="villager"),
        },
    )
    selected = free_discussion({
        "game_state": wolf_state,
        "engine": _new_engine(),
        "agent_registry": Registry(),
        "speech_order": ["wolf"],
        "speech_index": 0,
    })
    selected_events = {event.type: event for event in selected["game_state"].events}
    assert selected_events["self_destruct_opportunity"].visibility is EventVisibility.MODERATOR_ONLY
    assert selected_events["self_destruct_selected"].visibility is EventVisibility.MODERATOR_ONLY

    resolved = resolve_self_destruct_node({
        **selected,
        "engine": _new_engine(),
    })["game_state"]
    assert event_visibility(next(
        event for event in resolved.events if event.type == "self_destruct_resolved"
    )) is EventVisibility.PUBLIC

    unavailable = free_discussion({
        "game_state": GameState(
            game_id="self_destruct_unavailable",
            phase="day",
            day_number=1,
            players={"villager": PlayerState(id="villager", role="villager")},
        ),
        "engine": _new_engine(),
        "speech_order": ["villager"],
        "speech_index": 0,
        "speech_text": "普通发言",
    })["game_state"]
    assert not [event for event in unavailable.events if event.type == "self_destruct_opportunity"]


def test_power_metrics_use_private_opportunity_events_as_the_denominator() -> None:
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )

    metrics = compute_power_acceptance_metrics([{
        "game_id": "power_opportunity_denominator",
        "status": "running",
        "players": {
            "hunter": {"role": "hunter"},
            "seer": {"role": "seer"},
            "wolf": {"role": "werewolf"},
        },
        "events": [
            {"type": "hunter_shot_opportunity", "payload": {}},
            {"type": "hunter_shot_selected", "payload": {}},
            {"type": "seer_check_opportunity", "payload": {}},
            {"type": "seer_check_skipped", "payload": {}},
        ],
    }])

    assert metrics["power_role_opportunity_count"] == 2
    assert metrics["power_role_selected_count"] == 1
    assert metrics["power_role_selection_rate"] == 0.5
