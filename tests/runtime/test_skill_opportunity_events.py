# -*- coding: utf-8 -*-
"""
验证技能机会链的权威事件与角色私有投影。

作者: Project contributors
创建日期: 2026-07-18
修改日期: 2026-07-27
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
    import pytest

    from werewolf_agent.runtime.exposure_audit import (
        is_safe_public_skill_resolution_payload,
    )
    event = build_public_skill_resolution(
        "hunter_shot_resolved",
        actor_id="hunter",
        target_id="wolf",
        public_result="target_died",
        day_number=2,
        private_reason="the model supplied a private reason",
        target_true_role="werewolf",
    )

    assert event_visibility(event) is EventVisibility.PUBLIC
    assert event.payload == {
        "actor_id": "hunter",
        "target_id": "wolf",
        "public_result": "target_died",
        "day_number": 2,
    }
    assert is_safe_public_skill_resolution_payload(event.payload)
    assert not is_safe_public_skill_resolution_payload({
        **event.payload,
        "private_reason": "sensitive",
    })
    for unsafe_day_number in (True, -1, "2"):
        assert not is_safe_public_skill_resolution_payload({
            **event.payload,
            "day_number": unsafe_day_number,
        })
    legacy_event = build_public_skill_resolution(
        "hunter_shot_resolved",
        actor_id="hunter",
        target_id="wolf",
        public_result="target_died",
    )
    assert "day_number" not in legacy_event.payload
    with pytest.raises(ValueError, match="public_skill_resolution_event_type"):
        build_public_skill_resolution(
            "seer_check_resolved",
            actor_id="seer",
            target_id="wolf",
            public_result="werewolf",
        )


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


def test_night_resolution_rejects_forged_or_cross_role_seer_choice() -> None:
    from werewolf_agent.runtime.graph import _new_engine
    from werewolf_agent.runtime.nodes.night_resolution import resolve_night

    gs = GameState(
        game_id="forged_seer_choice",
        night_number=2,
        players={
            "seer": PlayerState(id="seer", role="seer"),
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "villager": PlayerState(id="villager", role="villager"),
        },
        events=list(build_private_skill_event(
            "seer_check_selected",
            actor_id="wolf",
            night_number=2,
            target_id="villager",
        )),
    )

    result = resolve_night({
        "game_state": gs,
        "engine": _new_engine(),
        "seer_target_id": "villager",
    })["game_state"]

    assert not [
        event for event in result.events
        if event.type == "seer_check_resolved"
    ]


def test_night_seer_same_night_reentry_does_not_duplicate_choice_chain(
    monkeypatch,
) -> None:
    from importlib import import_module

    night_specialists = import_module(
        "werewolf_agent.runtime.nodes.night_specialists"
    )
    monkeypatch.setattr(
        night_specialists,
        "_dispatch_agent",
        lambda *_args, **_kwargs: {"seer_target_id": "villager"},
    )
    initial = {
        "game_state": GameState(
            game_id="seer_reentry",
            night_number=2,
            players={
                "seer": PlayerState(id="seer", role="seer"),
                "villager": PlayerState(id="villager", role="villager"),
            },
        ),
        "seer_target_id": "villager",
    }
    first = night_specialists.night_seer(initial)
    replay = night_specialists.night_seer({
        **initial,
        "game_state": first["game_state"],
    })

    for event_type in ("seer_check_opportunity", "seer_check_selected"):
        assert len([
            event for event in replay["game_state"].events
            if event.type == event_type
            and event.visibility is EventVisibility.MODERATOR_ONLY
        ]) == 1


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


def test_sheriff_speech_records_private_wolf_choice_and_routes_to_resolver(
    monkeypatch,
) -> None:
    from importlib import import_module

    from werewolf_agent.runtime.graph import route_after_sheriff_speech
    sheriff_speech_module = import_module(
        "werewolf_agent.runtime.nodes.sheriff_speech"
    )

    gs = GameState(
        game_id="sheriff_speech_self_destruct",
        day_number=1,
        sheriff_candidates=["wolf", "villager"],
        players={
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "villager": PlayerState(id="villager", role="villager"),
        },
    )

    class Registry:
        def get_agent(self, _player_id):
            return None

    monkeypatch.setattr(
        sheriff_speech_module,
        "_dispatch_agent",
        lambda *_args, **_kwargs: {"self_destruct": True},
    )
    result = sheriff_speech_module.sheriff_speech({
        "game_state": gs,
        "agent_registry": Registry(),
    })

    assert result["self_destruct_wolf_id"] == "wolf"
    assert route_after_sheriff_speech(result) == "resolve_self_destruct"
    events = result["game_state"].events
    opportunity = [
        event for event in events
        if event.type == "self_destruct_opportunity"
        and event.visibility is EventVisibility.MODERATOR_ONLY
    ]
    selected = [
        event for event in events
        if event.type == "self_destruct_selected"
        and event.visibility is EventVisibility.MODERATOR_ONLY
    ]
    assert [event.payload["actor_id"] for event in opportunity] == ["wolf"]
    assert [event.payload["actor_id"] for event in selected] == ["wolf"]
    assert not [event for event in events if event.type == "self_destruct_declined"]

    policy = VisibilityPolicy()
    wolf_facts = policy.filter_visible_facts(
        build_world_state(result["game_state"]), "wolf", "werewolf"
    )
    villager_facts = policy.filter_visible_facts(
        build_world_state(result["game_state"]), "villager", "villager"
    )
    assert {
        fact.fact_type for fact in wolf_facts
        if fact.fact_type.startswith("self_destruct_")
    } == {
        "self_destruct_opportunity_actor_view",
        "self_destruct_selected_actor_view",
    }
    assert not {
        fact.fact_type for fact in villager_facts
        if fact.fact_type.startswith("self_destruct_")
    }


def test_self_destruct_resolution_ignores_stale_reentry_after_canonical_result() -> None:
    from werewolf_agent.runtime.graph import _new_engine
    from werewolf_agent.runtime.nodes.skills import resolve_self_destruct_node

    gs = GameState(
        game_id="self_destruct_reentry",
        day_number=2,
        players={"wolf": PlayerState(id="wolf", role="werewolf")},
        events=[
            *build_private_skill_event(
                "self_destruct_opportunity",
                actor_id="wolf",
                day_number=2,
                opportunity_phase="sheriff_speech",
            ),
            *build_private_skill_event(
                "self_destruct_selected",
                actor_id="wolf",
                day_number=2,
                opportunity_phase="sheriff_speech",
            ),
        ],
    )
    first = resolve_self_destruct_node({
        "game_state": gs,
        "engine": _new_engine(),
        "self_destruct_wolf_id": "wolf",
    })["game_state"]
    replay = resolve_self_destruct_node({
        "game_state": first,
        "engine": _new_engine(),
        "self_destruct_wolf_id": "wolf",
    })["game_state"]

    assert replay.events == first.events
    assert replay.sheriff_interrupt_count == first.sheriff_interrupt_count


def test_day_speech_reentry_does_not_duplicate_canonical_self_destruct_chain() -> None:
    from werewolf_agent.runtime.graph import _new_engine, free_discussion

    class Registry:
        def get_agent(self, _player_id):
            return None

    state = {
        "game_state": GameState(
            game_id="day_self_destruct_reentry",
            day_number=2,
            players={"wolf": PlayerState(id="wolf", role="werewolf")},
        ),
        "agent_registry": Registry(),
        "speech_order": ["wolf"],
        "speech_index": 0,
    }
    first = free_discussion({
        **state,
        "engine": _new_engine(),
    })
    replay = free_discussion({
        **state,
        **first,
        "engine": _new_engine(),
        "speech_index": 0,
    })

    for event_type in ("self_destruct_opportunity", "self_destruct_declined"):
        assert len([
            event for event in replay["game_state"].events
            if event.type == event_type
            and event.visibility is EventVisibility.MODERATOR_ONLY
        ]) == 1


def test_sheriff_speech_declines_once_for_wolf_fallback_but_not_nonwolf(
    monkeypatch,
) -> None:
    from importlib import import_module

    sheriff_speech_module = import_module(
        "werewolf_agent.runtime.nodes.sheriff_speech"
    )

    class Registry:
        def get_agent(self, _player_id):
            return None

    def run(candidates, players, dispatch_result):
        monkeypatch.setattr(
            sheriff_speech_module,
            "_dispatch_agent",
            lambda *_args, **_kwargs: dispatch_result,
        )
        return sheriff_speech_module.sheriff_speech({
            "game_state": GameState(
                game_id="sheriff_speech_declined",
                day_number=1,
                sheriff_candidates=candidates,
                players=players,
            ),
            "agent_registry": Registry(),
        })["game_state"].events

    wolf_events = run(
        ["wolf"],
        {"wolf": PlayerState(id="wolf", role="werewolf")},
        None,
    )
    assert len([
        event for event in wolf_events
        if event.type == "self_destruct_opportunity"
        and event.visibility is EventVisibility.MODERATOR_ONLY
    ]) == 1
    assert len([
        event for event in wolf_events
        if event.type == "self_destruct_declined"
        and event.visibility is EventVisibility.MODERATOR_ONLY
    ]) == 1

    continued_events = run(
        ["wolf"],
        {"wolf": PlayerState(id="wolf", role="werewolf")},
        {"speech_text": "continued speech"},
    )
    assert len([
        event for event in continued_events
        if event.type == "self_destruct_declined"
        and event.visibility is EventVisibility.MODERATOR_ONLY
    ]) == 1

    villager_events = run(
        ["villager"],
        {"villager": PlayerState(id="villager", role="villager")},
        {"self_destruct": True},
    )
    assert not [
        event for event in villager_events
        if event.type.startswith("self_destruct_")
    ]


def test_power_metrics_use_private_opportunity_events_as_the_denominator() -> None:
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )

    def event(sequence_number, event_type, payload, visibility):
        return {
            "type": event_type,
            "payload": payload,
            "visibility": visibility,
            "event_id": f"power_opportunity_denominator:e{sequence_number:06d}",
            "sequence_number": sequence_number,
            "game_id": "power_opportunity_denominator",
            "schema_version": "2",
            "occurred_at": "2026-07-18T00:00:00+00:00",
        }

    metrics = compute_power_acceptance_metrics([{
        "game_id": "power_opportunity_denominator",
        "status": "finished",
        "winning_faction": "good",
        "players": {
            "hunter": {"role": "hunter"},
            "seer": {"role": "seer"},
            "wolf": {"role": "werewolf"},
        },
        "events": [
            event(0, "hunter_shot_opportunity", {"actor_id": "hunter", "night_number": 1}, "moderator_only"),
            event(1, "hunter_shot_selected", {"actor_id": "hunter", "night_number": 1}, "moderator_only"),
            event(2, "hunter_shot_resolved", {"actor_id": "hunter"}, "public"),
            event(3, "seer_check_opportunity", {"actor_id": "seer", "night_number": 1}, "moderator_only"),
            event(4, "seer_check_skipped", {"actor_id": "seer", "night_number": 1}, "moderator_only"),
            event(5, "seer_check_resolved", {"actor_id": "seer", "night_number": 1}, "moderator_only"),
        ],
    }])

    assert metrics["power_role_opportunity_count"] == 2
    assert metrics["power_role_selected_count"] == 1
    assert metrics["power_role_selection_rate"] == 0.5


def test_power_opportunity_metrics_reject_unfinished_or_forged_event_chain() -> None:
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )

    def event(sequence_number, event_type, visibility):
        return {
            "type": event_type,
            "payload": {"actor_id": "seer", "night_number": 2},
            "visibility": visibility,
            "event_id": f"duplicate_power_chain:e{sequence_number:06d}",
            "sequence_number": sequence_number,
            "game_id": "duplicate_power_chain",
            "schema_version": "2",
            "occurred_at": "2026-07-18T00:00:00+00:00",
        }

    metrics = compute_power_acceptance_metrics([{
        "game_id": "unfinished_power_metrics",
        "status": "running",
        "players": {"villager": {"role": "villager"}},
        "events": [{
            "type": "seer_check_opportunity",
            "payload": {"actor_id": "villager", "night_number": 1},
            "visibility": "public",
        }],
    }])

    assert metrics["power_role_opportunity_metrics_supported"] is False
    assert metrics["power_role_opportunity_count"] is None

    out_of_order = compute_power_acceptance_metrics([{
        "game_id": "duplicate_power_chain",
        "status": "finished",
        "winning_faction": "good",
        "players": {"seer": {"role": "seer"}},
        "events": [
            event(2, "seer_check_opportunity", "moderator_only"),
            event(1, "seer_check_selected", "moderator_only"),
            event(0, "seer_check_resolved", "moderator_only"),
        ],
    }])
    assert out_of_order["power_role_opportunity_metrics_supported"] is False

    assert metrics["power_role_selection_rate"] is None


def test_power_opportunity_metrics_reject_duplicate_or_out_of_order_v2_chain() -> None:
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )

    def event(sequence_number, event_type, visibility):
        return {
            "type": event_type,
            "payload": {"actor_id": "seer", "night_number": 2},
            "visibility": visibility,
            "event_id": f"duplicate_power_chain:e{sequence_number:06d}",
            "sequence_number": sequence_number,
            "game_id": "duplicate_power_chain",
            "schema_version": "2",
            "occurred_at": "2026-07-18T00:00:00+00:00",
        }

    metrics = compute_power_acceptance_metrics([{
        "game_id": "duplicate_power_chain",
        "status": "finished",
        "winning_faction": "good",
        "players": {"seer": {"role": "seer"}},
        "events": [
            event(0, "seer_check_opportunity", "moderator_only"),
            event(1, "seer_check_selected", "moderator_only"),
            event(2, "seer_check_selected", "moderator_only"),
            event(3, "seer_check_resolved", "moderator_only"),
        ],
    }])

    assert metrics["power_role_opportunity_metrics_supported"] is False
    assert metrics["power_role_opportunity_count"] is None
