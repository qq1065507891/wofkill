# -*- coding: utf-8 -*-
"""
验证所有结算前空刀统一经过确定性的 NoKillPolicy。

作者: Project contributors
创建日期: 2026-07-16
修改日期: 2026-07-18

使用示例:
    >>> python -m pytest tests/runtime/test_wolf_no_kill_policy.py -q
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.evaluation.trace_identity import DecisionIdentity


def _expected_trace_id(game_state: GameState, event_type: str, index: int) -> str:
    """按统一狼人决策身份契约计算生产事件的稳定 trace ID。"""
    return DecisionIdentity(
        game_id=game_state.game_id,
        player_id="werewolf_team",
        phase="wolf_consensus",
        day_number=game_state.day_number,
        night_number=game_state.night_number,
        task_type=event_type,
        action_index=index,
    ).trace_id()


def _game_state(*, events: list[GameEvent] | None = None) -> GameState:
    return GameState(
        game_id="no-kill-policy",
        night_number=3,
        players={
            "w1": PlayerState(id="w1", role="werewolf"),
            "p3": PlayerState(id="p3", role="villager"),
            "p1": PlayerState(id="p1", role="seer"),
            "p2": PlayerState(id="p2", role="witch"),
        },
        events=events or [],
    )


@pytest.mark.parametrize(
    "reason_code",
    [
        "strategic_abstain",
        "true_tie",
        "insufficient_quorum",
        "invalid_primary",
        "invalid_backup",
        "plan_generation_failed",
        "provider_unavailable",
    ],
)
def test_every_no_kill_reason_has_one_decision_schema(reason_code: str) -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    result = NoKillPolicy().resolve(_game_state(), reason_code=reason_code)

    assert result["wolf_kill_target_id"] is None
    event = result["game_state"].events[-1]
    assert event.type == "wolf_no_kill_timeout"
    assert event.visibility is EventVisibility.WEREWOLF_TEAM_ONLY
    assert event.schema_version == "2"
    assert event.trace_id == _expected_trace_id(
        result["game_state"], "wolf_no_kill_timeout", 0
    )
    assert event.payload["reason"] == reason_code
    assert event.payload["no_kill_decision"] == {
        "reason_code": reason_code,
        "consecutive_pre_resolution_no_kill_count": 1,
        "forced_recovery_applied": False,
        "recovered_target_id": None,
    }


def test_third_consecutive_pre_resolution_no_kill_forces_deterministic_target() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    prior_events = [
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={
                "reason": "provider_unavailable",
                "night_number": 1,
                "no_kill_decision": {
                    "reason_code": "provider_unavailable",
                    "consecutive_pre_resolution_no_kill_count": 1,
                    "forced_recovery_applied": False,
                    "recovered_target_id": None,
                },
            },
        ),
        GameEvent(
            type="wolf_no_kill_declared",
            payload={
                "reason": "strategic_abstain",
                "night_number": 2,
                "no_kill_decision": {
                    "reason_code": "strategic_abstain",
                    "consecutive_pre_resolution_no_kill_count": 2,
                    "forced_recovery_applied": False,
                    "recovered_target_id": None,
                },
            },
        ),
    ]

    result = NoKillPolicy(max_consecutive_pre_resolution_no_kill=2).resolve(
        _game_state(events=prior_events),
        reason_code="true_tie",
        primary_positive_support={"p2": 1, "p1": 1},
        backup_positive_support={"p2": 1, "p1": 2},
    )

    assert result["wolf_kill_target_id"] == "p1"
    recovery, selected = result["game_state"].events[-2:]
    assert recovery.type == "wolf_kill_forced_recovery"
    assert recovery.payload == {
        "night_number": 3,
        "original_reasons": [
            "provider_unavailable",
            "strategic_abstain",
            "true_tie",
        ],
        "consecutive_pre_resolution_no_kill_count": 3,
        "candidate_scores": {
            "p1": [1, 2, -3],
            "p2": [1, 1, -4],
            "p3": [0, 0, -2],
        },
        "final_target_id": "p1",
    }
    assert recovery.visibility is EventVisibility.MODERATOR_ONLY
    assert recovery.trace_id == _expected_trace_id(
        result["game_state"], "wolf_kill_forced_recovery", 0
    )
    assert selected.type == "wolf_kill_selected"
    assert selected.visibility is EventVisibility.WEREWOLF_TEAM_ONLY
    assert selected.trace_id == _expected_trace_id(
        result["game_state"], "wolf_kill_selected", 1
    )
    assert recovery.trace_id != selected.trace_id
    assert selected.payload["target_id"] == "p1"
    assert selected.payload["reason"] == "forced_recovery"


def test_selected_kill_immediately_resets_counter_even_after_skill_cancellation() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    events = [
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={"reason": "true_tie"},
        ),
        GameEvent(
            type="wolf_kill_selected",
            payload={"target_id": "p1"},
        ),
        GameEvent(type="witch_antidote_used", payload={"target_id": "p1"}),
        GameEvent(type="guard_protected", payload={"target_id": "p1"}),
    ]

    result = NoKillPolicy().resolve(
        _game_state(events=events),
        reason_code="provider_unavailable",
    )

    decision = result["game_state"].events[-1].payload["no_kill_decision"]
    assert decision["consecutive_pre_resolution_no_kill_count"] == 1
    assert decision["forced_recovery_applied"] is False


def test_duplicate_no_kill_events_in_one_night_count_only_once() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    events = [
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={"night_number": 1, "reason": "provider_unavailable"},
        ),
        GameEvent(
            type="wolf_no_kill_declared",
            payload={"night_number": 1, "reason": "strategic_abstain"},
        ),
    ]

    result = NoKillPolicy().resolve(
        _game_state(events=events),
        reason_code="true_tie",
    )

    assert result["wolf_kill_target_id"] is None
    decision = result["game_state"].events[-1].payload["no_kill_decision"]
    assert decision["consecutive_pre_resolution_no_kill_count"] == 2
    assert decision["forced_recovery_applied"] is False


def test_repeated_resolve_in_same_night_is_idempotent() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    first = NoKillPolicy().resolve(
        _game_state(),
        reason_code="strategic_abstain",
    )
    first_state = first["game_state"]

    second = NoKillPolicy().resolve(
        first_state,
        reason_code="provider_unavailable",
    )

    assert second["game_state"] is first_state
    assert second["wolf_kill_target_id"] is None
    assert len(second["game_state"].events) == 1
    assert (
        second["game_state"].events[-1]
        .payload["no_kill_decision"]
        ["consecutive_pre_resolution_no_kill_count"]
        == 1
    )


def test_no_kill_trace_identity_is_stable_on_replay_and_declared_route() -> None:
    """同一输入重放保持 trace 稳定，主动空刀使用独立语义身份。"""
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    initial = _game_state()
    first = NoKillPolicy().resolve(initial, reason_code="provider_unavailable")
    replay = NoKillPolicy().resolve(initial, reason_code="provider_unavailable")
    declared = NoKillPolicy().resolve(
        initial,
        reason_code="strategic_abstain",
        event_type="wolf_no_kill_declared",
    )

    timeout_event = first["game_state"].events[-1]
    replay_event = replay["game_state"].events[-1]
    declared_event = declared["game_state"].events[-1]
    assert timeout_event.trace_id == replay_event.trace_id
    assert timeout_event.trace_id == _expected_trace_id(
        initial, "wolf_no_kill_timeout", 0
    )
    assert declared_event.trace_id == _expected_trace_id(
        initial, "wolf_no_kill_declared", 0
    )
    assert declared_event.trace_id != timeout_event.trace_id

    from werewolf_agent.runtime.event_metadata import (
        deserialize_game_event,
        serialize_game_event,
    )

    restored = deserialize_game_event(serialize_game_event(timeout_event))
    assert restored.trace_id == timeout_event.trace_id


def test_new_no_kill_trace_does_not_backfill_legacy_v1_event() -> None:
    """V1 历史事件保持只读无 trace，新写 V2 事件才获得权威身份。"""
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    legacy = GameEvent(
        type="wolf_no_kill_timeout",
        payload={"night_number": 2, "reason": "provider_unavailable"},
    )
    initial = _game_state(events=[legacy])

    result = NoKillPolicy().resolve(initial, reason_code="true_tie")

    assert result["game_state"].events[0] is legacy
    assert result["game_state"].events[0].trace_id is None
    assert result["game_state"].events[-1].trace_id == _expected_trace_id(
        initial, "wolf_no_kill_timeout", 0
    )


def test_threshold_one_does_not_recover_on_second_same_night_resolve() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    policy = NoKillPolicy(max_consecutive_pre_resolution_no_kill=1)
    first = policy.resolve(_game_state(), reason_code="true_tie")

    repeated = policy.resolve(
        first["game_state"],
        reason_code="plan_generation_failed",
    )

    assert repeated["game_state"] is first["game_state"]
    assert repeated["wolf_kill_target_id"] is None
    assert len(repeated["game_state"].events) == 1

    next_night = replace(first["game_state"], night_number=4)
    recovered = policy.resolve(
        next_night,
        reason_code="plan_generation_failed",
    )
    assert recovered["wolf_kill_target_id"] is not None
    assert recovered["game_state"].events[-2].type == "wolf_kill_forced_recovery"


def test_repeated_resolve_after_forced_recovery_preserves_selected_target() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    policy = NoKillPolicy(max_consecutive_pre_resolution_no_kill=1)
    first_night = policy.resolve(_game_state(), reason_code="true_tie")
    next_night = replace(first_night["game_state"], night_number=4)
    recovered = policy.resolve(
        next_night,
        reason_code="plan_generation_failed",
    )
    recovered_state = recovered["game_state"]

    repeated = policy.resolve(
        recovered_state,
        reason_code="provider_unavailable",
    )

    assert repeated["game_state"] is recovered_state
    assert repeated["wolf_kill_target_id"] == recovered["wolf_kill_target_id"]
    assert len(repeated["game_state"].events) == len(recovered_state.events)


def test_repeated_resolve_preserves_selected_target_after_resolution_death() -> None:
    from werewolf_agent.runtime.event_metadata import new_game_event
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state()
    selected = new_game_event(
        gs,
        "wolf_kill_selected",
        {"night_number": 3, "target_id": "p1"},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )
    resolved_state = replace(
        gs,
        players={
            **gs.players,
            "p1": replace(gs.players["p1"], alive=False),
        },
        events=[selected],
    )

    repeated = NoKillPolicy().resolve(
        resolved_state,
        reason_code="provider_unavailable",
    )

    assert repeated["game_state"] is resolved_state
    assert repeated["wolf_kill_target_id"] == "p1"
    assert repeated["game_state"].events == [selected]


def test_v1_checkpoint_no_kill_forms_trigger_third_night_recovery() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state(events=[
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={"night_number": 1},
        ),
        GameEvent(
            type="wolf_no_kill_declared",
            payload={
                "night_number": 2,
                "wolf_action_reason": "create peace-night pressure",
            },
        ),
    ])

    result = NoKillPolicy().resolve(gs, reason_code="true_tie")

    assert result["wolf_kill_target_id"] is not None
    recovery = result["game_state"].events[-2]
    assert recovery.payload["original_reasons"] == [
        "provider_unavailable",
        "strategic_abstain",
        "true_tie",
    ]


def test_v1_invalid_plan_forms_have_stable_reason_mapping() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state(events=[
        GameEvent(
            type="wolf_plan_invalid_no_kill",
            payload={"night_number": 1, "reason": "old plan text"},
        ),
        GameEvent(
            type="wolf_plan_invalid_no_kill",
            payload={"night_number": 2, "reason": "arbitrary old text"},
        ),
    ])

    result = NoKillPolicy().resolve(gs, reason_code="invalid_backup")

    recovery = result["game_state"].events[-2]
    assert recovery.payload["original_reasons"] == [
        "plan_generation_failed",
        "plan_generation_failed",
        "invalid_backup",
    ]


def test_legacy_no_kill_without_night_identity_is_not_counted() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state(events=[
        GameEvent(type="wolf_no_kill_timeout", payload={}),
        GameEvent(
            type="wolf_no_kill_declared",
            payload={"wolf_action_reason": "old checkpoint"},
        ),
    ])

    result = NoKillPolicy().resolve(gs, reason_code="true_tie")

    decision = result["game_state"].events[-1].payload["no_kill_decision"]
    assert decision["consecutive_pre_resolution_no_kill_count"] == 1
    assert decision["forced_recovery_applied"] is False


def test_legacy_adapter_rejects_unknown_visibility_alias() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state(events=[
        GameEvent(
            type="wolf_plan_invalid_no_kill",
            payload={"night_number": 1, "visibility": "forged-private"},
        ),
        GameEvent(
            type="wolf_plan_invalid_no_kill",
            payload={"night_number": 2, "visibility": "forged-private"},
        ),
    ])

    result = NoKillPolicy().resolve(gs, reason_code="true_tie")

    decision = result["game_state"].events[-1].payload["no_kill_decision"]
    assert decision["consecutive_pre_resolution_no_kill_count"] == 1


@pytest.mark.parametrize(
    "reserved_key",
    [
        "night_number",
        "reason",
        "reason_code",
        "no_kill_decision",
        "target_id",
        "original_reasons",
        "consecutive_pre_resolution_no_kill_count",
        "forced_recovery_applied",
        "recovered_target_id",
        "candidate_scores",
        "final_target_id",
    ],
)
def test_extra_payload_cannot_overwrite_canonical_no_kill_fields(
    reserved_key: str,
) -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state()

    with pytest.raises(ValueError, match="reserved no-kill payload"):
        NoKillPolicy().resolve(
            gs,
            reason_code="true_tie",
            extra_payload={reserved_key: "forged"},
        )

    assert gs.events == []


def test_current_night_v1_choice_requires_explicit_safe_legacy_shape() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import (
        _current_night_wolf_choice,
    )

    valid = _game_state(events=[GameEvent(
        type="wolf_kill_selected",
        payload={"night_number": 3, "target_id": "p1"},
    )])
    illegal = _game_state(events=[GameEvent(
        type="wolf_kill_selected",
        payload={"night_number": 3, "target_id": "w1"},
    )])
    forged = _game_state(events=[GameEvent(
        type="unrelated_event",
        payload={"night_number": 3, "target_id": "p1"},
    )])

    assert _current_night_wolf_choice(valid) == (True, "p1")
    assert _current_night_wolf_choice(illegal) == (False, None)
    assert _current_night_wolf_choice(forged) == (False, None)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: replace(event, game_id="other-game"),
        lambda event: replace(event, event_id="forged:e999999"),
        lambda event: replace(event, visibility=EventVisibility.PUBLIC),
        lambda event: replace(event, payload={**event.payload, "reason": "true_tie"}),
        lambda event: replace(event, payload={**event.payload, "no_kill_decision": {}}),
    ],
    ids=[
        "cross_game",
        "noncanonical_event_id",
        "public_visibility",
        "reason_mismatch",
        "incomplete_decision",
    ],
)
def test_current_night_v2_no_kill_requires_trusted_complete_schema(mutate) -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import (
        NoKillPolicy,
        _current_night_wolf_choice,
    )

    resolved = NoKillPolicy().resolve(_game_state(), reason_code="strategic_abstain")
    gs = resolved["game_state"]
    gs = replace(gs, events=[mutate(gs.events[-1])])

    assert _current_night_wolf_choice(gs) == (False, None)


@pytest.mark.parametrize(
    "target_id",
    ["w1", "missing"],
)
def test_current_night_v2_selected_target_must_be_legal(target_id: str) -> None:
    from werewolf_agent.runtime.event_metadata import new_game_event
    from werewolf_agent.runtime.wolf_no_kill_policy import (
        _current_night_wolf_choice,
    )

    gs = _game_state()
    selected = new_game_event(
        gs,
        "wolf_kill_selected",
        {"night_number": 3, "target_id": target_id},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )
    gs = replace(gs, events=[selected])

    assert _current_night_wolf_choice(gs) == (False, None)


def test_current_night_v2_selected_rejects_unexpected_malformed_decision() -> None:
    from werewolf_agent.runtime.event_metadata import new_game_event
    from werewolf_agent.runtime.wolf_no_kill_policy import (
        _current_night_wolf_choice,
    )

    gs = _game_state()
    selected = new_game_event(
        gs,
        "wolf_kill_selected",
        {
            "night_number": 3,
            "target_id": "p1",
            "no_kill_decision": {},
        },
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )
    gs = replace(gs, events=[selected])

    assert _current_night_wolf_choice(gs) == (False, None)


def test_duplicate_v2_identity_cannot_suppress_current_night_action() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import (
        NoKillPolicy,
        _current_night_wolf_choice,
    )

    resolved = NoKillPolicy().resolve(_game_state(), reason_code="true_tie")
    gs = resolved["game_state"]
    gs = replace(gs, events=[gs.events[0], gs.events[0]])

    assert _current_night_wolf_choice(gs) == (False, None)


def test_legacy_consensus_new_selected_write_is_private_v2() -> None:
    from werewolf_agent.runtime.nodes.wolf_consensus import (
        _legacy_wolf_consensus,
    )

    result = _legacy_wolf_consensus({
        "game_state": _game_state(),
        "wolf_action": "kill",
        "wolf_kill_target_id": "p1",
    })

    event = result["game_state"].events[-1]
    assert event.type == "wolf_kill_selected"
    assert event.schema_version == "2"
    assert event.visibility is EventVisibility.WEREWOLF_TEAM_ONLY
    assert event.game_id == result["game_state"].game_id


def test_forced_recovery_without_legal_target_keeps_auditable_count() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state(events=[
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={"night_number": 1, "reason": "true_tie"},
        ),
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={"night_number": 2, "reason": "invalid_backup"},
        ),
    ])
    gs = replace(
        gs,
        players={
            player_id: (
                player
                if player.role == "werewolf"
                else replace(player, alive=False)
            )
            for player_id, player in gs.players.items()
        },
    )

    result = NoKillPolicy().resolve(gs, reason_code="plan_generation_failed")

    assert result["wolf_kill_target_id"] is None
    event = result["game_state"].events[-1]
    assert event.type == "forced_recovery_no_legal_target"
    assert event.visibility is EventVisibility.MODERATOR_ONLY
    assert event.trace_id == _expected_trace_id(
        result["game_state"], "forced_recovery_no_legal_target", 0
    )
    assert event.payload["consecutive_pre_resolution_no_kill_count"] == 3
    assert event.payload["original_reasons"] == [
        "true_tie",
        "invalid_backup",
        "plan_generation_failed",
    ]
    assert event.payload["candidate_scores"] == {}
    assert event.payload["final_target_id"] is None
    assert event.payload["no_kill_decision"]["forced_recovery_applied"] is True


def test_ruleset_loader_exposes_validated_no_kill_threshold(tmp_path) -> None:
    from werewolf_agent.engine.ruleset_loader import load_ruleset_from_yaml

    path = tmp_path / "ruleset.yaml"
    path.write_text(
        "\n".join([
            "player_count: 2",
            "roles:",
            "  werewolf:",
            "    count: 1",
            "  villager:",
            "    count: 1",
            "constraints:",
            "  max_consecutive_pre_resolution_no_kill: 2",
        ]),
        encoding="utf-8",
    )

    ruleset = load_ruleset_from_yaml(path)

    assert ruleset.max_consecutive_pre_resolution_no_kill == 2


def test_custom_ruleset_validator_normalizes_no_kill_threshold() -> None:
    from werewolf_agent.customization.validators import validate_ruleset_yaml

    result = validate_ruleset_yaml(
        "\n".join([
            "ruleset_id: no_kill_threshold",
            "player_count: 2",
            "roles:",
            "  werewolf:",
            "    count: 1",
            "  villager:",
            "    count: 1",
            "constraints:",
            "  max_consecutive_pre_resolution_no_kill: 2",
        ])
    )

    assert result.valid
    assert (
        result.normalized["constraints"]
        ["max_consecutive_pre_resolution_no_kill"]
        == 2
    )


def test_custom_ruleset_validator_rejects_invalid_no_kill_threshold() -> None:
    from werewolf_agent.customization.validators import validate_ruleset_yaml

    result = validate_ruleset_yaml(
        "\n".join([
            "ruleset_id: no_kill_threshold",
            "player_count: 1",
            "roles:",
            "  werewolf:",
            "    count: 1",
            "constraints:",
            "  max_consecutive_pre_resolution_no_kill: 0",
        ])
    )

    assert not result.valid
    assert any(
        issue.field
        == "constraints.max_consecutive_pre_resolution_no_kill"
        for issue in result.errors
    )


def _runtime_state_with_authoritative_stances(
    stances: tuple[tuple[str, str | None, str, str], ...],
    *,
    fallback_reason: str | None = None,
    fallback_night: int = 1,
    fallback_visibility: EventVisibility = EventVisibility.WEREWOLF_TEAM_ONLY,
    forged_fallback: bool = False,
    dead_after_stances: tuple[str, ...] = (),
) -> dict[str, Any]:
    from werewolf_agent.runtime.event_metadata import new_game_event
    from werewolf_agent.runtime.wolf_discussion_directives import (
        build_validated_wolf_target_stance,
    )

    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "w3": PlayerState(id="w3", role="werewolf"),
        "p1": PlayerState(id="p1", role="villager"),
        "p2": PlayerState(id="p2", role="seer"),
        "p3": PlayerState(id="p3", role="witch"),
    }
    gs = GameState(
        game_id="route-no-kill-reasons",
        players=players,
        night_number=1,
    )
    if fallback_reason is not None:
        payload = {
            "night_number": fallback_night,
            "reason": fallback_reason,
        }
        fallback = (
            GameEvent(
                type="wolf_team_plan_fallback",
                payload={**payload, "visibility": fallback_visibility.value},
            )
            if forged_fallback
            else new_game_event(
                gs,
                "wolf_team_plan_fallback",
                payload,
                visibility=fallback_visibility,
            )
        )
        gs = replace(gs, events=[*gs.events, fallback])

    for round_number, (wolf_id, target_id, stance, priority) in enumerate(
        stances,
        start=1,
    ):
        payload = {
            "wolf_id": wolf_id,
            "round": round_number,
            "night_number": 1,
            "text": "",
        }
        discussion = new_game_event(
            gs,
            "wolf_discussion",
            payload,
            visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        )
        target_stance = build_validated_wolf_target_stance(
            gs,
            discussion,
            wolf_id=wolf_id,
            round_number=round_number,
            raw_stance={
                "target_id": target_id,
                "stance": stance,
                "priority": priority,
            },
        )
        discussion = replace(
            discussion,
            payload={
                **payload,
                "target_stance": target_stance.model_dump(),
            },
        )
        gs = replace(gs, events=[*gs.events, discussion])

    if dead_after_stances:
        gs = replace(
            gs,
            players={
                player_id: (
                    replace(player, alive=False)
                    if player_id in dead_after_stances
                    else player
                )
                for player_id, player in gs.players.items()
            },
        )
    return {"game_state": gs}


@pytest.mark.parametrize(
    ("state", "runtime_function", "expected_reason"),
    [
        (
            _runtime_state_with_authoritative_stances((
                ("w1", None, "abstain", "primary"),
                ("w2", None, "abstain", "primary"),
                ("w3", None, "abstain", "primary"),
            )),
            "planned",
            "strategic_abstain",
        ),
        (
            _runtime_state_with_authoritative_stances((
                ("w1", "p1", "support", "primary"),
                ("w2", "p2", "support", "primary"),
                ("w3", "p3", "support", "primary"),
            )),
            "planned",
            "true_tie",
        ),
        (
            _runtime_state_with_authoritative_stances((
                ("w1", "p1", "support", "primary"),
            )),
            "planned",
            "insufficient_quorum",
        ),
        (
            _runtime_state_with_authoritative_stances(
                (
                    ("w1", "p1", "support", "primary"),
                    ("w2", "p1", "support", "primary"),
                    ("w1", "p2", "support", "backup"),
                    ("w2", "p2", "support", "backup"),
                ),
                dead_after_stances=("p1", "p2"),
            ),
            "planned",
            "invalid_backup",
        ),
        (
            _runtime_state_with_authoritative_stances(()),
            "legacy_invalid",
            "invalid_primary",
        ),
        (
            _runtime_state_with_authoritative_stances(
                (),
                fallback_reason="schema_validation_failed",
            ),
            "planned",
            "plan_generation_failed",
        ),
        (
            _runtime_state_with_authoritative_stances(
                (),
                fallback_reason="llm_failed_or_unavailable",
            ),
            "planned",
            "provider_unavailable",
        ),
    ],
)
def test_all_reason_codes_flow_through_real_runtime_routes(
    state: dict[str, Any],
    runtime_function: str,
    expected_reason: str,
) -> None:
    if runtime_function == "planned":
        from werewolf_agent.runtime.nodes.node_helpers import _planned_wolf_kill

        result = _planned_wolf_kill(state)
    else:
        from werewolf_agent.runtime.nodes.wolf_consensus import (
            _legacy_wolf_consensus,
        )

        result = _legacy_wolf_consensus({
            **state,
            "wolf_action": "kill",
            "wolf_kill_target_id": "missing",
        })

    assert result is not None
    event = result["game_state"].events[-1]
    assert event.payload["reason"] == expected_reason
    assert event.payload["no_kill_decision"]["reason_code"] == expected_reason


@pytest.mark.parametrize(
    "state",
    [
        _runtime_state_with_authoritative_stances(
            (),
            fallback_reason="schema_validation_failed",
            fallback_night=0,
        ),
        _runtime_state_with_authoritative_stances(
            (),
            fallback_reason="schema_validation_failed",
            fallback_visibility=EventVisibility.PUBLIC,
        ),
        _runtime_state_with_authoritative_stances(
            (),
            fallback_reason="schema_validation_failed",
            forged_fallback=True,
        ),
    ],
)
def test_untrusted_fallback_metadata_cannot_override_genuine_abstain(
    state: dict[str, Any],
) -> None:
    from werewolf_agent.runtime.nodes.node_helpers import _planned_wolf_kill

    result = _planned_wolf_kill(state)

    assert result is not None
    event = result["game_state"].events[-1]
    assert event.payload["reason"] == "strategic_abstain"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: replace(event, game_id="other-game"),
        lambda event: replace(event, event_id="forged:e999999"),
    ],
    ids=["cross_game", "noncanonical_event_id"],
)
def test_failure_classifier_rejects_mismatched_v2_fallback_identity(
    mutate,
) -> None:
    state = _runtime_state_with_authoritative_stances(
        (),
        fallback_reason="schema_validation_failed",
    )
    gs = state["game_state"]
    gs = replace(gs, events=[mutate(gs.events[0]), *gs.events[1:]])

    from werewolf_agent.runtime.nodes.node_helpers import (
        _trusted_wolf_plan_failure_reason,
    )

    assert _trusted_wolf_plan_failure_reason(gs) is None


@pytest.mark.parametrize("corruption", ["duplicate", "out_of_order"])
def test_failure_classifier_rejects_non_authoritative_v2_event_log(
    corruption: str,
) -> None:
    from werewolf_agent.runtime.event_metadata import new_game_event
    from werewolf_agent.runtime.nodes.node_helpers import (
        _trusted_wolf_plan_failure_reason,
    )

    gs = _runtime_state_with_authoritative_stances(
        (),
        fallback_reason="schema_validation_failed",
    )["game_state"]
    marker = new_game_event(
        gs,
        "wolf_consensus_plan_mismatch",
        {"night_number": 1},
        visibility=EventVisibility.MODERATOR_ONLY,
    )
    if corruption == "duplicate":
        marker = replace(
            marker,
            event_id=gs.events[0].event_id,
            sequence_number=gs.events[0].sequence_number,
        )
        events = [*gs.events, marker]
    else:
        events = [marker, *gs.events]
    gs = replace(gs, events=events)

    assert _trusted_wolf_plan_failure_reason(gs) is None


def test_trusted_failure_metadata_cannot_authorize_its_own_target() -> None:
    state = _runtime_state_with_authoritative_stances(
        (
            ("w1", "p1", "support", "primary"),
            ("w2", "p1", "support", "primary"),
        ),
        fallback_reason="schema_validation_failed",
    )
    gs = state["game_state"]
    fallback = replace(
        gs.events[0],
        payload={**gs.events[0].payload, "target_id": "p3"},
    )
    state = {"game_state": replace(gs, events=[fallback, *gs.events[1:]])}

    from werewolf_agent.runtime.nodes.node_helpers import _planned_wolf_kill

    result = _planned_wolf_kill(state)

    assert result is not None
    assert result["wolf_kill_target_id"] == "p1"
    selected = result["game_state"].events[-1]
    assert selected.payload["target_id"] == "p1"
    assert selected.schema_version == "2"
    assert selected.visibility is EventVisibility.WEREWOLF_TEAM_ONLY


@pytest.mark.parametrize("value", [True, 0, -1, "2"])
def test_ruleset_loader_rejects_invalid_no_kill_threshold(tmp_path, value) -> None:
    from werewolf_agent.engine.ruleset_loader import load_ruleset_from_yaml

    path = tmp_path / "ruleset.yaml"
    path.write_text(
        "\n".join([
            "player_count: 2",
            "roles: {}",
            "constraints:",
            f"  max_consecutive_pre_resolution_no_kill: {value!r}",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_consecutive_pre_resolution_no_kill"):
        load_ruleset_from_yaml(path)
