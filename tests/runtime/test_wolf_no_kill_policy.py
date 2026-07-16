# -*- coding: utf-8 -*-
"""
验证所有结算前空刀统一经过确定性的 NoKillPolicy。

作者: Project contributors
创建日期: 2026-07-16

使用示例:
    >>> python -m pytest tests/runtime/test_wolf_no_kill_policy.py -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.core.event_visibility import EventVisibility


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
    assert selected.type == "wolf_kill_selected"
    assert selected.visibility is EventVisibility.WEREWOLF_TEAM_ONLY
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


def test_forced_recovery_without_legal_target_keeps_auditable_count() -> None:
    from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy

    gs = _game_state(events=[
        GameEvent(type="wolf_no_kill_timeout", payload={"reason": "true_tie"}),
        GameEvent(type="wolf_no_kill_timeout", payload={"reason": "invalid_backup"}),
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
