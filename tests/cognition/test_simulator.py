# -*- coding: utf-8 -*-
"""
验证有界模拟器仅导出有公开证据支撑的可能世界引用。

作者: Project contributors
修改日期: 2026-07-13
"""

from __future__ import annotations

from werewolf_agent.cognition.worlds import PossibleWorld, PossibleWorldSet


def _worlds() -> PossibleWorldSet:
    return PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=9,
        worlds=[
            PossibleWorld(
                world_id="World A",
                probability=0.7,
                roles={
                    "p01": "villager",
                    "p02": "werewolf",
                    "p03": "seer",
                    "p04": "villager",
                },
                supporting_evidence=["event:g:1"],
            ),
            PossibleWorld(
                world_id="World B",
                probability=0.3,
                roles={
                    "p01": "villager",
                    "p02": "villager",
                    "p03": "seer",
                    "p04": "werewolf",
                },
                supporting_evidence=["event:g:1"],
            ),
        ],
        marginal_role_probs={},
        public_evidence_ids={"event:g:1"},
    )


def test_simulator_returns_bounded_prediction_cards() -> None:
    from werewolf_agent.cognition.simulator import BoundedSimulator

    result = BoundedSimulator().simulate(
        viewer_id="p01",
        possible_worlds=_worlds(),
        alive_players=["p01", "p02", "p03", "p04"],
        day_number=2,
        pressure_summaries={
            "p02": {"pressure_score": 1.4, "defense_score": 0.2},
            "p04": {"pressure_score": 0.3, "defense_score": 0.1},
        },
        top_k=2,
    )

    assert result.viewer_id == "p01"
    assert result.horizon == "next_turn"
    assert 1 <= len(result.predictions) <= 2
    assert all(0.0 <= item.probability <= 1.0 for item in result.predictions)
    assert result.predictions[0].event_type == "next_day_vote_pressure"
    assert result.predictions[0].affected_players == ["p02"]
    assert result.predictions[0].world_ids

    prompt_dict = result.to_prompt_dict()

    assert prompt_dict["type"] == "simulation"
    assert prompt_dict["warning"] == "Prediction, not fact."
    assert "roles" not in str(prompt_dict)
    assert "world_" in str(prompt_dict)


def test_simulator_omits_invalid_or_empty_inputs() -> None:
    from werewolf_agent.cognition.simulator import BoundedSimulator

    result = BoundedSimulator().simulate(
        viewer_id="p01",
        possible_worlds=None,
        alive_players=[],
        day_number=1,
    )

    assert result.predictions == []
    assert result.to_prompt_dict()["predictions"] == []


def test_simulator_rejects_worlds_without_known_public_evidence() -> None:
    from werewolf_agent.cognition.simulator import BoundedSimulator

    worlds = PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=1,
        worlds=[PossibleWorld("unknown", 1.0, {"p01": "villager", "p02": "werewolf"}, supporting_evidence=["event:g:404"])],
        marginal_role_probs={},
        public_evidence_ids={"event:g:1"},
    )
    result = BoundedSimulator().simulate(
        viewer_id="p01",
        possible_worlds=worlds,
        alive_players=["p01", "p02", "p03", "p04"],
        day_number=1,
    )

    assert result.predictions == []


def test_simulation_export_drops_prediction_with_any_unknown_world_id() -> None:
    from werewolf_agent.cognition.simulator import (
        FutureEventPrediction,
        SimulationResult,
    )

    result = SimulationResult(
        viewer_id="p01",
        horizon="next_turn",
        predictions=[
            FutureEventPrediction(
                event_type="next_day_vote_pressure",
                probability=0.8,
                affected_players=["p02"],
                world_ids=["World A", "unknown", "unknown"],
            ),
            FutureEventPrediction(
                event_type="night_kill_pressure",
                probability=0.6,
                affected_players=["p03"],
                world_ids=["World B", "World A", "World B"],
            ),
        ],
        retained_promptable_world_ids=["World A", "World B"],
    )

    exported = result.to_prompt_dict()

    assert [item["event"] for item in exported["predictions"]] == [
        "night_kill_pressure"
    ]
    assert exported["predictions"][0]["world_ids"] == ["World B", "World A"]
    assert exported["rejected_unknown_world_id_count"] == 2


def test_simulation_export_drops_unknown_only_prediction() -> None:
    from werewolf_agent.cognition.simulator import (
        FutureEventPrediction,
        SimulationResult,
    )

    result = SimulationResult(
        viewer_id="p01",
        horizon="next_turn",
        predictions=[
            FutureEventPrediction(
                event_type="next_day_vote_pressure",
                probability=0.7,
                affected_players=["p02"],
                world_ids=["unknown"],
            )
        ],
        retained_promptable_world_ids=["World A"],
    )

    exported = result.to_prompt_dict()

    assert exported["predictions"] == []
    assert exported["rejected_unknown_world_id_count"] == 1


def test_bounded_simulator_exports_only_retained_promptable_world_ids() -> None:
    from werewolf_agent.cognition.simulator import BoundedSimulator

    result = BoundedSimulator().simulate(
        viewer_id="p01",
        possible_worlds=_worlds(),
        alive_players=["p01", "p02", "p03", "p04"],
        day_number=2,
        pressure_summaries={"p02": {"pressure_score": 1.4}},
        top_k=2,
    )

    exported = result.to_prompt_dict()
    allowed_order = [world.world_id for world in _worlds().promptable_worlds()]
    allowed = set(allowed_order)

    assert result.retained_promptable_world_ids == allowed_order
    assert all(
        set(prediction["world_ids"]).issubset(allowed)
        for prediction in exported["predictions"]
    )
    assert exported["rejected_unknown_world_id_count"] == 0
