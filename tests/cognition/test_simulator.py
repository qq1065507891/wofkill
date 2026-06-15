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
                supporting_evidence=["p02 pressure aligns"],
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
            ),
        ],
        marginal_role_probs={},
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
    assert "World A" in str(prompt_dict)


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
