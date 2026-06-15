from __future__ import annotations

from werewolf_agent.cognition.worlds import PossibleWorldsEngine


def test_worlds_respect_role_counts() -> None:
    engine = PossibleWorldsEngine()

    worlds = engine.generate(
        viewer_id="p01",
        viewer_role="seer",
        player_ids=["p01", "p02", "p03", "p04"],
        role_counts={"seer": 1, "villager": 2, "werewolf": 1},
        top_k=3,
    )

    assert worlds.worlds
    for world in worlds.worlds:
        counts = {
            role: list(world.roles.values()).count(role)
            for role in set(world.roles.values())
        }
        assert counts == {"seer": 1, "villager": 2, "werewolf": 1}
        assert world.roles["p01"] == "seer"
    assert round(sum(world.probability for world in worlds.worlds), 6) == 1.0


def test_worlds_rank_belief_aligned_assignments_higher() -> None:
    engine = PossibleWorldsEngine()

    worlds = engine.generate(
        viewer_id="p01",
        viewer_role="seer",
        player_ids=["p01", "p02", "p03", "p04"],
        role_counts={"seer": 1, "villager": 2, "werewolf": 1},
        belief_summary={
            "my_suspects": [
                {
                    "player": "p03",
                    "top_role_guess": "werewolf",
                    "top_role_prob": 0.86,
                    "trust": 0.18,
                }
            ],
            "my_trusted": [],
        },
        top_k=3,
    )

    assert worlds.worlds[0].roles["p03"] == "werewolf"
    assert worlds.marginal_role_probs["p03"]["werewolf"] > 0.5


def test_worlds_prompt_dict_is_hypothesis_only() -> None:
    engine = PossibleWorldsEngine()
    worlds = engine.generate(
        viewer_id="p01",
        viewer_role="villager",
        player_ids=["p01", "p02", "p03", "p04"],
        role_counts={"villager": 3, "werewolf": 1},
        top_k=1,
    )

    prompt_dict = worlds.to_prompt_dict()

    assert prompt_dict["warning"]
    assert "hypotheses" in prompt_dict["warning"]
    assert "roles" not in prompt_dict["top_worlds"][0]
    assert "key_assignments" in prompt_dict["top_worlds"][0]
