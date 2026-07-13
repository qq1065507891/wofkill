from __future__ import annotations

from werewolf_agent.cognition.worlds import PossibleWorld, PossibleWorldSet, PossibleWorldsEngine


def test_world_set_collapses_assignment_order_and_renormalizes() -> None:
    worlds = PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=2,
        worlds=[
            PossibleWorld("ignored-a", 0.2, {"p02": "werewolf", "p01": "seer"}, supporting_evidence=["event:g:1"]),
            PossibleWorld("ignored-b", 0.3, {"p01": "seer", "p02": "werewolf"}, supporting_evidence=["event:g:1"]),
            PossibleWorld("ignored-c", 0.5, {"p01": "seer", "p02": "villager"}, supporting_evidence=["event:g:1"]),
        ],
        marginal_role_probs={},
        public_evidence_ids={"event:g:1"},
    )

    assert len(worlds.worlds) == 2
    assert [world.probability for world in worlds.worlds] == [0.5, 0.5]
    assert all(world.world_id.startswith("world_") for world in worlds.worlds)
    assert worlds.worlds[0].world_id != worlds.worlds[1].world_id


def test_prompt_rejects_empty_unknown_and_duplicate_assignments() -> None:
    worlds = PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=2,
        worlds=[
            PossibleWorld("empty", 0.4, {"p01": "seer", "p02": "werewolf"}),
            PossibleWorld("unknown", 0.3, {"p01": "seer", "p03": "werewolf"}, supporting_evidence=["event:g:404"]),
            PossibleWorld("valid", 0.3, {"p01": "seer", "p04": "werewolf"}, supporting_evidence=["claim:g:1"]),
        ],
        marginal_role_probs={},
        public_evidence_ids={"claim:g:1"},
    )

    prompt = worlds.to_prompt_dict()

    assert len(prompt["top_worlds"]) == 1
    assert prompt["top_worlds"][0]["why"] == ["claim:g:1"]
    assert "p02" not in str(prompt) and "p03" not in str(prompt)


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
        belief_summary={
            "my_suspects": [{
                "player": "p02",
                "top_role_guess": "werewolf",
                "top_role_prob": 0.8,
                "evidence_ids": ["event:g:1"],
            }],
        },
        public_evidence_ids={"event:g:1"},
        top_k=1,
    )

    prompt_dict = worlds.to_prompt_dict()

    assert prompt_dict["warning"]
    assert "hypotheses" in prompt_dict["warning"]
    assert "roles" not in prompt_dict["top_worlds"][0]
    assert "key_assignments" in prompt_dict["top_worlds"][0]
