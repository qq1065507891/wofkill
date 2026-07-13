# -*- coding: utf-8 -*-
"""验证可能世界的规范化、概率、证据门控与不可变边界。

作者: Project contributors
修改日期: 2026-07-13
"""

from __future__ import annotations

import math

import pytest

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


def test_canonical_identity_does_not_merge_distinct_assignments_on_hash_collision(monkeypatch) -> None:
    """完整规范字节必须参与等价判断，不能只依赖截断哈希。"""
    import werewolf_agent.cognition.worlds as worlds_module

    class _CollisionHash:
        def hexdigest(self) -> str:
            return "0" * 64

    monkeypatch.setattr(worlds_module.hashlib, "sha256", lambda _value: _CollisionHash())
    result = PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=1,
        worlds=[
            PossibleWorld("a", 0.4, {"p01": "seer", "p02": "werewolf"}),
            PossibleWorld("b", 0.6, {"p01": "seer", "p02": "villager"}),
        ],
        marginal_role_probs={},
    )

    assert len(result.worlds) == 2
    assert len({world.world_id for world in result.worlds}) == 2


def test_assignment_normalization_precedes_sort_and_rejects_duplicate_player() -> None:
    normalized = PossibleWorldSet(
        viewer_id="P01",
        generated_at_event_index=1,
        worlds=[PossibleWorld("x", 1.0, {1: " SEER ", " P02 ": "WEREWOLF"})],
        marginal_role_probs={},
    )
    assert dict(normalized.worlds[0].roles) == {"1": "seer", "p02": "werewolf"}

    with pytest.raises(ValueError, match="duplicate player"):
        PossibleWorldSet(
            viewer_id="p01",
            generated_at_event_index=1,
            worlds=[PossibleWorld("x", 1.0, {"P02": "seer", " p02 ": "werewolf"})],
            marginal_role_probs={},
        )


@pytest.mark.parametrize("probability", [math.nan, math.inf, -math.inf, -0.01])
def test_invalid_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        PossibleWorldSet(
            viewer_id="p01",
            generated_at_event_index=1,
            worlds=[PossibleWorld("x", probability, {"p01": "seer"})],
            marginal_role_probs={},
        )


def test_zero_total_is_uniform_and_marginals_are_recomputed_after_dedup() -> None:
    result = PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=1,
        worlds=[
            PossibleWorld("a", 0.0, {"p01": "seer", "p02": "werewolf"}),
            PossibleWorld("b", 0.0, {"p01": "seer", "p02": "villager"}),
        ],
        marginal_role_probs={"p02": {"hunter": 99.0}},
    )
    assert [world.probability for world in result.worlds] == [0.5, 0.5]
    assert result.marginal_role_probs["p02"] == {"villager": 0.5, "werewolf": 0.5}


def test_possible_world_values_are_deeply_immutable() -> None:
    result = PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=1,
        worlds=[PossibleWorld(
            "x", 1.0, {"p01": "seer"},
            score_breakdown={"base": 1.0},
            supporting_evidence=["claim:g:1"],
        )],
        marginal_role_probs={},
        public_evidence_ids={"claim:g:1"},
    )
    world = result.worlds[0]
    with pytest.raises(TypeError):
        world.roles["p01"] = "werewolf"
    with pytest.raises(AttributeError):
        world.supporting_evidence.append("claim:g:2")
    with pytest.raises(AttributeError):
        result.worlds.append(world)
    with pytest.raises(AttributeError):
        result.public_evidence_ids.add("claim:g:2")
    with pytest.raises(TypeError):
        result.marginal_role_probs["p01"]["seer"] = 0.0


def test_no_public_evidence_exports_uniform_faction_hypothesis_only() -> None:
    worlds = PossibleWorldsEngine().generate(
        viewer_id="p01",
        viewer_role="seer",
        player_ids=["p01", "p02", "p03"],
        role_counts={"seer": 1, "werewolf": 1, "villager": 1},
    )
    prompt = worlds.to_prompt_dict()

    assert prompt["top_worlds"] == []
    assert prompt["faction_hypothesis"]["good"] + prompt["faction_hypothesis"]["werewolf"] == 1.0
    assert not any(player_id in str(prompt) for player_id in ("p01", "p02", "p03"))
    assert not any(role in str(prompt).lower() for role in ("seer", "witch", "hunter", "idiot"))
