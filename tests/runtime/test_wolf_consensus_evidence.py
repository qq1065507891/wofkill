# -*- coding: utf-8 -*-
"""
验证狼人结构化立场的确定性主备刀聚合。

作者: Project contributors
创建日期: 2026-07-16
"""

from __future__ import annotations

import pickle
from dataclasses import replace

import pytest

from werewolf_agent.agents.schemas import WolfTargetStance
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameState, PlayerState


def _stance(
    wolf_id: str,
    target_id: str | None,
    *,
    stance: str = "support",
    priority: str = "primary",
    round_number: int = 1,
) -> WolfTargetStance:
    return WolfTargetStance.model_validate({
        "wolf_id": wolf_id,
        "target_id": target_id,
        "stance": stance,
        "priority": priority,
        "source_event_id": f"g:e{round_number:06d}",
        "round_number": round_number,
    })


def _derive(*stances: WolfTargetStance, wolves: tuple[str, ...] = ("w1", "w2", "w3")):
    from werewolf_agent.runtime.wolf_consensus_evidence import (
        derive_wolf_consensus_evidence,
    )

    return derive_wolf_consensus_evidence(
        night_number=1,
        alive_wolf_ids=wolves,
        stances=stances,
    )


def test_three_wolves_two_to_one_produces_strict_primary_majority() -> None:
    evidence = _derive(
        _stance("w1", "p1", stance="propose"),
        _stance("w2", "p1"),
        _stance("w3", "p2"),
    )

    assert evidence.quorum == 2
    assert evidence.primary.status == "majority"
    assert evidence.primary.target_id == "p1"
    assert evidence.primary.supporters_by_target == {
        "p1": ("w1", "w2"),
        "p2": ("w3",),
    }


def test_three_wolves_one_one_one_is_true_tie_without_target() -> None:
    evidence = _derive(
        _stance("w1", "p1"),
        _stance("w2", "p2"),
        _stance("w3", "p3"),
    )

    assert evidence.primary.status == "tie"
    assert evidence.primary.target_id is None


def test_three_wolves_one_positive_vote_is_insufficient_without_target() -> None:
    evidence = _derive(_stance("w1", "p1"))

    assert evidence.primary.status == "insufficient"
    assert evidence.primary.target_id is None
    assert evidence.primary.supporters_by_target == {"p1": ("w1",)}


def test_all_oppose_and_abstain_is_all_abstain_for_each_priority() -> None:
    evidence = _derive(
        _stance("w1", "p1", stance="oppose"),
        _stance("w2", None, stance="abstain"),
        _stance("w3", None, stance="abstain", priority="backup"),
    )

    assert evidence.primary.status == "all_abstain"
    assert evidence.backup.status == "all_abstain"
    assert evidence.primary.supporters_by_target == {}


def test_single_wolf_primary_and_backup_are_independent_single_wolf_results() -> None:
    evidence = _derive(
        _stance("w1", "p1", stance="propose"),
        _stance("w1", "p2", stance="support", priority="backup"),
        wolves=("w1",),
    )

    assert evidence.quorum == 1
    assert (evidence.primary.status, evidence.primary.target_id) == (
        "single_wolf",
        "p1",
    )
    assert (evidence.backup.status, evidence.backup.target_id) == (
        "single_wolf",
        "p2",
    )


def test_latest_stance_replaces_old_support_in_same_wolf_priority_slot() -> None:
    evidence = _derive(
        _stance("w1", "p1", round_number=1),
        _stance("w2", "p1", round_number=1),
        _stance("w1", "p1", stance="oppose", round_number=2),
    )

    assert evidence.primary.status == "insufficient"
    assert evidence.primary.target_id is None
    assert evidence.primary.supporters_by_target == {"p1": ("w2",)}
    assert len(evidence.stances) == 3


def test_primary_majority_does_not_compare_peer_level_backup_tie() -> None:
    evidence = _derive(
        _stance("w1", "p1"),
        _stance("w2", "p1"),
        _stance("w1", "p2", priority="backup"),
        _stance("w2", "p3", priority="backup"),
    )

    assert evidence.primary.status == "majority"
    assert evidence.primary.target_id == "p1"
    assert evidence.backup.status == "tie"
    assert evidence.backup.target_id is None


def test_consensus_evidence_is_immutable_at_collection_boundaries() -> None:
    evidence = _derive(_stance("w1", "p1"), _stance("w2", "p1"))

    with pytest.raises(TypeError):
        evidence.primary.supporters_by_target["p1"] = ("w3",)
    with pytest.raises(AttributeError):
        evidence.alive_wolf_ids = ("w9",)
    with pytest.raises(Exception, match="frozen"):
        evidence.stances[0].target_id = "p9"


def test_two_simultaneous_strict_majorities_fail_closed() -> None:
    from werewolf_agent.runtime.wolf_consensus_evidence import (
        ConsensusInvariantViolation,
        _consensus_from_supporters,
    )

    with pytest.raises(
        ConsensusInvariantViolation,
        match="consensus_invariant_violation",
    ):
        _consensus_from_supporters(
            priority="primary",
            alive_wolf_count=3,
            supporters_by_target={
                "p1": {"w1", "w2"},
                "p2": {"w2", "w3"},
            },
        )


def _runtime_state_with_stances(
    entries: tuple[tuple[str, str, str], ...],
) -> dict[str, object]:
    from werewolf_agent.runtime.event_metadata import new_game_event
    from werewolf_agent.runtime.wolf_discussion_directives import (
        build_validated_wolf_target_stance,
    )

    players = {
        **{
            wolf_id: PlayerState(id=wolf_id, role="werewolf", alive=True)
            for wolf_id in ("w1", "w2", "w3")
        },
        "p1": PlayerState(id="p1", role="villager", alive=True),
        "p2": PlayerState(id="p2", role="seer", alive=True),
    }
    gs = GameState(game_id="authoritative-runtime", players=players, night_number=1)
    for round_number, (wolf_id, target_id, stance_name) in enumerate(
        entries,
        start=1,
    ):
        payload = {
            "wolf_id": wolf_id,
            "round": round_number,
            "night_number": 1,
            "text": "",
        }
        event = new_game_event(
            gs,
            "wolf_discussion",
            payload,
            visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        )
        stance = build_validated_wolf_target_stance(
            gs,
            event,
            wolf_id=wolf_id,
            round_number=round_number,
            raw_stance={
                "target_id": target_id,
                "stance": stance_name,
                "priority": "primary",
            },
        )
        event = replace(
            event,
            payload={**payload, "target_stance": stance.model_dump()},
        )
        gs = replace(gs, events=[*gs.events, event])
    return {"game_state": gs}


def test_no_plan_valid_stance_majority_still_executes_authoritative_target() -> None:
    from werewolf_agent.runtime.nodes.node_helpers import _planned_wolf_kill

    state = _runtime_state_with_stances((
        ("w1", "p1", "support"),
        ("w2", "p1", "support"),
        ("w3", "p2", "support"),
    ))

    result = _planned_wolf_kill(state)

    assert result is not None
    assert result["wolf_kill_target_id"] == "p1"


def test_execution_reaggregates_later_stance_changes_instead_of_using_cache() -> None:
    from werewolf_agent.runtime.nodes.node_helpers import _planned_wolf_kill

    initial = _runtime_state_with_stances((
        ("w1", "p1", "support"),
        ("w2", "p1", "support"),
        ("w3", "p2", "support"),
    ))
    initial_result = _planned_wolf_kill(initial)
    assert initial_result is not None
    assert initial_result["wolf_kill_target_id"] == "p1"

    changed = _runtime_state_with_stances((
        ("w1", "p1", "support"),
        ("w2", "p1", "support"),
        ("w3", "p2", "support"),
        ("w1", "p2", "support"),
        ("w2", "p2", "support"),
    ))
    changed["wolf_consensus_evidence"] = _derive(
        _stance("w1", "p1"),
        _stance("w2", "p1"),
        _stance("w3", "p2"),
    )

    changed_result = _planned_wolf_kill(changed)

    assert changed_result is not None
    assert changed_result["wolf_kill_target_id"] == "p2"


def test_hand_constructed_cached_majority_cannot_authorize_without_events() -> None:
    from werewolf_agent.runtime.nodes.node_helpers import _planned_wolf_kill
    from werewolf_agent.runtime.wolf_consensus_evidence import (
        WolfConsensusEvidenceV2,
        WolfPriorityConsensus,
    )

    state = _runtime_state_with_stances(())
    state["wolf_consensus_evidence"] = WolfConsensusEvidenceV2(
        night_number=1,
        alive_wolf_ids=("w1", "w2", "w3"),
        stances=(),
        quorum=2,
        primary=WolfPriorityConsensus(
            priority="primary",
            target_id="p1",
            status="majority",
            supporters_by_target={},
        ),
        backup=WolfPriorityConsensus(
            priority="backup",
            target_id=None,
            status="all_abstain",
            supporters_by_target={},
        ),
    )

    result = _planned_wolf_kill(state)

    assert result is not None
    assert result["wolf_kill_target_id"] is None
    assert result["game_state"].events[-1].payload["reason"] == "strategic_abstain"


def test_evidence_is_pickle_safe_and_retained_by_real_checkpointed_graph() -> None:
    from werewolf_agent.runtime.checkpoints import make_checkpointer
    from werewolf_agent.runtime.graph import build_game_graph_with_checkpoint
    from werewolf_agent.runtime.wolf_consensus_evidence import (
        deserialize_wolf_consensus_evidence,
        serialize_wolf_consensus_evidence,
    )

    evidence = _derive(_stance("w1", "p1"), _stance("w2", "p1"))
    assert pickle.loads(pickle.dumps(evidence)) == evidence
    serialized = serialize_wolf_consensus_evidence(evidence)
    assert deserialize_wolf_consensus_evidence(serialized) == evidence

    graph = build_game_graph_with_checkpoint(make_checkpointer())
    config = {"configurable": {"thread_id": "wolf-consensus-channel"}}
    graph.update_state(
        config,
        {"wolf_consensus_evidence": serialized},
        as_node="wolf_team_plan",
    )
    restored = graph.get_state(config)

    assert restored.values["wolf_consensus_evidence"] == serialized
    assert deserialize_wolf_consensus_evidence(
        restored.values["wolf_consensus_evidence"]
    ) == evidence
