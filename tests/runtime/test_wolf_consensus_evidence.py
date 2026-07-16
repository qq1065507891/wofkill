# -*- coding: utf-8 -*-
"""
验证狼人结构化立场的确定性主备刀聚合。

作者: Project contributors
创建日期: 2026-07-16
"""

from __future__ import annotations

import pytest

from werewolf_agent.agents.schemas import WolfTargetStance


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
