# -*- coding: utf-8 -*-
"""
从狼队私有结构化立场确定性聚合主刀与备刀共识。

作者: Project contributors
创建日期: 2026-07-16

使用示例:
    >>> derive_wolf_consensus_evidence(1, ("w1",), ())
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from pydantic import ConfigDict

from werewolf_agent.agents.schemas import WolfTargetStance

WolfPriority = Literal["primary", "backup"]
WolfConsensusStatus = Literal[
    "majority",
    "single_wolf",
    "tie",
    "insufficient",
    "all_abstain",
]


class ConsensusInvariantViolation(ValueError):
    """表示共识聚合违反一狼一槽位所保证的严格多数不变量。"""

    reason_code = "consensus_invariant_violation"

    def __init__(self, priority: WolfPriority, targets: Sequence[str]) -> None:
        self.priority = priority
        self.targets = tuple(sorted(targets))
        super().__init__(
            f"{self.reason_code}: priority={priority}, targets={self.targets}"
        )


class _ImmutableWolfTargetStance(WolfTargetStance):
    """隔离调用方可变对象，防止聚合后立场历史被原地篡改。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class WolfPriorityConsensus:
    """一个优先级上由当前有效立场形成的不可变聚合结果。"""

    priority: WolfPriority
    target_id: str | None
    status: WolfConsensusStatus
    supporters_by_target: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        frozen_supporters = {
            target_id: tuple(sorted(set(supporters)))
            for target_id, supporters in sorted(self.supporters_by_target.items())
        }
        object.__setattr__(
            self,
            "supporters_by_target",
            MappingProxyType(frozen_supporters),
        )


@dataclass(frozen=True)
class WolfConsensusEvidenceV2:
    """本夜主刀与备刀的完整不可变权威证据。"""

    night_number: int
    alive_wolf_ids: tuple[str, ...]
    stances: tuple[WolfTargetStance, ...]
    quorum: int
    primary: WolfPriorityConsensus
    backup: WolfPriorityConsensus


def _consensus_from_supporters(
    *,
    priority: WolfPriority,
    alive_wolf_count: int,
    supporters_by_target: Mapping[str, set[str]],
) -> WolfPriorityConsensus:
    """根据已去重支持者集合产生一个优先级的严格共识。"""
    normalized = {
        target_id: tuple(sorted(supporters))
        for target_id, supporters in supporters_by_target.items()
        if supporters
    }
    if not normalized:
        return WolfPriorityConsensus(
            priority=priority,
            target_id=None,
            status="all_abstain",
            supporters_by_target={},
        )

    quorum = alive_wolf_count // 2 + 1
    qualified = [
        target_id
        for target_id, supporters in normalized.items()
        if len(supporters) >= quorum
    ]
    if len(qualified) > 1:
        raise ConsensusInvariantViolation(priority, qualified)
    if qualified:
        target_id = qualified[0]
        status: WolfConsensusStatus = (
            "single_wolf" if alive_wolf_count == 1 else "majority"
        )
        return WolfPriorityConsensus(
            priority=priority,
            target_id=target_id,
            status=status,
            supporters_by_target=normalized,
        )

    highest_support = max(len(supporters) for supporters in normalized.values())
    leaders = [
        target_id
        for target_id, supporters in normalized.items()
        if len(supporters) == highest_support
    ]
    return WolfPriorityConsensus(
        priority=priority,
        target_id=None,
        status="tie" if len(leaders) > 1 else "insufficient",
        supporters_by_target=normalized,
    )


def derive_wolf_consensus_evidence(
    night_number: int,
    alive_wolf_ids: Sequence[str],
    stances: Sequence[WolfTargetStance],
) -> WolfConsensusEvidenceV2:
    """按事件顺序保留每狼每优先级最后立场，并分别聚合主备刀。"""
    alive_wolves = tuple(dict.fromkeys(alive_wolf_ids))
    if not alive_wolves:
        raise ValueError("alive_wolf_ids must not be empty")
    if len(alive_wolves) != len(tuple(alive_wolf_ids)):
        raise ValueError("alive_wolf_ids must be unique")

    stance_history = tuple(
        _ImmutableWolfTargetStance.model_validate(stance.model_dump())
        for stance in stances
    )
    alive_wolf_set = set(alive_wolves)
    slots: dict[tuple[str, WolfPriority], WolfTargetStance] = {}
    for stance in stance_history:
        if stance.wolf_id in alive_wolf_set:
            slots[(stance.wolf_id, stance.priority)] = stance

    priority_results: dict[WolfPriority, WolfPriorityConsensus] = {}
    for priority in ("primary", "backup"):
        supporters: dict[str, set[str]] = {}
        for (wolf_id, slot_priority), stance in slots.items():
            if (
                slot_priority == priority
                and stance.stance in {"propose", "support"}
                and stance.target_id is not None
            ):
                supporters.setdefault(stance.target_id, set()).add(wolf_id)
        priority_results[priority] = _consensus_from_supporters(
            priority=priority,
            alive_wolf_count=len(alive_wolves),
            supporters_by_target=supporters,
        )

    return WolfConsensusEvidenceV2(
        night_number=night_number,
        alive_wolf_ids=alive_wolves,
        stances=stance_history,
        quorum=len(alive_wolves) // 2 + 1,
        primary=priority_results["primary"],
        backup=priority_results["backup"],
    )


__all__ = [
    "ConsensusInvariantViolation",
    "WolfConsensusEvidenceV2",
    "WolfPriorityConsensus",
    "derive_wolf_consensus_evidence",
]
