# -*- coding: utf-8 -*-
"""
从狼队私有结构化立场确定性聚合主刀与备刀共识。

作者: Project contributors
创建日期: 2026-07-16

使用示例:
    >>> derive_wolf_consensus_evidence(1, ("w1",), ())
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

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


class ImmutableWolfTargetStance(WolfTargetStance):
    """隔离调用方可变对象，防止聚合后立场历史被原地篡改。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class FrozenSupportersByTarget(Mapping[str, tuple[str, ...]]):
    """使用纯 tuple 保存支持者映射，兼容 pickle/checkpoint 且不可原地修改。"""

    entries: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entries",
            tuple(
                (str(target_id), tuple(str(wolf_id) for wolf_id in supporters))
                for target_id, supporters in self.entries
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        supporters_by_target: Mapping[str, Sequence[str]],
    ) -> "FrozenSupportersByTarget":
        return cls(tuple(
            (
                target_id,
                tuple(sorted(set(supporters))),
            )
            for target_id, supporters in sorted(supporters_by_target.items())
        ))

    def __getitem__(self, target_id: str) -> tuple[str, ...]:
        for current_target, supporters in self.entries:
            if current_target == target_id:
                return supporters
        raise KeyError(target_id)

    def __iter__(self) -> Iterator[str]:
        return (target_id for target_id, _supporters in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())


@dataclass(frozen=True)
class WolfPriorityConsensus:
    """一个优先级上由当前有效立场形成的不可变聚合结果。"""

    priority: WolfPriority
    target_id: str | None
    status: WolfConsensusStatus
    supporters_by_target: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.supporters_by_target, FrozenSupportersByTarget):
            object.__setattr__(
                self,
                "supporters_by_target",
                FrozenSupportersByTarget.from_mapping(
                    self.supporters_by_target
                ),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "alive_wolf_ids", tuple(self.alive_wolf_ids))
        object.__setattr__(
            self,
            "stances",
            tuple(
                stance
                if isinstance(stance, ImmutableWolfTargetStance)
                else ImmutableWolfTargetStance.model_validate(
                    stance.model_dump()
                    if isinstance(stance, WolfTargetStance)
                    else stance
                )
                for stance in self.stances
            ),
        )


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
        ImmutableWolfTargetStance.model_validate(stance.model_dump())
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


def serialize_wolf_consensus_evidence(
    evidence: WolfConsensusEvidenceV2,
) -> str:
    """把不可变证据编码为 checkpoint 安全的规范 JSON 字符串。"""
    def priority_payload(priority: WolfPriorityConsensus) -> dict[str, Any]:
        return {
            "priority": priority.priority,
            "target_id": priority.target_id,
            "status": priority.status,
            "supporters_by_target": {
                target_id: list(supporters)
                for target_id, supporters
                in priority.supporters_by_target.items()
            },
        }

    return json.dumps(
        {
            "night_number": evidence.night_number,
            "alive_wolf_ids": list(evidence.alive_wolf_ids),
            "stances": [
                stance.model_dump(mode="json")
                for stance in evidence.stances
            ],
            "quorum": evidence.quorum,
            "primary": priority_payload(evidence.primary),
            "backup": priority_payload(evidence.backup),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deserialize_wolf_consensus_evidence(
    serialized: str,
) -> WolfConsensusEvidenceV2:
    """从规范 JSON 恢复不可变证据，仅供审计、检查点与回放读取。"""
    payload = json.loads(serialized)

    def priority_from_payload(raw: Mapping[str, Any]) -> WolfPriorityConsensus:
        return WolfPriorityConsensus(
            priority=raw["priority"],
            target_id=raw.get("target_id"),
            status=raw["status"],
            supporters_by_target=raw.get("supporters_by_target") or {},
        )

    return WolfConsensusEvidenceV2(
        night_number=payload["night_number"],
        alive_wolf_ids=tuple(payload["alive_wolf_ids"]),
        stances=tuple(
            ImmutableWolfTargetStance.model_validate(raw_stance)
            for raw_stance in payload["stances"]
        ),
        quorum=payload["quorum"],
        primary=priority_from_payload(payload["primary"]),
        backup=priority_from_payload(payload["backup"]),
    )


__all__ = [
    "ConsensusInvariantViolation",
    "FrozenSupportersByTarget",
    "ImmutableWolfTargetStance",
    "WolfConsensusEvidenceV2",
    "WolfPriorityConsensus",
    "deserialize_wolf_consensus_evidence",
    "derive_wolf_consensus_evidence",
    "serialize_wolf_consensus_evidence",
]
