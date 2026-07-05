# -*- coding: utf-8 -*-
"""
功能描述：结构化关系图谱——以带类型谓词的可查询事件表存储投票链、身份声明和攻防关系（设计文档 §10）。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.cognition.world_state import StructuredFact, StructuredWorldState
from werewolf_agent.memory.schemas import RelationEvent, RelationType


class RelationGraph:
    """In-memory structured relation graph backed by an event list."""

    def __init__(self) -> None:
        self._events: list[RelationEvent] = []

    def add_event(self, event: RelationEvent) -> None:
        self._events.append(event)

    def add_events(self, events: list[RelationEvent]) -> None:
        self._events.extend(events)

    def all_events(self) -> list[RelationEvent]:
        return list(self._events)

    def count(self) -> int:
        return len(self._events)

    # --- query helpers ---

    def by_predicate(self, predicate: RelationType) -> list[RelationEvent]:
        return [e for e in self._events if e.predicate == predicate]

    def by_source(self, player_id: str) -> list[RelationEvent]:
        return [e for e in self._events if e.source == player_id]

    def by_target(self, player_id: str) -> list[RelationEvent]:
        return [e for e in self._events if e.target == player_id]

    def by_day(self, day: int) -> list[RelationEvent]:
        return [e for e in self._events if e.day == day]

    def query(
        self,
        predicate: RelationType | None = None,
        source: str | None = None,
        target: str | None = None,
        day: int | None = None,
    ) -> list[RelationEvent]:
        results = self._events
        if predicate is not None:
            results = [e for e in results if e.predicate == predicate]
        if source is not None:
            results = [e for e in results if e.source == source]
        if target is not None:
            results = [e for e in results if e.target == target]
        if day is not None:
            results = [e for e in results if e.day == day]
        return results

    def spoke_against(self, source: str, target: str, day: int) -> bool:
        return any(
            e.predicate == RelationType.SPOKE_AGAINST
            and e.source == source
            and e.target == target
            and e.day == day
            for e in self._events
        )

    def voted_for(self, source: str, target: str) -> list[RelationEvent]:
        return [
            e for e in self._events
            if e.predicate == RelationType.VOTED
            and e.source == source
            and e.target == target
        ]

    def claimed_roles(self, player_id: str) -> list[RelationEvent]:
        return [
            e for e in self._events
            if e.predicate == RelationType.CLAIMED_ROLE
            and e.source == player_id
        ]

    def defenses(self, target_id: str) -> list[RelationEvent]:
        return [
            e for e in self._events
            if e.predicate == RelationType.DEFENDED
            and e.target == target_id
        ]

    # --- import from world state ---

    def import_from_world_state(
        self,
        world_state: StructuredWorldState,
        day: int = 0,
    ) -> int:
        """Convert structured facts to relation events. Returns count added."""
        count = 0
        for fact in world_state.facts:
            events = self._fact_to_events(fact, day)
            self._events.extend(events)
            count += len(events)
        return count

    def _fact_to_events(
        self,
        fact: StructuredFact,
        default_day: int = 0,
    ) -> list[RelationEvent]:
        day = fact.day or default_day
        events: list[RelationEvent] = []

        if fact.fact_type == "vote":
            events.append(RelationEvent(
                predicate=RelationType.VOTED,
                source=fact.source_player or "?",
                target=fact.target_player,
                day=day,
                value="voted_for",
            ))

        elif fact.fact_type.startswith("claimed_"):
            role = fact.value or fact.fact_type.replace("claimed_", "")
            events.append(RelationEvent(
                predicate=RelationType.CLAIMED_ROLE,
                source=fact.source_player or "?",
                target=fact.target_player,
                day=day,
                value=role,
            ))

        elif fact.fact_type == "speech":
            text = fact.value.lower()
            src = fact.source_player or "?"
            # Detect attack/defend patterns from speech content
            # These are heuristic; richer NLP can improve later
            if any(w in text for w in ("怀疑", "狼坑", "投出", "suspect", "wolf")):
                if fact.target_player:
                    events.append(RelationEvent(
                        predicate=RelationType.SPOKE_AGAINST,
                        source=src,
                        target=fact.target_player,
                        day=day,
                        value=text[:100],
                    ))
            if any(w in text for w in ("保", "信任", "好人", "protect", "trust")):
                if fact.target_player:
                    events.append(RelationEvent(
                        predicate=RelationType.DEFENDED,
                        source=src,
                        target=fact.target_player,
                        day=day,
                        value=text[:100],
                    ))

        elif fact.fact_type == "seer_check":
            events.append(RelationEvent(
                predicate=RelationType.NIGHT_RESULT_CLAIMED,
                source=fact.source_player or "?",
                target=fact.target_player,
                day=day,
                value=fact.value,
                metadata={"night": fact.night},
            ))

        elif fact.fact_type == "seer_check_claim":
            events.append(RelationEvent(
                predicate=RelationType.NIGHT_RESULT_CLAIMED,
                source=fact.source_player or "?",
                target=fact.target_player,
                day=day,
                value=fact.value,
                metadata={"claim_type": fact.metadata.get("claim_type", "")},
            ))

        if not events:
            return events
        from werewolf_agent.memory.relation_scoring import annotate_relation_event

        event_ref = fact.metadata.get("event_ref") or (
            f"fact:{fact.fact_type}:{fact.source_player or ''}:"
            f"{fact.target_player or ''}:d{day}"
        )
        return [
            annotate_relation_event(event, event_ref=event_ref)
            for event in events
        ]

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [
                {
                    "predicate": e.predicate.value,
                    "source": e.source,
                    "target": e.target,
                    "day": e.day,
                    "value": e.value,
                    "metadata": e.metadata,
                }
                for e in self._events
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationGraph:
        graph = cls()
        for e_data in data.get("events", []):
            graph.add_event(RelationEvent(
                predicate=RelationType(e_data["predicate"]),
                source=e_data["source"],
                target=e_data["target"],
                day=e_data.get("day", 0),
                value=e_data.get("value", ""),
                metadata=e_data.get("metadata"),
            ))
        return graph
