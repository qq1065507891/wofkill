"""Deterministic scoring helpers for relation graph evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from werewolf_agent.memory.schemas import RelationEvent, RelationType


_PRESSURE_PREDICATES = {RelationType.SPOKE_AGAINST, RelationType.VOTED}
_DEFENSE_PREDICATES = {RelationType.DEFENDED}


def score_relation_event(event: RelationEvent) -> float:
    """Return a bounded evidence weight for a relation event."""
    existing = event.metadata.get("weight") if isinstance(event.metadata, dict) else None
    if isinstance(existing, (int, float)):
        return _clamp(float(existing))

    if event.predicate == RelationType.VOTED:
        return 0.6
    if event.predicate == RelationType.CLAIMED_ROLE:
        return 0.5
    if event.predicate == RelationType.NIGHT_RESULT_CLAIMED:
        return 0.75
    if event.predicate in (RelationType.SPOKE_AGAINST, RelationType.DEFENDED):
        return _speech_relation_weight(event.value)
    return 0.4


def annotate_relation_event(
    event: RelationEvent,
    *,
    event_ref: str = "",
    visibility: str = "public",
) -> RelationEvent:
    """Return a copy of ``event`` with scoring metadata filled in."""
    metadata = dict(event.metadata or {})
    if event_ref:
        metadata.setdefault("event_ref", event_ref)
    metadata.setdefault("weight", score_relation_event(event))
    metadata.setdefault("confidence", metadata["weight"])
    metadata.setdefault("visibility", visibility)
    metadata.setdefault("basis", event.predicate.value)
    return replace(event, metadata=metadata)


def relation_strength(
    graph: Any,
    source: str,
    target: str,
    predicates: Iterable[RelationType] | None = None,
    day_window: tuple[int, int] | None = None,
) -> float:
    allowed = set(predicates) if predicates is not None else None
    score = 0.0
    for event in graph.all_events():
        if event.source != source or event.target != target:
            continue
        if allowed is not None and event.predicate not in allowed:
            continue
        if not _day_in_window(event.day, day_window):
            continue
        score += score_relation_event(event)
    return round(score, 3)


def player_pressure_summary(
    graph: Any,
    player_id: str,
    day_window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    pressure_events: list[RelationEvent] = []
    defense_events: list[RelationEvent] = []
    for event in graph.by_target(player_id):
        if not _day_in_window(event.day, day_window):
            continue
        if event.predicate in _PRESSURE_PREDICATES:
            pressure_events.append(event)
        elif event.predicate in _DEFENSE_PREDICATES:
            defense_events.append(event)

    return {
        "player": player_id,
        "pressure_score": round(sum(score_relation_event(e) for e in pressure_events), 3),
        "defense_score": round(sum(score_relation_event(e) for e in defense_events), 3),
        "attackers": _ordered_sources(
            e for e in pressure_events if e.predicate == RelationType.SPOKE_AGAINST
        ),
        "voters": _ordered_sources(
            e for e in pressure_events if e.predicate == RelationType.VOTED
        ),
        "defenders": _ordered_sources(defense_events),
    }


def coalition_edges(
    graph: Any,
    day_window: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for event in graph.all_events():
        if event.target is None or not _day_in_window(event.day, day_window):
            continue
        edges.append({
            "source": event.source,
            "target": event.target,
            "predicate": event.predicate.value,
            "day": event.day,
            "weight": score_relation_event(event),
            "event_ref": event.metadata.get("event_ref", ""),
        })
    edges.sort(key=lambda edge: (-float(edge["weight"]), edge["source"], edge["target"]))
    return edges


def _speech_relation_weight(text: str) -> float:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("强烈", "定狼", "查杀", "must", "confirmed")):
        return 0.7
    if any(token in lowered for token in ("怀疑", "保", "信任", "suspect", "trust")):
        return 0.5
    return 0.4


def _ordered_sources(events: Iterable[RelationEvent]) -> list[str]:
    ordered = sorted(events, key=lambda e: (-score_relation_event(e), e.source))
    seen: set[str] = set()
    result: list[str] = []
    for event in ordered:
        if event.source not in seen:
            seen.add(event.source)
            result.append(event.source)
    return result


def _day_in_window(day: int, day_window: tuple[int, int] | None) -> bool:
    if day_window is None:
        return True
    start, end = day_window
    return start <= day <= end


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
