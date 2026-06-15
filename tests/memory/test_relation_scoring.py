from __future__ import annotations

from werewolf_agent.cognition.world_state import StructuredFact, StructuredWorldState
from werewolf_agent.memory.relation_graph import RelationGraph
from werewolf_agent.memory.review import ReviewGenerator
from werewolf_agent.memory.schemas import RelationEvent, RelationType
from werewolf_agent.memory.store import MemoryStore


def test_scores_relation_events_with_event_refs() -> None:
    ws = StructuredWorldState()
    ws.append(StructuredFact(
        fact_type="vote",
        source_player="p01",
        target_player="p03",
        day=1,
        metadata={"event_ref": "event:7:vote"},
    ))
    store = MemoryStore()

    added = store.import_world_state(ws, day=1)

    assert added == 1
    event = store.relation_graph.all_events()[0]
    assert event.metadata["event_ref"] == "event:7:vote"
    assert event.metadata["weight"] == 0.6
    assert 0.0 <= event.metadata["confidence"] <= 1.0
    assert event.metadata["visibility"] == "public"


def test_relation_strength_pressure_and_coalition_summaries() -> None:
    from werewolf_agent.memory.relation_scoring import (
        coalition_edges,
        player_pressure_summary,
        relation_strength,
    )

    graph = RelationGraph()
    graph.add_events([
        RelationEvent(
            predicate=RelationType.VOTED,
            source="p01",
            target="p03",
            day=1,
            metadata={"weight": 0.6},
        ),
        RelationEvent(
            predicate=RelationType.SPOKE_AGAINST,
            source="p02",
            target="p03",
            day=1,
            value="强烈怀疑 p03",
            metadata={"weight": 0.7},
        ),
        RelationEvent(
            predicate=RelationType.DEFENDED,
            source="p04",
            target="p03",
            day=1,
            value="保 p03",
            metadata={"weight": 0.4},
        ),
    ])

    assert relation_strength(graph, "p01", "p03") == 0.6
    summary = player_pressure_summary(graph, "p03")
    assert summary["player"] == "p03"
    assert summary["pressure_score"] == 1.3
    assert summary["defense_score"] == 0.4
    assert summary["attackers"] == ["p02"]
    assert summary["voters"] == ["p01"]
    assert summary["defenders"] == ["p04"]

    edges = coalition_edges(graph)
    assert edges[0]["source"] == "p02"
    assert edges[0]["target"] == "p03"
    assert edges[0]["weight"] == 0.7


def test_review_deception_prefers_stronger_relation_evidence() -> None:
    graph = RelationGraph()
    graph.add_events([
        RelationEvent(
            predicate=RelationType.VOTED,
            source="p01",
            target="p02",
            day=1,
            metadata={"weight": 0.6},
        ),
        RelationEvent(
            predicate=RelationType.SPOKE_AGAINST,
            source="p05",
            target="p02",
            day=1,
            metadata={"weight": 0.2},
        ),
        RelationEvent(
            predicate=RelationType.SPOKE_AGAINST,
            source="p06",
            target="p02",
            day=1,
            metadata={"weight": 0.9},
        ),
    ])

    report = ReviewGenerator().generate(
        game_id="g",
        player_id="p01",
        role="villager",
        faction_won=False,
        ground_truth={
            "p01": "villager",
            "p02": "villager",
            "p05": "werewolf",
            "p06": "werewolf",
        },
        relation_graph=graph,
    )

    assert report.deceived_by[:2] == ["p06", "p05"]
