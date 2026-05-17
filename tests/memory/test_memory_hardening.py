"""Memory hardening tests: persistence through storage interface.

Covers Task 8 Step 1:
- Cognition matrix save/load round-trip
- Relation graph events save/load
- Reflection memory save/load
- Profile store save/load
- Full MemoryStore snapshot and restore
"""

from __future__ import annotations

import json

import pytest

from werewolf_agent.memory.schemas import (
    CognitionMatrixEntry,
    CrossGameQuery,
    PlayerProfile,
    ReflectionEntry,
    RelationEvent,
    RelationType,
)
from werewolf_agent.memory.cognition_matrix import CognitionMatrix
from werewolf_agent.memory.relation_graph import RelationGraph
from werewolf_agent.memory.reflection import ReflectionMemory
from werewolf_agent.memory.profile import ProfileStore
from werewolf_agent.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Cognition Matrix persistence
# ---------------------------------------------------------------------------


class TestCognitionMatrixPersistence:
    def test_matrix_to_dict_and_back(self) -> None:
        matrix = CognitionMatrix("viewer1")
        matrix.initialize(["p1", "p2", "p3"], ["seer", "werewolf", "villager"])
        matrix.add_evidence("p1", "claimed seer")

        data = matrix.to_dict()
        restored = CognitionMatrix.from_dict(data)

        assert restored.viewer_id == "viewer1"
        entry = restored.get("p1")
        assert entry is not None
        assert "claimed seer" in entry.key_evidence

    def test_matrix_json_roundtrip(self) -> None:
        matrix = CognitionMatrix("viewer2")
        matrix.initialize(["p1"], ["werewolf"])

        data = matrix.to_dict()
        json_str = json.dumps(data, ensure_ascii=False)
        parsed = json.loads(json_str)
        restored = CognitionMatrix.from_dict(parsed)

        assert restored.get("p1") is not None


# ---------------------------------------------------------------------------
# Relation Graph persistence
# ---------------------------------------------------------------------------


class TestRelationGraphPersistence:
    def test_graph_serialization_roundtrip(self) -> None:
        graph = RelationGraph()
        graph.add_event(RelationEvent(
            predicate=RelationType.SPOKE_AGAINST,
            source="p1", target="p2", day=1, value="accused",
        ))
        graph.add_event(RelationEvent(
            predicate=RelationType.VOTED,
            source="p1", target="p2", day=1, value="voted_exile",
        ))

        data = graph.to_dict()
        restored = RelationGraph.from_dict(data)

        assert restored.count() == 2
        assert len(restored.by_predicate(RelationType.SPOKE_AGAINST)) == 1

    def test_graph_json_roundtrip(self) -> None:
        graph = RelationGraph()
        graph.add_event(RelationEvent(
            predicate=RelationType.CLAIMED_ROLE,
            source="p1", target="p1", day=1, value="seer",
        ))

        data = graph.to_dict()
        json_str = json.dumps(data, ensure_ascii=False)
        parsed = json.loads(json_str)
        restored = RelationGraph.from_dict(parsed)

        assert restored.count() == 1


# ---------------------------------------------------------------------------
# Reflection Memory persistence
# ---------------------------------------------------------------------------


class TestReflectionPersistence:
    def test_reflection_save_and_load(self) -> None:
        from werewolf_agent.memory.persistence import save_reflections, load_reflections

        mem = ReflectionMemory()
        mem.store(ReflectionEntry(
            entry_id="r1", game_id="g1", player_id="p1",
            role="seer", faction_won=True,
            text="Good game", tags=["seer", "win"],
        ))

        data = save_reflections(mem)
        loaded = load_reflections(data)

        new_mem = ReflectionMemory()
        for entry in loaded:
            new_mem.store(entry)
        assert new_mem.count() == 1
        assert new_mem.get("r1") is not None

    def test_reflection_json_roundtrip(self) -> None:
        from werewolf_agent.memory.persistence import save_reflections, load_reflections

        mem = ReflectionMemory()
        mem.store(ReflectionEntry(
            entry_id="j1", game_id="g1", player_id="p1",
            role="werewolf", faction_won=False,
            text="Lost because too aggressive", tags=["wolf", "fail"],
        ))

        data = save_reflections(mem)
        json_str = json.dumps(data, ensure_ascii=False)
        parsed = json.loads(json_str)
        loaded = load_reflections(parsed)

        assert loaded[0].role == "werewolf"
        assert "wolf" in loaded[0].tags


# ---------------------------------------------------------------------------
# Profile Store persistence
# ---------------------------------------------------------------------------


class TestProfilePersistence:
    def test_profile_save_and_load(self) -> None:
        from werewolf_agent.memory.persistence import save_profiles, load_profiles

        store = ProfileStore()
        store.get_or_create("p1")
        store.update_after_game("p1", role="seer", faction_won=True)

        data = save_profiles(store)
        loaded = load_profiles(data)

        new_store = ProfileStore()
        for profile in loaded:
            new_store.get_or_create(profile.player_id)
            # Apply loaded state
            new_store._profiles[profile.player_id] = profile

        p = new_store.get("p1")
        assert p is not None
        assert p.games_played == 1

    def test_profile_json_roundtrip(self) -> None:
        from werewolf_agent.memory.persistence import save_profiles, load_profiles

        store = ProfileStore()
        store.get_or_create("p1")
        store.update_after_game("p1", role="werewolf", faction_won=False,
                                ability_deltas={"deception": 0.1})

        data = save_profiles(store)
        json_str = json.dumps(data, ensure_ascii=False)
        parsed = json.loads(json_str)
        loaded = load_profiles(parsed)

        assert loaded[0].deception == pytest.approx(0.6, abs=0.01)


# ---------------------------------------------------------------------------
# Full MemoryStore snapshot and restore
# ---------------------------------------------------------------------------


class TestMemoryStoreSnapshot:
    def test_store_snapshot_and_restore(self) -> None:
        store = MemoryStore()

        # Add cognition matrix
        store.init_matrix("viewer1", ["p1", "p2"], ["seer", "werewolf"])

        # Add relation
        store.add_relation(RelationEvent(
            predicate=RelationType.SPOKE_AGAINST,
            source="p1", target="p2", day=1, value="sus",
        ))

        # Add reflection
        store.store_reflection(ReflectionEntry(
            entry_id="snap_r1", game_id="g1", player_id="p1",
            role="seer", faction_won=True, text="Good",
            tags=["seer"],
        ))

        # Add profile
        store.get_or_create_profile("p1")

        # Snapshot
        from werewolf_agent.memory.persistence import save_memory_store, restore_memory_store
        snapshot = save_memory_store(store)

        # Verify snapshot is serializable
        json_str = json.dumps(snapshot, ensure_ascii=False)
        parsed = json.loads(json_str)

        # Restore
        restored = restore_memory_store(parsed)

        # Verify cognition matrix
        m = restored.get_matrix("viewer1")
        assert m is not None
        assert "p1" in m.player_ids()

        # Verify relation
        assert restored.relation_graph.count() == 1

        # Verify reflection
        assert len(restored.reflections.by_player("p1")) == 1

        # Verify profile
        p = restored.get_profile("p1")
        assert p is not None
