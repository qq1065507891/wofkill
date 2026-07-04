"""Memory persistence: serialize/deserialize memory subsystems.

Works with CognitionMatrix, RelationGraph, ReflectionMemory,
ProfileStore, and MemoryStore. Returns JSON-serializable dicts.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from werewolf_agent.memory.cognition_matrix import CognitionMatrix
from werewolf_agent.memory.relation_graph import RelationGraph
from werewolf_agent.memory.reflection import ReflectionMemory
from werewolf_agent.memory.profile import ProfileStore
from werewolf_agent.memory.schemas import ReflectionEntry, PlayerProfile

if TYPE_CHECKING:
    from werewolf_agent.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Reflection Memory
# ---------------------------------------------------------------------------


def save_reflections(mem: ReflectionMemory) -> list[dict[str, Any]]:
    return [e.to_dict() for e in mem.all_entries()]


def load_reflections(data: list[dict[str, Any]]) -> list[ReflectionEntry]:
    return [ReflectionEntry.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Profile Store
# ---------------------------------------------------------------------------


def save_profiles(store: ProfileStore) -> list[dict[str, Any]]:
    return [p.to_dict() for p in store.all_profiles()]


def load_profiles(data: list[dict[str, Any]]) -> list[PlayerProfile]:
    return [PlayerProfile(**d) for d in data]


# ---------------------------------------------------------------------------
# Full MemoryStore snapshot
# ---------------------------------------------------------------------------


def save_memory_store(store: MemoryStore) -> dict[str, Any]:
    """Serialize entire MemoryStore to a dict."""
    matrices = {}
    for viewer_id in store.cognition_matrices:
        m = store.get_matrix(viewer_id)
        if m is not None:
            matrices[viewer_id] = m.to_dict()
    reflection_ids = [e.entry_id for e in store.reflections.all_entries()]
    if hasattr(store.reflections, "all_v2_entries"):
        reflection_ids.extend(
            e.entry_id for e in store.reflections.all_v2_entries()
        )

    return {
        "cognition_matrices": matrices,
        "relation_graph": store.relation_graph.to_dict(),
        "reflections": reflection_ids,
        "profiles": save_profiles(store.profiles),
    }


def restore_memory_store(data: dict[str, Any], repo: Any = None) -> MemoryStore:
    """Restore MemoryStore from a snapshot dict."""
    from werewolf_agent.memory.store import MemoryStore as _MS
    store = _MS(repo=repo)

    # Restore cognition matrices
    for viewer_id, matrix_data in data.get("cognition_matrices", {}).items():
        matrix = CognitionMatrix.from_dict(matrix_data)
        store.cognition_matrices[viewer_id] = matrix

    # Restore relation graph
    rg_data = data.get("relation_graph", {})
    if rg_data:
        store.relation_graph = RelationGraph.from_dict(rg_data)

    # Snapshot reflections are IDs only in V2. Dirty legacy snapshots may still
    # contain full bodies; they are intentionally ignored so old snapshots cannot
    # rehydrate rejected/review-only text into live memory or write it back to
    # the reflections table.

    # Restore profiles
    for prof_data in data.get("profiles", []):
        profile = PlayerProfile(**prof_data)
        store.profiles.store(profile)

    return store
