"""Persistent memory coordinator: auto-save/restore MemoryStore and RAG via repository.

Design doc §13: V1 uses SQLite as local persistent backend.
This module wires the in-memory MemoryStore and StrategyRetriever
to a GameRepository for durable storage across process restarts.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.memory.persistence import (
    save_memory_store,
    restore_memory_store,
)
from werewolf_agent.rag.persistence import save_rag_entries, load_rag_entries
from werewolf_agent.rag.schemas import RAGEntry


class PersistentMemoryCoordinator:
    """Bridges in-memory subsystems with a persistent repository.

    Usage:
        coord = PersistentMemoryCoordinator(repo)
        coord.save_all(memory_store, retriever)  # persist everything
        mem, rag = coord.restore_all()  # restore from last snapshot
    """

    def __init__(self, repo: Any) -> None:
        self._repo = repo

    # -- RAG entries ---------------------------------------------------------

    def save_rag(self, retriever: Any) -> None:
        """Persist all RAG entries from a StrategyRetriever or list."""
        if retriever is None:
            return
        entries = save_rag_entries(retriever)
        if entries:
            self._repo.save_rag_entries(entries)

    def restore_rag(self) -> list[RAGEntry]:
        """Load persisted RAG entries."""
        data = self._repo.load_rag_entries()
        return load_rag_entries(data)

    def delete_rag_entry(self, entry_id: str) -> None:
        self._repo.delete_rag_entry(entry_id)

    # -- Memory store --------------------------------------------------------

    def save_memory(self, memory_store: Any, snapshot_id: str = "latest") -> None:
        """Persist MemoryStore snapshot and update latest alias.

        Reflection bodies are written to the durable ``reflections`` table, not
        embedded in the snapshot. The snapshot stores only lightweight IDs.
        """
        self._persist_reflection_rows(memory_store)
        data = save_memory_store(memory_store)
        self._repo.save_memory_snapshot(snapshot_id, data)
        if snapshot_id != "latest":
            self._repo.save_memory_snapshot("latest", data)

    def restore_memory(
        self,
        snapshot_id: str = "latest",
        *,
        fallback_to_latest: bool = False,
    ) -> Any:
        """Restore MemoryStore from a snapshot. Returns MemoryStore or None."""
        from werewolf_agent.memory.store import MemoryStore
        data = self._repo.load_memory_snapshot(snapshot_id)
        if data is None and fallback_to_latest and snapshot_id != "latest":
            data = self._repo.load_memory_snapshot("latest")
        if data is None:
            return MemoryStore(repo=self._repo)
        return restore_memory_store(data, repo=self._repo)

    def restore_latest_memory(self) -> Any:
        return self.restore_memory("latest")

    def restore_for_new_game(self, game_id: str) -> Any:
        return self.restore_memory(game_id, fallback_to_latest=True)

    def list_snapshots(self) -> list[dict[str, Any]]:
        return self._repo.list_memory_snapshots()

    def delete_snapshot(self, snapshot_id: str) -> None:
        self._repo.delete_memory_snapshot(snapshot_id)

    # -- Combined save/restore -----------------------------------------------

    def save_all(self, memory_store: Any, retriever: Any, snapshot_id: str = "latest") -> None:
        self.save_rag(retriever)
        self.save_memory(memory_store, snapshot_id)

    def restore_all(self, snapshot_id: str = "latest") -> tuple[Any, list[RAGEntry]]:
        mem = self.restore_memory(snapshot_id)
        rag = self.restore_rag()
        return mem, rag

    def _persist_reflection_rows(self, memory_store: Any) -> None:
        if memory_store is None or not hasattr(self._repo, "save_reflection"):
            return
        reflections = getattr(memory_store, "reflections", None)
        if reflections is None:
            return
        for entry in getattr(reflections, "all_entries", lambda: [])():
            entry_id = getattr(entry, "entry_id", "")
            existing = None
            if entry_id and hasattr(self._repo, "load_reflection"):
                existing = self._repo.load_reflection(entry_id)
            if existing is None:
                self._repo.save_reflection(entry.to_dict())
        for entry in getattr(reflections, "all_v2_entries", lambda: [])():
            self._repo.save_reflection(entry.to_dict())
