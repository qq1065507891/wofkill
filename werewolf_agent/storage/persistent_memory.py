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
        """Persist full MemoryStore snapshot."""
        data = save_memory_store(memory_store)
        self._repo.save_memory_snapshot(snapshot_id, data)

    def restore_memory(self, snapshot_id: str = "latest") -> Any:
        """Restore MemoryStore from a snapshot. Returns MemoryStore or None."""
        from werewolf_agent.memory.store import MemoryStore
        data = self._repo.load_memory_snapshot(snapshot_id)
        if data is None:
            return MemoryStore(repo=self._repo)
        return restore_memory_store(data, repo=self._repo)

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
