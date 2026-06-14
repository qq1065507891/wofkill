"""RAG persistence: serialize/deserialize RAG entries for storage.

Works with StrategyRetriever, CaseIngester, or plain list[RAGEntry].
Returns JSON-serializable dicts that can be stored via GameRepository
or any persistence backend.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.rag.schemas import RAGEntry


def save_rag_entries(source: list[RAGEntry] | Any) -> list[dict[str, Any]]:
    """Serialize RAG entries from a list, retriever, or ingester."""
    if hasattr(source, "all_entries"):
        entries = source.all_entries()
    elif hasattr(source, "_entries"):
        entries = list(source._entries.values())
    elif isinstance(source, list):
        entries = source
    else:
        entries = list(source)
    return [e.model_dump() for e in entries]


def load_rag_entries(data: list[dict[str, Any]]) -> list[RAGEntry]:
    """Deserialize RAG entries from dicts."""
    entries: list[RAGEntry] = []
    for raw in data:
        item = dict(raw)
        item.setdefault("schema_version", 1)
        entries.append(RAGEntry(**item))
    return entries
