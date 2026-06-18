"""Post-game attribution engine for evaluation feedback traces.

Annotates cognition-module exposures (rag / reflection / possible_worlds /
simulator) with cited_by_decision / aligned_with_decision / harmful_transfer,
and runs the consistency judge per trace with rebuilt public_facts. Pure
post-game: no runtime change, no audit payload growth.
"""

from __future__ import annotations

from typing import Any, Mapping

from werewolf_agent.evaluation.feedback_schemas import ModuleExposure


_RAG_TEXT_FIELDS = (
    "title",
    "situation_signature",
    "transferable_lesson",
    "recommended_action",
    "misuse_risk",
)
_REFLECTION_TEXT_FIELDS = (
    "theme",
    "lesson",
    "recommended_action",
    "misuse_risk",
)


class AttributionTextResolver:
    """Resolve compact exposure records to the prompt-safe card text.

    Production wiring wraps ``RAGRepository.get(entry_id)`` /
    ``ReflectionMemory.all_v2_entries()`` (or a service-level cache). Tests
    pass fixture dicts. Returns ``None`` when the entry cannot be resolved —
    the engine then marks that exposure ``MetricSupport.UNSUPPORTED``.
    """

    def __init__(
        self,
        *,
        rag_entries: Mapping[str, Mapping[str, Any]] | None = None,
        reflection_entries: Mapping[str, Mapping[str, Any]] | None = None,
        rag_provider=None,
        reflection_provider=None,
    ) -> None:
        self._rag_entries = rag_entries
        self._reflection_entries = reflection_entries
        self._rag_provider = rag_provider
        self._reflection_provider = reflection_provider

    def rag_text(self, exposure: ModuleExposure) -> str | None:
        data = self._resolve("rag", exposure.item_id)
        if not data:
            return None
        return " ".join(str(data.get(f, "") or "") for f in _RAG_TEXT_FIELDS).strip()

    def reflection_text(self, exposure: ModuleExposure) -> str | None:
        data = self._resolve("reflection", exposure.item_id)
        if not data:
            return None
        card = data.get("prompt_card", data)
        return " ".join(str(card.get(f, "") or "") for f in _REFLECTION_TEXT_FIELDS).strip()

    def _resolve(self, module: str, item_id: str) -> Mapping[str, Any] | None:
        if module == "rag":
            if self._rag_provider is not None:
                return self._rag_provider(item_id)
            if self._rag_entries is not None:
                return self._rag_entries.get(item_id)
        elif module == "reflection":
            if self._reflection_provider is not None:
                return self._reflection_provider(item_id)
            if self._reflection_entries is not None:
                return self._reflection_entries.get(item_id)
        return None
