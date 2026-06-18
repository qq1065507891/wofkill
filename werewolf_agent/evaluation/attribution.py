"""Post-game attribution engine for evaluation feedback traces.

Annotates cognition-module exposures (rag / reflection / possible_worlds /
simulator) with cited_by_decision / aligned_with_decision / harmful_transfer,
and runs the consistency judge per trace with rebuilt public_facts. Pure
post-game: no runtime change, no audit payload growth.
"""

from __future__ import annotations

import re as _re
from typing import Any, Mapping

from werewolf_agent.evaluation.feedback_schemas import ModuleExposure
from werewolf_agent.evaluation.text_similarity import jaccard, tokenize


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


_CITED_THRESHOLD = 0.15
_ACTION_VERBS = ("先", "不要", "避免", "必须", "优先", "核验", "比较", "列")


def speech_from_decision(decision) -> str:
    """Read the public speech text from DecisionSnapshot.raw.

    EvaluationTrace does not retain a standalone parsed_action; speech lives
    in decision.raw (set by EvaluationTraceBuilder._decision_snapshot).
    """
    if decision is None:
        return ""
    raw = decision.raw or {}
    return str(raw.get("speech") or raw.get("public_story") or "")


def exposure_representative_text(
    exposure: ModuleExposure,
    resolver: AttributionTextResolver,
) -> str | None:
    """Prompt-safe representative text for an exposure, or None if unresolved.

    None signals the engine to mark the exposure MetricSupport.UNSUPPORTED
    (RAG/reflection whose card text cannot be resolved post-game).
    possible_worlds/simulator always resolve from their structured metadata.
    """
    module = exposure.module
    meta = exposure.metadata
    if module == "rag":
        return resolver.rag_text(exposure)
    if module == "reflection":
        return resolver.reflection_text(exposure)
    if module == "possible_worlds":
        assignments = meta.get("key_assignments") or {}
        return " ".join(f"{pid}={role}" for pid, role in assignments.items())
    if module == "simulator":
        affected = meta.get("affected_players") or []
        return f"{exposure.item_id} {' '.join(str(p) for p in affected)}".strip()
    return None


def cited(decision, exposure: ModuleExposure, resolver: AttributionTextResolver) -> bool:
    exp_text = exposure_representative_text(exposure, resolver)
    if not exp_text:
        return False  # unresolved → engine marks UNSUPPORTED; not cited
    decision_text = f"{decision.reason or ''} {speech_from_decision(decision)}"
    return jaccard(decision_text, exp_text) >= _CITED_THRESHOLD


_PLAYER_ID_RE = _re.compile(r"p\d{1,3}")
_WOLF_ROLES = frozenset({"werewolf", "wolf"})


def _reason_players(decision) -> set[str]:
    if decision is None or not decision.reason:
        return set()
    return set(_PLAYER_ID_RE.findall(decision.reason))


def aligned(
    decision,
    exposure: ModuleExposure,
    faction: str,
    resolver: AttributionTextResolver | None = None,
) -> bool:
    """Per-module direction rule: did the decision follow the exposure?"""
    module = exposure.module
    meta = exposure.metadata
    target = decision.target_id if decision else None
    mentioned = _reason_players(decision)
    relevant_players = ({target} if target else set()) | mentioned

    if module == "possible_worlds":
        assignments = meta.get("key_assignments") or {}
        wolves = {pid for pid, role in assignments.items() if role in _WOLF_ROLES}
        return bool(relevant_players & wolves)
    if module == "simulator":
        affected = set(meta.get("affected_players") or [])
        return bool(relevant_players & affected)
    if module in ("rag", "reflection"):
        if resolver is None:
            return False
        exp_text = exposure_representative_text(exposure, resolver)
        if not exp_text:
            return False
        reason = decision.reason or "" if decision else ""
        # the decision adopted a recommended action verb that the card also mentions
        return any(verb in reason and verb in exp_text for verb in _ACTION_VERBS)
    return False
