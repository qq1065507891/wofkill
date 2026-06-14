"""Prompt-safe tactical text helpers for RAG V2 frames."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from werewolf_agent.rag.schemas import RAGEntry, RAGHit, RAGTacticalFrame


TACTICAL_FRAME_FIELDS: tuple[str, ...] = (
    "situation_signature",
    "transferable_lesson",
    "applicability",
    "counter_signals",
    "recommended_use",
    "misuse_risk",
)


def _read_value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _candidate_frame(item: Any) -> Any:
    if isinstance(item, RAGTacticalFrame):
        return item
    if isinstance(item, RAGEntry | RAGHit):
        return item.tactical_frame
    if isinstance(item, Mapping):
        nested = item.get("tactical_frame")
        if nested is not None:
            return nested
        if all(field in item for field in TACTICAL_FRAME_FIELDS):
            return item
    return None


def _legacy_summary(item: Any) -> str:
    summary = _read_value(item, "summary", "")
    if summary is None:
        return ""
    return str(summary).strip()[:800]


def _legacy_key_decisions(item: Any) -> list[str]:
    decisions = _read_value(item, "key_decisions", [])
    if decisions is None:
        return []
    if isinstance(decisions, list):
        return [
            str(decision).strip()
            for decision in decisions
            if str(decision).strip()
        ]
    text = str(decisions).strip()
    return [text] if text else []


def _legacy_fallback_frame(item: Any) -> dict[str, str | list[str]]:
    summary = _legacy_summary(item)
    lesson = summary or "旧版RAG条目缺少V2战术框架。"
    return {
        "situation_signature": "旧版RAG条目缺少V2战术框架。",
        "transferable_lesson": lesson,
        "applicability": [
            "仅在当前局面与旧摘要明确匹配时参考。",
        ],
        "counter_signals": [
            "当前局面与旧摘要描述不一致。",
        ],
        "recommended_use": "作为谨慎参考，不作为直接指令。",
        "misuse_risk": (
            "旧摘要缺少结构化反信号，过度套用可能误导发言。"
        ),
    }


def to_prompt_safe_tactical_frame(item: Any) -> dict[str, str | list[str]]:
    """Return only prompt-safe tactical frame fields for an entry, hit, or dict."""
    candidate = _candidate_frame(item)
    if candidate is None:
        return _legacy_fallback_frame(item)

    if isinstance(candidate, RAGTacticalFrame):
        frame = candidate
    elif isinstance(candidate, Mapping):
        frame = RAGTacticalFrame(
            **{field: candidate.get(field) for field in TACTICAL_FRAME_FIELDS}
        )
    elif hasattr(candidate, "model_dump"):
        dumped = candidate.model_dump()
        frame = RAGTacticalFrame(
            **{field: dumped.get(field) for field in TACTICAL_FRAME_FIELDS}
        )
    else:
        frame = RAGTacticalFrame(**{
            field: getattr(candidate, field)
            for field in TACTICAL_FRAME_FIELDS
        })

    return frame.model_dump()


def get_prompt_tactical_frame(item: Any) -> dict[str, str | list[str]]:
    """Return a prompt-safe tactical frame, with a legacy fallback when needed."""
    return to_prompt_safe_tactical_frame(item)


def _text_values(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def build_rag_retrieval_text(item: Any, *, max_chars: int = 1500) -> str:
    """Build compact retrieval text from V2 frames or legacy fields."""
    parts: list[str] = []
    title = str(_read_value(item, "title", "")).strip()
    if title:
        parts.append(title)
    if _candidate_frame(item) is None:
        summary = _legacy_summary(item)
        if summary:
            parts.append(summary)
        parts.extend(_legacy_key_decisions(item))
        text = "\n".join(parts)
        return text[:max(0, int(max_chars))]

    frame = get_prompt_tactical_frame(item)
    for field in TACTICAL_FRAME_FIELDS:
        parts.extend(_text_values(frame[field]))
    text = "\n".join(parts)
    return text[:max(0, int(max_chars))]


prompt_safe_tactical_frame_dict = to_prompt_safe_tactical_frame
