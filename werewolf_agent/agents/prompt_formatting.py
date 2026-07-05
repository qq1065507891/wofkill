# -*- coding: utf-8 -*-
"""
清洗 prompt 文本并压缩 JSON 上下文内容。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.agents.prompt_formatting import clean_prompt_text
    >>> clean_prompt_text("p01 发言")
"""

from __future__ import annotations

import json
import re
from typing import Any


MAX_JSON_CONTEXT_CHARS = 1800
MAX_PERSONA_LINE_CHARS = 180
MAX_LEARNING_TEXT_CHARS = 160
PLAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def clean_prompt_text(
    value: Any,
    *,
    max_chars: int = MAX_PERSONA_LINE_CHARS,
) -> str:
    text = str(value or "").strip()
    text = PLAYER_ID_RE.sub("历史玩家", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 6)] + "…已截断"


def compact_json(
    value: Any,
    *,
    max_chars: int = MAX_JSON_CONTEXT_CHARS,
) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text

    structured = structured_json_summary(value)
    if structured is not None:
        structured_text = json.dumps(
            structured,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(structured_text) <= max_chars:
            return structured_text

    prefix = text[: max_chars // 3]
    suffix = text[-(max_chars // 3):]
    while True:
        rendered = json.dumps(
            {
                "truncated": True,
                "original_type": type(value).__name__,
                "original_length": len(text),
                "omitted_middle": True,
                "content_prefix": prefix,
                "content_suffix": suffix,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(rendered) <= max_chars or (not prefix and not suffix):
            return rendered
        overflow = len(rendered) - max_chars
        trim_prefix = min(len(prefix), overflow // 2 + 8)
        trim_suffix = min(len(suffix), overflow - trim_prefix + 8)
        prefix = prefix[: max(0, len(prefix) - trim_prefix)]
        suffix = suffix[min(len(suffix), trim_suffix):]


def structured_json_summary(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        items = list(value.items())
        if not items:
            return None
        head_count = min(6, len(items))
        tail_count = min(4, max(0, len(items) - head_count))
        head = {
            key: summarize_json_value(val)
            for key, val in items[:head_count]
        }
        tail_items = items[-tail_count:] if tail_count else []
        tail = {
            key: summarize_json_value(val)
            for key, val in tail_items
        }
        return {
            "truncated": True,
            "original_type": "dict",
            "original_length": len(items),
            "omitted_middle": len(items) > head_count + tail_count,
            "head": head,
            "tail": tail,
        }
    if isinstance(value, list):
        if not value:
            return None
        head_count = min(4, len(value))
        tail_count = min(3, max(0, len(value) - head_count))
        return {
            "truncated": True,
            "original_type": "list",
            "original_length": len(value),
            "omitted_middle": len(value) > head_count + tail_count,
            "head": [
                summarize_json_value(item)
                for item in value[:head_count]
            ],
            "tail": [
                summarize_json_value(item)
                for item in (value[-tail_count:] if tail_count else [])
            ],
        }
    return None


def summarize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_prompt_text(value, max_chars=MAX_LEARNING_TEXT_CHARS)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [
            summarize_json_value(item)
            for item in value[:3]
        ]
    if isinstance(value, dict):
        slim: dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 5:
                break
            slim[str(key)] = summarize_json_value(val)
        return slim
    return clean_prompt_text(value, max_chars=MAX_LEARNING_TEXT_CHARS)


def truncate_text(
    text: str,
    max_chars: int,
    *,
    marker: str = "...（已截断）",
    prefer_sentence_boundary: bool = True,
) -> str:
    if len(text) <= max_chars:
        return text
    if prefer_sentence_boundary:
        slack = max(1, max_chars // 10)
        search_start = max_chars - slack
        best_cut = text.rfind("\n", search_start, max_chars)
        if best_cut < search_start:
            for ch in ".。!！?？":
                idx = text.rfind(ch, search_start, max_chars)
                if idx > best_cut:
                    best_cut = idx
        if best_cut >= search_start:
            return text[: best_cut + 1] + marker
    return text[:max_chars] + marker
