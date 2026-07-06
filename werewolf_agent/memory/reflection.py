# -*- coding: utf-8 -*-
"""
长期反思记忆的兼容 facade，重新导出拆分后的质量门、合成器和存储类。

作者：Mike
创建日期：2025-01-15
修改日期：2026-07-06
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from werewolf_agent.memory.reflection_quality import ReflectionQualityGate, _GENERIC_PHRASES
from werewolf_agent.memory.reflection_repository import ReflectionMemory
from werewolf_agent.memory.reflection_repository import _LOG
from werewolf_agent.memory.reflection_sanitization import (
    _LEADING_ITEM_PREFIX_RE,
    _LLM_TRUTH_TOKENS,
    _PLAYER_ID_RE,
    _SOURCE_TEXT_CAP,
    _cap_source_text,
    _iter_section_items,
    _scrub_ids,
)
from werewolf_agent.memory.reflection_synthesis import (
    ReflectionSynthesizer,
    _LLM_MISTAKE_HEADER_CATEGORY,
    _LLM_MISTAKE_SECTION_RE,
    _LLM_STRENGTH_SECTION_RE,
)
from werewolf_agent.memory.schemas import (
    CrossGameQuery,
    ReflectionEntry,
    ReflectionEntryV2,
    ReflectionMistakePattern,
    ReflectionPreservedStrength,
    ReflectionQualityStatus,
    ReviewReport,
)

__all__ = [
    "CrossGameQuery",
    "ReflectionEntry",
    "ReflectionEntryV2",
    "ReflectionMemory",
    "ReflectionMistakePattern",
    "ReflectionPreservedStrength",
    "ReflectionQualityGate",
    "ReflectionQualityStatus",
    "ReflectionSynthesizer",
    "ReviewReport",
    "_GENERIC_PHRASES",
    "_LEADING_ITEM_PREFIX_RE",
    "_LLM_MISTAKE_HEADER_CATEGORY",
    "_LLM_MISTAKE_SECTION_RE",
    "_LLM_STRENGTH_SECTION_RE",
    "_LLM_TRUTH_TOKENS",
    "_LOG",
    "_PLAYER_ID_RE",
    "_SOURCE_TEXT_CAP",
    "_cap_source_text",
    "_iter_section_items",
    "_scrub_ids",
]
