# -*- coding: utf-8 -*-
"""
构建玩家提示词，将稳定规则和动态上下文分别写入对应消息。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    >>> PlayerPromptBuilder(...)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.directive_priority import (
    HARD_CONSTRAINT_KEYS,
    REFERENCE_KEYS,
    SUGGESTION_KEYS,
)
from werewolf_agent.agents.prompt_formatting import (
    clean_prompt_text,
    compact_json,
    structured_json_summary,
    summarize_json_value,
    truncate_text,
)
from werewolf_agent.agents import prompt_composer
from werewolf_agent.agents.prompt_memory import (
    REFLECTION_CARD_BUDGET,
    PromptMemoryMixin,
    _MAX_LEARNING_CONTEXT_CHARS,
    _MAX_LEARNING_TEXT_CHARS,
    _MAX_RAG_TEXT_CHARS,
)
from werewolf_agent.agents.prompt_persona import PromptPersonaMixin
from werewolf_agent.agents.prompt_output import (
    PromptOutputMixin,
    _ACTION_TYPE_GUARD,
    _OUTPUT_SCHEMA_SKILL_FIELDS,
    _OUTPUT_SCHEMA_SPEECH_FIELDS,
    _OUTPUT_SCHEMA_VOTE_FIELDS,
    _SPEECH_INTENT_TASKS,
    _VOTE_AUDIT_FIELDS,
    _VOTE_REASON_PRIVACY_GUARD,
)
from werewolf_agent.agents.prompt_salience import (
    PromptSalienceMixin,
    _MAX_SALIENCE_ITEMS,
    _SALIENCE_PRIVATE_KEYS,
    _SALIENCE_PUBLIC_FIELDS,
    _slim_salience_item,
)
from werewolf_agent.agents.prompt_sections import (
    PromptSectionMixin,
    _NEVER_DROP_TIER,
    _SectionSpec,
    _USER_PROMPT_BUDGET_CHARS,
)
from werewolf_agent.agents.prompt_strategy import (
    PromptStrategyMixin,
    _MAX_SKILL_TACTICAL_ADVICE_CHARS,
    _MAX_SKILL_TACTICAL_ADVICE_ITEMS,
    _STRATEGY_GROUP_ORDER,
)
from werewolf_agent.agents.prompt_system import PromptSystemMixin
from werewolf_agent.agents.prompt_user_context import (
    PromptUserContextMixin,
    _MAX_PERSONA_LINE_CHARS,
    _MAX_PUBLIC_SUMMARY_CHARS,
    _MAX_TRANSCRIPT_ITEMS,
    _MAX_TRANSCRIPT_TEXT_CHARS,
    _clean_current_game_list_items,
    _clean_current_game_token,
    _safe_float,
)
from werewolf_agent.agents.schemas import (
    AgentContext,
    RetryInfo,
)

__all__ = [
    "PlayerPromptBuilder",
    "HARD_CONSTRAINT_KEYS",
    "REFERENCE_KEYS",
    "REFLECTION_CARD_BUDGET",
    "SUGGESTION_KEYS",
    "_MAX_LEARNING_CONTEXT_CHARS",
    "_MAX_LEARNING_TEXT_CHARS",
    "_MAX_PERSONA_LINE_CHARS",
    "_MAX_PUBLIC_SUMMARY_CHARS",
    "_MAX_RAG_TEXT_CHARS",
    "_MAX_SALIENCE_ITEMS",
    "_MAX_SKILL_TACTICAL_ADVICE_CHARS",
    "_MAX_SKILL_TACTICAL_ADVICE_ITEMS",
    "_MAX_TRANSCRIPT_ITEMS",
    "_MAX_TRANSCRIPT_TEXT_CHARS",
    "_NEVER_DROP_TIER",
    "_SALIENCE_PRIVATE_KEYS",
    "_SALIENCE_PUBLIC_FIELDS",
    "_SectionSpec",
    "_STRATEGY_GROUP_ORDER",
    "_USER_PROMPT_BUDGET_CHARS",
    "_ACTION_TYPE_GUARD",
    "_OUTPUT_SCHEMA_SKILL_FIELDS",
    "_OUTPUT_SCHEMA_SPEECH_FIELDS",
    "_OUTPUT_SCHEMA_VOTE_FIELDS",
    "_SPEECH_INTENT_TASKS",
    "_VOTE_AUDIT_FIELDS",
    "_VOTE_REASON_PRIVACY_GUARD",
    "_clean_current_game_list_items",
    "_clean_current_game_token",
    "_safe_float",
    "_slim_salience_item",
]

# P0-K1: skill catalog removed (tool path is dead code). Skill analyses
# are pre-injected via skill_analysis_hints — no separate tool catalog.

_MAX_JSON_CONTEXT_CHARS = 1800






class PlayerPromptBuilder(
    PromptSectionMixin,
    PromptSalienceMixin,
    PromptMemoryMixin,
    PromptStrategyMixin,
    PromptPersonaMixin,
    PromptSystemMixin,
    PromptUserContextMixin,
    PromptOutputMixin,
):
    """Assembles player prompts as a pipeline of independently-built sections.

    Per s10:
      system_prompt = core + rules + role_guide + skills + output_contract
      user_prompt   = boundary + phase + belief + summary + state
                      + events + directive + persona + transcript
                      + retry + task + contract

    Each _build_* method owns exactly one data source.  Stable sections
    (identity, rules, output format) go into the system prompt; per-turn
    dynamic context goes into the user message.
    """

    def __init__(self, context: AgentContext, player_name: str = "") -> None:
        self.context = context
        self.player_name = player_name or context.agent_id

    # ═══════════════════════════════════════════════════════════════
    #  System prompt: stable identity, rules, skill catalog
    # ═══════════════════════════════════════════════════════════════

    def build_system_prompt(self) -> str:
        return prompt_composer.compose_system_prompt(self)

    # ═══════════════════════════════════════════════════════════════
    #  User prompt: per-turn dynamic context (system reminder)
    # ═══════════════════════════════════════════════════════════════

    def build_user_prompt(self, retry: RetryInfo) -> str:
        return prompt_composer.compose_user_prompt(self, retry)


    @staticmethod
    def _clean_prompt_text(
        value: Any,
        *,
        max_chars: int = _MAX_PERSONA_LINE_CHARS,
    ) -> str:
        return clean_prompt_text(value, max_chars=max_chars)

    # ── Utility ──

    def _compact_json(self, value: Any) -> str:
        return compact_json(value)

    @staticmethod
    def _structured_json_summary(value: Any) -> dict[str, Any] | None:
        return structured_json_summary(value)

    @staticmethod
    def _summarize_json_value(value: Any) -> Any:
        return summarize_json_value(value)

    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int,
        *,
        marker: str = "...（已截断）",
        prefer_sentence_boundary: bool = True,
    ) -> str:
        return truncate_text(
            text,
            max_chars,
            marker=marker,
            prefer_sentence_boundary=prefer_sentence_boundary,
        )
