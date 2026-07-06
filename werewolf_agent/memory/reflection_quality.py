# -*- coding: utf-8 -*-
"""
评估 V2 反思质量并阻止不安全提示卡进入实时提示。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.memory.reflection_quality import ReflectionQualityGate
    >>> ReflectionQualityGate()
"""

from __future__ import annotations

import re

from werewolf_agent.evaluation.text_similarity import jaccard as _jaccard
from werewolf_agent.memory.reflection_sanitization import (
    _LLM_TRUTH_TOKENS,
    _PLAYER_ID_RE,
    _SOURCE_TEXT_CAP,
)
from werewolf_agent.memory.schemas import ReflectionEntryV2, ReflectionQualityStatus

_GENERIC_PHRASES = (
    "复盘失败对局，关注关键转折点的信息缺失",
    "关注关键转折点的信息缺失",
    "下局继续努力",
    "总结经验",
)


class ReflectionQualityGate:
    """Deterministic V2 reflection scorer and prompt-safety gate."""

    def __init__(self, existing_entries: list[ReflectionEntryV2] | None = None) -> None:
        self._existing_entries = list(existing_entries or [])

    def evaluate(self, entry: ReflectionEntryV2) -> ReflectionEntryV2:
        flags: list[str] = []
        score = 0.0

        visible_blob = "\n".join(entry.prompt_visible_texts())
        hard_reject = False

        if not entry.game_id.strip() or not entry.player_id.strip() or not entry.role.strip():
            flags.append("missing_identity")
            hard_reject = True
        if not entry.prompt_card.theme.strip() or not entry.prompt_card.recommended_action.strip():
            flags.append("missing_prompt_card")
            hard_reject = True
        if _PLAYER_ID_RE.search(visible_blob):
            flags.append("player_id_leak")
            hard_reject = True
        if self._has_unsafe_truth_claim(entry, visible_blob):
            flags.append("unsafe_truth_claim")
            hard_reject = True

        if any(p.trigger.strip() and p.better_action.strip() for p in entry.mistake_patterns):
            score += 0.25
        if any(s.reuse_condition.strip() for s in entry.preserved_strengths):
            score += 0.15
        if self._has_complete_prompt_card(entry):
            score += 0.25
        if (
            entry.situation_signature.role.strip()
            and (
                entry.situation_signature.phase_focus
                or entry.situation_signature.game_patterns
            )
        ):
            score += 0.10
        if (
            entry.prompt_card.auto_verified
            or entry.prompt_card.fact_basis == "llm_transferable"
        ):
            score += 0.10
        if any(self._looks_actionable(text) for text in entry.actionable_advice):
            score += 0.10
        if self._looks_role_specific(entry):
            score += 0.05

        if self._is_generic(visible_blob):
            flags.append("generic_text")
            score -= 0.25
        if self._prompt_card_content_len(entry) < 80:
            flags.append("short_prompt_card")
            score -= 0.15
        if not (
            entry.situation_signature.phase_focus
            or entry.situation_signature.game_patterns
            or entry.prompt_card.trigger_signals
        ):
            flags.append("missing_trigger")
            score -= 0.15
        duplicate = self._find_duplicate(entry)
        if duplicate is not None:
            flags.append("duplicate")
            score -= 0.20
        source_blob = (
            entry.source.llm_self_review
            + entry.source.auto_review_summary
        )
        if len(source_blob) > _SOURCE_TEXT_CAP * 2:
            flags.append("source_truncated")
            score -= 0.05

        score = max(0.0, min(1.0, round(score, 2)))
        if duplicate is not None and score <= duplicate.quality_score:
            score = min(score, 0.69)

        if hard_reject:
            status = ReflectionQualityStatus.REJECTED
        elif score >= 0.70:
            status = ReflectionQualityStatus.APPROVED
        elif score >= 0.40:
            status = ReflectionQualityStatus.REVIEW_ONLY
        else:
            status = ReflectionQualityStatus.REJECTED

        return entry.model_copy(
            update={
                "quality_score": score,
                "quality_status": status,
                "quality_flags": sorted(set(flags)),
            },
            deep=True,
        )

    @staticmethod
    def _has_complete_prompt_card(entry: ReflectionEntryV2) -> bool:
        card = entry.prompt_card
        return bool(
            card.theme.strip()
            and card.lesson.strip()
            and card.trigger_signals
            and card.recommended_action.strip()
            and card.misuse_risk.strip()
        )

    @staticmethod
    def _prompt_card_content_len(entry: ReflectionEntryV2) -> int:
        card = entry.prompt_card
        return len(
            card.theme
            + card.lesson
            + "".join(card.trigger_signals)
            + card.recommended_action
            + card.misuse_risk
        )

    @staticmethod
    def _looks_actionable(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        return any(
            token in stripped
            for token in ("先", "不要", "避免", "必须", "优先", "核验", "比较", "列")
        )

    @staticmethod
    def _looks_role_specific(entry: ReflectionEntryV2) -> bool:
        text = "\n".join(entry.prompt_visible_texts())
        role_keywords = {
            "seer": ("预言家", "验人", "警徽流", "对跳"),
            "werewolf": ("狼人", "悍跳", "冲票", "倒钩", "深水"),
            "witch": ("女巫", "解药", "毒药", "银水"),
            "hunter": ("猎人", "开枪", "带走"),
            "idiot": ("白痴", "翻牌", "遗言", "出局"),
            "villager": ("村民", "平民", "站边", "票型"),
            "hybrid": ("混血儿", "主人", "阵营"),
        }
        return any(kw in text for kw in role_keywords.get(entry.role, (entry.role,)))

    @staticmethod
    def _is_generic(text: str) -> bool:
        stripped = str(text or "")
        return any(phrase in stripped for phrase in _GENERIC_PHRASES)

    @staticmethod
    def _has_unsafe_truth_claim(entry: ReflectionEntryV2, visible_blob: str) -> bool:
        if entry.prompt_card.auto_verified:
            return False
        return any(token in visible_blob for token in _LLM_TRUTH_TOKENS)

    def _find_duplicate(self, entry: ReflectionEntryV2) -> ReflectionEntryV2 | None:
        key = self._duplicate_key(entry)
        body = entry.prompt_card.lesson + entry.prompt_card.recommended_action
        for existing in self._existing_entries:
            if existing.quality_status != ReflectionQualityStatus.APPROVED:
                continue
            if self._duplicate_key(existing) != key:
                continue
            existing_body = (
                existing.prompt_card.lesson
                + existing.prompt_card.recommended_action
            )
            if _jaccard(body, existing_body) >= 0.70:
                return existing
        return None

    @staticmethod
    def _duplicate_key(entry: ReflectionEntryV2) -> tuple[str, str, str, str]:
        category = (
            entry.mistake_patterns[0].category
            if entry.mistake_patterns
            else ""
        )
        theme = re.sub(r"\s+", "", entry.prompt_card.theme.lower())
        return (entry.player_id, entry.role, category, theme)
