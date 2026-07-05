# -*- coding: utf-8 -*-
"""
功能描述：校验、清洗并存储 RAG 条目，强制执行 RAG 边界安全策略。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import re
from datetime import datetime

from werewolf_agent.rag.schemas import (
    FORBIDDEN_RAG_KEYWORDS,
    CaseType,
    QualityGrade,
    RAGEntry,
)


def _entry_safety_text(entry: RAGEntry) -> str:
    """Collect all RAG text fields that must pass safety validation."""
    parts: list[str] = [
        entry.title,
        entry.summary,
        *entry.key_decisions,
        *entry.short_quotes,
        *entry.metadata.tags,
    ]
    if entry.tactical_frame is not None:
        frame = entry.tactical_frame
        parts.extend([
            frame.situation_signature,
            frame.transferable_lesson,
            *frame.applicability,
            *frame.counter_signals,
            frame.recommended_use,
            frame.misuse_risk,
        ])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Ingestion validation
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Raised when an entry fails ingestion validation."""


class CaseIngester:
    """Validates and ingests RAG entries with boundary enforcement."""

    def __init__(self) -> None:
        self._entries: dict[str, RAGEntry] = {}

    def ingest(self, entry: RAGEntry) -> RAGEntry:
        """Validate and store an entry. Raises IngestionError on violation."""
        try:
            validate_rag_entry_prompt_safe(entry)
        except ValueError as exc:
            raise IngestionError(str(exc)) from exc
        self._validate_source_metadata(entry)
        self._validate_quality(entry)

        # Auto-timestamp if missing
        if not entry.metadata.source.collected_at:
            entry = entry.model_copy(update={
                "metadata": entry.metadata.model_copy(update={
                    "source": entry.metadata.source.model_copy(update={
                        "collected_at": datetime.now().isoformat(),
                    }),
                }),
            })

        self._entries[entry.entry_id] = entry
        return entry

    def get(self, entry_id: str) -> RAGEntry | None:
        return self._entries.get(entry_id)

    def all_entries(self) -> list[RAGEntry]:
        return list(self._entries.values())

    def count(self) -> int:
        return len(self._entries)

    def _validate_forbidden_content(self, entry: RAGEntry) -> None:
        """Check entry content for forbidden patterns.

        rag-hardening-2: also scan for ``pNN`` player-ID references.
        RAG entries are strategy cases / public community knowledge —
        they MUST NOT name a specific player slot. A seed that says
        "p05 查杀是真预言家" leaks past-game role information that
        could match a real player in the current game; the LLM would
        then ground its decisions on stale cross-game identity. The
        12-player V1 board uses ``pNN`` exclusively, so
        ``\\bp\\d{2}\\b`` is a precise match — no false positives
        against words like "pre" or "step" (those don't have a digit
        after the ``p``).
        """
        # R16: also scan ``metadata.tags``. The audit contract is
        # "no RAG entry may carry a forbidden keyword anywhere" —
        # tags are user-supplied free text and used to be silently
        # skipped, which let an entry with a clean title/summary
        # but a ``moderator_knows`` tag pass ingestion.
        text = _entry_safety_text(entry)
        text_lower = text.lower()
        for kw in FORBIDDEN_RAG_KEYWORDS:
            if kw in text_lower:
                raise IngestionError(
                    f"Forbidden keyword '{kw}' found in entry '{entry.entry_id}'. "
                    f"RAG must not contain ground truth or rule adjudication."
                )
        # rag-hardening-2: PII / player-ID filter.
        pii_matches = re.findall(r"\bp\d{2}\b", text_lower)
        if pii_matches:
            raise IngestionError(
                f"RAG entry '{entry.entry_id}' contains player-ID "
                f"reference(s) {sorted(set(pii_matches))}. RAG is for "
                f"strategy / community knowledge and must not name "
                f"specific player slots — past-game identity may match "
                f"current-game players and leak cross-game information."
            )

    def _validate_source_metadata(self, entry: RAGEntry) -> None:
        """Ensure source metadata is present for external cases."""
        meta = entry.metadata
        if meta.case_type in (
            CaseType.EXTERNAL_HIGH_END_CASE,
            CaseType.EXTERNAL_TACTICS,
        ):
            if not meta.source.source_type:
                raise IngestionError(
                    f"External case '{entry.entry_id}' must have source_type"
                )
            if meta.quality_grade == QualityGrade.UNREVIEWED:
                raise IngestionError(
                    f"External case '{entry.entry_id}' must have a quality grade "
                    f"(not UNREVIEWED)"
                )

    def _validate_quality(self, entry: RAGEntry) -> None:
        """Quality checks based on case type."""
        if entry.metadata.quality_grade == QualityGrade.SELF_PLAY_CANDIDATE:
            if entry.metadata.case_type in (
                CaseType.EXTERNAL_HIGH_END_CASE,
                CaseType.EXTERNAL_TACTICS,
            ):
                raise IngestionError(
                    f"External case '{entry.entry_id}' cannot use SELF_PLAY_CANDIDATE quality"
                )
        if entry.metadata.quality_grade == QualityGrade.PRO_MATCH:
            if entry.metadata.case_type in (
                CaseType.PROJECT_HISTORY,
                CaseType.PROJECT_REVIEW,
            ):
                raise IngestionError(
                    f"Project-internal case '{entry.entry_id}' cannot claim PRO_MATCH quality"
                )

    def _validate_not_rule_truth(self, entry: RAGEntry) -> None:
        """Ensure entry does not contain base rule truth.

        rag-hardening-3: extended catch-all patterns block generic
        identity assertions of the form ``pNN 是 <role>`` /
        ``pNN 查杀`` / ``pNN 金水``. The 16 base patterns above
        cover the specific V1 rule statements (女巫不能自救 etc.)
        but did NOT cover generic "X is wolf" / "X is seer" — a
        future seed that wrote "p05 是狼" would pass the rule
        scan yet still leak role truth to the LLM. The catch-all
        block closes that gap by rejecting any direct identity
        statement bound to a player slot.

        Note: PII / player-ID filter (``\\bp\\d{2}\\b``) lives in
        ``_validate_forbidden_content`` rather than here, so the
        ingestion contract stays: "any forbidden token anywhere in
        the user-visible text triggers ``IngestionError``".
        """
        rule_truth_patterns = [
            r"witch\s+cannot\s+self[\s-]save",
            r"seer\s+checks?\s+hybrid\s+as\s+good",
            r"idiot\s+reveals?\s+only\s+when\s+exiled",
            r"second\s+tie\s+means\s+no\s+exile",
            r"hunter\s+cannot\s+shoot\s+if\s+poisoned",
            r"女巫.{0,8}(不能|不得|无法|不可).{0,6}(自救|救自己)",
            r"预言家.{0,8}(查验|验).{0,4}(混血儿|混混).{0,8}(好人|金水)",
            r"猎人.{0,8}(被毒|毒死|女巫毒|吃毒).{0,8}(不能|不得|无法|不可).{0,4}(开枪|带人)",
            r"女巫.{0,8}(不能|不得|无法|不可).{0,4}自救",
            r"预言家.{0,8}查验.{0,4}混血儿.{0,8}(好人|金水)",
            r"白痴.{0,8}(被放逐|放逐).{0,8}(翻牌|揭示|留在场上)",
            r"猎人.{0,8}(被毒|毒死|女巫毒).{0,8}(不能|不得|无法|不可).{0,4}(开枪|带人)",
            r"(第二次平票|二次平票|第二轮平票).{0,8}(无人出局|无人放逐|不放逐|进入黑夜)",
            r"混血儿.{0,8}(胜负|输赢).{0,8}(跟随|绑定|取决于).{0,8}(主人|榜样).{0,8}(原始阵营|阵营)",
            r"(基础规则|胜负条件|裁判结算)",
        ]
        rule_truth_patterns.extend([
            r"\u72fc\u4eba.{0,8}\u4e3b\u52a8.{0,4}\u7a7a\u5200",
            r"\u72fc\u961f.{0,8}\u8d85\u65f6.{0,8}\u9ed8\u8ba4.{0,4}\u7a7a\u5200",
            r"wolf_kill_selected.{0,12}\u5973\u5deb.{0,8}\u5200\u53e3",
            r"\u5973\u5deb.{0,8}(\u4e0d\u5141\u8bb8|\u4e0d\u80fd|\u4e0d\u53ef|\u7981\u6b62).{0,4}\u81ea\u6551",
            r"\u72fc.{0,8}(\u6ca1\u5b9a|\u672a\u5b9a|\u6ca1\u6709\u51b3\u5b9a).{0,8}\u5200\u53e3.{0,8}\u9ed8\u8ba4.{0,4}\u7a7a\u5200",
            r"\u72fc\u961f.{0,8}(\u53ef\u4ee5|\u80fd).{0,8}(\u9009\u62e9\u4e0d\u5200\u4eba|\u4e0d\u5200\u4eba|\u7a7a\u5200)",
        ])
        # rag-hardening-3: catch-all identity-leak patterns. The base
        # rules above reject canonical V1 rule statements; these
        # below reject generic role assertions that aren't covered
        # by the rule set but still leak role truth to the LLM.
        rule_truth_patterns.extend([
            # English
            r"\bp\d{2}\s+is\s+(a\s+)?(werewolf|wolf|villager|seer|witch|hunter|idiot|hybrid|good|bad)\b",
            r"\bp\d{2}\s+(wolf[\s-]?checked|checked[\s-]?wolf|\u67e5\u6740|\u91d1\u6c34|\u9a8c\u4e86|\u67e5\u9a8c)\b",
            # Chinese
            r"\bp\d{2}\s*(\u662f|\u4e3a|\u5f53)\s*(\u72fc|\u72fc\u4eba|\u9884\u8a00\u5bb6|\u5973\u5deb|\u730e\u4eba|\u767d\u75f4|\u6df7\u8840\u513f|\u5e73\u6c11|\u6751\u6c11)\b",
            r"\bp\d{2}\s*(\u67e5\u6740|\u91d1\u6c34|\u94f6\u6c34|\u9a8c\u51fa|\u67e5\u9a8c|\u9a8c\u4e86)\b",
        ])
        # N4: include ``metadata.tags`` in the scanned text. R16 added
        # tags to the parallel ``_validate_forbidden_content`` scan,
        # but the rule-truth scan missed it — so an entry with a
        # clean title/summary/key_decisions/short_quotes but a
        # rule-truth pattern in its tags used to pass ingestion. The
        # RAG contract is "no RAG entry may carry a rule-truth
        # statement anywhere", which includes tags.
        text = _entry_safety_text(entry).lower()
        for pattern in rule_truth_patterns:
            if re.search(pattern, text):
                raise IngestionError(
                    f"Entry '{entry.entry_id}' contains base rule truth "
                    f"(pattern: {pattern}). RAG must not answer base rules."
                )


def create_seed_entries():  # type: ignore[no-redef]
    """Backward-compatible re-export from seed_data (lazy to avoid circular import)."""
    from werewolf_agent.rag.seed_data import create_seed_entries as _real
    return _real()


def validate_rag_entry_prompt_safe(entry: RAGEntry) -> None:
    """Validate text that can become prompt-visible RAG content."""
    checker = CaseIngester()
    try:
        checker._validate_forbidden_content(entry)
        checker._validate_not_rule_truth(entry)
    except IngestionError as exc:
        raise ValueError(str(exc)) from exc
