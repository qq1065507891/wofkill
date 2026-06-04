"""Case ingestion: validate, sanitize, and store RAG entries.

Enforces the RAG boundary: no base rules, no role-skill truth,
no victory truth, no live game adjudication. Every entry must have
source metadata and quality grading.
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
        self._validate_forbidden_content(entry)
        self._validate_source_metadata(entry)
        self._validate_quality(entry)
        self._validate_not_rule_truth(entry)

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
        """Check entry content for forbidden patterns."""
        # R16: also scan ``metadata.tags``. The audit contract is
        # "no RAG entry may carry a forbidden keyword anywhere" —
        # tags are user-supplied free text and used to be silently
        # skipped, which let an entry with a clean title/summary
        # but a ``moderator_knows`` tag pass ingestion.
        text = (
            f"{entry.title} {entry.summary} "
            f"{' '.join(entry.key_decisions)} "
            f"{' '.join(entry.short_quotes)} "
            f"{' '.join(entry.metadata.tags)}"
        )
        text_lower = text.lower()
        for kw in FORBIDDEN_RAG_KEYWORDS:
            if kw in text_lower:
                raise IngestionError(
                    f"Forbidden keyword '{kw}' found in entry '{entry.entry_id}'. "
                    f"RAG must not contain ground truth or rule adjudication."
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
        """Ensure entry does not contain base rule truth."""
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
        text = f"{entry.title} {entry.summary} {' '.join(entry.key_decisions)} {' '.join(entry.short_quotes)}".lower()
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
