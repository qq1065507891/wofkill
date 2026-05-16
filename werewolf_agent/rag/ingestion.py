"""Case ingestion: validate, sanitize, and store RAG entries.

Enforces the RAG boundary: no base rules, no role-skill truth,
no victory truth, no live game adjudication. Every entry must have
source metadata and quality grading.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from werewolf_agent.rag.schemas import (
    FORBIDDEN_RAG_CONTENT_TYPES,
    FORBIDDEN_RAG_KEYWORDS,
    CaseMetadata,
    CaseType,
    QualityGrade,
    RAGEntry,
    ReviewStatus,
    SourceMetadata,
    SourceType,
    VisibilityBoundary,
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
        text = f"{entry.title} {entry.summary} {' '.join(entry.key_decisions)} {' '.join(entry.short_quotes)}"
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


# ---------------------------------------------------------------------------
# Seed data — cold start cases
# ---------------------------------------------------------------------------

def create_seed_entries() -> list[RAGEntry]:
    """Create initial seed entries for cold start (Phase A/B)."""
    ingester = CaseIngester()
    entries: list[RAGEntry] = []

    # Phase A: External high-end case (seed)
    entries.append(RAGEntry(
        entry_id="seed_ext_seer_claim_01",
        title="预言家起跳声明警徽流标准打法",
        summary=(
            "预言家在警上起跳，报查验结果和警徽流。先报查验，再声明今晚验X，"
            "如果死亡警徽给Y。标准打法要求预言家在第一天警上发言时起跳，"
            "报出第一晚查验结果和后续验人计划。"
        ),
        key_decisions=[
            "第一天警上起跳报查验",
            "声明警徽流：今晚验X，死亡给Y",
            "通过查验结果和站边逻辑建立可信度",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_HIGH_END_CASE,
            quality_grade=QualityGrade.HIGH_RANK_GAME,
            review_status=ReviewStatus.APPROVED,
            reviewer="seed_validator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="sheriff_speech",
            role_perspective="seer",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_title="预言家标准打法参考",
                source_author="community_guide",
            ),
            tags=["seer", "claim", "badge_flow", "sheriff_speech"],
        ),
    ))

    # External tactics: wolf deep hook
    entries.append(RAGEntry(
        entry_id="seed_ext_wolf_deep_hook_01",
        title="狼人倒钩打法：隐蔽保护队友",
        summary=(
            "倒钩狼的核心是在好人阵营中建立可信度，通过适度攻击狼队友来获取信任。"
            "避免在关键投票中直接保护队友，而是通过引导话题和站边来间接帮助狼队。"
        ),
        key_decisions=[
            "初期站边好人，建立可信度",
            "适度攻击狼队友（非致命）",
            "关键投票时引导目标偏离队友",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.EXPERT_REVIEW,
            review_status=ReviewStatus.APPROVED,
            reviewer="seed_validator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="werewolf",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_title="狼人倒钩战术参考",
                source_author="community_guide",
            ),
            tags=["werewolf", "deep_hook", "deception"],
        ),
    ))

    # Phase B: Rule-derived seed
    entries.append(RAGEntry(
        entry_id="seed_rule_seer_badge_flow_01",
        title="预言家警徽流声明模板",
        summary=(
            "预言家拿到警徽后，标准声明模板：'我昨晚验了X，结果是好人/狼人。"
            "今晚我验Y，如果我死亡，警徽给Z。' 这是结构化的声明规范，"
            "帮助预言家组织信息输出。"
        ),
        key_decisions=[
            "报查验结果",
            "声明今晚验人目标",
            "指定警徽移交对象",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.ROLE_STRATEGY,
            quality_grade=QualityGrade.RULE_DERIVED_SEED,
            review_status=ReviewStatus.APPROVED,
            reviewer="auto_seed",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="sheriff_speech",
            role_perspective="seer",
            visibility_boundary=VisibilityBoundary.PUBLIC_ONLY,
            source=SourceMetadata(
                source_type=SourceType.RULE_DERIVED,
                source_title="警徽流声明模板",
            ),
            tags=["seer", "badge_flow", "template"],
        ),
    ))

    # Speech template
    entries.append(RAGEntry(
        entry_id="seed_speech_wolf_defense_01",
        title="狼人被怀疑时的防守发言模板",
        summary=(
            "被怀疑时的发言策略：先表示理解大家的疑虑，然后用逻辑反驳关键指控，"
            "最后给出建设性的站边建议。避免过度防御或攻击质疑者。"
        ),
        key_decisions=[
            "承认疑虑合理性",
            "逻辑反驳关键指控",
            "提出建设性站边",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.SPEECH_TEMPLATE,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="seed_validator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="defense_speech",
            role_perspective="werewolf",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.MANUAL_ENTRY,
                source_title="防守发言模板",
            ),
            tags=["werewolf", "defense", "speech_template"],
        ),
    ))

    # Witch strategy
    entries.append(RAGEntry(
        entry_id="seed_ext_witch_poison_timing_01",
        title="女巫毒药使用时机参考",
        summary=(
            "女巫毒药的常见使用时机：第一天如果有明确狼人倾向的玩家可以考虑毒，"
            "但也可能留到后面更确定时使用。重点是评估信息确定性和用药收益。"
        ),
        key_decisions=[
            "评估毒药目标确定性",
            "权衡即时使用vs保留",
            "避免毒到好人阵营神职",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.ROLE_STRATEGY,
            quality_grade=QualityGrade.HIGH_RANK_GAME,
            review_status=ReviewStatus.APPROVED,
            reviewer="seed_validator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="night_action",
            role_perspective="witch",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_title="女巫用药策略",
                source_author="community_guide",
            ),
            tags=["witch", "poison", "night_action"],
        ),
    ))

    # God-view review (only usable in review context)
    entries.append(RAGEntry(
        entry_id="seed_godview_review_01",
        title="复盘案例：狼队配合失误导致屠边失败",
        summary=(
            "[GOD_VIEW] 此案例仅供复盘阶段使用。狼队在第二天错误地冲票了一名神职，"
            "导致好人阵营掌握了关键信息链，最终所有狼人被找出。"
            "复盘要点：冲票选择应考虑信息暴露风险。"
        ),
        key_decisions=[
            "狼队冲票目标选择失误",
            "冲票暴露了狼队投票链",
            "好人通过投票链锁定了狼人",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_HIGH_END_CASE,
            quality_grade=QualityGrade.EXPERT_REVIEW,
            review_status=ReviewStatus.APPROVED,
            reviewer="seed_validator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="review",
            role_perspective="god_view",
            visibility_boundary=VisibilityBoundary.GOD_VIEW,
            source=SourceMetadata(
                source_type=SourceType.PUBLIC_REVIEW,
                source_title="复盘案例分析",
                source_author="expert_reviewer",
            ),
            tags=["review", "god_view", "wolf_mistake"],
        ),
    ))

    # Hybrid strategy
    entries.append(RAGEntry(
        entry_id="seed_hybrid_survive_01",
        title="混血儿生存策略：低调跟随主人阵营",
        summary=(
            "混血儿不知道主人身份和阵营，只能通过主人白天的发言和行为推测方向。"
            "核心策略是保持低调，观察主人倾向，跟随主人站边但不过度表现。"
        ),
        key_decisions=[
            "观察主人白天的站边倾向",
            "低调跟随，不过度表现",
            "避免引起双方阵营的怀疑",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.ROLE_STRATEGY,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="seed_validator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="hybrid",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.MANUAL_ENTRY,
                source_title="混血儿生存策略",
            ),
            tags=["hybrid", "survival", "strategy"],
        ),
    ))

    # Ingest all seeds
    for entry in entries:
        ingester.ingest(entry)

    return entries
