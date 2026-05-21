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

    entries.append(RAGEntry(
        entry_id="seed_timeline_first_night_before_day_one_01",
        title="时间线基础纠偏：首夜在第一天之前",
        summary=(
            "12人预女猎白混的行动顺序应按 N1 首夜 -> D1 第一天 -> N2 第二夜 -> D2 第二天理解。"
            "首夜发生在第一天之前；第一天是首夜结算后的第一个白天。"
            "首夜阶段通常已经产生狼人刀口、预言家首验、女巫用药判断、混血儿选主人等信息，"
            "所以第一天警上发言时不要说成“第一天之后才进入首夜”。"
        ),
        key_decisions=[
            "看到 D1 / 第一天 时，默认 N1 / 首夜已经发生并结算。",
            "首夜行动包括狼人刀人、预言家验人、女巫用药、混血儿选主人。",
            "如果发言中出现“第一天之后才首夜”，应立刻修正为“首夜在第一天之前”。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.ROLE_STRATEGY,
            quality_grade=QualityGrade.RULE_DERIVED_SEED,
            review_status=ReviewStatus.APPROVED,
            reviewer="timeline_guard",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="general",
            role_perspective="all",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.RULE_DERIVED,
                source_title="项目内时间轴纠偏种子",
                source_author="wofkill",
            ),
            tags=["timeline", "cold_start", "anti_confusion"],
        ),
    ))

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

    entries.append(RAGEntry(
        entry_id="seed_seer_counterclaim_vote_push_01",
        title="预言家对跳局归票悍跳位是高概率主线",
        summary=(
            "在预女猎白混的对跳局里，真预言家白天发言或警长归票时把票归到悍跳位，"
            "是高概率且合理的主线事件。这个动作的价值是把好人注意力集中到真假预言家对抗，"
            "迫使悍跳位解释验人、警徽流、站边和票型，而不是让讨论被零散怀疑带偏。"
            "但它不是硬规则：如果场上已有明确查杀、爆点发言、关键票型矛盾或更高优先级目标，"
            "预言家可以说明理由后调整归票目标。"
        ),
        key_decisions=[
            "真预言家在对跳局里归票悍跳位，通常不是异常激进，而是常见主线推进。",
            "归票时要同时解释悍跳位的验人、警徽流、站边和发言漏洞，避免只喊身份。",
            "好人可以把“跟可信预言家归票悍跳位”视为合理投票依据，但仍要结合查杀、爆点和票型。",
            "狼悍跳位应预期真预言家会归自己，需要提前准备反打逻辑和公共叙事。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.ROLE_STRATEGY,
            quality_grade=QualityGrade.EXPERT_REVIEW,
            review_status=ReviewStatus.APPROVED,
            reviewer="strategy_seed",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="seer",
            visibility_boundary=VisibilityBoundary.PUBLIC_ONLY,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_title="预女猎白混对跳局归票策略",
                source_author="project_strategy_review",
            ),
            tags=[
                "seer",
                "counterclaim",
                "fake_seer",
                "vote_push",
                "day_vote",
                "speech",
                "high_probability_event",
            ],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_tutorial_yumindao_seer_beginner_450",
        title="新手预言家：报验人、讲心路、留警徽流",
        summary=(
            "新手预言家的核心不是喊身份，而是把信息交代完整：先报昨夜查验，"
            "再解释为什么验这个人，随后给出警徽流和当天归票方向。对跳局里，"
            "要把悍跳位的验人、警徽流、发言心路和票型承接逐项拆开，让好人知道为什么站你。"
            "如果第一天被抗推出局，遗言仍然要整理查验、狼坑和重新站边理由。"
        ),
        key_decisions=[
            "警上优先报查验结果，再讲验人心路和警徽流。",
            "对跳时攻击悍跳位的具体逻辑漏洞，而不是只说“我才是真预言家”。",
            "放逐发言要同步给出查验信息、下一轮视角和明确归票。",
            "被抗推出局时用遗言纠偏站错边的好人，留下可执行的票型建议。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_curator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="sheriff_speech",
            role_perspective="seer",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_url="https://www.youmindao.com/langrensha/566.html",
                source_title="狼人杀各个身份发言技巧",
                source_author="游民岛",
                publish_date="2022-10-24",
            ),
            tags=["beginner_tutorial", "seer", "sheriff_speech", "badge_flow", "counterclaim"],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_tutorial_yumindao_witch_beginner_450",
        title="新手女巫：隐藏到能带队，药水发言要有理由",
        summary=(
            "新手女巫前期不要急着暴露身份，发言可以按低视角好人表水，"
            "重点观察谁在穿女巫衣服、谁在强行带错队。需要起跳带队时，"
            "要说明自己为什么可信、为什么怀疑目标、药水压力如何配合投票，"
            "避免无理由撒毒或只靠身份压人。"
        ),
        key_decisions=[
            "起跳前以好人表水和观察为主，减少无意义身份暴露。",
            "有人穿女巫衣服或场上无人带队时，再考虑用身份和药水压力接管节奏。",
            "使用毒药压力时必须给出目标发言、站边或票型依据。",
            "不要因为情绪或单点怀疑随意撒毒，药水决策要能被好人复盘理解。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_curator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="witch",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_url="https://www.youmindao.com/langrensha/566.html",
                source_title="狼人杀各个身份发言技巧",
                source_author="游民岛",
                publish_date="2022-10-24",
            ),
            tags=["beginner_tutorial", "witch", "speech", "poison_pressure", "leadership"],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_tutorial_yumindao_hunter_idiot_civilian_488",
        title="新手猎白民：先表水，再找狼，别抢神职节奏",
        summary=(
            "猎人、白痴和平民都是容易被新手玩成“乱带队”的位置。前期应先表水，"
            "用发言、站边和对票型的理解让别人知道你为什么像好人；身份或视角不足时，"
            "不要急着踩死别人，也不要和女巫、预言家的主线无理由对冲。"
            "当自己身份被认可或局面需要你起跳带队时，再明确给出怀疑对象和投票理由。"
        ),
        key_decisions=[
            "平民和白痴前期以干净表水为主，找狼要给具体发言漏洞。",
            "猎人前期不要因为技能强就乱打强势，先避免误导好人主线。",
            "身份坐高后再扩大找狼范围，不能直接把低视角怀疑说成铁狼。",
            "遗言或关键发言要留下狼坑、站边和投票建议，避免只报身份。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_curator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="villager",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_url="https://www.youmindao.com/langrensha/450.html",
                source_title="狼人杀12人标准局预女猎白板子规则玩法技巧解读",
                source_author="游民岛",
                publish_date="2022-10-11",
            ),
            tags=[
                "beginner_tutorial",
                "villager",
                "hunter",
                "idiot",
                "speech",
                "self_clear",
            ],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_tutorial_yumindao_wolf_roles_883",
        title="新手狼人：悍跳、冲锋、倒钩、深水要分工",
        summary=(
            "预女猎白混的狼队新手最常见错误是全员同一种发言。狼队夜聊应先分工："
            "一名悍跳位负责预言家对抗，一名冲锋位帮悍跳位建立公共叙事，"
            "一名倒钩或阴阳倒钩位保留回旋空间，一名深水位按低视角好人发言。"
            "白天发言要服务同一个故事，避免队友之间互相拆台或突然换目标。"
        ),
        key_decisions=[
            "夜聊先确定谁悍跳、谁冲锋、谁倒钩、谁深水。",
            "悍跳位要模仿预言家结构：报验人、留警徽流、讲心路。",
            "冲锋位负责放大悍跳位可信点，但要避免无脑硬冲导致双狼暴露。",
            "倒钩和深水位要保存身份空间，为悍跳失败后的残局服务。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_curator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="night_discussion",
            role_perspective="werewolf",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_url="https://www.youmindao.com/langrensha/883.html",
                source_title="狼人杀12人局狼人打法",
                source_author="游民岛",
                publish_date="2022-11-24",
            ),
            tags=[
                "beginner_tutorial",
                "werewolf",
                "fake_seer",
                "pusher",
                "deep_hook",
                "deep_cover",
            ],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_tutorial_yumindao_hybrid_beginner_488",
        title="新手混血儿：隐藏榜样信息，先按低视角发言",
        summary=(
            "混血儿新手不应一上来公开自己的私密选择，也不要在没有公共逻辑时突然强站边。"
            "更稳的做法是先按低视角玩家表水，观察榜样的发言、站边和票型，再逐步调整自己的公共立场。"
            "发言要能被场上玩家理解为正常好人或低视角位置的推理，避免暴露额外信息导致被狼队或好人同时排斥。"
        ),
        key_decisions=[
            "不要无理由公开榜样信息，先保护自己的生存空间。",
            "先用低视角发言表水，再根据榜样的公共表现调整站边。",
            "如果榜样发言很差，不要盲目硬保，要用可解释的公共逻辑处理。",
            "混血儿的目标是让自己的站边看起来有信息来源，但不是泄露私密信息。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_curator",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="hybrid",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
                source_url="https://langrensha.163.com/2021/0917/34741_973105.html",
                source_title="预女猎白混板子规则",
                source_author="口袋狼人杀",
                publish_date="2021-09-17",
            ),
            tags=["beginner_tutorial", "hybrid", "speech", "self_clear", "survival"],
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

    # Jingcheng Master Tournament: pre_witch_hunter_idiot_mixed public cases.
    entries.append(RAGEntry(
        entry_id="seed_jingcheng_villager_fake_seer_250709",
        title="京城大师赛250709：平民代跳预言家扰乱狼队视角",
        summary=(
            "警上：公开切片标题记录牛肉干以平民身份跳预言家，核心训练点是代跳必须给出可追踪的验人视角、站边理由和压力目标，不能只喊身份。"
            "第一天：代跳发言持续扰乱狼队对真实神职分布的判断，其他好人需要观察狼队是否因代跳出现反应变形。"
            "夜聊：公开资料未给出完整狼队夜聊，训练时可把重点放在狼队如何复盘白天代跳收益、如何重新判断刀口和次日叙事。"
            "投票：该案例用于提示好人不要机械相信身份口号，应结合发言强度、站边连贯性和被压力者反应决定票型。"
            "复盘结论：平民代跳的价值在于压缩狼队判断空间、保护真实神职发言窗口；风险在于过度承诺会反向污染好人视角。"
        ),
        key_decisions=[
            "平民代跳必须给出连贯的站边理由，避免只喊身份不交逻辑。",
            "发言重点放在压缩狼队判断空间，而不是替真实神职做不可撤回的承诺。",
            "其他好人要复盘代跳收益：是否逼出了狼人反应、是否保护了关键神职发言空间。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_HIGH_END_CASE,
            quality_grade=QualityGrade.PRO_MATCH,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_2026_05_20",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="villager",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.PUBLIC_TOURNAMENT,
                source_url="https://www.bilibili.com/video/BV1GEGMz4EQm/",
                source_title="京城大师赛，牛肉干平民跳预言家，秀翻狼队，扰乱狼队思路，骗的狼队晕头转向。狼人杀大师赛预女猎白混250709",
                source_author="萌小蚕蚕",
                publish_date="2025-07-10",
            ),
            tags=[
                "jingcheng_master",
                "pre_witch_hunter_idiot_mixed",
                "villager",
                "fake_seer",
                "pressure_speech",
                "good_side_deception",
            ],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_jingcheng_wolf_antiprophet_push_250415",
        title="京城大师赛250415：狼队抗推预言家后继续推进好人混",
        summary=(
            "警上：公开简介显示本局围绕预言家牛肉干形成对抗，狼队需要在警上阶段建立可延续的反预言家叙事，而不是只做单点悍跳。"
            "第一天：狼队取得抗推预言家的节奏收益后，继续把焦点推进到李斯好人混身上，体现连续压缩好人身份空间的打法。"
            "夜聊：训练重点是狼队在夜间同步第二天话术：谁负责冲锋制造确定性，谁负责倒钩解释票型，谁负责补充身份链。"
            "投票：投票承接必须沿用前一天站边逻辑，把抗推预言家的票型解释成公共叙事的一部分，避免突然换目标导致断层。"
            "复盘结论：狼队领先时最容易犯的低级错误是只讨论刀口、不规划次日发言；高质量打法会把白天叙事、夜间分工和后续票型连成一条线。"
        ),
        key_decisions=[
            "狼队拿到抗推收益后要立刻规划下一天的公共叙事，避免夜晚只讨论刀口。",
            "冲锋位负责制造确定性，倒钩位负责解释票型和缓冲队友压力。",
            "推动第二目标时要接住前一天的站边逻辑，不能突然换一套理由。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.PRO_MATCH,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_2026_05_20",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="werewolf",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.PUBLIC_TOURNAMENT,
                source_url="https://www.bilibili.com/video/BV1Yx5fz6Emi/",
                source_title="京城大师赛，大G狼队打得好人披头散发，狼队直接抗推预言家牛肉干，接着抗推好人混李斯。狼人杀大师赛250415预女猎白混",
                source_author="萌小蚕",
                publish_date="2025-04-20",
            ),
            tags=[
                "jingcheng_master",
                "pre_witch_hunter_idiot_mixed",
                "werewolf",
                "anti_prophet_push",
                "vote_chain",
                "team_coordination",
            ],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_jingcheng_review_double_bomb_badge_loss_241218",
        title="京城大师赛241218：双自爆导致警徽流失后的好人复盘",
        summary=(
            "警上：公开录像简介记录首夜刀口被救，警上出现狼人自爆；后续再次自爆导致警徽流失，警上信息链被打断。"
            "第一天：好人不能因为警徽流失就停止盘逻辑，需要把前置发言、死亡顺序、查验遗留和自爆动机放在一起复盘。"
            "夜聊：公开流程显示后续夜间出现连续落刀和女巫压毒/撒毒决策，训练重点是夜间行动要能服务次日公共解释。"
            "投票：第三天和第四天通过公投逐步收束狼坑，说明警徽缺失后仍可依靠发言遗留、死亡信息和票型建立归因链。"
            "复盘结论：自爆会中断警上信息，但也暴露狼队急于处理的信息压力；好人应把信息中断转化为行为压力，而不是陷入无警徽恐慌。"
        ),
        key_decisions=[
            "警徽流失后不要只依赖身份口号，要把前置发言、死亡顺序和票型连起来。",
            "女巫夜间行动要服务于白天公共逻辑，次日能解释收益和风险。",
            "好人阵营需要把自爆造成的信息中断转化为行为压力，而不是停止盘逻辑。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_HIGH_END_CASE,
            quality_grade=QualityGrade.HIGH_RANK_GAME,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_2026_05_20",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="review",
            role_perspective="general",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.PUBLIC_TOURNAMENT,
                source_url="https://www.bilibili.com/video/BV1djkcYuEVf/",
                source_title="〖京城大师赛〗12.18换场地后第一场预女猎白混",
                source_author="薄尘yo",
                publish_date="2024-12-18",
            ),
            tags=[
                "jingcheng_master",
                "pre_witch_hunter_idiot_mixed",
                "review",
                "badge_loss",
                "wolf_bomb",
                "good_side_rebuild",
            ],
        ),
    ))

    entries.append(RAGEntry(
        entry_id="seed_jingcheng_wolf_god_hunt_260227",
        title="京城大师赛260227：抗推预言家后的狼队神牌信息交换",
        summary=(
            "警上：233乐园导读记录该局为20260227第一局预女猎白混，狼队目标是围绕预言家对抗制造可抗推空间。"
            "第一天：狼队成功抗推预言家后获得节奏优势，此时关键不是庆祝收益，而是继续整理场上神职线索和好人站边裂缝。"
            "夜聊：导读明确提到狼队友间精准交流神牌信息、准备拍刀；训练重点是夜间先汇总公开发言线索，再同步每名狼人的次日身份包装。"
            "投票：抗推预言家的票型要被狼队包装成可解释的公共选择，次日继续借票型和站边压力推进下一目标。"
            "复盘结论：高质量狼队夜聊不是散聊刀口，而是把神职定位、拍刀收益、次日发言分工和投票叙事同时确定下来。"
        ),
        key_decisions=[
            "夜间狼队沟通要先汇总白天公开发言里的神职线索，再决定下一轮主叙事。",
            "拍刀前同步每名狼人的白天身份包装，避免次日发言互相拆台。",
            "抗推关键好人后不要松散发言，要继续用票型和站边压力锁定下一目标。",
        ],
        metadata=CaseMetadata(
            case_type=CaseType.EXTERNAL_TACTICS,
            quality_grade=QualityGrade.PRO_MATCH,
            review_status=ReviewStatus.APPROVED,
            reviewer="web_seed_2026_05_20",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="night_discussion",
            role_perspective="werewolf",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(
                source_type=SourceType.PUBLIC_REVIEW,
                source_url="https://www.233leyuan.com/post-detail/2011746799257293581",
                source_title="狼人杀 〖JY狼人杀〗有这样的狼队友怎么输？",
                source_author="静待另一片天",
                publish_date="2026-01-15",
            ),
            tags=[
                "jingcheng_master",
                "pre_witch_hunter_idiot_mixed",
                "werewolf",
                "night_discussion",
                "god_hunt",
                "team_coordination",
            ],
        ),
    ))

    # Ingest all seeds
    for entry in entries:
        ingester.ingest(entry)

    return entries
