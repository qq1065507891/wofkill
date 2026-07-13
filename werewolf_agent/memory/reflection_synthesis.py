# -*- coding: utf-8 -*-
"""
将 LLM 自我复盘和确定性评审报告合成 V2 反思。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.memory.reflection_synthesis import ReflectionSynthesizer
    >>> ReflectionSynthesizer()
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from werewolf_agent.core.models import GameState

from werewolf_agent.evaluation.text_similarity import jaccard as _jaccard
from werewolf_agent.memory.reflection_sanitization import (
    _LLM_TRUTH_TOKENS,
    _cap_source_text,
    _iter_section_items,
    _scrub_ids,
)
from werewolf_agent.memory.schemas import (
    ReflectionEntryV2,
    ReflectionMistakePattern,
    ReflectionPreservedStrength,
    ReviewReport,
)

# Captures the 【保留的优点】 section body up to the next 【...】 header or end.
_LLM_STRENGTH_SECTION_RE = re.compile(
    r"【保留的优点】[】:：\s]*(.*?)(?=【[^】]+】|$)",
    re.DOTALL,
)
# Captures each of the 6 mistake section headers + body up to the
# next 【...】 header or end. The captured header drives category
# mapping (not keyword heuristics), so a section like 【悍跳分析】
# is always decision_mistake regardless of bullet wording.
_LLM_MISTAKE_SECTION_RE = re.compile(
    r"【(投票错误|信息缺失|神职执行|悍跳分析|暴露原因|角色分工)】[】:：\s]*(.*?)(?=【[^】]+】|$)",
    re.DOTALL,
)
_LLM_MISTAKE_HEADER_CATEGORY = {
    "投票错误": "vote_mistake",
    "信息缺失": "info_miss",
    "神职执行": "role_execution",
    "悍跳分析": "decision_mistake",
    "暴露原因": "decision_mistake",
    "角色分工": "decision_mistake",
}


class ReflectionClaim(BaseModel):
    """LLM 提出的单条可由终局事件核验的事实声明。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    claim_id: str = Field(min_length=1)
    event_ref: str = Field(min_length=1)
    claim_type: Literal[
        "role", "faction", "vote", "death", "potion",
        "seer_check", "skill", "victory",
    ]
    subject_id: str = Field(min_length=1)
    target_id: str = ""
    value: str = ""


class ReflectionLesson(BaseModel):
    """只在全部依赖事实通过后才可跨局保存的抽象经验。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    lesson_id: str = Field(min_length=1)
    abstraction: str = Field(min_length=1)
    claim_dependencies: list[str] = Field(default_factory=list)
    lesson_kind: Literal["fact_dependent", "general_strategy"] = "fact_dependent"
    fact_independent: bool = False

    @model_validator(mode="after")
    def _require_fact_basis_or_explicit_independence(self) -> "ReflectionLesson":
        if (
            not self.claim_dependencies
            and self.lesson_kind != "general_strategy"
            and not self.fact_independent
        ):
            raise ValueError("dependency-free lessons must declare general_strategy or fact_independent")
        return self


class ReflectionDraft(BaseModel):
    """赛后模型输出的 moderator-only 结构化草稿。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[ReflectionClaim] = Field(default_factory=list)
    lessons: list[ReflectionLesson] = Field(default_factory=list)

    @model_validator(mode="after")
    def _claim_ids_must_be_unique(self) -> "ReflectionDraft":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("ReflectionDraft.claim_id must be unique")
        return self


class ReflectionVerification(BaseModel):
    """确定性事实门的结果，不包含原始草稿。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    verified_claims: list[ReflectionClaim] = Field(default_factory=list)
    verified_lessons: list[ReflectionLesson] = Field(default_factory=list)
    rejected_fact_count: int = 0
    rejected_lesson_count: int = 0


def parse_reflection_draft(text: str) -> ReflectionDraft | None:
    """解析严格 JSON 草稿；无法解析时不把自由文本提升为长期记忆。"""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return ReflectionDraft.model_validate(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _descriptor(
    *, event_ref: str, event_type: str, claim_type: str,
    subject_id: str, target_id: str = "", value: str = "",
) -> dict[str, str]:
    return {
        "event_ref": event_ref,
        "event_type": event_type,
        "visibility": "moderator_postgame",
        "claim_type": claim_type,
        "subject_id": subject_id,
        "target_id": target_id,
        "value": value,
    }


def _valid_death(state: GameState, player_id: str, reason: str) -> bool:
    return any(
        death.player_id == player_id and death.reason == reason
        for death in state.deaths
    )


def _player_faction(state: GameState, player_id: str) -> str:
    player = state.players[player_id]
    if player.role == "hybrid":
        return str(state.hybrid_master_faction or "")
    return str(player.faction or "")


def iter_verifiable_claim_descriptors(state: GameState) -> list[dict[str, str]]:
    """按真实事件 schema 生成唯一的可核验事实注册表。"""
    refs: list[dict[str, str]] = []
    witch_ids = sorted(
        pid for pid, player in state.players.items() if player.role == "witch"
    )
    seer_ids = sorted(
        pid for pid, player in state.players.items() if player.role == "seer"
    )
    for index, event in enumerate(state.events):
        event_ref = f"{state.game_id}:{index}"
        payload = event.payload or {}

        def add(claim_type: str, subject_id: object, target_id: object = "", value: object = "") -> None:
            subject = str(subject_id or "")
            target = str(target_id or "")
            normalized_value = str(value or "")
            if subject:
                refs.append(_descriptor(
                    event_ref=event_ref, event_type=event.type,
                    claim_type=claim_type, subject_id=subject,
                    target_id=target, value=normalized_value,
                ))

        if event.type == "roles_assigned":
            for pid, player in sorted(state.players.items()):
                add("role", pid, value=player.role)
                faction = _player_faction(state, pid)
                if faction:
                    add("faction", pid, value=faction)
        elif event.type == "role_revealed":
            pid = str(payload.get("player_id") or "")
            role = str(payload.get("role") or "")
            player = state.players.get(pid)
            if player is not None and player.role == role:
                add("role", pid, value=role)
                faction = _player_faction(state, pid)
                if faction:
                    add("faction", pid, value=faction)
        elif event.type == "vote":
            add("vote", payload.get("voter"), payload.get("target"))
        elif event.type in {"vote_resolved", "sheriff_vote_record", "sheriff_vote_resolved"}:
            for vote in payload.get("votes", []):
                if isinstance(vote, dict):
                    add("vote", vote.get("voter"), vote.get("target"))
        elif event.type == "player_died":
            pid = str(payload.get("player_id") or "")
            reason = str(payload.get("reason") or "")
            if _valid_death(state, pid, reason):
                add("death", pid, value=reason)
        elif event.type in {"witch_antidote_used", "witch_poison_used"} and len(witch_ids) == 1:
            add(
                "potion", witch_ids[0], payload.get("target_id"),
                "antidote" if event.type == "witch_antidote_used" else "poison",
            )
        elif event.type == "seer_check":
            seer_id = str(payload.get("seer_id") or "")
            if not seer_id and len(seer_ids) == 1:
                seer_id = seer_ids[0]
            player = state.players.get(seer_id)
            target_id = str(payload.get("target_id") or "")
            alignment = str(payload.get("alignment") or payload.get("result") or "")
            if player is not None and player.role == "seer" and target_id and alignment:
                add(
                    "seer_check", seer_id, target_id, alignment,
                )
        elif event.type == "hunter_shot_public":
            hunter_id = str(payload.get("hunter_id") or "")
            hunter = state.players.get(hunter_id)
            target_id = str(payload.get("target_id") or "")
            if hunter is not None and hunter.role == "hunter" and target_id:
                add("skill", hunter_id, target_id, "hunter_shot")
        elif event.type == "idiot_revealed":
            idiot_id = str(payload.get("player_id") or "")
            idiot = state.players.get(idiot_id)
            if idiot is not None and idiot.role == "idiot":
                add("skill", idiot_id, value="idiot_reveal")
        elif event.type == "hybrid_master_chosen":
            hybrid_id = str(payload.get("hybrid_id") or "")
            hybrid = state.players.get(hybrid_id)
            master_id = str(payload.get("master_id") or "")
            if (
                hybrid is not None
                and hybrid.role == "hybrid"
                and master_id
                and master_id == state.hybrid_master_id
            ):
                add("skill", hybrid_id, master_id, "hybrid_bind")
        elif event.type == "victory":
            winner = str(payload.get("winner") or payload.get("winning_faction") or "")
            reason = str(payload.get("reason") or "")
            if winner and winner == state.winning_faction:
                add("victory", winner, value=reason)
    return refs


def _claim_matches(claim: ReflectionClaim, state: GameState) -> bool:
    expected = {
        "event_ref": claim.event_ref,
        "claim_type": claim.claim_type,
        "subject_id": claim.subject_id,
        "target_id": claim.target_id,
        "value": claim.value,
    }
    return any(
        all(descriptor.get(key) == value for key, value in expected.items())
        for descriptor in iter_verifiable_claim_descriptors(state)
    )


_PLAYER_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_EVENT_REF_RE = re.compile(r"\b[\w-]+:\d+\b")
_CONCRETE_FACT_RE = re.compile(
    r"(?:预言家|女巫|猎人|狼人|村民|平民|白痴|混血儿).{0,8}(?:是|身份|阵营|获胜|死亡|被放逐|被毒|开枪)"
    r"|(?:好人|狼人)阵营.{0,4}获胜"
    r"|(?:解药|毒药).{0,8}(?:救了|毒了|使用|目标)"
)


def _safe_fact_independent_lesson(lesson: ReflectionLesson) -> bool:
    text = lesson.abstraction
    return not (
        _PLAYER_ID_TOKEN_RE.search(text)
        or _EVENT_REF_RE.search(text)
        or _CONCRETE_FACT_RE.search(text)
    )


def verify_reflection_draft(
    draft: ReflectionDraft,
    state: GameState,
) -> ReflectionVerification:
    """对最终 GameState 做确定性核验，并按依赖关系筛选经验。"""
    verified_claims = [
        claim for claim in draft.claims if _claim_matches(claim, state)
    ]
    accepted = {claim.claim_id for claim in verified_claims}
    rejected_fact_count = len(draft.claims) - len(accepted)
    verified_lessons = []
    for lesson in draft.lessons:
        if lesson.claim_dependencies:
            if all(dependency in accepted for dependency in lesson.claim_dependencies):
                verified_lessons.append(lesson)
        elif _safe_fact_independent_lesson(lesson):
            verified_lessons.append(lesson)
    return ReflectionVerification(
        verified_claims=verified_claims,
        verified_lessons=verified_lessons,
        rejected_fact_count=rejected_fact_count,
        rejected_lesson_count=len(draft.lessons) - len(verified_lessons),
    )


class ReflectionSynthesizer:
    """Merge LLM self-review and deterministic ReviewReport into V2."""

    def synthesize(
        self,
        *,
        llm_self_review: str,
        review_report: ReviewReport,
        faction: str = "",
    ) -> ReflectionEntryV2:
        corrected = bool(
            review_report.error_analysis
            and re.search(
                r"没有犯错|没犯错|无错误|都正确|没什么问题|没有明显失误|都还好|"
                r"判断都挺准(?:的)?|没(?:有)?失误|都对",
                llm_self_review or "",
            )
        )
        mistake_patterns = self._mistake_patterns(review_report, corrected)
        # Deterministic mistake_patterns take precedence; LLM-extracted
        # mistakes supplement them up to a total cap of 3, deduped by
        # jaccard on wrong_action (mirrors the strengths merge below).
        for llm_mistake in self._extract_llm_mistakes(llm_self_review, review_report.role):
            if any(_jaccard(llm_mistake.wrong_action, m.wrong_action) >= 0.6 for m in mistake_patterns):
                continue
            mistake_patterns.append(llm_mistake)
            if len(mistake_patterns) >= 3:
                break
        strengths = self._preserved_strengths(review_report)
        for llm_strength in self._extract_llm_strengths(llm_self_review):
            if any(_jaccard(llm_strength.behavior, s.behavior) >= 0.6 for s in strengths):
                continue
            strengths.append(llm_strength)
            if len(strengths) >= 3:
                break
        advice = [
            _scrub_ids(s)
            for s in review_report.improvement_suggestions[:3]
            if str(s or "").strip()
        ]
        if not advice:
            advice = [self._default_advice(review_report.role)]

        theme = self._theme(review_report, mistake_patterns)
        trigger_signals = [
            p.trigger for p in mistake_patterns[:2] if p.trigger.strip()
        ] or ["局势与历史反思相似"]
        lesson = self._lesson(review_report, mistake_patterns)
        recommended_action = advice[0]
        misuse_risk = "不要把历史经验直接映射到本局玩家身份。"

        outcome = "win" if review_report.faction_won else "loss"
        return ReflectionEntryV2(
            schema_version=2,
            entry_id=f"reflection_{review_report.game_id}_{review_report.player_id}",
            game_id=review_report.game_id,
            player_id=review_report.player_id,
            role=review_report.role,
            faction=faction,
            faction_won=review_report.faction_won,
            situation_signature={
                "role": review_report.role,
                "faction": faction,
                "outcome": outcome,
                "phase_focus": self._phase_focus(review_report),
                "game_patterns": self._game_patterns(review_report),
            },
            mistake_patterns=mistake_patterns,
            preserved_strengths=strengths,
            actionable_advice=advice,
            avoid_next_time=self._avoid_next_time(mistake_patterns),
            prompt_card={
                "theme": theme,
                "lesson": lesson,
                "trigger_signals": trigger_signals,
                "recommended_action": recommended_action,
                "misuse_risk": misuse_risk,
                "fact_basis": "auto_review" if mistake_patterns else "llm_transferable",
                "auto_verified": bool(mistake_patterns),
            },
            source={
                "llm_self_review": _cap_source_text(llm_self_review),
                "auto_review_summary": _cap_source_text(review_report.summary),
                "merged_by": "reflection_synthesizer_v2",
            },
        )

    def _mistake_patterns(
        self,
        report: ReviewReport,
        corrected: bool,
    ) -> list[ReflectionMistakePattern]:
        patterns: list[ReflectionMistakePattern] = []
        for raw in report.error_analysis[:3]:
            text = _scrub_ids(raw)
            category = self._category(text)
            better = (
                _scrub_ids(report.improvement_suggestions[0])
                if report.improvement_suggestions
                else self._default_advice(report.role)
            )
            patterns.append(ReflectionMistakePattern(
                category=category,
                trigger=self._trigger_for_category(category, text),
                wrong_action=text,
                better_action=better,
                fact_basis="auto_review",
                auto_verified=True,
                corrected_from_llm=corrected,
            ))
        return patterns

    @staticmethod
    def _preserved_strengths(report: ReviewReport) -> list[ReflectionPreservedStrength]:
        strengths: list[ReflectionPreservedStrength] = []
        for item in report.successful_strategies[:2]:
            text = _scrub_ids(item)
            strengths.append(ReflectionPreservedStrength(
                category="speech_quality" if "发言" in text else "strategy",
                behavior=text,
                reuse_condition="本局公开事实支持同类判断时",
            ))
        return strengths

    @staticmethod
    def _extract_llm_strengths(llm_self_review: str) -> list[ReflectionPreservedStrength]:
        """Parse the 【保留的优点】 section into fact-free preserved strengths.

        Spec Synthesis rule 1: subjective review must become structured
        strengths. Spec rule 7: LLM-only fact-bound claims (votes/roles/
        deaths/checks) must not be promoted — so any item (bullet/numeric/
        prose via :func:`_iter_section_items`) containing a truth token is
        dropped. ``ReflectionPreservedStrength`` has no ``fact_basis``
        field, so LLM provenance is implicit (the section source is
        recorded in ``source.llm_self_review``). Returns at most 2 strengths.
        """
        match = _LLM_STRENGTH_SECTION_RE.search(str(llm_self_review or ""))
        if not match:
            return []
        body = match.group(1)
        strengths: list[ReflectionPreservedStrength] = []
        for line in _iter_section_items(body):
            line = _scrub_ids(line)
            if any(token in line for token in _LLM_TRUTH_TOKENS):
                continue
            strengths.append(ReflectionPreservedStrength(
                category="speech_quality" if any(k in line for k in ("发言", "质疑", "表达")) else "strategy",
                behavior=line,
                reuse_condition="本局公开事实支持同类判断时",
            ))
            if len(strengths) >= 2:
                break
        return strengths

    @staticmethod
    def _extract_llm_mistakes(
        llm_self_review: str, role: str
    ) -> list[ReflectionMistakePattern]:
        """Parse the 6 LLM mistake sections into fact-free mistake patterns.

        Mirrors :meth:`_extract_llm_strengths`: section-item parsing
        (bullet/numeric/prose via :func:`_iter_section_items`), ID
        scrubbing, truth-token drop, length floor. Category is driven by
        the section header (not keyword heuristics) so e.g. 【悍跳分析】
        is always ``decision_mistake``. Returns at most 3 patterns.

        Safety: ``auto_verified`` is always False. Setting it True would
        bypass the truth-token gate at ``_has_unsafe_truth_claim``, so an
        item leaking a forbidden token would reach the live prompt
        unchecked. ``corrected_from_llm`` is also False — this flag means
        a *deterministic* review cleared the mistake, which an LLM
        self-assessment cannot assert.
        """
        cls = ReflectionSynthesizer
        patterns: list[ReflectionMistakePattern] = []
        for match in _LLM_MISTAKE_SECTION_RE.finditer(str(llm_self_review or "")):
            header = match.group(1)
            body = match.group(2)
            category = _LLM_MISTAKE_HEADER_CATEGORY[header]
            for line in _iter_section_items(body):
                line = _scrub_ids(line)
                if any(token in line for token in _LLM_TRUTH_TOKENS):
                    continue
                patterns.append(ReflectionMistakePattern(
                    category=category,
                    trigger=cls._trigger_for_category(category, line),
                    wrong_action=line,
                    # LLM bullets describe what went wrong, not how to
                    # improve — fall back to the role default advice.
                    better_action=cls._default_advice(role),
                    fact_basis="llm_transferable",
                    auto_verified=False,
                    corrected_from_llm=False,
                ))
                if len(patterns) >= 3:
                    return patterns
        return patterns

    @staticmethod
    def _category(text: str) -> str:
        if any(token in text for token in ("投", "票", "站边")):
            return "vote_mistake"
        if any(token in text for token in ("验", "警徽", "神职")):
            return "role_execution"
        if any(token in text for token in ("误判", "信息", "忽略")):
            return "info_miss"
        return "decision_mistake"

    @staticmethod
    def _trigger_for_category(category: str, text: str) -> str:
        if category == "vote_mistake":
            return "投票或站边前证据链不完整"
        if category == "role_execution":
            return "神职信息或警徽流需要兑现时"
        if category == "info_miss":
            return "出现误判或关键信息缺失时"
        return text[:80] or "历史错误模式再次出现时"

    @staticmethod
    def _default_advice(role: str) -> str:
        if role == "seer":
            return "发言或投票前先核验验人时间线、警徽流和票型承接。"
        if role == "werewolf":
            return "行动前先统一狼队叙事，避免票型和发言目标互相冲突。"
        return "决策前先核验公开证据链，再给出结论。"

    @staticmethod
    def _theme(
        report: ReviewReport,
        patterns: list[ReflectionMistakePattern],
    ) -> str:
        if patterns:
            mapping = {
                "vote_mistake": "投票前核验证据链",
                "role_execution": "角色能力先对齐公开逻辑",
                "info_miss": "先补齐关键信息再下结论",
            }
            return mapping.get(patterns[0].category, "复盘历史决策错误")
        if report.successful_strategies:
            return "保留可复用成功策略"
        return "保持证据优先"

    @staticmethod
    def _lesson(
        report: ReviewReport,
        patterns: list[ReflectionMistakePattern],
    ) -> str:
        if patterns:
            return (
                f"你过去作为{report.role}时出现过{patterns[0].category}。"
                f"下次遇到类似局面，先处理触发信号：{patterns[0].trigger}。"
            )
        if report.successful_strategies:
            return (
                f"你过去作为{report.role}时有可复用的成功策略："
                f"{_scrub_ids(report.successful_strategies[0])}。"
            )
        return "历史反思只作为参考，当前局仍以公开事实和身份规则为准。"

    @staticmethod
    def _phase_focus(report: ReviewReport) -> list[str]:
        text = " ".join(report.error_analysis + report.improvement_suggestions)
        phases: list[str] = []
        if any(token in text for token in ("投", "票", "站边")):
            phases.append("vote")
        if any(token in text for token in ("发言", "警徽", "悍跳")):
            phases.append("speech")
        return phases or ["general"]

    @staticmethod
    def _game_patterns(report: ReviewReport) -> list[str]:
        text = " ".join(report.error_analysis + report.improvement_suggestions)
        patterns: list[str] = []
        if "警徽" in text or "对跳" in text:
            patterns.append("badge_flow")
        if "误判" in text:
            patterns.append("misread")
        if "被欺骗" in text or report.deceived_by:
            patterns.append("deception")
        return patterns or ["post_game_review"]

    @staticmethod
    def _avoid_next_time(
        patterns: list[ReflectionMistakePattern],
    ) -> list[str]:
        if not patterns:
            return []
        return [
            "不要在证据链不完整时把历史经验当成本局事实。",
            patterns[0].wrong_action[:80],
        ]
