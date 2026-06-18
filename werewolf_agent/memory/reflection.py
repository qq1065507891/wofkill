"""Long-term reflection memory: post-game unstructured experience storage.

Design doc §10: unstructured reflections (e.g. "last time I trusted
emotional speech and got misaligned") go into long-term memory.
Vote chains, claims, and attack/defense logic stay structured in RelationGraph.

This module stores reflections with tag-based retrieval. Vector search
is a future extension — current implementation uses exact tag/role matching.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from typing import Any

from werewolf_agent.memory.schemas import (
    CrossGameQuery,
    ReflectionEntry,
    ReflectionEntryV2,
    ReflectionMistakePattern,
    ReflectionPreservedStrength,
    ReflectionQualityStatus,
    ReviewReport,
)


_LOG = logging.getLogger(__name__)

_PLAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# Captures the 【保留的优点】 section body up to the next 【...】 header or end.
_LLM_STRENGTH_SECTION_RE = re.compile(
    r"【保留的优点】[】:：\s]*(.*?)(?=【[^】]+】|$)",
    re.DOTALL,
)
_LLM_TRUTH_TOKENS = ("实际", "真实身份", "底牌", "查验结果", "死亡原因")
_GENERIC_PHRASES = (
    "复盘失败对局，关注关键转折点的信息缺失",
    "关注关键转折点的信息缺失",
    "下局继续努力",
    "总结经验",
)
_SOURCE_TEXT_CAP = 800


def _scrub_ids(text: str) -> str:
    return _PLAYER_ID_RE.sub("[玩家ID已省略]", str(text or ""))


def _cap_source_text(text: str, max_chars: int = _SOURCE_TEXT_CAP) -> str:
    cleaned = _scrub_ids(text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 6] + "…已截断"


def _token_set(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
    return set(tokens)


def _jaccard(left: str, right: str) -> float:
    a = _token_set(left)
    b = _token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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
            "idiot": ("白痴", "翻牌", "免死"),
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
        truth_tokens = ("实际", "真实身份", "底牌", "查验结果", "死亡原因")
        return any(token in visible_blob for token in truth_tokens)

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
        deaths/checks) must not be promoted — so any bullet containing a
        truth token is dropped. ``ReflectionPreservedStrength`` has no
        ``fact_basis`` field, so LLM provenance is implicit (the section
        source is recorded in ``source.llm_self_review``). Returns at
        most 2 strengths.
        """
        match = _LLM_STRENGTH_SECTION_RE.search(str(llm_self_review or ""))
        if not match:
            return []
        body = match.group(1)
        strengths: list[ReflectionPreservedStrength] = []
        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if not stripped.startswith(("-", "•", "*")):
                # Only bullet lines are real strengths; the section
                # preamble (e.g. "本局做对的:") is skipped.
                continue
            line = _scrub_ids(stripped).lstrip("-•*").strip()
            if len(line) < 6:
                continue
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


class ReflectionMemory:
    """Stores and retrieves post-game reflections across games.

    When a ``repo`` (PostgresGameRepository) is provided, reflections are
    persisted to the ``reflections`` table.  Otherwise entries are kept
    in memory only (suitable for tests and single-run scenarios).
    """

    def __init__(self, repo: Any | None = None) -> None:
        self._entries: dict[str, ReflectionEntry] = {}
        self._v2_entries: dict[str, ReflectionEntryV2] = {}
        self._repo = repo
        if self._repo is not None:
            self._load_all()

    # -- Persistence --------------------------------------------------------

    def _load_all(self) -> None:
        try:
            rows = self._repo.load_all_reflections()
        except Exception:
            return
        for data in rows:
            try:
                if data.get("schema_version") == 2:
                    entry_v2 = ReflectionEntryV2.from_dict(data)
                    self._v2_entries[entry_v2.entry_id] = entry_v2
                else:
                    entry = ReflectionEntry.from_dict(data)
                    self._entries[entry.entry_id] = entry
            except Exception:
                pass

    def _persist(self, entry: ReflectionEntry, raise_on_failure: bool = False) -> None:
        if self._repo is None:
            return
        try:
            self._repo.save_reflection(entry.to_dict())
        except Exception:
            # MEM-25: by default, keep the legacy silent-on-failure
            # behavior so production callers don't start seeing new
            # exceptions after a dependency upgrade. Tests /
            # migration scripts can opt in via raise_on_failure=True
            # to surface the underlying error.
            if raise_on_failure:
                raise
            _LOG.warning(
                "Failed to persist reflection %s for player %s",
                entry.entry_id, entry.player_id, exc_info=True,
            )

    def _persist_v2(
        self,
        entry: ReflectionEntryV2,
        raise_on_failure: bool = False,
    ) -> None:
        if self._repo is None:
            return
        try:
            self._repo.save_reflection(entry.to_dict())
        except Exception:
            if raise_on_failure:
                raise
            _LOG.warning(
                "Failed to persist V2 reflection %s for player %s",
                entry.entry_id, entry.player_id, exc_info=True,
            )

    # -- CRUD ---------------------------------------------------------------

    def store(
        self,
        entry_or_game_id: ReflectionEntry | str = "",
        *,
        player_id: str = "",
        role: str = "",
        faction_won: str | bool = "",
        text: str = "",
        tags: list[str] | None = None,
        situation: str | dict | None = None,
        raise_on_failure: bool = False,
    ) -> None:
        """Store a reflection entry. Accepts either a ReflectionEntry or keyword args.

        MEM-25: ``raise_on_failure`` is forwarded to ``_persist``;
        when True, a DB write failure propagates out of ``store``
        instead of being logged. Default is False to preserve the
        legacy silent-on-failure behavior.
        """
        if isinstance(entry_or_game_id, ReflectionEntry):
            entry = entry_or_game_id
        else:
            # MEM-NEW-7: faction_won MUST be a bool. The pre-fix code
            # accepted strings via ``faction_won == "werewolf"``, which
            # silently mapped "true" / "yes" / "win" to False and
            # corrupted the reflection's win/loss record. Drop the
            # string fallback; raise TypeError so the bug surfaces at
            # the call site, not deep inside a cross-game query.
            if not isinstance(faction_won, bool):
                raise TypeError(
                    f"ReflectionMemory.store: faction_won must be bool, "
                    f"got {type(faction_won).__name__}: {faction_won!r}"
                )
            entry = ReflectionEntry(
                entry_id=uuid.uuid4().hex[:12],
                game_id=str(entry_or_game_id),
                player_id=player_id,
                role=role,
                faction_won=faction_won,
                text=text,
                tags=tags or [],
                situation=json.dumps(situation, ensure_ascii=False) if isinstance(situation, dict) else str(situation or ""),
            )
        self._entries[entry.entry_id] = entry
        self._persist(entry, raise_on_failure=raise_on_failure)

    def store_v2(
        self,
        entry: ReflectionEntryV2,
        *,
        raise_on_failure: bool = False,
    ) -> None:
        self._v2_entries[entry.entry_id] = entry
        self._persist_v2(entry, raise_on_failure=raise_on_failure)

    def get(self, entry_id: str) -> ReflectionEntry | None:
        return self._entries.get(entry_id)

    def all_entries(self) -> list[ReflectionEntry]:
        return list(self._entries.values())

    def all_v2_entries(self) -> list[ReflectionEntryV2]:
        return list(self._v2_entries.values())

    def count(self) -> int:
        return len(self._entries) + len(self._v2_entries)

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            if self._repo is not None:
                try:
                    self._repo.delete_reflection(entry_id)
                except Exception:
                    pass
            return True
        if entry_id in self._v2_entries:
            del self._v2_entries[entry_id]
            if self._repo is not None:
                try:
                    self._repo.delete_reflection(entry_id)
                except Exception:
                    pass
            return True
        return False

    # -- Query --------------------------------------------------------------

    def query(
        self,
        query: CrossGameQuery,
        vector_index: Any | None = None,
    ) -> list[ReflectionEntry]:
        """Retrieve reflections matching query criteria.

        P0-M6: when a ``vector_index`` is supplied, the query proceeds
        in two stages.

        1. **Filter** by hard constraints (player_id, role, tags,
           faction_won). These are required equality / membership
           predicates; semantic similarity cannot satisfy them.
        2. **Rank** by cosine similarity against ``query.situation`` (or
           ``query.text``). Entries absent from the index fall back to
           score 0 and are appended after scored entries.

        Without a vector index (or with an empty one) the path is the
        pure exact-match behavior used before P0-M6.
        """
        candidates = self._filter_candidates(query)

        # MEM-NEW-11: an object with __len__ but no similarity method
        # is not a valid vector index. Catch it HERE, in the guard,
        # so the rest of the function never sees it — the prior
        # post-guard ``hasattr`` check at the call site was a
        # defensive but easy-to-miss fallback, and an object with a
        # `similarity` attribute that raises on call would have
        # leaked through with an AttributeError deep inside the
        # rank loop. The contract for a valid index is BOTH
        # __len__ AND similarity; missing either one falls
        # through to the exact-match path with a warning.
        if (
            vector_index is None
            or not getattr(vector_index, "__len__", lambda: 0)()
            or not hasattr(vector_index, "similarity")
        ):
            if vector_index is not None and not hasattr(vector_index, "similarity"):
                # MEM-13: warn the caller — without a similarity
                # method every entry scores 0 and falls to the
                # unindexed tail, so a "vector search" silently
                # degrades to exact-match order. Make the fallback
                # loud so the upstream caller can fix the index
                # wiring.
                _LOG.warning(
                    "vector index has no similarity method, falling back"
                )
            # P0-M6: when situation is set, keep the pre-existing substring
            # match against the *situation* field; otherwise no-op.
            return self._apply_situation_filter(candidates, query)[: query.max_results]

        query_text = query.situation or ""
        scores = vector_index.similarity(query_text)

        scored: list[tuple[float, int, ReflectionEntry]] = []
        unindexed: list[ReflectionEntry] = []
        for idx, entry in enumerate(candidates):
            score = float(scores.get(entry.entry_id, 0.0))
            if score > 0.0 or entry.entry_id in scores:
                scored.append((score, -idx, entry))  # tie-break: original order
            else:
                unindexed.append(entry)
        # Higher similarity first; preserve original order on ties.
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        ranked = [entry for _, _, entry in scored] + unindexed
        return ranked[: query.max_results]

    def _filter_candidates(self, query: CrossGameQuery) -> list[ReflectionEntry]:
        """Apply exact-match *filters* (player_id, role, tags, faction_won).

        MEM-06: also sort by game_id descending (newest first) so the
        truncation step in ``query`` keeps the most recent experience
        when the candidate set is larger than ``max_results``.
        """
        results = list(self._entries.values())
        if query.player_id:
            results = [e for e in results if e.player_id == query.player_id]
        if query.role:
            results = [e for e in results if e.role == query.role]
        if query.tags:
            results = [
                e for e in results
                if any(t in e.tags for t in query.tags)
            ]
        if query.faction_won is not None:
            results = [e for e in results if e.faction_won == query.faction_won]
        # Newest first: sort by game_id descending. ``game_id`` is a
        # string; for game ids of the form ``g123`` lex order matches
        # numeric order once the prefix is aligned. For arbitrary ids,
        # reverse-sort is still a stable proxy for "newer first" when
        # ids are timestamp-derived.
        results.sort(key=lambda e: e.game_id, reverse=True)
        return results

    def _apply_situation_filter(
        self,
        candidates: list[ReflectionEntry],
        query: CrossGameQuery,
    ) -> list[ReflectionEntry]:
        if not query.situation:
            return list(candidates)
        return [
            e for e in candidates
            if query.situation.lower() in e.situation.lower()
        ]

    def by_player(self, player_id: str) -> list[ReflectionEntry]:
        return [e for e in self._entries.values() if e.player_id == player_id]

    def by_role(self, role: str) -> list[ReflectionEntry]:
        return [e for e in self._entries.values() if e.role == role]

    def by_game(self, game_id: str) -> list[ReflectionEntry]:
        return [e for e in self._entries.values() if e.game_id == game_id]

    def tag_index(self) -> dict[str, int]:
        """Return tag → count mapping for observability."""
        tag_counts: dict[str, int] = {}
        for entry in self._entries.values():
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    def query_live(self, query: CrossGameQuery) -> list[ReflectionEntryV2]:
        """Retrieve approved V2 reflections eligible for live prompts."""
        results = [
            entry for entry in self._v2_entries.values()
            if entry.quality_status == ReflectionQualityStatus.APPROVED
        ]
        if query.player_id:
            results = [e for e in results if e.player_id == query.player_id]
        if query.role:
            results = [e for e in results if e.role == query.role]
        if query.faction_won is not None:
            results = [e for e in results if e.faction_won == query.faction_won]
        if query.tags:
            results = [
                e for e in results
                if any(self._v2_has_tag(e, tag) for tag in query.tags)
            ]
        if query.situation:
            needle = query.situation.lower()
            results = [
                e for e in results
                if needle in json.dumps(e.to_dict(), ensure_ascii=False).lower()
            ]
        results.sort(key=self._live_rank_key)
        return results[: query.max_results]

    @staticmethod
    def _v2_has_tag(entry: ReflectionEntryV2, tag: str) -> bool:
        fields = [
            entry.role,
            entry.faction,
            *entry.situation_signature.phase_focus,
            *entry.situation_signature.game_patterns,
            *(p.category for p in entry.mistake_patterns),
        ]
        return tag in fields

    @staticmethod
    def _live_rank_key(entry: ReflectionEntryV2) -> tuple[float, str, str]:
        return (-entry.quality_score, str(entry.game_id), str(entry.entry_id))

    def live_error_pattern(
        self,
        player_id: str,
        role: str = "",
    ) -> dict[str, Any]:
        entries = self.query_live(CrossGameQuery(
            player_id=player_id,
            role=role,
            max_results=100,
        ))
        counter: Counter[str] = Counter()
        preserved = 0
        strength_labels: list[str] = []
        for entry in entries:
            for pattern in entry.mistake_patterns:
                if pattern.category:
                    counter[pattern.category] += 1
            if entry.preserved_strengths:
                preserved += 1
                for strength in entry.preserved_strengths[:1]:
                    if strength.category:
                        strength_labels.append(strength.category)
        total_mistakes = sum(counter.values())
        dominant = (
            round(counter.most_common(1)[0][1] / total_mistakes, 2)
            if total_mistakes and counter
            else 0.0
        )
        return {
            "top_mistakes": counter.most_common(2),
            "preserved_strength_count": preserved,
            "preserved_strength_labels": strength_labels[:2],
            "total_reflections": len(entries),
            "same_role_reflections": sum(1 for e in entries if e.role == role),
            "dominant_mistake_ratio": dominant,
            "current_role": role,
        }
