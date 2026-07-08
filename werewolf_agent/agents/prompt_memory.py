# -*- coding: utf-8 -*-
"""
渲染本局私有记忆、跨局学习、RAG 和认知校准 prompt 片段。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.prompt_memory import PromptMemoryMixin
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.agents.prompt_rag_memory import (
    build_rag_hints,
    render_rag_hint_cards,
    slim_rag_hint_items,
)
from werewolf_agent.runtime.context import REFLECTION_CARD_BUDGET

_MAX_LEARNING_TEXT_CHARS = 160
_MAX_RAG_TEXT_CHARS = 220
_MAX_LEARNING_CONTEXT_CHARS = 3_600
_PLAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


class PromptMemoryMixin:
    def _build_private_memory_hints(self) -> str:
        ctx = self.context
        # P0-M7: read only from private_memory_hints. The previous code
        # also fell back to ctx.visible_world_state.get("private_memory"),
        # which caused duplicate injection if both fields were populated
        # and risked surfacing stale data from an older code path.
        memory = ctx.private_memory_hints
        if not memory:
            return ""
        # P0-M1: prepend a "本局·第N轮·私有记忆" label so the LLM cannot
        # confuse this section with cross-game reflection memory or
        # with public speech.
        day_label = f"第{ctx.day_number}轮" if ctx.day_number else "首轮"
        # MEM-02: emit the P1-M10 caveat BEFORE the JSON payload so
        # the LLM sees the warning in the same paragraph as the
        # keyword-signal categories. Omitted when logic_flaws /
        # valid_points are both empty (caveat would be noise).
        caveat = ctx.private_memory_caveat or ""
        # Phase-1 audit: wrap the caveat in ``---`` markers so the
        # LLM cannot mistake the warning for a JSON key or treat it
        # as a hint payload entry.  Without the marker, the LLM has
        # been observed to fold caveat text into the JSON dict
        # (e.g. parsing "私有记忆" as a key).
        if caveat:
            caveat_block = f"---\n{caveat}\n---\n"
        else:
            caveat_block = ""
        return (
            f"【本局·{day_label}·私有记忆】以下只代表你在本局形成的观察、站边和私有思考，"
            "不是公开记录。"
            "【严禁】在公开发言中复述以下任何角色身份信息或暗示你从私有记忆中获知的身份。"
            "你在公开发言中只能使用公开可见的信息。\n"
            + caveat_block
            + self._compact_json(memory)
        )

    def _build_rag_hints(self) -> str:
        return build_rag_hints(self.context)

    _slim_rag_hint_items = staticmethod(slim_rag_hint_items)
    _render_rag_hint_cards = staticmethod(render_rag_hint_cards)

    def _build_learning_context(self) -> str:
        header = (
            "跨局学习参考: 以下内容只是历史经验与自我校准，"
            "不代表本局任何玩家真实身份。"
        )
        parts: list[tuple[str, int, str]] = []
        error_pattern = self._build_error_pattern_hint()
        if error_pattern:
            parts.append(("error_pattern", 4, error_pattern))
        reflection = self._build_reflection_memory_hints()
        if reflection:
            parts.append(("reflection", 3, reflection))
        profile = self._build_profile_memory_hint()
        if profile:
            parts.append(("profile", 2, profile))
        cognition = self._build_cognition_matrix_hint()
        if cognition:
            parts.append(("cognition", 1, cognition))
        rag = self._build_rag_hints()
        if rag:
            parts.append(("rag", 0, rag))
        if not parts:
            return ""

        active = list(parts)

        def render(items: list[tuple[str, int, str]]) -> str:
            return header + "\n\n" + "\n\n".join(text for _, _, text in items)

        while active and len(render(active)) > _MAX_LEARNING_CONTEXT_CHARS:
            lowest_priority = min(priority for _, priority, _ in active)
            for idx in range(len(active) - 1, -1, -1):
                if active[idx][1] == lowest_priority:
                    del active[idx]
                    break

        return render(active) if active else ""

    def _build_reflection_memory_hints(self) -> str:
        ctx = self.context
        if not ctx.reflection_memory_hints:
            return ""
        hints = self._slim_reflection_hints(
            ctx.reflection_memory_hints[:REFLECTION_CARD_BUDGET]
        )
        if not hints:
            return ""
        if any("recommended_action" in hint or "trigger_signals" in hint for hint in hints):
            return (
                "跨局反思记忆: 以下是你过往对局后的经验总结，不代表本局任何玩家真实身份。\n"
                "历史玩家 ID、身份真相和决策链不得映射到本局玩家。\n"
                + self._render_reflection_hint_cards(hints)
            )
        return (
            "跨局反思记忆: 以下是你过往对局后的经验总结，不代表本局任何玩家真实身份。\n"
            "历史玩家 ID、身份真相和决策链不得映射到本局玩家。\n"
            + self._compact_json(hints)
        )

    @staticmethod
    def _render_reflection_hint_cards(hints: list[dict[str, Any]]) -> str:
        def join_values(value: Any, *, fallback: str) -> str:
            if isinstance(value, list):
                text = "；".join(str(part).strip() for part in value if str(part).strip())
                return text or fallback
            text = str(value or "").strip()
            return text or fallback

        cards: list[str] = []
        for idx, hint in enumerate(hints, start=1):
            title = join_values(hint.get("theme"), fallback="历史反思")
            cards.append(
                f"\n\n反思 {idx}: {title}\n"
                f"- 触发信号: {join_values(hint.get('trigger_signals'), fallback='仅当本局公开事实相似时参考。')}\n"
                f"- 历史教训: {join_values(hint.get('lesson') or hint.get('summary') or hint.get('text'), fallback='历史经验仅供参考。')}\n"
                f"- 本局做法: {join_values(hint.get('recommended_action') or hint.get('actionable_advice'), fallback='先核验本局公开证据，再决定是否采用。')}\n"
                f"- 误用风险: {join_values(hint.get('misuse_risk'), fallback='不要把历史经验直接映射到本局玩家身份。')}"
            )
        return "".join(cards)

    def _build_error_pattern_hint(self) -> str:
        """reflect-cross-1: 跨局错误模式聚合,顶部强提示 LLM 自我修正。

        当 LLM 拿到"你历史最常犯的错误是 X"这种聚合信号,比单条反思更
        容易驱动行为改变。同时"保留优点 N 次"提醒 LLM 不要只学错误。
        """
        ctx = self.context
        ep = ctx.error_pattern_hint
        if not ep or ep.get("total_reflections", 0) == 0:
            return ""
        parts: list[str] = ["【跨局错误模式(基于你过往反思)】"]
        top = ep.get("top_mistakes") or []
        if top:
            mistake_strs = [
                f"{self._normalize_error_category(cat)}({count}次)"
                for cat, count in top
            ]
            parts.append(f"你最常犯的 2 类错误: {'、'.join(mistake_strs)}。")
        strengths = [
            self._clean_prompt_text(label, max_chars=40)
            for label in (ep.get("preserved_strength_labels") or [])
            if str(label or "").strip()
        ][:2]
        preserved = ep.get("preserved_strength_count", 0)
        if strengths:
            parts.append(f"可复用优点: {'、'.join(strengths)}。")
        elif preserved:
            parts.append(f"过去 {preserved} 局反思中出现过可保留优点,本局先核验再复用。")
        same = ep.get("same_role_reflections", 0)
        if same:
            parts.append(
                f"其中 {same} 局你拿过当前角色({ep.get('current_role', '?')}),"
                f"历史经验对当前角色特别相关。"
            )
        return "\n".join(parts)

    def _build_profile_memory_hint(self) -> str:
        ctx = self.context
        if not ctx.profile_memory_hint:
            return ""
        allowed_fields = (
            "games_played",
            "current_role",
            "current_role_games",
            "current_role_win_rate_pct",
            "win_rate_confidence",
            "summary",
        )
        live_hint = {}
        for key in allowed_fields:
            if key not in ctx.profile_memory_hint:
                continue
            value = ctx.profile_memory_hint[key]
            if isinstance(value, str):
                value = self._clean_prompt_text(
                    value,
                    max_chars=_MAX_LEARNING_TEXT_CHARS,
                )
            live_hint[key] = value
        if not live_hint:
            return ""
        return (
            "历史角色经验: 以下仅是样本量与当前角色经历，不代表本局能力高低。\n"
            + self._compact_json(live_hint)
        )

    def _build_cognition_matrix_hint(self) -> str:
        ctx = self.context
        if not ctx.cognition_matrix_hint:
            return ""
        suspects = ctx.cognition_matrix_hint.get("suspects")
        trusted = ctx.cognition_matrix_hint.get("trusted")
        if suspects is None and trusted is None:
            suspects = [
                value for value in ctx.cognition_matrix_hint.values()
                if isinstance(value, dict)
                and (
                    value.get("faction_lean") in {"wolf", "wolf_lean", "werewolf"}
                    or float(value.get("trust", 0.5) or 0.5) < 0.4
                )
            ]
            trusted = [
                value for value in ctx.cognition_matrix_hint.values()
                if isinstance(value, dict)
                and (
                    value.get("faction_lean") in {"good", "good_lean", "villager"}
                    or float(value.get("trust", 0.5) or 0.5) > 0.6
                )
            ]
        suspects = suspects or []
        trusted = trusted or []
        live_hint = {
            "tracked_suspect_count": len(suspects),
            "tracked_trusted_count": len(trusted),
            "open_question_count": sum(
                len(item.get("open_questions") or [])
                for item in list(suspects) + list(trusted)
                if isinstance(item, dict)
            ),
            "evidence_anchor_count": sum(
                len(item.get("key_evidence") or [])
                for item in list(suspects) + list(trusted)
                if isinstance(item, dict)
            ),
            "calibration_warning": "只复核证据充分性，不复用历史玩家名单。",
        }
        return (
            "认知校准摘要: 历史矩阵只用于提醒你重新核验判断，不提供本局嫌疑名单。\n"
            + self._compact_json(live_hint)
        )

    @classmethod
    def _slim_reflection_hints(cls, hints: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed_fields = (
            "role",
            "result",
            "theme",
            "summary",
            "text",
            "lesson",
            "trigger_signals",
            "recommended_action",
            "misuse_risk",
            "actionable_advice",
        )
        slimmed: list[dict[str, Any]] = []
        for item in hints:
            if not isinstance(item, dict):
                continue
            slim: dict[str, Any] = {}
            for key in allowed_fields:
                if key not in item:
                    continue
                value = item[key]
                if isinstance(value, str):
                    value = cls._clean_prompt_text(
                        value,
                        max_chars=_MAX_LEARNING_TEXT_CHARS,
                    )
                elif isinstance(value, list):
                    value = [
                            cls._clean_prompt_text(
                            part,
                            max_chars=_MAX_LEARNING_TEXT_CHARS,
                        )
                        for part in value
                        if str(part or "").strip()
                    ][:4]
                slim[key] = value
            if slim:
                slimmed.append(slim)
        return slimmed

    @staticmethod
    def _normalize_error_category(category: Any) -> str:
        raw = str(category or "").strip().lower()
        mapping = {
            "vote_mistake": "投票错误",
            "info_miss": "漏读信息",
            "role_execution": "角色执行",
            "speech_quality": "发言质量",
            "format_error": "格式错误",
            "parse_error": "格式错误",
        }
        if raw in mapping:
            return mapping[raw]
        cleaned = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", raw)
        cleaned = _PLAYER_ID_RE.sub("历史玩家", cleaned)
        return cleaned[:24] or "other"
