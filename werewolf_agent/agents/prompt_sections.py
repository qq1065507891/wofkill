# -*- coding: utf-8 -*-
"""
管理玩家 user prompt 的分段元数据与预算裁剪逻辑。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.prompt_sections import PromptSectionMixin
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _SectionSpec:
    """Single source of truth for user-prompt section metadata."""

    builder_name: str
    label: str
    display_name: str
    info_kind: str
    drop_tier: int | None
    public_record: bool = False


_NEVER_DROP_TIER: int | None = None
_USER_PROMPT_BUDGET_CHARS = 20_000
USER_PROMPT_BUDGET_CHARS = _USER_PROMPT_BUDGET_CHARS

USER_SECTION_SPECS: tuple[_SectionSpec, ...] = (
    _SectionSpec("_build_phase_context", "【辅助】", "阶段上下文", "action_context", _NEVER_DROP_TIER),
    _SectionSpec("_build_public_summary", "【场上记录】", "当前局公开事实", "public_record", _NEVER_DROP_TIER, True),
    _SectionSpec("_build_visible_state", "【辅助】", "可见世界状态", "public_record", _NEVER_DROP_TIER, True),
    _SectionSpec("_build_salience_events", "【辅助】", "关键事件", "public_record", _NEVER_DROP_TIER, True),
    _SectionSpec("_build_recent_transcript", "【场上记录】", "近期发言", "public_record", _NEVER_DROP_TIER, True),
    _SectionSpec("_build_persona", "【人格】", "人格设定", "style", _NEVER_DROP_TIER),
    _SectionSpec("_build_belief_state", "【辅助】", "我的判断", "private_reasoning", 2),
    _SectionSpec("_build_contradiction_alerts", "【辅助】", "公开矛盾点", "private_reasoning", 2),
    _SectionSpec("_build_seer_credibility", "【辅助】", "预言家线可信度", "private_reasoning", 2),
    _SectionSpec("_build_possible_worlds", "【辅助】", "可能世界假设", "private_reasoning", 2),
    _SectionSpec("_build_simulation_predictions", "【辅助】", "未来预测", "private_reasoning", 2),
    _SectionSpec("_build_private_memory_hints", "【辅助】", "本局·私有记忆", "private_memory", 1),
    _SectionSpec("_build_learning_context", "【参考】", "跨局学习参考", "cross_game_reference", 0),
    _SectionSpec("_build_strategy_directive", "【策略指令】", "策略指令", "directive", _NEVER_DROP_TIER),
    _SectionSpec("_build_final_output_guard", "【硬约束】", "最终输出约束", "output_constraint", _NEVER_DROP_TIER),
)
SECTION_SPEC_BY_NAME: dict[str, _SectionSpec] = {
    spec.builder_name: spec for spec in USER_SECTION_SPECS
}
NEVER_DROP_SECTIONS: frozenset[str] = frozenset(
    spec.builder_name
    for spec in USER_SECTION_SPECS
    if spec.drop_tier is _NEVER_DROP_TIER
)
LOW_VALUE_SECTIONS: frozenset[str] = frozenset(
    spec.builder_name for spec in USER_SECTION_SPECS if spec.drop_tier == 0
)
SECTION_PRIORITIES: dict[str, str] = {
    spec.builder_name: spec.label
    for spec in USER_SECTION_SPECS
}


class PromptSectionMixin:
    _USER_SECTION_SPECS = USER_SECTION_SPECS
    _SECTION_SPEC_BY_NAME = SECTION_SPEC_BY_NAME
    _NEVER_DROP = NEVER_DROP_SECTIONS
    _LOW_VALUE_SECTIONS = LOW_VALUE_SECTIONS
    _SECTION_PRIORITIES = SECTION_PRIORITIES

    def _label_section(self, builder_name: str, body: str) -> str:
        """Prepend the priority label to a section's body.

        P1-S3: Empty bodies are returned unchanged so the section
        just disappears from the prompt (preserving the existing
        `for p in parts if p` filter behavior).
        """
        if not body:
            return body
        label = self._SECTION_PRIORITIES.get(builder_name, "")
        if not label:
            return body
        return f"{label} {body}"

    def _enforce_budget(
        self,
        parts: list[tuple[str, str]],
    ) -> str:
        """Join parts with blank-line separator, then trim if over budget."""
        full_joined = "\n\n".join(p for _, p in parts if p)
        if len(full_joined) <= _USER_PROMPT_BUDGET_CHARS:
            return full_joined
        droppable: list[tuple[int, int]] = []
        for idx, (name, _) in enumerate(parts):
            if not name:
                continue
            spec = self._SECTION_SPEC_BY_NAME.get(name)
            if spec is None or spec.drop_tier is None:
                continue
            droppable.append((spec.drop_tier, idx))
        droppable.sort(key=lambda x: x[0])
        total = len(full_joined)
        dropped: set[int] = set()
        for tier, idx in droppable:
            if total <= _USER_PROMPT_BUDGET_CHARS:
                break
            if idx in dropped:
                continue
            spec = self._SECTION_SPEC_BY_NAME.get(parts[idx][0])
            if spec is None or spec.drop_tier != tier:
                continue
            body = parts[idx][1]
            if body:
                dropped.add(idx)
                total -= len(body) + 2
            else:
                dropped.add(idx)
        active = [
            p for i, (_, p) in enumerate(parts) if i not in dropped and p
        ]
        return "\n\n".join(active)
