# -*- coding: utf-8 -*-
"""
渲染策略指令和技能战术建议 prompt 片段。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.prompt_strategy import PromptStrategyMixin
"""

from __future__ import annotations

import json
from typing import Any

from werewolf_agent.agents.directive_priority import (
    HARD_CONSTRAINT_KEYS,
    REFERENCE_KEYS,
    SUGGESTION_KEYS,
)
from werewolf_agent.agents.prompt_formatting import clean_prompt_text

_MAX_SKILL_TACTICAL_ADVICE_ITEMS = 3
_MAX_SKILL_TACTICAL_ADVICE_CHARS = 180
_STRATEGY_GROUP_ORDER: tuple[frozenset[str], str, str] = (
    (HARD_CONSTRAINT_KEYS, "【硬约束】", "以下指令必须遵守（MUST）："),
    (SUGGESTION_KEYS, "【建议】", "以下指令为建议（SHOULD），偏离时需有充分理由："),
    (REFERENCE_KEYS, "【参考】", "以下为背景信息（REFERENCE），仅供决策参考："),
)


class PromptStrategyMixin:
    def _build_strategy_directive(self) -> str:
        """Render strategy_directive split into 3 priority sections.

        P0-S5: LLM previously saw a single flat JSON block with 20+ keys
        and no priority signal — had to guess hard vs soft. Now keys are
        grouped into 【硬约束】 (MUST), 【建议】 (SHOULD), 【参考】
        (REFERENCE). Unknown keys fall through to 参考 for forward-compat.

        NEW-R4-P1-1: ``skill_tactical_advice`` is a structured list of
        ``[{skill, advice, confidence}, ...]`` (S-07 contract). Running
        the whole section through ``_compact_json`` wraps it in a
        ``{"skill_tactical_advice":[{...}, ...]}`` JSON envelope that
        the LLM has to parse before reading the advice — wasted tokens
        and a parse-failure surface. We render that key with a
        dedicated human-readable bullet renderer:
        ``- [skill_name/confidence] advice_text``.
        """
        ctx = self.context
        if not ctx.strategy_directive:
            return ""

        grouped: dict[str, dict[str, Any]] = {
            header: {} for _, header, _ in _STRATEGY_GROUP_ORDER
        }
        reference_fallback_idx = next(
            i for i, (keys, header, _) in enumerate(_STRATEGY_GROUP_ORDER)
            if header == "【参考】"
        )
        reference_header = _STRATEGY_GROUP_ORDER[reference_fallback_idx][1]

        for key, value in ctx.strategy_directive.items():
            placed = False
            for i, (keys, header, _) in enumerate(_STRATEGY_GROUP_ORDER):
                if key in keys:
                    grouped[header][key] = value
                    placed = True
                    break
            if not placed:
                grouped[reference_header][key] = value

        parts: list[str] = ["本轮策略指令:"]
        for keys_set, header, label in _STRATEGY_GROUP_ORDER:
            section = grouped[header]
            if not section:
                continue
            parts.append(
                f"{header} {label}\n" + self._render_strategy_section(section)
            )
        return "\n\n".join(parts)

    def _render_strategy_section(self, section: dict[str, Any]) -> str:
        """Render a single strategy_directive section.

        NEW-R4-P1-1: if the section contains ``skill_tactical_advice``
        (a structured list per S-07), use a dedicated bullet renderer
        for that key and render the remaining keys via ``_compact_json``.
        This avoids the raw JSON envelope the LLM would otherwise have
        to parse before reading the advice.

        PR1: ``reflection_task`` is a free-form role-family reflection
        template whose section headers (【投票错误】 / 【保留的优点】 / …)
        downstream aggregation parses. Running it through
        ``_compact_json`` wraps the whole section as a JSON object
        ``{"reflection_task":"...【投票错误】...","game_outcome":"..."}``
        so the headers end up as escaped content inside a JSON string
        value — the LLM saw a background field, not MUST text, and
        emitted in-game speech instead of sectioned reflection
        (game g_415624166, 12/12 reflections lacked headers). When the
        section contains a string ``reflection_task``, render that key
        verbatim as plain text and render the rest via ``_compact_json``.
        """
        advice = section.get("skill_tactical_advice")
        if isinstance(advice, list) and advice:
            # Render the advice as a human-readable bullet list. Other
            # keys in the section still go through _compact_json — only
            # the structured advice is humanized.
            bullets = self._render_skill_tactical_advice(advice)
            rest = {k: v for k, v in section.items() if k != "skill_tactical_advice"}
            if not rest:
                return bullets
            return bullets + "\n" + self._compact_json(rest)

        reflection_task = section.get("reflection_task")
        if isinstance(reflection_task, str) and reflection_task:
            # Render the template verbatim as plain text so its section
            # headers are top-level readable lines, not a JSON string.
            rest = {k: v for k, v in section.items() if k != "reflection_task"}
            text = f"反思指令:\n{reflection_task.rstrip()}"
            if not rest:
                return text
            return text + "\n" + self._compact_json(rest)

        return self._compact_json(section)

    @staticmethod
    def _render_skill_tactical_advice(advice: list[Any]) -> str:
        """Render the S-07 ``skill_tactical_advice`` list as bullets.

        Format: ``- [skill_name/confidence] advice_text`` (one per line).
        Entries that are not dicts or that lack the expected keys are
        rendered defensively via ``json.dumps`` so a single bad entry
        doesn't break the whole list.

        NEW-R4-P1-1: bypasses ``_compact_json`` so the LLM sees plain
        text rather than a JSON envelope.
        """
        lines: list[str] = ["技能战术建议:"]
        rendered = advice[:_MAX_SKILL_TACTICAL_ADVICE_ITEMS]
        for entry in rendered:
            if not isinstance(entry, dict):
                text = clean_prompt_text(
                    json.dumps(entry, ensure_ascii=False),
                    max_chars=_MAX_SKILL_TACTICAL_ADVICE_CHARS,
                )
                lines.append(f"- {text}")
                continue
            skill = clean_prompt_text(
                entry.get("skill", ""),
                max_chars=40,
            )
            conf = entry.get("confidence", "")
            text = clean_prompt_text(
                entry.get("advice", ""),
                max_chars=_MAX_SKILL_TACTICAL_ADVICE_CHARS,
            )
            try:
                conf_str = f"{float(conf):.2f}"
            except (TypeError, ValueError):
                conf_str = str(conf)
            frame_lines = [f"- [{skill}/{conf_str}] {text}"]
            situation = clean_prompt_text(
                entry.get("situation_signature", ""),
                max_chars=120,
            )
            recommended_use = clean_prompt_text(
                entry.get("recommended_use", ""),
                max_chars=_MAX_SKILL_TACTICAL_ADVICE_CHARS,
            )
            risk_alerts = [
                clean_prompt_text(item, max_chars=100)
                for item in list(entry.get("risk_alerts") or [])[:2]
            ]
            counter_signals = [
                clean_prompt_text(item, max_chars=100)
                for item in list(entry.get("counter_signals") or [])[:2]
            ]
            forbidden_use = clean_prompt_text(
                entry.get("forbidden_use", ""),
                max_chars=120,
            )
            if situation:
                frame_lines.append(f"  适用局面：{situation}")
            if recommended_use:
                frame_lines.append(f"  本轮建议：{recommended_use}")
            if risk_alerts:
                frame_lines.append(f"  风险：{'；'.join(risk_alerts)}")
            if counter_signals:
                frame_lines.append(f"  不适用信号：{'；'.join(counter_signals)}")
            if forbidden_use:
                frame_lines.append(f"  禁止套用：{forbidden_use}")
            lines.extend(frame_lines)
        omitted = len(advice) - len(rendered)
        if omitted > 0:
            lines.append(f"- 其余 {omitted} 条技能战术建议已省略。")
        return "\n".join(lines)
