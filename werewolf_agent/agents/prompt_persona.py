# -*- coding: utf-8 -*-
"""
渲染玩家人格相关的 prompt 片段。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.prompt_persona import PromptPersonaMixin
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.persona_runtime.router import sanitize_persona_snapshot


class PromptPersonaMixin:
    def _build_persona(self) -> str:
        ctx = self.context
        if not ctx.persona_snapshot:
            return ""
        sanitized_snapshot = sanitize_persona_snapshot(
            ctx.persona_snapshot,
            own_role=ctx.own_role or "",
            task_type=ctx.task_type.value,
        )
        lines = ["人格设定:"]
        text_fields = (
            ("personality", "人格核心"),
            ("speech_style", "表达风格"),
            ("task_style", "任务风格"),
            ("tone", "语气"),
        )
        for key, label in text_fields:
            value = sanitized_snapshot.get(key)
            if value:
                lines.append(f"- {label}: {self._clean_prompt_text(value)}")
        effective = self._slim_numeric_params(
            sanitized_snapshot.get("effective_params")
        )
        if effective:
            lines.append(f"- 稳定倾向: {self._compact_json(effective)}")
        adjustments = self._slim_numeric_params(
            sanitized_snapshot.get("dynamic_adjustments")
        )
        if adjustments:
            lines.append(f"- 本轮调整: {self._compact_json(adjustments)}")
        if len(lines) == 1:
            return ""
        lines.append("注意: 人格只影响表达和决策风格，不代表身份信息、公开事实或固定战术。")
        return "\n".join(lines)

    @staticmethod
    def _slim_numeric_params(value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        slim: dict[str, float] = {}
        for key, raw in value.items():
            if key == "deception_skill" or key.endswith("_rank"):
                continue
            if not isinstance(raw, (int, float)):
                continue
            slim[str(key)] = round(float(raw), 2)
        return slim
