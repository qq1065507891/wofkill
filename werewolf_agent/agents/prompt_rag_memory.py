# -*- coding: utf-8 -*-
"""
渲染跨局 RAG 案例提示的安全摘要和卡片文本。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.prompt_rag_memory import slim_rag_hint_items
    >>> slim_rag_hint_items([{"type": "rag_hit", "title": "case"}])[0]["title"]
    'case'
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.prompt_formatting import clean_prompt_text
from werewolf_agent.agents.schemas import AgentContext

_MAX_LEARNING_TEXT_CHARS = 160
_MAX_RAG_TEXT_CHARS = 220


def build_rag_hints(context: AgentContext) -> str:
    """构建 live prompt 中可见的 RAG 案例参考段落。"""
    if not context.rag_hints:
        return ""
    # 只保留 rag_hit，避免调试或旁路元数据进入 live prompt。
    rag_only = [
        item for item in context.rag_hints
        if isinstance(item, dict) and item.get("type") == "rag_hit"
    ]
    if not rag_only:
        return ""

    from werewolf_agent.rag.prompt_renderer import RAG_LIVE_PROMPT_CAP

    slim_items = slim_rag_hint_items(rag_only[:RAG_LIVE_PROMPT_CAP])
    warning = (
        "⚠️ RAG 案例中的玩家 ID 与战术选择仅供启发；"
        "本局的玩家 ID、票型、遗言均与案例无关；"
        "不得直接套用案例中具体玩家的动作、票型或决策链。\n"
    )
    case_cards = render_rag_hint_cards(slim_items)
    tail = "（以上案例仅供参考，不得作为本局事实或硬性指令。）"
    if "…已截断" in case_cards:
        tail = tail + "（部分字段已截断，案例未完整呈现。）"
    return (
        "知识库提示: 知识库提示不是当前局事实，只能作为玩法经验和案例参考。\n"
        + warning
        + case_cards
        + "\n"
        + tail
    )


def slim_rag_hint_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """移除审计字段，只保留可进入 live prompt 的战术框架。"""
    from werewolf_agent.rag.tactical_text import prompt_safe_tactical_frame_dict

    slim: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        frame = prompt_safe_tactical_frame_dict(item)
        slim.append({
            "title": clean_prompt_text(
                item.get("title", ""),
                max_chars=_MAX_LEARNING_TEXT_CHARS,
            ),
            "situation_signature": clean_prompt_text(
                frame["situation_signature"],
                max_chars=_MAX_RAG_TEXT_CHARS,
            ),
            "transferable_lesson": clean_prompt_text(
                frame["transferable_lesson"],
                max_chars=_MAX_RAG_TEXT_CHARS,
            ),
            "applicability": [
                clean_prompt_text(
                    value,
                    max_chars=_MAX_RAG_TEXT_CHARS,
                )
                for value in frame["applicability"]
            ],
            "counter_signals": [
                clean_prompt_text(
                    value,
                    max_chars=_MAX_RAG_TEXT_CHARS,
                )
                for value in frame["counter_signals"]
            ],
            "recommended_use": clean_prompt_text(
                frame["recommended_use"],
                max_chars=_MAX_RAG_TEXT_CHARS,
            ),
            "misuse_risk": clean_prompt_text(
                frame["misuse_risk"],
                max_chars=_MAX_RAG_TEXT_CHARS,
            ),
        })
    return slim


def render_rag_hint_cards(items: list[dict[str, Any]]) -> str:
    """把 prompt-safe RAG 条目渲染为低优先级参考卡片。"""

    def join_values(value: Any, *, fallback: str) -> str:
        if isinstance(value, list):
            text = "；".join(str(part).strip() for part in value if str(part).strip())
            return text or fallback
        text = str(value or "").strip()
        return text or fallback

    cards: list[str] = []
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title") or "未命名案例")
        cards.append(
            f"案例 {idx}：{title}\n"
            f"- 适用局面：{join_values(item.get('situation_signature'), fallback='缺少局面描述。')}\n"
            f"- 可迁移原则：{join_values(item.get('transferable_lesson'), fallback='仅作为谨慎参考。')}\n"
            f"- 适用条件：{join_values(item.get('applicability'), fallback='仅当本局公开事实与案例局面相似时参考。')}\n"
            f"- 不适用信号：{join_values(item.get('counter_signals'), fallback='若当前局面不匹配则不要套用。')}\n"
            f"- 本局参考方式：{join_values(item.get('recommended_use'), fallback='作为参考，不作为直接指令。')}\n"
            f"- 误用风险：{join_values(item.get('misuse_risk'), fallback='过度套用可能误导判断。')}"
        )

    return "\n\n".join(cards)


__all__ = [
    "build_rag_hints",
    "render_rag_hint_cards",
    "slim_rag_hint_items",
]
