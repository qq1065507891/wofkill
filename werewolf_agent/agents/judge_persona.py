# -*- coding: utf-8 -*-
"""
法官人格提示词和事实边界注入辅助函数。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> inject_persona_prompt("请宣布结果", profile_router=None)[1] is not None
    True
"""

from __future__ import annotations

from werewolf_agent.persona_runtime.judge_router import (
    JudgePersonaSnapshot,
    JudgeProfileRouter,
)


JUDGE_FACT_ONLY_SYSTEM_PROMPT = (
    "你只能根据调用中明确提供的公开字段播报结果，不得补充隐藏身份、技能使用或夜间行动。"
    "不得推断平安夜原因；不得把无人死亡归因于守护、解药、空刀或任何未公开行动。"
)

JUDGE_FACT_ONLY_USER_BOUNDARY = (
    "仅陈述上文明确给出的公开结果，不得补充未提供的身份、技能或夜间原因。"
)


def resolve_persona(
    profile_router: JudgeProfileRouter | None,
    profile_id: str,
    task_type: str = "judge_phase",
) -> JudgePersonaSnapshot | None:
    """解析当前法官人格；未配置 router 时返回 None。"""
    if profile_router is None:
        return None
    return profile_router.resolve(profile_id=profile_id, task_type=task_type)


def build_persona_system_prompt(
    profile_router: JudgeProfileRouter | None,
    profile_id: str,
    task_type: str = "judge_phase",
) -> str:
    """构造人格 system prompt，并始终追加事实边界约束。"""
    persona = resolve_persona(profile_router, profile_id, task_type)
    persona_prompt = persona.system_prompt if persona is not None else ""
    return "\n\n".join(
        part for part in (persona_prompt, JUDGE_FACT_ONLY_SYSTEM_PROMPT)
        if part
    )


def inject_persona_prompt(
    prompt: str,
    *,
    profile_router: JudgeProfileRouter | None,
    profile_id: str = "tournament_referee",
    task_type: str = "judge_phase",
) -> tuple[str, str | None]:
    """返回带事实边界的 user prompt 和独立 system prompt。"""
    bounded_prompt = f"{prompt}\n{JUDGE_FACT_ONLY_USER_BOUNDARY}"
    system_prompt = build_persona_system_prompt(
        profile_router,
        profile_id,
        task_type,
    ) or None
    return bounded_prompt, system_prompt


__all__ = [
    "JUDGE_FACT_ONLY_SYSTEM_PROMPT",
    "JUDGE_FACT_ONLY_USER_BOUNDARY",
    "JudgePersonaSnapshot",
    "JudgeProfileRouter",
    "build_persona_system_prompt",
    "inject_persona_prompt",
    "resolve_persona",
]
