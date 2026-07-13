# -*- coding: utf-8 -*-
"""
组合玩家提示词，并把 persona 固定在最终 system 消息中。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.agents.prompt_composer import compose_system_prompt
    >>> compose_system_prompt(builder)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import RetryInfo


def compose_system_prompt(builder: Any) -> str:
    """按稳定区段顺序组合 system prompt。"""
    parts: list[str] = []
    parts.append(builder._build_core_identity())
    # Persona 必须位于最终 system 消息；request 层会在完整组装后取证。
    parts.append(builder._build_persona())
    parts.append(builder._build_game_rules())
    parts.append(builder._build_role_guide())
    parts.append(builder._build_information_boundaries())
    parts.append(builder._build_reasoning_method())
    # skill analyses 已预先注入，system prompt 只保留边界说明。
    parts.append(builder._build_skill_policy())
    parts.append(builder._build_output_contract())
    return "\n\n".join(p for p in parts if p)


def compose_user_prompt(builder: Any, retry: RetryInfo) -> str:
    """按动态区段顺序组合 user prompt，并委托 builder 执行预算裁剪。"""
    parts: list[tuple[str, str]] = []
    parts.append(("", "=== DYNAMIC_BOUNDARY ==="))
    parts.append((
        "_build_phase_context",
        builder._label_section("_build_phase_context", builder._build_phase_context()),
    ))
    parts.append((
        "_build_public_summary",
        builder._label_section("_build_public_summary", builder._build_public_summary()),
    ))
    parts.append((
        "_build_visible_state",
        builder._label_section("_build_visible_state", builder._build_visible_state()),
    ))
    parts.append((
        "_build_salience_events",
        builder._label_section("_build_salience_events", builder._build_salience_events()),
    ))
    parts.append((
        "_build_recent_transcript",
        builder._label_section("_build_recent_transcript", builder._build_recent_transcript()),
    ))
    parts.append((
        "_build_belief_state",
        builder._label_section("_build_belief_state", builder._build_belief_state()),
    ))
    parts.append((
        "_build_contradiction_alerts",
        builder._label_section(
            "_build_contradiction_alerts",
            builder._build_contradiction_alerts(),
        ),
    ))
    parts.append((
        "_build_seer_credibility",
        builder._label_section("_build_seer_credibility", builder._build_seer_credibility()),
    ))
    parts.append((
        "_build_possible_worlds",
        builder._label_section("_build_possible_worlds", builder._build_possible_worlds()),
    ))
    parts.append((
        "_build_simulation_predictions",
        builder._label_section(
            "_build_simulation_predictions",
            builder._build_simulation_predictions(),
        ),
    ))
    parts.append((
        "_build_private_memory_hints",
        builder._label_section(
            "_build_private_memory_hints",
            builder._build_private_memory_hints(),
        ),
    ))
    parts.append((
        "_build_learning_context",
        builder._label_section("_build_learning_context", builder._build_learning_context()),
    ))
    parts.append((
        "_build_strategy_directive",
        builder._label_section(
            "_build_strategy_directive",
            builder._build_strategy_directive(),
        ),
    ))
    parts.append(("", builder._build_task_prompt()))
    parts.append((
        "_build_final_output_guard",
        builder._label_section(
            "_build_final_output_guard",
            builder._build_final_output_guard(retry),
        ),
    ))
    return builder._enforce_budget(parts)
