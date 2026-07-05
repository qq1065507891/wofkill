# -*- coding: utf-8 -*-
"""
整理狼队计划适配器使用的提示文本、成员校验和审计证据。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.wolf_team_plan_support import build_prior_plan_summary
    >>> build_prior_plan_summary({})
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from werewolf_agent.core.models import GameState


def collect_current_wolf_discussion_text(gs: GameState) -> str:
    """收集当前夜的狼队夜聊文本，没有文本时返回稳定占位内容。"""
    night_num = gs.night_number
    discussion_lines: list[str] = []
    for event in gs.events:
        text = (event.payload.get("text") or "").strip()
        if (
            event.type == "wolf_discussion"
            and event.payload.get("night_number") == night_num
            and text
        ):
            wolf_id = event.payload.get("wolf_id", "?")
            round_num = event.payload.get("round", "?")
            discussion_lines.append(f"[第{round_num}轮 {wolf_id}]: {text}")
    return "\n".join(discussion_lines) or "(本夜无夜聊文本)"


def build_prior_plan_summary(prior_plan: Mapping[str, Any]) -> str:
    """把上一夜狼队计划压缩成提示词中的一行摘要。"""
    if not prior_plan:
        return "无上局计划 (首夜)"
    return (
        f"上夜计划: fake_seer={prior_plan.get('fake_seer')}, "
        f"pusher={prior_plan.get('pusher')}, hooker={prior_plan.get('hooker')}, "
        f"deep_cover={prior_plan.get('deep_cover')}, "
        f"primary={prior_plan.get('night_kill_primary')}"
    )


def build_wolf_role_definitions(role_strategy: Mapping[str, str]) -> str:
    """把狼队角色策略说明压缩为提示词中的字段定义列表。"""
    return "\n".join(
        f"- {field_name}: {description.split(chr(10))[0]}"
        for field_name, description in role_strategy.items()
    )


def validate_wolf_team_plan_membership(
    plan: Any,
    alive_wolves: Sequence[str],
    alive_non_wolves: Sequence[str],
) -> str | None:
    """校验计划中的狼队分工和击杀目标是否落在合法候选集中。"""
    alive_wolf_ids = list(alive_wolves)
    alive_target_ids = list(alive_non_wolves)
    valid_wolves = set(alive_wolf_ids)
    valid_targets = set(alive_target_ids)
    for role in ("fake_seer", "pusher", "hooker", "deep_cover"):
        value = getattr(plan, role)
        if value is not None and value not in valid_wolves:
            return f"{role}={value} not in alive_wolves={alive_wolf_ids}"
    for field_name in ("night_kill_primary", "night_kill_backup"):
        value = getattr(plan, field_name)
        if value is not None and value not in valid_targets:
            return f"{field_name}={value} not in alive_non_wolves={alive_target_ids}"
    return None


def build_wolf_team_plan_evidence(
    plan_dict: Mapping[str, Any],
    captain_id: str,
) -> list[dict[str, Any]]:
    """为 LLM 队长选择的击杀目标生成下游审计需要的合成证据。"""
    synthetic_evidence: list[dict[str, Any]] = []
    if plan_dict.get("night_kill_primary"):
        synthetic_evidence.append({
            "target": plan_dict["night_kill_primary"],
            "wolf_id": captain_id,
            "reason": "llm_captain_decision",
        })
    if plan_dict.get("night_kill_backup"):
        synthetic_evidence.append({
            "target": plan_dict["night_kill_backup"],
            "wolf_id": captain_id,
            "reason": "llm_captain_backup",
        })
    return synthetic_evidence
