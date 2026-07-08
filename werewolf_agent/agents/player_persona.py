# -*- coding: utf-8 -*-
"""
解析并记录 PlayerAgent 每回合 persona snapshot。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.player_persona import attach_persona_snapshot
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from werewolf_agent.agents.schemas import AgentContext
from werewolf_agent.persona_runtime.router import GameContext, sanitize_persona_snapshot


def attach_persona_snapshot(agent: Any, context: AgentContext) -> AgentContext:
    """在渲染 prompt 前解析本回合 persona 快照。"""
    if context.persona_snapshot:
        sanitized = sanitize_persona_snapshot(
            context.persona_snapshot,
            own_role=context.own_role or "",
            task_type=context.task_type.value,
        )
        attached = (
            context
            if sanitized == context.persona_snapshot
            else context.model_copy(update={"persona_snapshot": sanitized})
        )
        record_persona_exposure(attached)
        return attached
    if not agent.persona_key or agent.persona_router is None:
        return context

    visible = context.visible_world_state or {}
    alive_players = set(visible.get("alive_players") or [])
    wolf_teammates = set(visible.get("wolf_teammates") or [])
    public_fragments = [
        str(item.get("text") or "")
        for item in context.recent_transcript
        if isinstance(item, dict)
    ]
    public_fragments.append(json.dumps(context.strategy_directive, ensure_ascii=False))
    player_id = re.escape(context.agent_id)
    suspicion_pattern = re.compile(
        rf"(?:(?:怀疑|质疑|施压).{{0,8}}{player_id}|"
        rf"{player_id}.{{0,8}}(?:可疑|狼面|有问题|矛盾|需要回应|承受压力))"
    )
    player_is_suspected = any(
        suspicion_pattern.search(fragment)
        for fragment in public_fragments
    )
    teammate_exiled = bool(
        context.own_role == "werewolf"
        and alive_players
        and any(teammate not in alive_players for teammate in wolf_teammates)
    )
    snapshot = agent.persona_router.resolve(
        agent.agent_id,
        context.task_type.value,
        GameContext(
            phase=context.phase,
            day_number=context.day_number,
            night_number=context.night_number,
            player_is_suspected=player_is_suspected,
            teammate_exiled=teammate_exiled,
            has_badge=visible.get("sheriff_id") == context.agent_id,
            own_role=context.own_role or "",
            alive=not alive_players or context.agent_id in alive_players,
        ),
    )
    data = asdict(snapshot)
    data.pop("agent_id", None)
    data.pop("base_params", None)
    attached = context.model_copy(update={"persona_snapshot": data})
    record_persona_exposure(attached)
    return attached


def record_persona_exposure(context: AgentContext) -> None:
    """把 persona 快照写入曝光采集器。"""
    identity = getattr(context, "decision_identity", None)
    collector = getattr(context, "exposure_collector", None)
    if identity is not None and collector is not None:
        collector.record_persona(identity, context.persona_snapshot)
