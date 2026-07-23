# -*- coding: utf-8 -*-
"""
处理女巫夜间节点的唤醒、行动派发和审计事件。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.nodes.night_witch_node import night_witch
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import agent_night_witch
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _ensure_runtime_audit_state,
    _find_role,
    _jb,
    _judge_broadcast,
    _player_display,
    logger,
)


def _compat(name: str, fallback: Any) -> Any:
    """读取旧 facade 上的 monkeypatch，保持测试和外部补丁路径兼容。"""
    try:
        from werewolf_agent.runtime.nodes import night as night_mod

        return getattr(night_mod, name, fallback)
    except (ImportError, AttributeError):
        return fallback


def _witch_role_payload(
    state: RuntimeState,
    gs: GameState,
    *,
    extra_fields: dict[str, Any] | None = None,
    context_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造女巫私有提示 payload，避免多处重复查找角色。"""
    witch_id = _find_role(gs, "witch") or ""
    payload: dict[str, Any] = {
        "role": "witch",
        "player_id": witch_id,
        "player_name": _player_display(state, witch_id),
        "available_actions": ["use_antidote", "use_poison", "no_action"],
    }
    if extra_fields:
        payload.update(extra_fields)
    if context_hints:
        payload["context_hints"] = context_hints
    return payload


def night_witch(state: RuntimeState) -> dict[str, Any]:
    """执行女巫夜间行动节点。"""
    gs: GameState = state["game_state"]
    witch_id = _find_role(gs, "witch")
    if witch_id is None:
        return {"game_state": gs, "use_antidote": False, "poison_target_id": None}

    gs, _ = _jb(
        state,
        phase="witch_wake",
        message="女巫请睁眼",
        gs=gs,
        night_number=gs.night_number,
        visibility="moderator_only",
        judge_method="skill_guide",
        extra_payload=_witch_role_payload(state, gs),
    )

    wolf_target = state.get("wolf_kill_target_id")
    if wolf_target:
        gs, _ = _jb(
            state,
            phase="witch_kill_info",
            message=f"今晚{_player_display(state, wolf_target)}被狼人杀害了",
            gs=gs,
            night_number=gs.night_number,
            visibility="witch_private",
            extra_payload=_witch_role_payload(
                state,
                gs,
                extra_fields={"wolf_kill_target_id": wolf_target},
                context_hints={
                    "wolf_kill_target": _player_display(state, wolf_target),
                },
            ),
            judge_method="skill_guide",
        )

    gs, _ = _jb(
        state,
        phase="witch_choose",
        message="女巫请选择是否使用解药或毒药",
        gs=gs,
        night_number=gs.night_number,
        visibility="witch_private",
        judge_method="skill_guide",
        extra_payload=_witch_role_payload(state, gs),
    )

    _ensure_runtime_audit_state(state)
    state = {**state, "game_state": gs}

    decision_identity = _allocate_decision_identity(
        state,
        player_id=witch_id,
        phase="witch_choose",
        task_type="night_action",
        day_number=gs.day_number,
        night_number=gs.night_number,
    )
    exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
    result = _compat("_dispatch_agent", _dispatch_agent)(
        state,
        _compat("agent_night_witch", agent_night_witch),
        decision_identity=decision_identity,
        exposure_collector=exposure_collector,
    )
    if result is None:
        exposure_collector.flush_events()
        gs, _ = _judge_broadcast(
            phase="witch_sleep",
            message="女巫请闭眼",
            gs=gs,
            night_number=gs.night_number,
            visibility="moderator_only",
        )
        return {
            "use_antidote": state.get("use_antidote", False),
            "poison_target_id": state.get("poison_target_id"),
            "game_state": gs,
        }

    use_antidote = result.get("use_antidote", False)
    poison_target_id = result.get("poison_target_id")
    action_taken = "no_action"
    if use_antidote:
        action_taken = "use_antidote"
    elif poison_target_id:
        action_taken = "use_poison"

    if use_antidote:
        logger.debug(f"  [女巫] 使用解药救了 {_player_display(state, wolf_target)}")
    if poison_target_id:
        logger.debug(f"  [女巫] 使用毒药毒了 {_player_display(state, poison_target_id)}")
    if not use_antidote and not poison_target_id:
        logger.debug(
            f"  [女巫] 不使用药水 "
            f"(解药{'已用' if gs.antidote_used else '可用'}, "
            f"毒药{'已用' if gs.poison_used else '可用'})"
        )

    audit = GameEvent(
        type="witch_decision_audit",
        payload={
            "night_number": gs.night_number,
            "wolf_kill_target_id": state.get("wolf_kill_target_id"),
            "action_taken": action_taken,
            "poison_target_id": poison_target_id,
            "reason": "agent_decision",
            "visibility": "witch_private",
            "action_trace": result.get("witch_action_trace"),
        },
    )
    gs = replace(gs, events=gs.events + [audit])
    if result.get("witch_action_trace"):
        gs = replace(
            gs,
            events=gs.events + _action_audit_events(
                state=state,
                player_id=witch_id,
                phase="witch_choose",
                action_trace=result["witch_action_trace"],
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ),
        )
    else:
        exposure_collector.flush_events()
    gs, _ = _judge_broadcast(
        phase="witch_sleep",
        message="女巫请闭眼",
        gs=gs,
        night_number=gs.night_number,
        visibility="moderator_only",
    )

    return {"game_state": gs, **result}


__all__ = ["night_witch"]
