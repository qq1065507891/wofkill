# -*- coding: utf-8 -*-
"""
提供狼人夜晚统一击杀和旧共识兼容节点。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.nodes.wolf_consensus import wolf_consensus
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import agent_wolf_consensus
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    _action_audit_events,
    _alive_wolves,
    _allocate_decision_identity,
    _dispatch_agent,
    _ensure_runtime_audit_state,
    _judge_broadcast,
    _planned_wolf_kill,
    _player_display,
    _timer_expired,
    logger,
)
from werewolf_agent.runtime.wolf_no_kill_policy import no_kill_policy_for_state


def _compat(name: str, fallback: Any) -> Any:
    """读取旧 facade 上的 monkeypatch，保持测试和外部补丁路径兼容。"""
    try:
        from werewolf_agent.runtime.nodes import night as night_mod

        return getattr(night_mod, name, fallback)
    except (ImportError, AttributeError):
        return fallback


def _legacy_wolf_consensus(state: RuntimeState) -> dict[str, Any]:
    """兼容旧式 agent 或脚本动作，但所有空刀统一交给 NoKillPolicy。"""
    gs: GameState = state["game_state"]
    policy = no_kill_policy_for_state(state)

    if _timer_expired(state, "wolf_discussion"):
        logger.debug("  [狼人决策] 讨论超时，空刀")
        return policy.resolve(
            gs,
            reason_code="provider_unavailable",
            extra_payload={"legacy_reason": "timer_expired"},
        )

    if state.get("agent_registry") and not state.get("wolf_action"):
        wolves = _compat("_alive_wolves", _alive_wolves)(gs)
        decision_identities = {
            wolf_id: _allocate_decision_identity(
                state,
                player_id=wolf_id,
                phase="wolf_consensus",
                task_type="wolf_consensus",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            for wolf_id in wolves
        }
        exposure_collectors = {
            wolf_id: ModuleExposureAuditCollector()
            for wolf_id in wolves
        }
        result = _compat("_dispatch_agent", _dispatch_agent)(
            state,
            _compat("agent_wolf_consensus", agent_wolf_consensus),
            timeout_override=AGENT_TIMEOUTS.wolf_consensus,
            decision_identities=decision_identities,
            exposure_collectors=exposure_collectors,
        )
        if result is None:
            logger.debug("  [狼人决策] Agent调用超时，空刀")
            return policy.resolve(gs, reason_code="provider_unavailable")

        action = result.get("wolf_action", "kill")
        target = result.get("wolf_kill_target_id")
        audit_events: list[GameEvent] = []
        for wolf_id, action_trace in (result.get("action_traces") or {}).items():
            audit_events.extend(_action_audit_events(
                state=state,
                player_id=wolf_id,
                phase="wolf_consensus",
                action_trace=action_trace,
                decision_identity=(
                    result.get("action_decision_identities") or {}
                ).get(wolf_id),
                exposure_collector=(
                    result.get("action_exposure_collectors") or {}
                ).get(wolf_id),
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        audited_gs = replace(gs, events=[*gs.events, *audit_events])

        if action == "no_kill":
            logger.debug("  [狼人决策] 狼人主动选择空刀")
            return policy.resolve(
                audited_gs,
                reason_code="strategic_abstain",
                event_type="wolf_no_kill_declared",
                extra_payload={
                    "legacy_reason": result.get(
                        "wolf_action_reason",
                        "agent decision",
                    ),
                    "action_traces": result.get("action_traces", {}),
                },
            )
        if action == "kill" and target:
            target_state = gs.players.get(target)
            if (
                target_state is not None
                and target_state.alive
                and target_state.role != "werewolf"
            ):
                logger.debug(
                    "  [狼人决策] 击杀目标: %s",
                    _player_display(state, target),
                )
                event = GameEvent(
                    type="wolf_kill_selected",
                    payload={
                        "night_number": gs.night_number,
                        "target_id": target,
                        "action_traces": result.get("action_traces", {}),
                    },
                )
                return {
                    "game_state": replace(
                        audited_gs,
                        events=[*audited_gs.events, event],
                    ),
                    "wolf_kill_target_id": target,
                }
            return policy.resolve(
                audited_gs,
                reason_code="invalid_primary",
            )
        return policy.resolve(
            audited_gs,
            reason_code="plan_generation_failed",
        )

    action = state.get("wolf_action")
    target = state.get("wolf_kill_target_id")
    if action == "no_kill":
        return policy.resolve(
            gs,
            reason_code="strategic_abstain",
            event_type="wolf_no_kill_declared",
            extra_payload={
                "legacy_reason": state.get("wolf_action_reason", ""),
            },
        )

    if (action == "kill" or action is None) and target is not None:
        target_state = gs.players.get(target)
        if (
            target_state is None
            or not target_state.alive
            or target_state.role == "werewolf"
        ):
            return policy.resolve(gs, reason_code="invalid_primary")
        event = GameEvent(
            type="wolf_kill_selected",
            payload={"night_number": gs.night_number, "target_id": target},
        )
        return {
            "game_state": replace(gs, events=[*gs.events, event]),
            "wolf_kill_target_id": target,
        }

    return policy.resolve(gs, reason_code="plan_generation_failed")


def wolf_consensus(state: RuntimeState) -> dict[str, Any]:
    """优先执行权威私有结构化立场，否则兼容旧式显式动作。"""
    gs: GameState = state["game_state"]
    gs, _ = _judge_broadcast(
        phase="wolf_kill_choice",
        message="狼人请统一选择今晚的行动",
        gs=gs,
        night_number=gs.night_number,
        visibility="moderator_only",
    )
    _ensure_runtime_audit_state(state)
    state = {**state, "game_state": gs}
    planned = _planned_wolf_kill(state)

    if planned is not None and not state.get("wolf_action"):
        result = planned
    else:
        result = _legacy_wolf_consensus(state)

    result_gs = result.get("game_state", gs)
    result_gs, _ = _judge_broadcast(
        phase="wolf_sleep",
        message="狼人请闭眼",
        gs=result_gs,
        night_number=result_gs.night_number,
        visibility="moderator_only",
    )
    return {**result, "game_state": result_gs}


__all__ = ["_legacy_wolf_consensus", "wolf_consensus"]
