# -*- coding: utf-8 -*-
"""
提供女巫、预言家和混血儿夜晚身份行动节点。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.night_specialists import night_witch
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_hybrid_choose_master,
    agent_night_seer,
    agent_night_witch,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _ensure_runtime_audit_state,
    _find_role,
    _jb,
    _judge_broadcast,
    _player_display,
    _player_ids,
    _stable_seed,
)


def _compat(name: str, fallback: Any) -> Any:
    """读取旧 facade 上的 monkeypatch，保持测试和外部补丁路径兼容。"""
    try:
        from werewolf_agent.runtime.nodes import night as night_mod
        return getattr(night_mod, name, fallback)
    except (ImportError, AttributeError):
        return fallback

def night_witch(state: RuntimeState) -> dict[str, Any]:

    gs: GameState = state["game_state"]

    if _find_role(gs, "witch") is None:

        return {"game_state": gs, "use_antidote": False, "poison_target_id": None}

    gs, _ = _jb(

        state,

        phase="witch_wake",

        message="女巫请睁眼",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

        judge_method="skill_guide",

        extra_payload={

            "role": "witch",

            "player_id": _find_role(gs, "witch") or "",

            "player_name": _player_display(state, _find_role(gs, "witch") or ""),

            "available_actions": ["use_antidote", "use_poison", "no_action"],

        },

    )

    wolf_target = state.get("wolf_kill_target_id")

    if wolf_target:

        gs, _ = _jb(

            state,

            phase="witch_kill_info",

            message=f"今晚{_player_display(state, wolf_target)}被狼人杀害了",

            gs=gs, night_number=gs.night_number,

            visibility="witch_private",

            extra_payload={

                "wolf_kill_target_id": wolf_target,

                "role": "witch",

                "player_id": _find_role(gs, "witch") or "",

                "player_name": _player_display(state, _find_role(gs, "witch") or ""),

                "available_actions": ["use_antidote", "use_poison", "no_action"],

                "context_hints": {"wolf_kill_target": _player_display(state, wolf_target)},

            },

            judge_method="skill_guide",

        )

    gs, _ = _jb(

        state,

        phase="witch_choose",

        message="女巫请选择是否使用解药或毒药",

        gs=gs, night_number=gs.night_number,

        visibility="witch_private",

        judge_method="skill_guide",

        extra_payload={

            "role": "witch",

            "player_id": _find_role(gs, "witch") or "",

            "player_name": _player_display(state, _find_role(gs, "witch") or ""),

            "available_actions": ["use_antidote", "use_poison", "no_action"],

        },

    )

    _ensure_runtime_audit_state(state)
    state = {**state, "game_state": gs}

    # Try agent-driven decision first
    witch_id = _find_role(gs, "witch") or ""
    decision_identity = _allocate_decision_identity(
        state,
        player_id=witch_id,
        phase="witch_choose",
        task_type="night_action",
        day_number=gs.day_number,
        night_number=gs.night_number,
    )
    exposure_collector = ModuleExposureAuditCollector()
    result = _compat("_dispatch_agent", _dispatch_agent)(
        state,
        _compat("agent_night_witch", agent_night_witch),
        timeout_override=AGENT_TIMEOUTS.witch_action,
        decision_identity=decision_identity,
        exposure_collector=exposure_collector,
    )
    if result is not None:
        use_antidote = result.get("use_antidote", False)

        poison_target_id = result.get("poison_target_id")

        action_taken = "no_action"

        if use_antidote:

            action_taken = "use_antidote"

        elif poison_target_id:

            action_taken = "use_poison"

        wolf_target = state.get("wolf_kill_target_id")

        if use_antidote:

            logger.debug(f"  [女巫] 使用解药救了 {_player_display(state, wolf_target)}")

        if poison_target_id:

            logger.debug(f"  [女巫] 使用毒药毒了 {_player_display(state, poison_target_id)}")

        if not use_antidote and not poison_target_id:

            logger.debug(f"  [女巫] 不使用药水 (解药{'已用' if gs.antidote_used else '可用'}, 毒药{'已用' if gs.poison_used else '可用'})")

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
            gs = replace(gs, events=gs.events + _action_audit_events(
                state=state,
                player_id=witch_id,
                phase="witch_choose",
                action_trace=result["witch_action_trace"],
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        else:
            exposure_collector.flush_events()
        gs, _ = _judge_broadcast(
            phase="witch_sleep",
            message="女巫请闭眼",
            gs=gs, night_number=gs.night_number,

            visibility="moderator_only",

        )

        return {"game_state": gs, **result}
    exposure_collector.flush_events()

    # Scripted fallback
    gs, _ = _judge_broadcast(

        phase="witch_sleep",

        message="女巫请闭眼",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    return {"use_antidote": state.get("use_antidote", False),

            "poison_target_id": state.get("poison_target_id"),

            "game_state": gs}





def night_seer(state: RuntimeState) -> dict[str, Any]:

    gs: GameState = state["game_state"]

    if _find_role(gs, "seer") is None:

        return {"game_state": gs, "seer_target_id": None}

    gs, _ = _jb(

        state,

        phase="seer_wake",

        message="预言家请睁眼",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

        judge_method="skill_guide",

        extra_payload={

            "role": "seer",

            "player_id": _find_role(gs, "seer") or "",

            "player_name": _player_display(state, _find_role(gs, "seer") or ""),

            "available_actions": ["check_alignment"],

        },

    )

    gs, _ = _jb(

        state,

        phase="seer_choose",

        message="预言家请选择你要查验的玩家",

        gs=gs, night_number=gs.night_number,

        visibility="seer_private",

        judge_method="skill_guide",

        extra_payload={

            "role": "seer",

            "player_id": _find_role(gs, "seer") or "",

            "player_name": _player_display(state, _find_role(gs, "seer") or ""),

            "available_actions": ["check_alignment"],

        },

    )

    _ensure_runtime_audit_state(state)
    state = {**state, "game_state": gs}

    # Try agent-driven decision first
    seer_id = _find_role(gs, "seer") or ""
    decision_identity = _allocate_decision_identity(
        state,
        player_id=seer_id,
        phase="seer_choose",
        task_type="night_action",
        day_number=gs.day_number,
        night_number=gs.night_number,
    )
    exposure_collector = ModuleExposureAuditCollector()
    result = _compat("_dispatch_agent", _dispatch_agent)(
        state,
        _compat("agent_night_seer", agent_night_seer),
        timeout_override=AGENT_TIMEOUTS.seer_check,
        decision_identity=decision_identity,
        exposure_collector=exposure_collector,
    )
    if result is not None:
        target = result.get("seer_target_id")
        if target:
            logger.debug(f"  [预言家] 查验目标: {_player_display(state, target)}")
        if result.get("seer_action_trace"):
            gs = replace(gs, events=gs.events + _action_audit_events(
                state=state,
                player_id=seer_id,
                phase="seer_choose",
                action_trace=result["seer_action_trace"],
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        else:
            exposure_collector.flush_events()
        return {"game_state": gs, **result}
    exposure_collector.flush_events()


    # Scripted fallback

    return {"seer_target_id": state.get("seer_target_id"), "game_state": gs}







def first_night_hybrid_master(state: RuntimeState) -> dict[str, Any]:

    gs: GameState = state["game_state"]

    engine: RuleEngine = state["engine"]

    if gs.night_number != 1 or gs.hybrid_master_id is not None:

        return {}

    hybrid_id = _find_role(gs, "hybrid")

    if hybrid_id is None:

        return {}



    gs, _ = _judge_broadcast(

        phase="hybrid_wake",

        message=f"混血儿{_player_display(state, hybrid_id)}请睁眼，选择你的主人",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    gs, _ = _judge_broadcast(

        phase="hybrid_choose",

        message="混血儿请选择你的主人",

        gs=gs, night_number=gs.night_number,

        visibility="hybrid_private",

    )

    logger.debug(f"  [法官] 混血儿{_player_display(state, hybrid_id)}请睁眼，选择你的主人")



    master_target = state.get("hybrid_master_target_id")
    hybrid_decision_identity = None
    hybrid_exposure_collector = None
    hybrid_action_trace = None

    # Agent-driven: ask hybrid player to choose master
    if master_target is None:
        hybrid_decision_identity = _allocate_decision_identity(
            state,
            player_id=hybrid_id,
            phase="hybrid_choose",
            task_type="night_action",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        hybrid_exposure_collector = ModuleExposureAuditCollector()
        result = _compat("_dispatch_agent", _dispatch_agent)(
            state,
            _compat("agent_hybrid_choose_master", agent_hybrid_choose_master),
            hybrid_id,
            timeout_override=AGENT_TIMEOUTS.seer_check,
            decision_identity=hybrid_decision_identity,
            exposure_collector=hybrid_exposure_collector,
        )
        if result and result.get("master_target_id"):
            master_target = result["master_target_id"]
            hybrid_action_trace = result.get("action_trace")
        if not hybrid_action_trace:
            hybrid_exposure_collector.flush_events()


    # Fallback: random selection

    if master_target is None:

        import random

        candidates = [pid for pid in _player_ids(gs) if pid != hybrid_id]

        rng = random.Random(_stable_seed(gs.game_id, "hybrid_master"))

        master_target = rng.choice(candidates) if candidates else None



    if master_target is None:
        return {}
    gs, event = engine.choose_master(gs, hybrid_id=hybrid_id, master_id=master_target)
    gs = replace(gs, events=gs.events + [event])
    if hybrid_action_trace is not None:
        gs = replace(gs, events=gs.events + _action_audit_events(
            state=state,
            player_id=hybrid_id,
            phase="hybrid_choose",
            action_trace=hybrid_action_trace,
            decision_identity=hybrid_decision_identity,
            exposure_collector=hybrid_exposure_collector,
            day_number=gs.day_number,
            night_number=gs.night_number,
        ))
    gs, _ = _judge_broadcast(
        phase="hybrid_sleep",

        message="混血儿请闭眼",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    master_role = gs.players[master_target].role if master_target in gs.players else "?"

    logger.debug(f"  [混血儿] {_player_display(state, hybrid_id)} 选择了 {_player_display(state, master_target)}({master_role}) 作为主人")

    return {"game_state": gs}
