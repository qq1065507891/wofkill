# -*- coding: utf-8 -*-
"""
处理日间自由讨论的发言顺序和发言采集节点。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-18

使用示例:
    内部运行时节点模块。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import (
    agent_day_speech,
)
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
    _timer_expired,
)
from werewolf_agent.evaluation.balance_public_claims import (
    public_speech_history,
    sanitize_public_text,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
)
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.skill_opportunity_events import (
    append_private_skill_event,
    can_select_self_destruct,
    is_live_werewolf,
)
from werewolf_agent.runtime.agent_adapter import agent_sheriff_pick_speech_order
from werewolf_agent.runtime.agent_action_audit import (
    build_runtime_terminal_fallback_trace,
)


def _terminal_speech_trace(reason: str) -> dict[str, Any]:
    """构造不含模型私密内容的稳定发言机会审计。"""
    failure_stage = {
        "speech_timeout": "runtime",
        "pre_supplied_speech_text": "runtime",
        "agent_dispatch_error": "provider",
        "self_destruct_before_speech": "runtime",
        "missing_action_trace": "protocol",
        "agent_unavailable": "registry",
    }.get(reason, "runtime")
    return build_runtime_terminal_fallback_trace(
        reason_code=reason,
        failure_stage=failure_stage,
        fallback_kind="ordinary_speech",
        final_action_type="speech",
    )


def free_discussion(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    speech_order = state.get("speech_order", [])
    speech_index = state.get("speech_index", 0)
    speaker_id = state.get("current_speaker_id")

    # 首次进入时公告自由讨论开始。
    if speech_index == 0 and not speaker_id:
        if gs.sheriff_id and gs.sheriff_badge_state == "active":
            gs, _ = _judge_broadcast(
                phase="discussion_start",
                message=f"请警长{_player_display(state, gs.sheriff_id)}指定发言顺序，开始自由讨论",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )
        else:
            gs, _ = _judge_broadcast(
                phase="discussion_start",
                message="本局无警长，由法官随机指定发言顺序，开始自由讨论",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )

    # 根据警长状态自动填充发言顺序。
    if not speech_order:
        if gs.sheriff_id and gs.sheriff_badge_state == "active":
            # 警长 agent 选择首发言人；失败时回退到静态顺序。
            decision_identity = _allocate_decision_identity(
                state,
                player_id=gs.sheriff_id,
                phase="sheriff_speech_order",
                task_type="speech_order",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
            agent_order_result = _dispatch_agent(
                state,
                agent_sheriff_pick_speech_order,
                gs.sheriff_id,
                timeout_override=AGENT_TIMEOUTS.day_speech,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                include_action_trace=True,
            )
            action_trace = None
            if isinstance(agent_order_result, dict):
                agent_order = agent_order_result.get("speech_order")
                action_trace = agent_order_result.get("action_trace")
            else:
                agent_order = agent_order_result
            if action_trace:
                gs = replace(gs, events=gs.events + _action_audit_events(
                    state=state,
                    player_id=gs.sheriff_id,
                    phase="sheriff_speech_order",
                    action_trace=action_trace,
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            else:
                exposure_collector.flush_events()
            speech_order = agent_order or choose_sheriff_led_speech_order(gs, gs.sheriff_id)
        else:
            speech_order = choose_no_sheriff_speech_order(gs)

    # 从已有或自动生成的发言顺序中剔除已死亡玩家。
    speech_order = [pid for pid in speech_order if pid not in gs.players or gs.players[pid].alive]

    # P0-G3223805846-A3: 在下面分发给 `agent_day_speech` 前，先把刚计算出的
    # 顺序写回 `state`。否则当天首个发言会读到 `state.get("speech_order") == []`
    # （局部 `speech_order` 只写入返回的 `advance_speaker` 字典，而 LangGraph
    # 会在节点返回后才应用），导致预言家后置位“立即起跳”指令无法在首发言人处触发。
    # 原地写入是安全的：规范状态更新仍通过 `advance_speaker()` 的返回值传播。
    state["speech_order"] = speech_order

    if speech_index == 0 and not speaker_id:
        order_names = "、".join(_player_display(state, pid) for pid in speech_order)
        gs, _ = _judge_broadcast(
            phase="speech_order",
            message=f"本轮发言顺序: {order_names}",
            gs=gs,
            day_number=gs.day_number,
            visibility="public",
            extra_payload={"speech_order": speech_order},
        )

    if speaker_id is None and speech_index < len(speech_order):
        speaker_id = speech_order[speech_index]

    def advance_speaker() -> dict[str, Any]:
        next_index = speech_index + 1
        next_speaker = speech_order[next_index] if next_index < len(speech_order) else None
        return {
            "speech_index": next_index,
            "current_speaker_id": next_speaker,
            "speech_timed_out": False,
            "speech_text": "",
            "speech_order": speech_order,
        }

    timed_out = state.get("speech_timed_out", False) or (
        speaker_id is not None and _timer_expired(state, f"speech:{speaker_id}")
    )

    if speaker_id and timed_out:
        decision_identity = _allocate_decision_identity(
            state,
            player_id=speaker_id,
            phase="speech",
            task_type="speech",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
        events = _action_audit_events(
            state=state,
            player_id=speaker_id,
            phase="speech",
            action_trace=_terminal_speech_trace("speech_timeout"),
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        events.append(GameEvent(
            type="speech_timeout",
            payload={
                "player_id": speaker_id,
                "day_number": gs.day_number,
                "seconds_limit": state.get("speech_seconds_limit", 0),
            },
        ))
        gs = replace(gs, events=gs.events + events)
        return {"game_state": gs, **advance_speaker()}
    if speaker_id:
        # 法官公告当前发言人。
        gs, _ = _judge_broadcast(
            phase="speaker_turn",
            message=f"请{_player_display(state, speaker_id)}发言",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        speech_text = state.get("speech_text", "")
        decision_identity = _allocate_decision_identity(
            state,
            player_id=speaker_id,
            phase="speech",
            task_type="speech",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
        action_trace = (
            _terminal_speech_trace("pre_supplied_speech_text")
            if speech_text else None
        )
        seer_credibility_audit = None
        self_destruct_available = (
            not speech_text
            and is_live_werewolf(gs, speaker_id)
        )
        if not speech_text:
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_opportunity",
                    actor_id=speaker_id,
                    day_number=gs.day_number,
                    opportunity_phase="day_speech",
                )
            try:
                result = _dispatch_agent(
                    state,
                    agent_day_speech,
                    speaker_id,
                    timeout_override=AGENT_TIMEOUTS.day_speech,
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                )
            except Exception:
                logger.warning("发言 agent 调度失败，记录安全终止机会")
                result = None
                action_trace = _terminal_speech_trace("agent_dispatch_error")
            if result is not None:
                if (
                    result.get("self_destruct")
                    and self_destruct_available
                    and can_select_self_destruct(
                        gs,
                        actor_id=speaker_id,
                        day_number=gs.day_number,
                        opportunity_phase="day_speech",
                    )
                ):
                    gs = append_private_skill_event(
                        gs,
                        "self_destruct_selected",
                        actor_id=speaker_id,
                        day_number=gs.day_number,
                        opportunity_phase="day_speech",
                    )
                    trace = result.get("action_trace") or _terminal_speech_trace(
                        "self_destruct_before_speech"
                    )
                    gs = replace(gs, events=gs.events + _action_audit_events(
                        state=state,
                        player_id=speaker_id,
                        phase="speech",
                        action_trace=trace,
                        decision_identity=decision_identity,
                        exposure_collector=exposure_collector,
                        day_number=gs.day_number,
                        night_number=gs.night_number,
                    ))
                    return {"game_state": gs, "self_destruct_wolf_id": speaker_id}
                speech_text = result.get("speech_text", "")
                action_trace = result.get("action_trace") or _terminal_speech_trace(
                    "missing_action_trace"
                )
                seer_credibility_audit = result.get("seer_credibility_audit")
            elif action_trace is None:
                action_trace = _terminal_speech_trace("agent_unavailable")
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=speaker_id,
                    day_number=gs.day_number,
                    opportunity_phase="day_speech",
                    reason_code=("agent_unavailable" if result is None else "continued_speech"),
                )
        player_role = gs.players[speaker_id].role if speaker_id in gs.players else "?"
        logger.debug(f"  [{_player_display(state, speaker_id)}({player_role})]: {speech_text if speech_text else '(未发言)'}")
        if not speech_text.strip():
            # 空文本不公开，但保留 moderator-only 决策事实供真实分母审计。
            gs = replace(gs, events=gs.events + _action_audit_events(
                state=state,
                player_id=speaker_id,
                phase="speech",
                action_trace=action_trace,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
            advanced = advance_speaker()
            if advanced["current_speaker_id"] is None:
                gs, _ = _judge_broadcast(
                    phase="discussion_end",
                    message="所有玩家发言完毕，自由讨论结束",
                    gs=gs,
                    day_number=gs.day_number,
                    visibility="public",
                )
            return {"game_state": gs, **advanced}
        speech_text, redacted_claims = sanitize_public_text(
            speech_text,
            public_speech_history(gs.events),
        )
        payload = {
            "speaker": speaker_id,
            "day_number": gs.day_number,
            "text": speech_text,
        }
        if redacted_claims:
            payload["redacted_public_claims"] = redacted_claims
        events = _action_audit_events(
            state=state,
            player_id=speaker_id,
            phase="speech",
            action_trace=action_trace,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        if seer_credibility_audit:
            events.append(GameEvent(
                type="seer_credibility_audit",
                payload=seer_credibility_audit,
            ))
        events.append(GameEvent(type="speech", payload=payload))
        gs = replace(gs, events=gs.events + events)
        advanced = advance_speaker()
        if advanced["current_speaker_id"] is None:
            gs, _ = _judge_broadcast(
                phase="discussion_end",
                message="所有玩家发言完毕，自由讨论结束",
                gs=gs,
                day_number=gs.day_number,
                visibility="public",
            )
        return {"game_state": gs, **advanced}
    gs = replace(gs, events=gs.events + [GameEvent(type="free_discussion", payload={})])
    return {"game_state": gs}
