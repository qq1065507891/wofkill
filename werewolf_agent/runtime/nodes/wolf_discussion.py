# -*- coding: utf-8 -*-
"""
提供狼人夜晚讨论和队伍计划节点。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.nodes.wolf_discussion import wolf_discussion
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.runtime.event_metadata import new_game_event, stamp_new_events
from werewolf_agent.runtime.agent_adapter import agent_wolf_discussion
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _alive_wolves,
    _allocate_decision_identity,
    _build_wolf_team_plan,
    _dispatch_agent,
    _ensure_runtime_audit_state,
    _judge_broadcast,
    _player_display,
)
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_validated_wolf_target_stance,
)


def _append_stamped_events(gs: GameState, events: list[GameEvent]) -> GameState:
    """追加事件时统一补齐 V2 元数据，保持后续 source_event_id 单调唯一。"""
    stamped = stamp_new_events(
        gs.game_id,
        gs.events,
        [*gs.events, *events],
    )
    return replace(gs, events=stamped)


def _compat(name: str, fallback: Any) -> Any:
    """读取旧 facade 上的 monkeypatch，保持测试和外部补丁路径兼容。"""
    try:
        from werewolf_agent.runtime.nodes import night as night_mod
        return getattr(night_mod, name, fallback)
    except (ImportError, AttributeError):
        return fallback
def wolf_discussion(state: RuntimeState) -> dict[str, Any]:

    """Run multi-round private wolf strategy and produce a team plan."""

    gs: GameState = state["game_state"]

    gs, _ = _judge_broadcast(

        phase="wolf_wake",

        message="狼人请睁眼",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    gs, _ = _judge_broadcast(

        phase="wolf_discussion_start",

        message="狼人开始讨论今晚的行动",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    registry = state.get("agent_registry")



    if not registry:

        gs, _ = _judge_broadcast(

            phase="wolf_discussion_end",

            message="狼人讨论完毕",

            gs=gs, night_number=gs.night_number,

            visibility="moderator_only",

        )

        for wolf_id in _compat("_alive_wolves", _alive_wolves)(gs):
            base_payload = {
                "wolf_id": wolf_id,
                "round": 1,
                "night_number": gs.night_number,
                "text": "",
                "visibility": "werewolf_team_only",
            }
            event = new_game_event(
                gs,
                "wolf_discussion",
                base_payload,
                visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
            )
            stance = build_validated_wolf_target_stance(
                gs,
                event,
                wolf_id=wolf_id,
                round_number=1,
                raw_stance=None,
            )
            event = replace(
                event,
                payload={**base_payload, "target_stance": stance.model_dump()},
            )
            gs = _append_stamped_events(gs, [event])

        return {"game_state": gs}



    wolves = _compat("_alive_wolves", _alive_wolves)(gs)
    events: list[GameEvent] = []

    round_count = 3 if gs.night_number == 1 else 2

    logger.debug(f"  [狼人密谈] 狼人: {[_player_display(state, w) for w in wolves]}，共{round_count}轮")

    discussion_start = time.monotonic()
    _ensure_runtime_audit_state(state)
    for round_number in range(1, round_count + 1):
        # Check total discussion timeout

        elapsed = time.monotonic() - discussion_start

        if elapsed >= AGENT_TIMEOUTS.wolf_discussion_total:

            logger.debug(f"  [狼人密谈] 讨论总超时({elapsed:.0f}s/{AGENT_TIMEOUTS.wolf_discussion_total:.0f}s)，跳过剩余轮次")

            break

        round_state = dict(state)

        round_state["wolf_discussion_round"] = round_number

        for wolf_id in wolves:

            round_state["game_state"] = gs  # Latest gs with accumulated speeches

            decision_identity = _allocate_decision_identity(
                round_state,
                player_id=wolf_id,
                phase=f"wolf_discussion_round_{round_number}",
                task_type="wolf_discussion",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            exposure_collector = ModuleExposureAuditCollector()
            result = _compat("_dispatch_agent", _dispatch_agent)(
                round_state,
                _compat("agent_wolf_discussion", agent_wolf_discussion),
                wolf_id,
                timeout_override=AGENT_TIMEOUTS.wolf_discussion_per_player,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
            )
            speech_text = result.get("speech_text", "") if result else ""
            raw_stance = result.get("target_stance") if result else None

            logger.debug(

                f"    [第{round_number}轮] {_player_display(state, wolf_id)}(狼人): "

                f"{speech_text if speech_text else '(沉默)'}"

            )

            base_payload = {
                    "wolf_id": wolf_id,
                    "round": round_number,
                    "night_number": gs.night_number,
                    "text": speech_text,
                    "visibility": "werewolf_team_only",
            }
            disc_event = new_game_event(
                gs,
                "wolf_discussion",
                base_payload,
                visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
            )
            stance = build_validated_wolf_target_stance(
                gs,
                disc_event,
                wolf_id=wolf_id,
                round_number=round_number,
                raw_stance=raw_stance,
            )
            disc_event = replace(
                disc_event,
                payload={
                    **base_payload,
                    "target_stance": stance.model_dump(),
                },
            )

            # Immediately merge into gs so next wolf sees this speech
            gs = _append_stamped_events(gs, [disc_event])
            disc_event = gs.events[-1]

            events.append(disc_event)
            if result and result.get("action_trace"):
                trace_events = _action_audit_events(
                    state=round_state,
                    player_id=wolf_id,
                    phase=f"wolf_discussion_round_{round_number}",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                )
                before_count = len(gs.events)
                gs = _append_stamped_events(gs, trace_events)
                events.extend(gs.events[before_count:])
            else:
                exposure_collector.flush_events()


        # Check if wolves reached consensus after this round — end early if so

        if round_number < round_count:

            from werewolf_agent.runtime.wolf_strategy import (

                should_end_discussion_early,

                summarize_wolf_consensus,

            )

            mid_consensus = summarize_wolf_consensus(

                gs.events, wolves, night_number=gs.night_number

            )

            if should_end_discussion_early(mid_consensus, len(wolves)):

                logger.debug(f"  [狼人密谈] 第{round_number}轮已达成共识，提前结束讨论")

                break



    # Aggregate discussion into consensus plan, fallback to static plan

    # (events already merged incrementally into gs)

    #

    # NOTE: as of wolf-team-plan-llm-structured (2026-06-10), final

    # wolf_team_plan generation moved to the dedicated

    # `wolf_team_plan_node` (called after this node by the graph).

    # That node tries LLM captain decision first and falls back to

    # the legacy regex+static path implemented in

    # `_build_fallback_wolf_team_plan` below. This node no longer

    # emits `wolf_team_plan` events or returns the plan in state.

    gs, _ = _judge_broadcast(

        phase="wolf_discussion_end",

        message="狼人讨论完毕",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    return {"game_state": gs}

def _build_fallback_wolf_team_plan(

    state: RuntimeState,

    wolves: list[str],

) -> dict[str, Any]:

    """从结构化 stance 构建保守 fallback，绝不解析自由文本。"""
    from werewolf_agent.runtime.wolf_discussion_directives import (
        collect_current_wolf_target_stances,
    )

    gs: GameState = state["game_state"]
    plan = _build_wolf_team_plan(
        gs,
        previous_plan=state.get("wolf_team_plan"),
    )
    stances = collect_current_wolf_target_stances(gs)
    for priority, plan_key in (
        ("primary", "night_kill_primary"),
        ("backup", "night_kill_backup"),
    ):
        positive = [
            stance
            for stance in stances
            if (
                stance["priority"] == priority
                and stance["stance"] in {"propose", "support"}
                and stance["target_id"] is not None
            )
        ]
        plan[plan_key] = positive[-1]["target_id"] if positive else None

    plan["evidence_from_discussion"] = stances
    plan["day_push_target"] = None
    plan["evidence_quality"] = "weak" if any(
        plan.get(key) for key in ("night_kill_primary", "night_kill_backup")
    ) else "none"
    return plan

def wolf_team_plan_node(state: RuntimeState) -> dict[str, Any]:

    """Produce structured WolfTeamPlan via LLM captain, fallback to legacy.



    Replaces the legacy regex-extraction path inside wolf_discussion as

    the primary plan source. Graph wires this node between

    `wolf_discussion` and `wolf_consensus`.



    On success: emits `wolf_team_plan` event with consensus_method="llm".

    On fallback: emits `wolf_team_plan_fallback` audit event with reason,

    then emits `wolf_team_plan` with consensus_method="fallback".

    """

    from werewolf_agent.runtime.agent_adapter import agent_wolf_team_plan



    gs: GameState = state["game_state"]

    wolves = _compat("_alive_wolves", _alive_wolves)(gs)

    if not wolves:

        # No wolves alive — nothing to plan.

        return {"game_state": gs}



    registry = state.get("agent_registry")

    plan: dict[str, Any] | None = None

    fallback_reason: str | None = None



    if registry is None:

        fallback_reason = "no_registry"

    else:

        try:

            plan = agent_wolf_team_plan(

                state, engine=state.get("engine"), registry=registry,

            )

        except Exception as e:  # noqa: BLE001

            logger.debug(

                "[wolf_team_plan_node] agent_wolf_team_plan raised: %s", e

            )

            plan = None

            fallback_reason = f"agent_exception: {e}"



    events: list[GameEvent] = []

    if plan is None:

        # Fallback to legacy regex + static path

        if fallback_reason is None:

            failure_meta = state.get("wolf_team_plan_failure")
            if not isinstance(failure_meta, dict):
                failure_meta = {}

            fallback_reason = failure_meta.get("reason") or "llm_failed_or_unavailable"
        else:
            failure_meta = {}

        plan = _build_fallback_wolf_team_plan(state, wolves)

        plan["consensus_method"] = "fallback"

        plan.setdefault("captain_id", wolves[0] if wolves else None)

        fallback_payload = {
            "night_number": gs.night_number,
            "reason": fallback_reason,
            "visibility": "werewolf_team_only",
        }
        for key in (
            "stage",
            "attempts",
            "last_error",
            "captain_id",
            "normalization_triggered",
            "normalization_repairs",
        ):
            if key in failure_meta:
                fallback_payload[key] = failure_meta[key]

        if failure_meta.get("normalization_triggered") is True:
            plan["normalization_triggered"] = True
            plan["normalization_repairs"] = list(
                failure_meta.get("normalization_repairs") or []
            )

        events.append(GameEvent(

            type="wolf_team_plan_fallback",

            payload=fallback_payload,

        ))

        logger.debug(

            "[wolf_team_plan_node] fallback path used, reason=%s", fallback_reason,

        )



    events.append(GameEvent(

        type="wolf_team_plan",

        payload={**plan, "visibility": "werewolf_team_only"},

    ))

    gs = replace(gs, events=gs.events + events)

    return {"game_state": gs, "wolf_team_plan": plan}
