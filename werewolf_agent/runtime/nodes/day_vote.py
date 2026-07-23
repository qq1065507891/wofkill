# -*- coding: utf-8 -*-
"""
采集日间投票并结算投票、PK 和放逐结果。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-23

使用示例:
    内部运行时节点模块。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_day_vote,
)
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _action_trace_event,
    _allocate_decision_identity,
    _dispatch_agent,
    _judge_broadcast,
    _has_pending_hunter_shot,
    _jb,
    _ensure_runtime_audit_state,
    _player_display,
    _public_vote_reason,
    _with_vote_target_in_trace,
)
from werewolf_agent.runtime.nodes.day_finish import _commit_victory
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.evaluation.balance_public_claims import (
    public_speech_history,
    sanitize_public_text,
)


def day_vote(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]

    # 有警长时，警长归票节点已经公告过 vote_start。
    _already_announced = (
        gs.sheriff_id
        and gs.sheriff_badge_state == "active"
        and gs.players.get(gs.sheriff_id)
        and gs.players[gs.sheriff_id].alive
    )
    if not _already_announced:
        gs, _ = _jb(
            state,
            phase="vote_start",
            message="讨论结束，现在开始投票。所有人同时投票，投票时不能发言。",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )

    gs, _ = _jb(
        state,
        phase="vote_collect",
        message="请所有仍在场玩家同时投票",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    same_vote_window = (
        state.get("exile_vote_day") == gs.day_number
        and state.get("exile_vote_revote") == state.get("revote", False)
    )
    existing_votes = state.get("exile_votes", {}) if same_vote_window else {}
    is_revote = state.get("revote", False)
    if is_revote:
        logger.debug(f"\n  --- PK重新投票 ---")
    else:
        logger.debug(f"\n  --- 投票开始 ---")
    votes: dict[str, str] = {}
    vote_traces: dict[str, Any] = {}
    vote_identities: dict[str, Any] = {}
    pending_exposure_events = _ensure_runtime_audit_state(state)["pending_exposure_events_by_trace"]
    has_agents = False
    # 统计可投票玩家，供逐个唱票使用。
    eligible_voter_ids = [pid for pid, p in gs.players.items() if p.alive and p.vote_enabled]
    total_voters = len(eligible_voter_ids)
    sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
    if not existing_votes:
        for idx, pid in enumerate(eligible_voter_ids, start=1):
            # 法官逐个广播唱票。
            voter_name = _player_display(state, pid)
            gs, _ = _jb(
                state,
                phase="vote_calling",
                message=f"请{voter_name}投票，第{idx}/{total_voters}位",
                gs=gs, day_number=gs.day_number,
                visibility="public",
                judge_method="vote_calling",
                extra_payload={
                    "voter_id": pid,
                    "voter_name": voter_name,
                    "position": idx,
                    "total": total_voters,
                    "sheriff_weight": 1.5 if pid == sheriff_id else 1.0,
                },
            )
            decision_identity = _allocate_decision_identity(
                state,
                player_id=pid,
                phase="vote",
                task_type="vote",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
            result = _dispatch_agent(
                state,
                agent_day_vote,
                pid,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
            )
            if result is not None:
                has_agents = True
                if result.get("vote_target"):
                    votes[pid] = result["vote_target"]
                    logger.debug(f"    {_player_display(state, pid)} → {_player_display(state, result['vote_target'])}")
                else:
                    logger.debug(f"    {_player_display(state, pid)} 弃票")
                if result.get("action_trace"):
                    vote_traces[pid] = result["action_trace"]
                    vote_identities[pid] = decision_identity
                    pending_exposure_events[decision_identity.trace_id()] = (
                        exposure_collector.flush_events()
                    )
            else:
                exposure_collector.flush_events()
                has_agents = True
                logger.warning(f"    {_player_display(state, pid)} 投票超时（视为弃票）")

        if has_agents:
            gs = _broadcast_vote_details(state, gs, votes)
            return {
                "game_state": gs,
                "exile_votes": votes,
                "vote_action_traces": vote_traces,
                "vote_decision_identities": vote_identities,
                "pending_exposure_events_by_trace": pending_exposure_events,
                "exile_vote_day": gs.day_number,
                "exile_vote_revote": state.get("revote", False),
                "revote": state.get("revote", False),
            }
    if existing_votes:
        gs = _broadcast_vote_details(state, gs, existing_votes)
    return {
        "game_state": gs,
        "exile_votes": existing_votes,
        "exile_vote_day": gs.day_number,
        "exile_vote_revote": state.get("revote", False),
        "revote": state.get("revote", False),
    }


def _broadcast_vote_details(
    state: RuntimeState,
    gs: GameState,
    votes: dict[str, str],
) -> GameState:
    gs, _ = _jb(
        state,
        phase="vote_end",
        message="投票结束，开始统计票型",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
    # 构造票型统计，用于结构化法官公告。
    tally: dict[str, float] = {}
    vote_lines = []
    engine = state.get("engine")
    sheriff_vote_weight = (
        engine.sheriff_vote_weight()
        if isinstance(engine, RuleEngine)
        else 1.5
    )
    for voter_id, target_id in votes.items():
        weight = (
            engine.vote_weight(gs, voter_id)
            if isinstance(engine, RuleEngine)
            else (1.5 if voter_id == sheriff_id else 1.0)
        )
        tally[target_id] = tally.get(target_id, 0.0) + weight
        weight_label = f" (警长{sheriff_vote_weight}票)" if voter_id == sheriff_id else ""
        vote_lines.append(
            f"{_player_display(state, voter_id)}{weight_label} 投票给 {_player_display(state, target_id)}"
        )
    message = "投票结果："
    if vote_lines:
        message += "\n" + "\n".join(vote_lines)
    else:
        message += "无有效票"
    player_names = {pid: _player_display(state, pid) for pid in gs.players}
    gs, _ = _jb(
        state,
        phase="vote_result",
        message=message,
        gs=gs, day_number=gs.day_number,
        visibility="public",
        judge_method="vote_tally",
        extra_payload={
            "tally": tally,
            "player_names": player_names,
            "sheriff_id": sheriff_id,
            "sheriff_weight": sheriff_vote_weight,
        },
    )
    return gs


def resolve_vote(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    consecutive = state.get("consecutive_no_exile_days", 0)
    raw_votes = state.get("exile_votes", {})
    # P1-G3223805846-2: 前置过滤死人/vote_enabled=False 的 vote，与 engine.resolve_vote 行为一致。
    # 防止 checkpoint 恢复/重放时残留的死人票进入 vote_resolved payload 的
    # weighted_tally / vote_weights / votes 列表，避免审计事件虚高计数。
    eligible_voter_ids = {
        pid for pid, p in gs.players.items() if p.alive and p.vote_enabled
    }
    votes = {vid: tgt for vid, tgt in raw_votes.items() if vid in eligible_voter_ids}
    result = engine.resolve_vote(
        gs, votes=votes,
        revote=state.get("revote", False),
        consecutive_no_exile_days=consecutive,
        pk_candidates=state.get("pk_candidates"),
        rng_seed=f"{gs.game_id}-vote-d{gs.day_number}",
    )
    # 按带权票数记录票型，权重来自规则配置。
    sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
    base_vote_weight = engine.base_vote_weight()
    sheriff_vote_weight = engine.sheriff_vote_weight()
    weighted_tally: dict[str, float] = {}
    vote_weights: dict[str, float] = {}
    if votes:
        for voter_id, target_id in votes.items():
            weight = engine.vote_weight(gs, voter_id)
            vote_weights[voter_id] = weight
            weighted_tally[target_id] = weighted_tally.get(target_id, 0) + weight
        tally_items = sorted(weighted_tally.items(), key=lambda x: -x[1])
        tally_text = "  投票统计: " + ", ".join(
            f"{_player_display(state, t)}:{v}票" for t, v in tally_items
        )
    else:
        tally_text = "  投票统计: 无有效票"
    logger.debug(tally_text)
    if result.exiled_player_id:
        gs, _ = _judge_broadcast(
            phase="vote_result_announce",
            message=f"投票结果：{_player_display(state, result.exiled_player_id)} 以最高票被放逐出局",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] {_player_display(state, result.exiled_player_id)} 被放逐 (原因: {result.reason})")
    elif result.reason == "first_tie_pk":
        tied_names = "、".join(_player_display(state, t) for t in (result.tied_player_ids or []))
        gs, _ = _judge_broadcast(
            phase="vote_tie_pk",
            message=f"首次平票：{tied_names}进入PK发言",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] 首次平票，进入PK: {[_player_display(state, t) for t in (result.tied_player_ids or [])]}")
    elif result.reason == "second_tie_no_exile":
        gs, _ = _judge_broadcast(
            phase="vote_second_tie",
            message="二次平票，无人出局，直接进入黑夜",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] 二次平票，无人出局")
    elif result.reason == "anti_stall_empty_tally":
        gs, _ = _judge_broadcast(
            phase="vote_anti_stall",
            message=f"防死循环强制放逐：{_player_display(state, result.exiled_player_id)}出局",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] 防死循环强制放逐: {_player_display(state, result.exiled_player_id)}")
    else:
        logger.debug(f"  [投票结果] {result.reason}")
    public_history = public_speech_history(gs.events)
    public_votes: list[dict[str, Any]] = []
    for voter_id, target_id in sorted(votes.items()):
        reason, redacted_claims = sanitize_public_text(
            _public_vote_reason((state.get("vote_action_traces") or {}).get(voter_id)),
            public_history,
        )
        vote = {"voter": voter_id, "target": target_id, "reason": reason}
        if redacted_claims:
            vote["redacted_public_claims"] = redacted_claims
        public_votes.append(vote)
    payload: dict[str, Any] = {
        "exiled": result.exiled_player_id,
        "reason": result.reason,
        "day_number": gs.day_number,
        "sheriff_id": sheriff_id,
        "sheriff_vote_weight": sheriff_vote_weight if sheriff_id else base_vote_weight,
        "weighted_tally": weighted_tally,
        "vote_weights": vote_weights,
        "votes": public_votes,
    }
    if result.tied_player_ids:
        payload["tied"] = result.tied_player_ids
    vote_trace_events = []
    for pid, trace in (state.get("vote_action_traces") or {}).items():
        decision_identity = (state.get("vote_decision_identities") or {}).get(pid)
        trace_id = decision_identity.trace_id() if decision_identity else ""
        pending_by_trace = state.get("pending_exposure_events_by_trace", {})
        exposure_events = pending_by_trace.pop(trace_id, []) if trace_id else []
        audit_event = _action_trace_event(
            player_id=pid,
            phase="vote",
            action_trace=_with_vote_target_in_trace(trace, state.get("exile_votes", {}).get(pid, "")),
            day_number=gs.day_number,
            night_number=gs.night_number,
            decision_identity=decision_identity,
        )
        thought = audit_event.payload.get("private_vote_thought") or {}
        if thought:
            logger.debug(
                f"  [投票心理][仅主持人] {_player_display(state, pid)} -> "
                f"{_player_display(state, thought.get('target'))}: "
                f"站边={thought.get('standing_with_seer') or '未明确'}；"
                f"怀疑理由={thought.get('suspect_reason') or thought.get('public_reason') or '未说明'}；"
                f"排除理由={thought.get('not_voting_reason') or '未说明'}；"
                f"内心理由={thought.get('private_reason') or '未说明'}"
            )
        vote_trace_events.extend([*exposure_events, audit_event])
    gs = replace(gs, votes=votes,
                 events=gs.events + [GameEvent(
                     type="vote_resolved",
                     payload=payload,
                 )] + vote_trace_events)
    next_consecutive = (
        consecutive + 1 if result.reason == "second_tie_no_exile" else 0
    )
    next_state: dict[str, Any] = {
        "game_state": gs,
        "_vote_result": result,
        "consecutive_no_exile_days": next_consecutive,
    }
    if result.reason == "first_tie_pk":
        next_state["pk_candidates"] = result.tied_player_ids
    elif result.exiled_player_id is not None or result.reason == "second_tie_no_exile":
        next_state["pk_candidates"] = []
        next_state["exile_votes"] = {}
        next_state["vote_action_traces"] = {}
        next_state["exile_vote_revote"] = False
    return next_state


def resolve_exile(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    # 从最后一个 vote_resolved 事件读取被放逐玩家，而不是读 _vote_result；
    # 后者不在 RuntimeState 中，会被 LangGraph channel 丢弃。
    exiled_id = None
    for event in reversed(gs.events):
        if event.type == "vote_resolved":
            exiled_id = event.payload.get("exiled")
            break
    if exiled_id is None:
        return {"game_state": gs}
    # 先结算放逐，再公开广播，这样公告能反映白痴翻牌等特殊角色结果。
    gs, events = engine.resolve_exile(gs, target_id=exiled_id)
    gs = replace(gs, events=gs.events + events)
    exiled_role = gs.players.get(exiled_id, None)
    role_str = exiled_role.role if exiled_role else "?"
    idiot_revealed = any(ev.type == "idiot_revealed" for ev in events)
    if idiot_revealed:
        logger.debug(f"  [白痴亮牌] {_player_display(state, exiled_id)} 是白痴，翻牌后出局")
        gs, _ = _judge_broadcast(
            phase="idiot_revealed",
            message=(
                f"{_player_display(state, exiled_id)}亮出白痴身份，证明为好人，"
                "可发表遗言，随后出局"
            ),
            gs=gs, day_number=gs.day_number,
            extra_payload={"player_id": exiled_id},
            visibility="public",
        )
    else:
        gs, _ = _judge_broadcast(
            phase="exile",
            message=f"{_player_display(state, exiled_id)}被放逐出局",
            gs=gs, day_number=gs.day_number,
            extra_payload={"exiled": exiled_id},
            visibility="public",
        )
        logger.debug(f"  [放逐] {_player_display(state, exiled_id)}({role_str}) 被放逐出局")
    # 猎人死亡反应必须先完成；其余放逐在当前 step 原子提交胜负。
    if not _has_pending_hunter_shot(gs):
        gs = _commit_victory({**state, "game_state": gs})["game_state"]
    return {"game_state": gs}
