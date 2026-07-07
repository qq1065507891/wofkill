# -*- coding: utf-8 -*-
"""
狼人夜间讨论、计划和共识相关 agent action 适配器。

作者: Mike
创建日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.runtime.agent_wolf_actions import agent_wolf_discussion
    >>> agent_wolf_discussion(...)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.agent_action_audit import _audit_context_kwargs
from werewolf_agent.runtime.agent_registry import AgentRegistry
from werewolf_agent.runtime.context import (
    _action_trace_payload,
    build_agent_context,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.json_extract import (
    extract_first_balanced_json_object as _extract_first_balanced_json_object,
)
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_empty_wolf_discussion_fallback,
    build_teammate_transcript,
    build_wolf_discussion_instruction,
    build_wolf_discussion_strategy_directive,
    collect_wolf_discussion_speeches,
    living_wolf_ids,
    living_wolf_teammates,
    teammate_discussion_speeches,
)
from werewolf_agent.runtime.wolf_kill_support import (
    _build_wolf_kill_directive,
    _single_wolf_vote,
)
from werewolf_agent.runtime.wolf_team_plan_support import (
    build_prior_plan_summary,
    build_wolf_role_definitions,
    build_wolf_team_plan_evidence,
    collect_current_wolf_discussion_text,
    validate_wolf_team_plan_membership,
)

logger = logging.getLogger(__name__)


def agent_wolf_team_plan(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
) -> dict[str, Any] | None:
    """LLM wolf-team captain produces structured WolfTeamPlan once per night.

    Captain = sorted(alive_wolves)[0] (deterministic, reproducible across runs).
    Replaces the legacy regex-based extraction
    (wolf_strategy.summarize_wolf_consensus + build_wolf_team_plan_from_discussion)
    which silently dropped role assignments when LLM dialogue used synonyms
    not covered by the extractor's keyword set (e.g. "悍跳位" != "假预言家").

    Returns None on any failure (captain agent unavailable, LLM error,
    schema validation failure, retry exhausted) - caller is expected to
    fall back to the legacy regex + static plan path and emit a
    `wolf_team_plan_fallback` audit event with the failure reason.

    On success, returns plan dict with all WolfTeamPlan fields plus
    `consensus_method="llm"` and `captain_id` for audit/replay.
    """
    from werewolf_agent.agents.schemas import WolfTeamPlan
    from werewolf_agent.agents.tool_schema import wolf_team_plan_tool
    from werewolf_agent.runtime.directives.wolf import _WOLF_ROLE_STRATEGY

    gs: GameState = state["game_state"]
    alive_wolves = sorted(
        pid for pid, p in gs.players.items() if p.role == "werewolf" and p.alive
    )
    if not alive_wolves:
        return None

    captain_id = alive_wolves[0]
    captain_agent = registry.get_agent(captain_id)
    if captain_agent is None:
        logger.debug(
            "[wolf_team_plan] captain %s agent unavailable, fallback", captain_id
        )
        return None

    alive_non_wolves = sorted(
        pid for pid, p in gs.players.items() if p.role != "werewolf" and p.alive
    )

    night_num = gs.night_number
    discussion_text = collect_current_wolf_discussion_text(gs)
    prior_summary = build_prior_plan_summary(state.get("wolf_team_plan") or {})
    role_defs = build_wolf_role_definitions(_WOLF_ROLE_STRATEGY)

    system_prompt = (
        f"你是狼队队长 {captain_id}。本夜是 N{night_num}。"
        f"队友夜聊已完成,现在由你一次性产出团队作战计划。\n\n"
        f"【4 角色定义 (字段名 ↔ 中文)】\n{role_defs}\n\n"
        f"【硬约束】\n"
        f"- 4 角色字段 (fake_seer/pusher/hooker/deep_cover) 互不相同, 都从 "
        f"alive_wolves={alive_wolves} 中选; 任一字段可填 null (本夜不分配该位置)\n"
        f"- 击杀目标 (night_kill_primary/backup) 必须从 alive_non_wolves 中选 "
        f"或填 null (空刀); 不能是狼队成员\n"
        f"- public_story 1~120 字 (白天对外口径, 例: '昨夜平安, 我跟刀口去推 p01')\n"
        f"- reasoning 1~200 字 (审计用, 仅狼队可见, 不要泄露身份给好人)\n\n"
        f"【输出协议】必须通过 submit_wolf_team_plan 工具一次性提交完整 JSON, "
        f"不要在 reasoning / public_story 之外输出额外文字。"
    )

    user_prompt = (
        f"## 本夜 (N{night_num}) 狼队夜聊全文\n{discussion_text}\n\n"
        f"## 上局延续\n{prior_summary}\n\n"
        f"## alive_wolves 候选\n{alive_wolves}\n\n"
        f"## alive_non_wolves 候选 (击杀目标)\n{alive_non_wolves}\n\n"
        f"请基于夜聊共识和上局经验,合理分配 4 角色并确定击杀目标。"
        f"若夜聊未达成明确共识,根据队友能力倾向 (谁善辩 → fake_seer, "
        f"谁低调 → deep_cover) 自行决断。"
    )

    tool = wolf_team_plan_tool(alive_wolves, alive_non_wolves)
    max_retries = 3
    last_err: str | None = None

    for attempt in range(1, max_retries + 1):
        retry_suffix = (
            f"\n\n[重试 {attempt}/{max_retries}] 上次错误: {last_err}"
            if last_err
            else ""
        )
        try:
            result = captain_agent.model_router.generate(
                agent_id=captain_id,
                task_type=TaskType.WOLF_TEAM_PLAN.value,
                prompt=user_prompt + retry_suffix,
                system_prompt=system_prompt,
                tools=[tool],
                tool_choice={"type": "tool", "name": "submit_wolf_team_plan"},
            )
        except NotImplementedError:
            logger.debug(
                "[wolf_team_plan] provider does not support tool_choice, fallback"
            )
            return None
        except Exception as e:  # noqa: BLE001
            last_err = f"generate_error: {e}"
            logger.debug(
                "[wolf_team_plan] LLM generate failed attempt %d: %s", attempt, e
            )
            continue

        raw = (result.text or "").strip()
        if not raw:
            last_err = "empty_response"
            continue

        # 先尝试直接解析，再走修复器，最后做平衡 JSON 扫描。
        data: Any = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                from werewolf_agent.agents.output_parser import repair_json_text

                data = json.loads(repair_json_text(raw))
            except Exception:
                data = _extract_first_balanced_json_object(raw)
        if not isinstance(data, dict):
            last_err = f"json_parse_failed: raw[:80]={raw[:80]!r}"
            continue

        try:
            plan = WolfTeamPlan.model_validate(data)
        except Exception as e:  # noqa: BLE001
            last_err = f"schema_validation: {str(e)[:200]}"
            continue

        membership_err = validate_wolf_team_plan_membership(
            plan,
            alive_wolves=alive_wolves,
            alive_non_wolves=alive_non_wolves,
        )
        if membership_err is not None:
            last_err = membership_err
            continue

        plan_dict: dict[str, Any] = plan.model_dump()
        plan_dict["consensus_method"] = "llm"
        plan_dict["captain_id"] = captain_id
        plan_dict["evidence_from_discussion"] = build_wolf_team_plan_evidence(
            plan_dict,
            captain_id,
        )
        logger.debug(
            "[wolf_team_plan] captain %s produced plan in %d attempt(s)",
            captain_id,
            attempt,
        )
        return plan_dict

    logger.debug(
        "[wolf_team_plan] retry exhausted (%d), last_err=%s, fallback",
        max_retries,
        last_err,
    )
    return None


def agent_wolf_consensus(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    *,
    decision_identities: dict[str, DecisionIdentity] | None = None,
    exposure_collectors: dict[str, ModuleExposureAuditCollector] | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Try to get wolf consensus from agents. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    wolves = [pid for pid, p in gs.players.items() if p.role == "werewolf" and p.alive]
    if not wolves:
        return None

    kill_votes: dict[str, int] = {}
    no_kill_count = 0
    action_traces: dict[str, Any] = {}
    action_decision_identities: dict[str, DecisionIdentity] = {}
    action_exposure_collectors: dict[str, ModuleExposureAuditCollector] = {}

    for wolf_id in wolves:
        decision_identity = (decision_identities or {}).get(wolf_id)
        exposure_collector = (exposure_collectors or {}).get(wolf_id)
        vote = _single_wolf_vote(
            state,
            engine,
            registry,
            wolf_id,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
            decision_trace_sink=decision_trace_sink,
        )
        if vote is None:
            if exposure_collector is not None:
                exposure_collector.flush_events()
            no_kill_count += 1
            continue
        if vote.get("action_trace"):
            action_traces[wolf_id] = vote["action_trace"]
            if decision_identity is not None:
                action_decision_identities[wolf_id] = decision_identity
            if exposure_collector is not None:
                action_exposure_collectors[wolf_id] = exposure_collector
        elif exposure_collector is not None:
            exposure_collector.flush_events()
        if vote.get("wolf_action") == "kill" and vote.get("wolf_kill_target_id"):
            target = vote["wolf_kill_target_id"]
            kill_votes[target] = kill_votes.get(target, 0) + 1
        else:
            no_kill_count += 1

    total_kill = sum(kill_votes.values())
    if total_kill > no_kill_count and kill_votes:
        best_target = max(kill_votes, key=kill_votes.get)
        return {
            "wolf_action": "kill",
            "wolf_kill_target_id": best_target,
            "wolf_action_reason": f"majority({total_kill}/{total_kill + no_kill_count})",
            "action_traces": action_traces,
            "action_decision_identities": action_decision_identities,
            "action_exposure_collectors": action_exposure_collectors,
        }
    return {
        "wolf_action": "no_kill",
        "wolf_kill_target_id": None,
        "wolf_action_reason": f"no_kill_majority({no_kill_count}/{total_kill + no_kill_count})",
        "action_traces": action_traces,
        "action_decision_identities": action_decision_identities,
        "action_exposure_collectors": action_exposure_collectors,
    }


def agent_wolf_discussion(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    wolf_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Get wolf's private discussion speech. Returns None if agent unavailable."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(wolf_id)
    if agent is None:
        return None

    from werewolf_agent.runtime.wolf_strategy import round_requirements

    round_num = state.get("wolf_discussion_round", 1)
    requirements = round_requirements(gs.night_number, round_num)

    wolf_ids = living_wolf_ids(gs)
    prev_speeches = collect_wolf_discussion_speeches(gs, wolf_ids)
    wolf_teammates = living_wolf_teammates(gs, wolf_id)
    teammate_speeches = teammate_discussion_speeches(prev_speeches, wolf_id)
    discussion_instruction = build_wolf_discussion_instruction(
        wolf_id,
        night_number=gs.night_number,
        has_teammate_input=bool(teammate_speeches),
        has_previous_speeches=bool(prev_speeches),
    )
    strategy_directive = build_wolf_discussion_strategy_directive(
        discussion_instruction=discussion_instruction,
        round_focus=requirements.get("required", "讨论狼队策略。"),
        wolf_teammates=wolf_teammates,
        previous_speeches=prev_speeches,
    )
    # 注入高优先级击杀目标，帮助夜聊收敛到同一刀口。
    strategy_directive["wolf_high_priority_target"] = _build_wolf_kill_directive(
        gs,
        wolf_id=wolf_id,
        plan=state.get("wolf_team_plan"),
    )

    context = build_agent_context(
        engine,
        gs,
        wolf_id,
        TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )

    extra_transcript = build_teammate_transcript(teammate_speeches)
    merged_transcript = extra_transcript + list(context.recent_transcript)
    context = context.model_copy(
        update={
            "strategy_directive": strategy_directive,
            "recent_transcript": merged_transcript[-8:],
        }
    )

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""

    if not speech_text.strip():
        alive_non_wolves = [
            pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"
        ]
        fallback_target = alive_non_wolves[0] if alive_non_wolves else ""
        speech_text = build_empty_wolf_discussion_fallback(
            wolf_id,
            fallback_target,
            requirements.get("required", ""),
        )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


__all__ = [
    "agent_wolf_team_plan",
    "agent_wolf_consensus",
    "agent_wolf_discussion",
]
