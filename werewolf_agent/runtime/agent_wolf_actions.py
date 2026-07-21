# -*- coding: utf-8 -*-
"""
狼人夜间讨论、计划和共识相关 agent action 适配器。

作者: Mike
创建日期: 2026-07-07
修改日期: 2026-07-21

使用示例:
    >>> from werewolf_agent.runtime.agent_wolf_actions import agent_wolf_discussion
    >>> agent_wolf_discussion(...)

"""

from __future__ import annotations

import json
import logging
from typing import Any

from werewolf_agent.agents.wolf_prompt_contract import (
    WEREWOLF_CRITICAL_SEMANTIC_CLAUSES,
    WEREWOLF_TARGET_SEMANTICS_HEADER,
)
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
from werewolf_agent.runtime.strategy.seer import public_seer_claimants
from werewolf_agent.runtime.json_extract import (
    extract_first_balanced_json_object as _extract_first_balanced_json_object,
)
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_empty_wolf_discussion_fallback,
    build_layered_wolf_discussion_context,
    build_wolf_discussion_instruction,
    build_wolf_discussion_strategy_directive,
    collect_current_wolf_target_stances,
    collect_wolf_discussion_speeches,
    living_wolf_ids,
    living_wolf_teammates,
    teammate_discussion_speeches,
)
from werewolf_agent.runtime.wolf_kill_support import (
    _build_wolf_kill_directive,
    _single_wolf_vote,
)


# 2026-07-21 jitter 兜底：1a-verify 暴露 LLM 在 plan-envelope 模式下 ~6% 概率
# 静默漏 target_stance 字段，导致 _planned_wolf_kill 走 strategic_abstain 空刀。
# strategy_directive 已注入完整 contract，但某些模型组合仍漏；本常量用于重
# 试阶段的精简 hint，比完整 contract 短，让模型第二轮更容易按格式补字段。
_STANCE_RETRY_HINT = (
    "【重试必填 / RETRY MUST】上一次输出你漏了 `target_stance` 顶层键。"
    "本次必须输出如下结构 (对象或 null):\n"
    "{\"target_id\": \"p05\" 或 null, "
    "\"stance\": \"propose\" | \"support\" | \"oppose\" | \"abstain\", "
    "\"priority\": \"primary\" | \"backup\"}\n"
    "没决定就 target_id=null + stance=abstain + priority=primary。"
    "再次省略 = 空刀失败。"
)
from werewolf_agent.runtime.wolf_team_plan_support import (
    build_prior_plan_summary,
    build_wolf_role_definitions,
    build_wolf_team_plan_evidence,
    normalize_wolf_team_plan_payload,
    validate_wolf_team_plan_membership,
)
from werewolf_agent.agents.player_failures import (
    terminal_failure_code_for_task_failure,
)

logger = logging.getLogger(__name__)


_WOLF_TEAM_PLAN_FAILURE_KEY = "wolf_team_plan_failure"


def _record_wolf_team_plan_failure(
    state: dict[str, Any],
    *,
    reason: str,
    stage: str,
    attempts: int,
    last_error: str,
    captain_id: str | None,
    normalization_repairs: tuple[str, ...] = (),
) -> None:
    """记录狼队计划稳定失败元数据，不持久化原始异常正文。"""
    terminal_code = terminal_failure_code_for_task_failure(reason)
    failure = {
        "reason": reason,
        "stage": stage,
        "attempts": attempts,
        "captain_id": captain_id,
        "terminal_failure_code": terminal_code,
        "original_failure_code": terminal_code,
        "failure_stage": stage,
        "fallback_kind": "wolf_team_plan_structured_stance",
    }
    if normalization_repairs:
        failure.update({
            "normalization_triggered": True,
            "normalization_repairs": list(normalization_repairs),
        })
    state[_WOLF_TEAM_PLAN_FAILURE_KEY] = failure


def _wolf_team_plan_prompt_observer(
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    stance_source_event_ids: tuple[str, ...] = (),
) -> Any:
    """为队长计划的工具与纯 JSON 路径复用同一送达前合同门禁。"""
    from werewolf_agent.model_gateway.final_prompt_observer import (
        FinalPromptAssembly,
        FinalPromptContract,
        validate_final_prompt_contract,
    )

    contract = FinalPromptContract(
        contract_id="werewolf-team-plan",
        version="2026-07-18.v1",
        required_sections=(
            ("captain_role", "你是狼队队长".encode("utf-8")),
            ("hard_constraints", "【硬约束】".encode("utf-8")),
            ("output_protocol", "【输出协议】".encode("utf-8")),
            ("wolf_target_semantics", WEREWOLF_TARGET_SEMANTICS_HEADER.encode("utf-8")),
            *(
                (section_id, clause.encode("utf-8"))
                for section_id, clause in WEREWOLF_CRITICAL_SEMANTIC_CLAUSES
            ),
        ),
    )

    def _observe(assembly: FinalPromptAssembly) -> None:
        confirmations = validate_final_prompt_contract(assembly, contract)
        if decision_identity is not None and exposure_collector is not None:
            exposure_collector.record_provider_persona_prompt_proof(
                decision_identity,
                assembly.system_bytes,
                "",
                assembly.attempt_kind,
                attempt_ordinal=assembly.attempt_ordinal,
                provider=assembly.provider,
                model=assembly.model,
                final_system_location=assembly.final_system_location,
                final_system_message_index=assembly.final_system_message_index,
                prompt_contract_id=contract.contract_id,
                prompt_contract_version=contract.version,
                required_section_confirmations=confirmations,
                provider_payload_bytes=assembly.provider_payload_bytes,
                dynamic_content_tokens=stance_source_event_ids,
            )

    return _observe


def agent_wolf_team_plan(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
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
    from werewolf_agent.agents.wolf_team_plan_schema import wolf_team_plan_contract
    from werewolf_agent.agents.tool_schema import wolf_team_plan_tool
    from werewolf_agent.runtime.directives.wolf import _WOLF_ROLE_STRATEGY

    state.pop(_WOLF_TEAM_PLAN_FAILURE_KEY, None)
    gs: GameState = state["game_state"]
    alive_wolves = sorted(
        pid for pid, p in gs.players.items() if p.role == "werewolf" and p.alive
    )
    if not alive_wolves:
        _record_wolf_team_plan_failure(
            state,
            reason="no_alive_wolves",
            stage="runtime",
            attempts=0,
            last_error="no alive wolves",
            captain_id=None,
        )
        return None

    captain_id = alive_wolves[0]
    captain_agent = registry.get_agent(captain_id)
    if captain_agent is None:
        logger.debug(
            "[wolf_team_plan] captain %s agent unavailable, fallback", captain_id
        )
        _record_wolf_team_plan_failure(
            state,
            reason="captain_agent_missing",
            stage="registry",
            attempts=0,
            last_error=f"captain {captain_id} agent unavailable",
            captain_id=captain_id,
        )
        return None

    alive_non_wolves = sorted(
        pid for pid, p in gs.players.items() if p.role != "werewolf" and p.alive
    )

    night_num = gs.night_number
    discussion_stances = collect_current_wolf_target_stances(gs)
    prior_summary = build_prior_plan_summary(state.get("wolf_team_plan") or {})
    role_defs = build_wolf_role_definitions(_WOLF_ROLE_STRATEGY)
    contract = wolf_team_plan_contract()

    system_prompt = (
        f"你是狼队队长 {captain_id}。本夜是 N{night_num}。"
        f"队友夜聊已完成,现在由你一次性产出团队作战计划。\n\n"
        f"【4 角色定义 (字段名 ↔ 中文)】\n{role_defs}\n\n"
        f"【硬约束】\n"
        f"- 4 角色字段 (fake_seer/pusher/hooker/deep_cover) 互不相同, 都从 "
        f"alive_wolves={alive_wolves} 中选; 任一字段可填 null (本夜不分配该位置)\n"
        f"- 击杀目标 (night_kill_primary/backup) 必须从 alive_non_wolves 中选 "
        f"或填 null (空刀); 不能是狼队成员\n"
        f"【输出协议】必须通过 submit_wolf_team_plan 工具一次性提交完整 JSON, "
        f"不要在 reasoning / public_story 之外输出额外文字。"
    )
    system_prompt += (
        "\n\n【Schema 字段边界（权威）】\n"
        f"- public_story: {contract['public_story']['min_length']}~"
        f"{contract['public_story']['max_length']} 字\n"
        f"- reasoning: {contract['reasoning']['min_length']}~"
        f"{contract['reasoning']['max_length']} 字"
        + "\n"
        + WEREWOLF_TARGET_SEMANTICS_HEADER
        + "\n"
        + "\n".join(
            f"- {clause}" for _section_id, clause in WEREWOLF_CRITICAL_SEMANTIC_CLAUSES
        )
    )

    user_prompt = (
        f"## 本夜 (N{night_num}) 狼队结构化立场\n"
        f"{json.dumps(discussion_stances, ensure_ascii=False, sort_keys=True)}\n\n"
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
    last_reason = "retry_exhausted"
    last_stage = "unknown"
    use_tool_choice = True
    observed_normalization_repairs: list[str] = []
    final_prompt_observer = _wolf_team_plan_prompt_observer(
        decision_identity=decision_identity,
        exposure_collector=exposure_collector,
        stance_source_event_ids=tuple(
            str(stance.get("source_event_id") or "")
            for stance in discussion_stances
            if stance.get("source_event_id")
        ),
    )

    for attempt in range(1, max_retries + 1):
        retry_suffix = (
            f"\n\n[重试 {attempt}/{max_retries}] 上次错误: {last_err}"
            if last_err
            else ""
        )
        try:
            prompt_text = user_prompt + retry_suffix
            system_text = system_prompt
            tools_arg = [tool] if use_tool_choice else None
            tool_choice_arg = (
                {"type": "tool", "name": "submit_wolf_team_plan"}
                if use_tool_choice else None
            )
            if not use_tool_choice:
                prompt_text += (
                    "\n\n当前模型不支持工具调用；只输出一个完整JSON对象，"
                    "不要输出Markdown、解释或额外文本。"
                )
                system_text += "\n\n如果工具调用不可用，直接输出符合字段约束的JSON对象。"
            result = captain_agent.model_router.generate(
                agent_id=captain_id,
                task_type=TaskType.WOLF_TEAM_PLAN.value,
                prompt=prompt_text,
                system_prompt=system_text,
                tools=tools_arg,
                tool_choice=tool_choice_arg,
                final_prompt_observer=final_prompt_observer,
            )
        except NotImplementedError:
            logger.debug(
                "[wolf_team_plan] provider does not support tool_choice, "
                "retrying as plain JSON"
            )
            use_tool_choice = False
            try:
                result = captain_agent.model_router.generate(
                    agent_id=captain_id,
                    task_type=TaskType.WOLF_TEAM_PLAN.value,
                    prompt=(
                        user_prompt
                        + retry_suffix
                        + "\n\n当前模型不支持工具调用；只输出一个完整JSON对象，"
                        "不要输出Markdown、解释或额外文本。"
                    ),
                    system_prompt=(
                        system_prompt
                        + "\n\n如果工具调用不可用，直接输出符合字段约束的JSON对象。"
                    ),
                    tools=None,
                    tool_choice=None,
                    final_prompt_observer=final_prompt_observer,
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"plain_json_generate_error: {e}"
                last_reason = "provider_tool_choice_unsupported"
                last_stage = "provider"
                continue
        except Exception as e:  # noqa: BLE001
            last_err = f"generate_error: {e}"
            last_reason = "generate_error"
            last_stage = "provider"
            logger.debug(
                "[wolf_team_plan] LLM generate failed attempt %d: %s", attempt, e
            )
            continue

        raw = (result.text or "").strip()
        if not raw:
            last_err = "empty_response"
            last_reason = "empty_response"
            last_stage = "model_output"
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
            last_reason = "json_parse_failed"
            last_stage = "protocol"
            continue
        data, normalization_repairs = normalize_wolf_team_plan_payload(data)
        observed_normalization_repairs = list(dict.fromkeys([
            *observed_normalization_repairs,
            *normalization_repairs,
        ]))
        data.setdefault("night_number", night_num)

        try:
            plan = WolfTeamPlan.model_validate(data)
        except Exception as e:  # noqa: BLE001
            last_err = f"schema_validation: {str(e)[:200]}"
            last_reason = "schema_validation_failed"
            last_stage = "schema"
            continue

        membership_err = validate_wolf_team_plan_membership(
            plan,
            alive_wolves=alive_wolves,
            alive_non_wolves=alive_non_wolves,
        )
        if membership_err is not None:
            last_err = membership_err
            last_reason = "membership_validation_failed"
            last_stage = "semantic"
            continue

        plan_dict: dict[str, Any] = plan.model_dump()
        plan_dict["consensus_method"] = "llm"
        plan_dict["captain_id"] = captain_id
        if normalization_repairs:
            plan_dict["normalization_repairs"] = list(normalization_repairs)
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
    _record_wolf_team_plan_failure(
        state,
        reason=last_reason,
        stage=last_stage,
        attempts=max_retries,
        last_error=last_err or "retry exhausted",
        captain_id=captain_id,
        normalization_repairs=tuple(observed_normalization_repairs),
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
    layered_context = build_layered_wolf_discussion_context(gs, wolf_ids)
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
        layered_context=layered_context,
    )
    stored_plan = state.get("wolf_team_plan")
    current_plan = (
        stored_plan
        if isinstance(stored_plan, dict)
        and stored_plan.get("night_number") == gs.night_number
        else None
    )
    if current_plan is not None or public_seer_claimants(gs):
        # 公开跳预言家只作为讨论提示，最终击杀仍必须由本夜结构化 stance
        # 与共识节点授权；因此这里不会恢复旧计划的执行权威。
        strategy_directive["wolf_high_priority_target"] = _build_wolf_kill_directive(
            gs,
            wolf_id=wolf_id,
            plan=current_plan,
        )
    if current_plan is None and isinstance(stored_plan, dict) and stored_plan:
        # 跨夜计划只作为历史摘要，不能恢复执行权威。
        strategy_directive["wolf_plan_history"] = build_prior_plan_summary(stored_plan)

    context = build_agent_context(
        engine,
        gs,
        wolf_id,
        TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=current_plan,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )

    if decision_identity is not None and exposure_collector is not None:
        exposure_collector.record_wolf_prompt_context(
            decision_identity,
            layered_context["audit"],
        )
    context = context.model_copy(
        update={
            "strategy_directive": strategy_directive,
        }
    )

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    target_stance = getattr(action, "target_stance", None)

    # 2026-07-21 jitter 兜底：1a-verify 暴露某些模型组合（特别是 Kimi-K2.6 via Ark）
    # 即使在 strategy_directive 已注入 target_stance_contract，仍 ~6% 静默漏字段。
    # 这里一次性重试：缩短的 contract 注入到 directive，让 LLM 第二轮 act() 必须产出。
    if target_stance is None and decision_identity is not None:
        retry_directive = dict(strategy_directive)
        retry_directive["target_stance_retry_hint"] = _STANCE_RETRY_HINT
        retry_context = context.model_copy(
            update={
                "strategy_directive": retry_directive,
            },
        )
        retry_action, _ = agent.act(retry_context)
        retry_stance = getattr(retry_action, "target_stance", None)
        retry_speech = getattr(retry_action, "speech", "") or ""
        target_stance = retry_stance
        if retry_speech.strip():
            speech_text = retry_speech
        if exposure_collector is not None:
            exposure_collector._append(
                "wolf_target_stance_retry",
                decision_identity,
                {
                    "wolf_id": wolf_id,
                    "round": round_num,
                    "night_number": gs.night_number,
                    "retry_filled_stance": retry_stance is not None,
                },
            )

    if not speech_text.strip():
        speech_text = build_empty_wolf_discussion_fallback(
            wolf_id,
            requirements.get("required", ""),
        )
        target_stance = None

    return {
        "speech_text": speech_text,
        "target_stance": (
            target_stance.model_dump()
            if hasattr(target_stance, "model_dump")
            else target_stance
        ),
        "action_trace": _action_trace_payload(action),
    }


__all__ = [
    "agent_wolf_team_plan",
    "agent_wolf_consensus",
    "agent_wolf_discussion",
]
