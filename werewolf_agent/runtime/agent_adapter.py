# -*- coding: utf-8 -*-
"""
把 GameState 转换为 AgentContext，并把运行时决策委托给 PlayerAgent。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.agent_adapter import agent_day_speech
    >>> agent_day_speech(...)
"""


from __future__ import annotations

import logging
from typing import Any, Protocol

from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    TaskType,
)
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.badge_decision_directives import (
    build_badge_decision_directive,
    build_badge_decision_result,
)
from werewolf_agent.runtime.defense_speech_directives import (
    build_defense_context_directive,
    build_empty_defense_speech_fallback,
)
from werewolf_agent.runtime.hunter_shot_directives import (
    build_hunter_shot_directive,
    build_hunter_shot_result,
)
from werewolf_agent.runtime.hybrid_master_directives import (
    build_hybrid_master_candidates,
    build_hybrid_master_choice_directive,
    choose_hybrid_master_target,
)
from werewolf_agent.runtime.day_speech_directives import (
    build_day_speech_base_directive,
    build_empty_day_speech_fallback,
    build_sanitized_seer_claim_fallback,
    build_sheriff_election_record,
    build_sheriff_speech_directive,
    build_torn_badge_speech_state,
    collect_sheriff_election_speeches,
)
from werewolf_agent.runtime.day_vote_directives import (
    build_day_vote_base_directive,
    build_fallback_seer_vote_strategy,
    build_hunter_vote_strategy,
    build_hybrid_vote_strategy,
    build_seer_vote_strategy,
    build_villager_vote_strategy,
    build_vote_anti_herd_directive,
    build_witch_vote_strategy,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.json_extract import (
    extract_first_balanced_json_object as _extract_first_balanced_json_object,
)
from werewolf_agent.runtime.last_words_directives import build_exile_last_words_strategy
from werewolf_agent.runtime.pk_speech_directives import build_pk_speech_strategy
from werewolf_agent.runtime.reflection_prompt import (
    GOOD_REFLECTION_TEMPLATE,
    WOLF_REFLECTION_TEMPLATE,
    build_reflection_prompt,
)
from werewolf_agent.runtime.sheriff_election_directives import (
    build_previous_sheriff_speech_instruction,
    build_seer_verification_rationale,
    build_sheriff_badge_flow_instruction,
    build_sheriff_election_speech_directive,
    build_sheriff_role_speech_hint,
    build_sheriff_seer_context,
    build_wolf_sheriff_election_directives,
    collect_previous_sheriff_speeches,
    sheriff_uses_seer_protocol,
)
from werewolf_agent.runtime.sheriff_action_directives import (
    build_sheriff_endorse_directive,
    build_sheriff_endorse_result,
    build_sheriff_speech_order,
    build_sheriff_speech_order_directive,
    living_non_sheriff_ids,
)
from werewolf_agent.runtime.seer_night_directives import (
    build_badge_flow_next_targets,
    build_seer_legal_targets,
    build_seer_night_strategy_directive,
)
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.wolf_team_plan_support import (
    build_prior_plan_summary,
    build_wolf_role_definitions,
    build_wolf_team_plan_evidence,
    collect_current_wolf_discussion_text,
    validate_wolf_team_plan_membership,
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
from werewolf_agent.runtime.witch_night_directives import (
    build_witch_first_night_killed_directive,
    build_witch_legal_actions,
    build_witch_night_action_directive,
    build_witch_poison_candidates_directive,
    build_witch_poison_strategy,
    build_witch_pressure_directives,
    build_witch_strategy_hint,
)

# Backward-compatible re-exports from runtime.context (Task 3 extraction).
from werewolf_agent.runtime.context import (
    build_agent_context,
    _SPEECH_STYLE_HINTS,
    _SHERIFF_SPEECH_STYLE_OVERRIDES,
    _TASK_STYLE_HINTS,
    _get_persona_speech_style,
    _get_persona_task_style,
    _action_trace_payload,
    _merge_strategy_directive,
    _inject_skill_output as _inject_skill_output,
)

# Backward-compatible re-exports from runtime.directives package.
from werewolf_agent.runtime.directives import (
    build_hunter_directive as _build_hunter_day_speech_directive,
    build_hybrid_directive as _build_hybrid_day_speech_directive,
    build_idiot_directive as _build_idiot_day_speech_directive,
    build_seer_directive as _build_seer_day_speech_directive,
    build_villager_directive as _build_villager_day_speech_directive,
    build_witch_directive as _build_witch_day_speech_directive,
    build_wolf_day_directive as _build_wolf_day_speech_directive,
    build_wolf_night_directive as _build_wolf_night_directive,
    build_wolf_vote_directive as _build_wolf_vote_strategy,
)
from werewolf_agent.runtime.directives._shared import (
    build_sheriff_silent_directive as _build_sheriff_silent_directive,
)

logger = logging.getLogger(__name__)

# 兼容旧测试和调试入口：反思模板实现已移动到 runtime.reflection_prompt。
_build_reflection_prompt = build_reflection_prompt
_GOOD_REFLECTION_TEMPLATE = GOOD_REFLECTION_TEMPLATE
_WOLF_REFLECTION_TEMPLATE = WOLF_REFLECTION_TEMPLATE


def _audit_context_kwargs(
    decision_identity: DecisionIdentity | None,
    exposure_collector: ModuleExposureAuditCollector | None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any]:
    if decision_identity is None or exposure_collector is None:
        return {}
    return {
        "decision_identity": decision_identity,
        "exposure_collector": exposure_collector,
        "decision_trace_sink": decision_trace_sink,
    }


class AgentRegistry(Protocol):
    """Maps player_id to PlayerAgent. Return None for scripted fallback."""

    def get_agent(self, player_id: str) -> PlayerAgent | None: ...


class SimpleAgentRegistry:
    """Concrete registry: maps player_id -> PlayerAgent."""

    def __init__(self, agents: dict[str, PlayerAgent] | None = None) -> None:
        self._agents: dict[str, PlayerAgent] = agents or {}

    def register(self, player_id: str, agent: PlayerAgent) -> None:
        self._agents[player_id] = agent

    def get_agent(self, player_id: str) -> PlayerAgent | None:
        return self._agents.get(player_id)


# -- Backward-compatible re-exports from runtime.strategy (Task 2 extraction) --
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value as _estimate_witch_save_value,
    evaluate_seer_check_value as _evaluate_seer_check_value,
    evaluate_wolf_kill_target as _evaluate_wolf_kill_target,
    get_wolf_role_assignment as _get_wolf_role_assignment,
    has_publicly_claimed_seer as _has_publicly_claimed_seer,
)
from werewolf_agent.runtime.strategy.seer import public_seer_claimants as _public_seer_claimants  # noqa: F401


# M2-2: single-source guidance for vote/speech actions. Moved out
# of system prompt's role_guide (which is stable across turns and
# doesn't know task_type) to per-turn strategy_directive injection.
# Same wording as before (preserves LLM behavior for vote/speech).
# Seer is exempt (uses seer_check for own checks).
VOTE_BASIS_GUIDANCE = (
    "【投票时 vote_basis 选用 speech_logic / vote_pattern / "
    "seer_siding，不要用 seer_check。】"
)


def _inject_vote_basis_hint(
    strategy_directive: dict[str, Any],
    gs: GameState,
    player_id: str,
) -> None:
    """M2-2: per-turn VOTE_BASIS_GUIDANCE injection (seer exempt).

    Seer legitimately uses seer_check (its own checks); the
    guidance "vote_basis 选用 speech_logic / vote_pattern /
    seer_siding, 不要用 seer_check" doesn't apply to it.
    Hybrid also gets the guidance — it has no own-check ability.

    Mutates ``strategy_directive`` in place. Centralized so the
    seer-exempt rule is defined once and the prompt-tier
    registry (HARD_CONSTRAINT_KEYS) only needs to know the key.
    """
    role = gs.players[player_id].role if player_id in gs.players else ""
    if role != "seer":
        strategy_directive["vote_basis_hint"] = VOTE_BASIS_GUIDANCE


def _seer_credibility_audit_payload(
    context: AgentContext,
    day_number: int,
) -> dict[str, Any] | None:
    summary = context.seer_credibility or {}
    lines = summary.get("seer_lines")
    if not isinstance(lines, list) or not lines:
        return None
    safe_lines: list[dict[str, Any]] = []
    for item in lines[:3]:
        if not isinstance(item, dict):
            continue
        safe_lines.append({
            key: item[key]
            for key in (
                "claimant",
                "status",
                "score",
                "confidence",
                "checks",
                "evidence",
                "penalties",
            )
            if key in item
        })
    if not safe_lines:
        return None
    return {
        "day_number": day_number,
        "visibility": "moderator_only",
        "seer_lines": safe_lines,
    }


def _is_sheriff_silenced(gs: GameState, sheriff_id: str) -> bool:
    """Return True if the active sheriff is currently muted (cannot speak).

    P1-D4: a sheriff may hold the active badge but still be unable to
    speak — e.g., a witch poison mute or a self-destruct that lands the
    badge but freezes the day-speech action.  The pre-fix code only
    checked ``sheriff_id == speaker_id and sheriff_badge_state ==
    "active"`` and rendered a 归票 directive unconditionally, which
    contradicted the silence condition.

    The check is forward-compatible:
    - Looks for a ``sheriff_silenced`` event targeting this sheriff
      (e.g., emitted by a future skill resolver).
    - Falls back to badge states ``"silenced"`` / ``"frozen"`` if a
      caller sets them explicitly.
    """
    for ev in gs.events:
        if ev.type == "sheriff_silenced" and ev.payload.get("sheriff_id") == sheriff_id:
            return True
    if gs.sheriff_badge_state in {"silenced", "frozen"}:
        return True
    return False


def agent_night_witch(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Try to get witch decision from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    witch_id = next(
        (pid for pid, p in gs.players.items() if p.role == "witch" and p.alive),
        None,
    )
    if witch_id is None:
        return None

    agent = registry.get_agent(witch_id)
    if agent is None:
        return None

    wolf_kill_target_id = state.get("wolf_kill_target_id")

    legal_actions, legal_targets = build_witch_legal_actions(
        gs,
        engine,
        witch_id=witch_id,
        wolf_kill_target_id=wolf_kill_target_id,
    )

    context = build_agent_context(
        engine, gs, witch_id, TaskType.NIGHT_ACTION,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_kill_target_id=wolf_kill_target_id,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )

    witch_directive: dict[str, Any] = {
        "witch_night_action": build_witch_night_action_directive(
            wolf_kill_target_id=wolf_kill_target_id,
            witch_id=witch_id,
            antidote_used=gs.antidote_used,
            poison_used=gs.poison_used,
            can_use_antidote=ActionType.USE_ANTIDOTE in legal_actions,
            can_use_poison=ActionType.USE_POISON in legal_actions,
        ),
    }

    save_value = _estimate_witch_save_value(gs, wolf_kill_target_id)
    witch_directive["save_value_assessment"] = save_value
    witch_directive["witch_strategy_hint"] = build_witch_strategy_hint(
        save_value,
        poison_available=not gs.poison_used,
    )
    if not gs.poison_used:
        alive = sum(1 for p in gs.players.values() if p.alive)
        witch_directive["witch_poison_strategy"] = build_witch_poison_strategy(alive)

    if not gs.poison_used:
        try:
            from werewolf_agent.runtime.strategy.poison import collect_witch_poison_candidates
            cands = collect_witch_poison_candidates(gs, witch_id)
        except Exception:
            cands = []
        alive = sum(1 for p in gs.players.values() if p.alive)
        witch_directive["witch_poison_candidates"] = build_witch_poison_candidates_directive(
            cands,
            alive_count=alive,
        )

    first_night_killed = build_witch_first_night_killed_directive(
        wolf_kill_target_id=wolf_kill_target_id,
        witch_id=witch_id,
        poison_used=gs.poison_used,
    )
    if first_night_killed is not None:
        witch_directive["first_night_killed"] = first_night_killed

    poison_pressure = context.visible_world_state.get("poison_pressure_targets", [])
    witch_directive.update(build_witch_pressure_directives(poison_pressure))

    context = _merge_strategy_directive(context, witch_directive)

    action, retry_info = agent.act(context)

    use_antidote = action.action_type == ActionType.USE_ANTIDOTE
    poison_target_id = action.target_id if action.action_type == ActionType.USE_POISON else None

    return {
        "use_antidote": use_antidote,
        "poison_target_id": poison_target_id,
        "witch_action_trace": _action_trace_payload(action),
    }


def agent_night_seer(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Try to get seer decision from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    seer_id = next(
        (pid for pid, p in gs.players.items() if p.role == "seer" and p.alive),
        None,
    )
    if seer_id is None:
        return None

    agent = registry.get_agent(seer_id)
    if agent is None:
        return None

    counterclaiming_seers = _public_seer_claimants(gs) - {seer_id}
    legal_targets = build_seer_legal_targets(
        gs,
        seer_id=seer_id,
        counterclaiming_seers=counterclaiming_seers,
    )
    badge_flow_next = build_badge_flow_next_targets(
        gs,
        seer_id=seer_id,
        legal_targets=legal_targets,
    )
    check_value = _evaluate_seer_check_value(gs, seer_id, legal_targets)

    strategy_directive = build_seer_night_strategy_directive(
        night_number=gs.night_number,
        check_value=check_value,
        badge_flow_next=badge_flow_next,
        counterclaiming_seers=counterclaiming_seers,
    )

    context = build_agent_context(
        engine, gs, seer_id, TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    seer_target_id = action.target_id if action.action_type == ActionType.CHECK_ALIGNMENT else None
    return {
        "seer_target_id": seer_target_id,
        "seer_action_trace": _action_trace_payload(action),
    }


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
    not covered by the extractor's keyword set (e.g. "悍跳位" ≠ "假预言家").

    Returns None on any failure (captain agent unavailable, LLM error,
    schema validation failure, retry exhausted) — caller is expected to
    fall back to the legacy regex + static plan path and emit a
    `wolf_team_plan_fallback` audit event with the failure reason.

    On success, returns plan dict with all WolfTeamPlan fields plus
    `consensus_method="llm"` and `captain_id` for audit/replay.
    """
    import json

    from werewolf_agent.agents.schemas import WolfTeamPlan
    from werewolf_agent.agents.tool_schema import wolf_team_plan_tool
    from werewolf_agent.runtime.directives.wolf import _WOLF_ROLE_STRATEGY

    gs: GameState = state["game_state"]
    alive_wolves = sorted(
        pid for pid, p in gs.players.items()
        if p.role == "werewolf" and p.alive
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
        pid for pid, p in gs.players.items()
        if p.role != "werewolf" and p.alive
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
        retry_suffix = f"\n\n[重试 {attempt}/{max_retries}] 上次错误: {last_err}" if last_err else ""
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

        # Parse JSON: try direct, then output_parser repair, then balanced scan
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

        # Schema validation
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
            captain_id, attempt,
        )
        return plan_dict

    logger.debug(
        "[wolf_team_plan] retry exhausted (%d), last_err=%s, fallback",
        max_retries, last_err,
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

    # Collect votes from ALL alive wolves
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
        return {"wolf_action": "kill", "wolf_kill_target_id": best_target,
                "wolf_action_reason": f"majority({total_kill}/{total_kill + no_kill_count})",
                "action_traces": action_traces,
                "action_decision_identities": action_decision_identities,
                "action_exposure_collectors": action_exposure_collectors}
    return {"wolf_action": "no_kill", "wolf_kill_target_id": None,
            "wolf_action_reason": f"no_kill_majority({no_kill_count}/{total_kill + no_kill_count})",
            "action_traces": action_traces,
            "action_decision_identities": action_decision_identities,
            "action_exposure_collectors": action_exposure_collectors}


def _build_wolf_kill_directive(
    gs: GameState,
    *,
    wolf_id: str,
    plan: dict[str, Any] | None,
) -> str:
    """Build a kill-priority directive for the wolf kill/discussion prompts.

    If any non-wolf player has publicly claimed Seer (using
    ``has_publicly_claimed_seer``), the directive explicitly names them as the
    highest-priority kill target. Otherwise it falls back to the top-3 ranked
    targets from ``evaluate_wolf_kill_target`` so wolves still get concrete
    suggestions rather than a generic "kill the biggest threat" prompt.
    """
    from werewolf_agent.runtime.strategy.wolf import (
        evaluate_wolf_kill_target,
        has_publicly_claimed_seer,
    )

    parts: list[str] = []

    # Highest priority: anyone who has publicly claimed Seer (or any seer-flavored
    # keyword) must be explicitly named. We do not filter by ``role == "seer"``
    # because a fake-Seer jump also reveals that player as a top threat.
    claimed_seers: list[str] = []
    for pid, p in gs.players.items():
        if p.alive and p.role != "werewolf" and has_publicly_claimed_seer(gs, pid):
            claimed_seers.append(pid)

    if claimed_seers:
        names = ", ".join(claimed_seers)
        parts.append(
            f"高优先级击杀目标: {names} —— 该玩家已公开跳预言家，"
            "对狼队威胁最大，必须作为今晚的首选击杀目标。"
        )

    # Secondary: the wolf team plan's primary target (if set)
    if plan and plan.get("night_kill_primary"):
        primary = plan["night_kill_primary"]
        if primary in gs.players and gs.players[primary].alive:
            if primary not in claimed_seers:
                parts.append(
                    f"狼队讨论主目标: {primary}（备选: {plan.get('night_kill_backup') or '无'}）"
                )

    # Tertiary: top scoring candidates from evaluate_wolf_kill_target
    if not parts or len(claimed_seers) == 0:
        # No claimed Seer — show top-3 ranked by threat score
        scores = evaluate_wolf_kill_target(gs, wolf_id, [
            pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"
        ])
        if scores and scores.get("ranked_targets"):
            for entry in scores["ranked_targets"][:3]:
                parts.append(
                    f"击杀候选: {entry['target']}（威胁分={entry['value']}，"
                    f"信号: {', '.join(entry.get('signals', [])) or '无'}）"
                )

    if not parts:
        return "无明显优先目标，按战术需要自由选择击杀对象。"

    return "\n".join(parts)


def _single_wolf_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    wolf_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Get a single wolf's kill/no_kill vote.

    Each wolf gets an individual timeout.  Unknown action types are treated
    as wolf_no_kill (the agent chose something unexpected) rather than
    silently swallowed as None which would distort consensus.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(wolf_id)
    if agent is None:
        return None

    legal_targets = [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]

    # Kill target value assessment
    kill_assessment = _evaluate_wolf_kill_target(gs, wolf_id, legal_targets)
    wolf_plan = state.get("wolf_team_plan")
    # M3-3: build the night directive (kill/no_kill prompt) from the
    # shared module so day and night directives don't drift apart.
    strategy_directive: dict[str, Any] = _build_wolf_night_directive(
        gs, wolf_id, wolf_plan,
    )
    # Task 3 (Issue 4): explicitly name the claimed Seer as the top kill target
    # so all wolves converge on the same high-priority player.  Lives
    # here (not in directives/wolf.py) to avoid a circular import —
    # ``_build_wolf_kill_directive`` itself is defined in this module.
    strategy_directive["wolf_high_priority_target"] = _build_wolf_kill_directive(
        gs, wolf_id=wolf_id, plan=wolf_plan,
    )
    if kill_assessment:
        strategy_directive["kill_value_assessment"] = kill_assessment
    if wolf_plan and wolf_plan.get("night_kill_primary"):
        strategy_directive["wolf_plan_target"] = (
            f"狼队讨论确定的主目标: {wolf_plan['night_kill_primary']}"
            + (f"，备选: {wolf_plan['night_kill_backup']}" if wolf_plan.get("night_kill_backup") else "")
        )

    context = build_agent_context(
        engine, gs, wolf_id, TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=legal_targets,
        wolf_team_plan=wolf_plan,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    timeout = float(state.get("wolf_vote_timeout") or AGENT_TIMEOUTS.wolf_consensus)
    if timeout > 0:
        from werewolf_agent.runtime.timers import timed_call
        action_result = timed_call(agent.act, context, timeout=timeout, fallback=None)
    else:
        try:
            action_result = agent.act(context)
        except Exception as exc:
            logger.warning("Wolf vote failed for %s: %s: %s", wolf_id, type(exc).__name__, exc)
            action_result = None

    if action_result is None:
        # Timeout or exception — count as no_kill (strategy: skip this vote)
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None}

    action, retry_info = action_result
    action_trace = _action_trace_payload(action)

    if action.action_type == ActionType.WOLF_NO_KILL:
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}
    if action.action_type == ActionType.WOLF_KILL and action.target_id:
        # Validate target is alive and not a wolf teammate
        target_player = gs.players.get(action.target_id)
        if target_player and target_player.alive and target_player.role != "werewolf":
            return {"wolf_action": "kill", "wolf_kill_target_id": action.target_id, "action_trace": action_trace}
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}
    # Unknown action type — treat as no_kill rather than silently returning None
    return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}


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

    # Build round-specific requirements from wolf_strategy
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
    # Task 3 (Issue 4): inject claimed-Seer kill priority so the discussion
    # can converge on the same high-priority target.
    strategy_directive["wolf_high_priority_target"] = _build_wolf_kill_directive(
        gs, wolf_id=wolf_id, plan=state.get("wolf_team_plan"),
    )

    context = build_agent_context(
        engine, gs, wolf_id, TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )

    extra_transcript = build_teammate_transcript(teammate_speeches)
    merged_transcript = extra_transcript + list(context.recent_transcript)
    context = context.model_copy(update={
        "strategy_directive": strategy_directive,
        "recent_transcript": merged_transcript[-8:],
    })

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""

    # Reject empty/silent wolf speeches — retry with fallback
    if not speech_text.strip():
        alive_non_wolves = [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]
        fallback_target = alive_non_wolves[0] if alive_non_wolves else ""
        speech_text = build_empty_wolf_discussion_fallback(
            wolf_id,
            fallback_target,
            requirements.get("required", ""),
        )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_defense_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """D-8: handler for TaskType.DEFENSE_SPEECH.

    The defense-speech slot fires when a candidate is accused and gets
    the floor to defend themselves before the room votes.  Pre-fix the
    ``DEFENSE_SPEECH`` task type was registered in the schema and
    referenced in the RAG mapping, but no adapter handler existed —
    any node that tried to delegate defense speeches would crash or
    silently fall through to the scripted fallback.

    This handler delegates to the same machinery as ``agent_day_speech``
    (a public speech is a public speech) but tags the strategy
    directive with a ``defense_context`` block so the model knows it's
    on the spot, not just discussing freely.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine, gs, speaker_id, TaskType.DEFENSE_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        discussion_positions=state.get("discussion_positions"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )

    strategy_directive = context.strategy_directive or {}
    strategy_directive["defense_context"] = build_defense_context_directive()
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(strategy_directive, gs, speaker_id)

    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""

    # Fallback for empty defense speech
    if not speech_text.strip():
        speech_text = build_empty_defense_speech_fallback(speaker_id)

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_day_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Try to get day speech from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine, gs, speaker_id, TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        discussion_positions=state.get("discussion_positions"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )

    style_hint = ""
    ss = _get_persona_speech_style(agent)
    if ss and ss in _SPEECH_STYLE_HINTS:
        style_hint = f"\n- 你的发言风格：{_SPEECH_STYLE_HINTS[ss]}"
    strategy_directive = {
        **(context.strategy_directive or {}),
        **build_day_speech_base_directive(style_hint),
    }

    # Role-specific speech constraints
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    # Speech adapters also need it because the LLM often frames its
    # current speech in terms of who it intends to vote for.
    _inject_vote_basis_hint(strategy_directive, gs, speaker_id)
    player_role = gs.players[speaker_id].role if speaker_id in gs.players else ""
    if player_role == "werewolf":
        wolf_parts = _build_wolf_day_speech_directive(
            gs, speaker_id, state.get("wolf_team_plan"),
        )
        strategy_directive.update(wolf_parts)
    elif player_role == "seer":
        # P0-G3223805846-3: pass the day's speech order so the seer directive
        # can enforce the "jump immediately when speaking late" rule.  The
        # order lives on RuntimeState (populated by free_discussion); fall
        # back to None when not yet materialised so the directive still
        # works in unit tests / early-day planning contexts.
        seer_speech_parts = _build_seer_day_speech_directive(
            gs, speaker_id, speech_order=state.get("speech_order"),
        )
        strategy_directive.update(seer_speech_parts)
    elif player_role == "hunter":
        strategy_directive["hunter_speech_directive"] = _build_hunter_day_speech_directive(gs, speaker_id)
    elif player_role == "hybrid":
        strategy_directive["hybrid_speech_directive"] = _build_hybrid_day_speech_directive(gs, speaker_id)
    elif player_role == "witch":
        # D-1: delegate to the dedicated witch directive module so the
        # day-speech guidance is structured (and D-7 enriches it with
        # the witch's private view of public death-cause claims).
        strategy_directive.update(
            _build_witch_day_speech_directive(gs, speaker_id),
        )
    elif player_role == "idiot":
        strategy_directive.update(_build_idiot_day_speech_directive(gs, speaker_id))
    elif player_role == "villager":
        strategy_directive.update(_build_villager_day_speech_directive(gs, speaker_id))

    # Sheriff gets 归票 (vote push) directive — with silence fallback
    # (P1-D4) and torn-badge election-state directive (P1-D6).
    if gs.sheriff_id == speaker_id and gs.sheriff_badge_state == "active":
        alive_others = [pid for pid, p in gs.players.items() if p.alive and pid != speaker_id]
        strategy_directive.update(
            build_sheriff_speech_directive(
                is_silenced=_is_sheriff_silenced(gs, speaker_id),
                alive_others=alive_others,
            )
        )

    # After badge tear → no sheriff for the rest of the game.  Every
    # player (not just the previous sheriff) must know there is no
    # 归票人 and that speech order is now random (design doc §警长规则).
    if gs.sheriff_id is None and gs.sheriff_badge_state == "torn":
        strategy_directive["sheriff_election_state"] = build_torn_badge_speech_state()
        # P0-G3223805846-9: inject 归票 hint so players don't fall back
        # on "loudest voice wins".  Distinct key from `sheriff_silent`
        # (which is reserved for the silenced-but-alive sheriff case).
        strategy_directive.update(
            _build_sheriff_silent_directive(
                gs, sheriff_id=None, badge_state="torn",
            )
        )

    sheriff_election_record = build_sheriff_election_record(
        collect_sheriff_election_speeches(gs)
    )
    if sheriff_election_record:
        strategy_directive["sheriff_election_record"] = sheriff_election_record

    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"speech_text": "", "action_trace": {}, "self_destruct": True}

    speech_text = getattr(action, "speech", "") or ""

    # Reject empty day speeches — provide fallback
    if not speech_text.strip():
        alive_others = [pid for pid, p in gs.players.items() if p.alive and pid != speaker_id]
        target_hint = alive_others[0] if alive_others else ""
        speech_text = build_empty_day_speech_fallback(speaker_id, target_hint)

    # Guardrail: enforce the 1-check-per-night rule on public seer claims.
    # If a wolf (or anyone) generated a speech that violates this rule,
    # replace it with a sanitized fallback so the bad claim never reaches
    # the public timeline.
    # D-13: the validator used to be gated on ``player_role == "werewolf"``;
    # the rule is a domain rule (one check per night), not a wolf-specific
    # guard, so any speech — wolf impostor, confused LLM, hybrid, etc. —
    # must be screened.  We now run the validator on every public speech
    # that contains seer-style claim patterns.
    if speech_text:
        from werewolf_agent.runtime.seer_claim_validator import validate_seer_claim

        claim_err = validate_seer_claim(speech_text, day_number=gs.day_number)
        if claim_err:
            logger.warning(
                "Speaker %s (role=%s) speech violated seer claim rule: %s — applying fallback",
                speaker_id, player_role, claim_err,
            )
            alive_others = [
                pid for pid, p in gs.players.items()
                if p.alive and pid != speaker_id
            ]
            target_hint = alive_others[0] if alive_others else ""
            speech_text = build_sanitized_seer_claim_fallback(
                speaker_id,
                target_hint,
            )

    return {
        "speech_text": speech_text,
        "action_trace": _action_trace_payload(action),
        "seer_credibility_audit": _seer_credibility_audit_payload(
            context,
            gs.day_number,
        ),
        "self_destruct": False,
    }


def agent_sheriff_pick_speech_order(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
    include_action_trace: bool = False,
) -> list[str] | dict[str, Any] | None:
    """Ask the sheriff agent to choose the first speaker. Returns full speech order or None."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_players = living_non_sheriff_ids(gs, sheriff_id)
    if not alive_players:
        return None

    # A5 fix: build the context with VOTE action semantics from the start.
    # The sheriff picks a target (first speaker) via a VOTE-style action,
    # so task_type and legal_actions must align (no model_copy mutation).
    context = build_agent_context(
        engine, gs, sheriff_id, TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=alive_players,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    strategy_directive = build_sheriff_speech_order_directive(alive_players)
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    action_trace = _action_trace_payload(action)
    first_speaker = action.target_id if action.action_type == ActionType.VOTE else None

    speech_order = build_sheriff_speech_order(
        first_speaker=first_speaker,
        alive_players=alive_players,
        sheriff_id=sheriff_id,
    )
    if speech_order is not None:
        if include_action_trace:
            return {"speech_order": speech_order, "action_trace": action_trace}
        return speech_order
    if include_action_trace:
        return {"speech_order": None, "action_trace": action_trace}
    return None


def agent_sheriff_endorse(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Sheriff privately decides endorsement target via VOTE action.

    Returns dict with endorse_target / private_reason / action_trace
    (or None for scripted fallback when no agent is registered).
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_others = living_non_sheriff_ids(gs, sheriff_id)

    strategy_directive = build_sheriff_endorse_directive(alive_others)

    context = build_agent_context(
        engine, gs, sheriff_id, TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=alive_others,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    target = action.target_id if action.action_type == ActionType.VOTE else None

    action_trace = _action_trace_payload(action) if target and target in alive_others else None
    return build_sheriff_endorse_result(
        target=target,
        alive_others=alive_others,
        private_reason=getattr(action, "reason", "") or "",
        action_trace=action_trace,
    )


def agent_pk_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Get PK speech from a tied candidate. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    # Build context with prior vote tally info
    prior_tally = {}
    for e in gs.events:
        if e.type == "vote_resolved":
            prior_tally = e.payload
            break

    context = build_agent_context(
        engine, gs, speaker_id, TaskType.PK_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    # Add prior tally to visible state
    if prior_tally:
        updated_visible = {**context.visible_world_state, "prior_vote_tally": prior_tally}
        context = context.model_copy(update={"visible_world_state": updated_visible})

    pk_strategy = build_pk_speech_strategy(gs, speaker_id)
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(pk_strategy, gs, speaker_id)

    context = _merge_strategy_directive(context, pk_strategy)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_day_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    voter_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Try to get vote from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(voter_id)
    if agent is None:
        return None

    allow_abstain = engine.ruleset.raw["day_flow"]["vote"].get("allow_abstain", False)
    legal_actions = [ActionType.VOTE]
    if allow_abstain:
        legal_actions.append(ActionType.NO_ACTION)

    legal_targets = [pid for pid in engine.legal_exile_targets(gs) if pid != voter_id]
    voter_role = gs.players[voter_id].role if voter_id in gs.players else ""
    if state.get("revote") and state.get("pk_candidates"):
        pk_candidates = set(state.get("pk_candidates") or [])
        legal_targets = [pid for pid in legal_targets if pid in pk_candidates]

    consecutive_no_exile = state.get("consecutive_no_exile_days", 0)
    strategy_directive = build_day_vote_base_directive(
        voter_role,
        allow_abstain=allow_abstain,
        consecutive_no_exile=consecutive_no_exile,
    )
    _inject_vote_basis_hint(strategy_directive, gs, voter_id)
    try:
        from werewolf_agent.runtime.vote_quality import (
            build_day_discussion_summary,
            build_vote_pressure_context,
        )

        strategy_directive["day_discussion_summary"] = build_day_discussion_summary(
            gs, gs.day_number
        )
        strategy_directive["vote_pressure_context"] = build_vote_pressure_context(
            gs, voter_id, pk_candidates=state.get("pk_candidates")
        )
        strategy_directive["anti_herd"] = build_vote_anti_herd_directive()
    except Exception:
        logger.debug("Vote quality context build failed, skipping", exc_info=True)

    # Role-specific vote strategy (voter_role computed above for legal_targets filtering)
    if voter_role == "werewolf":
        wolf_vote_parts = _build_wolf_vote_strategy(
            gs, voter_id, state.get("wolf_team_plan"),
        )
        strategy_directive.update(wolf_vote_parts)
    elif voter_role == "hybrid" and gs.hybrid_master_id:
        strategy_directive["hybrid_vote_strategy"] = build_hybrid_vote_strategy(
            gs.hybrid_master_id
        )
    elif voter_role == "seer":
        try:
            strategy_directive["seer_vote_strategy"] = build_seer_vote_strategy(gs)
        except Exception:
            logger.debug("Failed to build seer vote strategy", exc_info=True)
            strategy_directive["seer_vote_strategy"] = build_fallback_seer_vote_strategy()
    elif voter_role == "witch":
        strategy_directive["witch_vote_strategy"] = build_witch_vote_strategy()
    elif voter_role == "hunter":
        strategy_directive["hunter_vote_strategy"] = build_hunter_vote_strategy()
    elif voter_role in ("villager", "idiot"):
        seer_claimants = _public_seer_claimants(gs)
        strategy_directive["villager_vote_strategy"] = build_villager_vote_strategy(
            seer_claimants
        )

    # Pre-compute evidence-based fallback target for structured failure
    non_self_legal = [t for t in legal_targets if t != voter_id]
    if non_self_legal:
        try:
            from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target
            fb = choose_vote_fallback_target(
                gs,
                voter_id,
                non_self_legal,
                require_evidence=True,
            )
            if fb:
                strategy_directive["_vote_fallback_target"] = fb
        except Exception:
            logger.warning("Failed to compute vote fallback target", exc_info=True)

    context = build_agent_context(
        engine, gs, voter_id, TaskType.VOTE,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    if strategy_directive:
        context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    target = action.target_id if action.action_type == ActionType.VOTE else None
    # Fallback: if agent returned wrong action type but has legal targets,
    # pick an evidence-aware target rather than abstaining silently.
    if target is None and legal_targets:
        target = choose_vote_fallback_target(
            gs,
            voter_id,
            legal_targets,
            require_evidence=True,
        )
    speech = getattr(action, "speech", "") or ""
    reason = getattr(action, "reason", "") or ""
    trace = getattr(action, "trace", None)
    return {
        "vote_target": target,
        "vote_speech": speech,
        "vote_reason": reason,
        "action_trace": trace.model_dump() if trace else None,
    }


# Re-export from runtime.strategy (Task 2 extraction)
from werewolf_agent.runtime.strategy import evaluate_hybrid_master_candidates as _evaluate_hybrid_master_candidates


def agent_hybrid_choose_master(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    hybrid_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Ask hybrid agent to choose their master. Returns None if agent unavailable."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(hybrid_id)
    if agent is None:
        return None

    candidates = build_hybrid_master_candidates(gs, hybrid_id)

    master_assessment = _evaluate_hybrid_master_candidates(gs, hybrid_id, candidates)

    strategy_directive = build_hybrid_master_choice_directive(master_assessment)

    context = build_agent_context(
        engine, gs, hybrid_id, TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHOOSE_MASTER],
        legal_targets=candidates,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    master_target_id = choose_hybrid_master_target(
        action_type=action.action_type,
        target_id=action.target_id,
        candidates=candidates,
    )

    return {
        "master_target_id": master_target_id,
        "action_trace": _action_trace_payload(action),
    }


def agent_exile_last_words(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    player_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Exiled player gives last words."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(player_id)
    if agent is None:
        return None

    player_role = gs.players[player_id].role if player_id in gs.players else ""
    context = build_agent_context(
        engine, gs, player_id, TaskType.LAST_WORDS,
        legal_actions=[ActionType.SPEECH],
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    alive_others = [
        pid for pid, player in gs.players.items()
        if player.alive and pid != player_id
    ]
    strategy_directive = build_exile_last_words_strategy(player_role, alive_others)
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_badge_decision(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Dying sheriff decides to transfer badge or tear it."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_others = living_non_sheriff_ids(gs, sheriff_id)
    context = build_agent_context(
        engine, gs, sheriff_id, TaskType.LAST_WORDS,
        legal_actions=[ActionType.BADGE_TRANSFER, ActionType.BADGE_TEAR],
        legal_targets=alive_others,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    player_role = gs.players[sheriff_id].role if sheriff_id in gs.players else ""
    strategy_directive = build_badge_decision_directive(player_role, alive_others)
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    return build_badge_decision_result(
        action_type=action.action_type,
        target_id=action.target_id,
        action_trace=_action_trace_payload(action),
    )


# Re-export from runtime.strategy (Task 2 extraction)
from werewolf_agent.runtime.strategy import evaluate_hunter_shot_target as _evaluate_hunter_shot_target


def agent_hunter_shot(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    hunter_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | str | None:
    """Get hunter shot target from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(hunter_id)
    if agent is None:
        return None

    death_reason = state.get("hunter_death_reason", "unknown")
    legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != hunter_id]

    # Evaluate shot target value
    try:
        shot_assessment = _evaluate_hunter_shot_target(
            gs, hunter_id, legal_targets, death_reason,
        )
    except Exception:
        logger.warning("Hunter shot target evaluation failed", exc_info=True)
        shot_assessment = None

    strategy_directive = build_hunter_shot_directive(
        death_reason=death_reason,
        shot_assessment=shot_assessment,
    )

    context = build_agent_context(
        engine, gs, hunter_id, TaskType.HUNTER_SHOT,
        legal_actions=[ActionType.HUNTER_SHOT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    return build_hunter_shot_result(
        action_type=action.action_type,
        target_id=action.target_id,
        action_trace=_action_trace_payload(action),
    )


def agent_sheriff_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    voter_id: str,
    candidates: list[str],
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Get sheriff vote from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(voter_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine, gs, voter_id, TaskType.VOTE,
        legal_actions=[ActionType.SHERIFF_VOTE, ActionType.NO_ACTION],
        legal_targets=candidates,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )

    # Wolf strategy for sheriff voting
    strategy_directive = context.strategy_directive or {}
    voter_role = gs.players[voter_id].role if voter_id in gs.players else ""
    # Sheriff voting is an election, not day exile voting; do not inject
    # vote_basis/seer_stance guidance into this action contract.
    if voter_role == "werewolf":
        wolf_teammates = [
            pid for pid, p in gs.players.items()
            if p.alive and p.role == "werewolf" and pid != voter_id
        ]
        teammate_candidates = [c for c in candidates if c in wolf_teammates]
        if teammate_candidates:
            strategy_directive["wolf_sheriff_vote"] = (
                f"你是狼人。你的队友 {', '.join(teammate_candidates)} 也在候选人中。"
                "投票时不要明显全部投给队友——这样会暴露你们的关系。"
                "如果场上有多个候选人，你应该分散投票，表现得像一个独立判断的好人。"
            )
    if strategy_directive:
        context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"vote_target": None, "self_destruct": True}
    target = action.target_id if action.action_type == ActionType.SHERIFF_VOTE else None
    return {
        "vote_target": target,
        "action_trace": _action_trace_payload(action),
        "self_destruct": False,
    }


def agent_sheriff_register(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    player_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Ask a player whether they want to register for sheriff election.

    Returns dict with registration result and self_destruct flag.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(player_id)
    if agent is None:
        return None

    player_role = gs.players[player_id].role if player_id in gs.players else ""
    # Build role-specific registration guidance
    wolf_plan = state.get("wolf_team_plan")
    if player_role == "seer":
        role_hint = (
            "你是预言家。上警通常有利于公开真实验人和建立警徽流，"
            "但应结合已有验人、发言顺序和场上声明决定；"
            "若上警，只能准确报告真实信息，不得为增强可信度编造结果。"
        )
    elif player_role == "werewolf":
        wolf_assignment = _get_wolf_role_assignment(wolf_plan, player_id)
        if wolf_assignment == "fake_seer":
            role_hint = (
                "【强制指令】你是团队安排的悍跳预言家！你必须上警！"
                "你需要在警上冒充预言家，报出假验人结果和警徽流，"
                "与真预言家争夺警徽。这是你的核心任务，必须上警。"
            )
        else:
            role_hint = (
                "你是狼人。如果团队安排你悍跳预言家，你必须上警与真预言家对抗。"
                "如果不悍跳，也可以上警发言获取信息或带节奏。"
            )
    else:
        # Good player (non-seer)
        role_hint = (
            "你是好人（非预言家），可以考虑上警发言表达观点、压制狼人发言空间。"
            "但注意：如果你不是预言家，不要在警上冒充预言家抢警徽，"
            "这会干扰真预言家的信息传递。上警主要目的是发言和表达立场。"
        )

    strategy_directive = {
        "sheriff_registration": (
            f"{role_hint}\n"
            "上警意味着你将在竞选环节发言，争取警长职位或表达观点。"
            "不上警则留在警下投票选出警长。"
        ),
    }

    context = build_agent_context(
        engine, gs, player_id, TaskType.SHERIFF_REGISTRATION,
        legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
        wolf_team_plan=wolf_plan,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    try:
        action, retry_info = agent.act(context)
        action_trace = _action_trace_payload(action)
        if action.action_type == ActionType.SELF_DESTRUCT:
            return {"registered": False, "self_destruct": True, "action_trace": action_trace}
        return {
            "registered": action.action_type == ActionType.SHERIFF_REGISTER,
            "self_destruct": False,
            "action_trace": action_trace,
        }
    except Exception:
        logger.warning("Sheriff registration failed for %s", player_id, exc_info=True)
        return {"registered": False, "self_destruct": False}


def agent_sheriff_withdraw(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Ask a sheriff candidate whether they want to withdraw.

    Returns dict with withdrawal result and self_destruct flag.
    """
    gs: GameState = state["game_state"]
    player_role = gs.players[candidate_id].role if candidate_id in gs.players else ""
    wolf_plan = state.get("wolf_team_plan")
    wolf_assignment = (
        _get_wolf_role_assignment(wolf_plan, candidate_id)
        if player_role == "werewolf"
        else ""
    )
    if player_role == "seer" or wolf_assignment == "fake_seer":
        return {"withdrew": False, "self_destruct": False}

    agent = registry.get_agent(candidate_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine, gs, candidate_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SHERIFF_WITHDRAW, ActionType.NO_ACTION],
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )

    try:
        action, retry_info = agent.act(context)
        action_trace = _action_trace_payload(action)
        if action.action_type == ActionType.SELF_DESTRUCT:
            return {"withdrew": False, "self_destruct": True, "action_trace": action_trace}
        return {
            "withdrew": action.action_type == ActionType.SHERIFF_WITHDRAW,
            "self_destruct": False,
            "action_trace": action_trace,
        }
    except Exception:
        logger.warning("Sheriff withdrawal failed for %s", candidate_id, exc_info=True)
        return {"withdrew": False, "self_destruct": False}


def agent_sheriff_election_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
    all_candidates: list[str],
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Get sheriff election speech from a candidate.

    The speech must explain why the candidate is running for sheriff,
    their badge flow plan (警徽流), and their initial stance.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(candidate_id)
    if agent is None:
        return None

    other_candidates = [c for c in all_candidates if c != candidate_id]

    # Badge flow is seer-exclusive. Only the true seer or the wolf explicitly
    # assigned to fake-seer duty should receive this private instruction.
    player_role = gs.players[candidate_id].role if candidate_id in gs.players else ""
    wolf_plan = state.get("wolf_team_plan")
    wolf_assignment = (
        _get_wolf_role_assignment(wolf_plan, candidate_id)
        if player_role == "werewolf"
        else ""
    )
    uses_seer_protocol = sheriff_uses_seer_protocol(player_role, wolf_assignment)
    badge_flow_instruction = build_sheriff_badge_flow_instruction(
        uses_seer_protocol
    )

    # Single-sided vs multi-seer context is derived only from public speeches.
    # The current candidate's own private assignment may be included because
    # this instruction is shown only to that candidate.
    public_seer_claimers = {
        candidate
        for candidate in all_candidates
        if _has_publicly_claimed_seer(gs, candidate)
    }
    if uses_seer_protocol:
        public_seer_claimers.add(candidate_id)
    seer_context = build_sheriff_seer_context(
        public_seer_claimers,
        uses_seer_protocol=uses_seer_protocol,
    )
    prev_speeches = collect_previous_sheriff_speeches(gs, candidate_id)
    prev_speech_instruction = build_previous_sheriff_speech_instruction(
        prev_speeches
    )

    # Persona-based speech style differentiation
    speech_style = _get_persona_speech_style(agent)
    task_style = _get_persona_task_style(agent, "sheriff_speech")

    merged_hints = {**_SPEECH_STYLE_HINTS, **_SHERIFF_SPEECH_STYLE_OVERRIDES}
    style_hint = merged_hints.get(speech_style, "从你自己的独特角度分析场上局势。")
    task_hint = _TASK_STYLE_HINTS.get(task_style, "")

    strategy_directive = build_sheriff_election_speech_directive(
        style_hint=style_hint,
        task_hint=task_hint,
        badge_flow_instruction=badge_flow_instruction,
        seer_context=seer_context,
        prev_speech_instruction=prev_speech_instruction,
        other_candidates=other_candidates,
    )
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(strategy_directive, gs, candidate_id)

    role_speech_hint = build_sheriff_role_speech_hint(player_role)
    if role_speech_hint:
        strategy_directive["role_speech_hint"] = role_speech_hint

    seer_verification_rationale = build_seer_verification_rationale(player_role)
    if seer_verification_rationale:
        strategy_directive["seer_verification_rationale"] = seer_verification_rationale

    # Wolf: inject role-specific strategy from wolf_team_plan
    if player_role == "werewolf":
        wolf_day_directive = _build_wolf_day_speech_directive(gs, candidate_id, wolf_plan)
        strategy_directive.update(wolf_day_directive)
        fake_seer_publicly_claimed = False
        if wolf_assignment != "fake_seer" and wolf_plan and wolf_plan.get("fake_seer"):
            fake_seer_publicly_claimed = _has_publicly_claimed_seer(
                gs,
                wolf_plan["fake_seer"],
            )
        strategy_directive.update(build_wolf_sheriff_election_directives(
            wolf_assignment=wolf_assignment,
            wolf_plan=wolf_plan,
            candidate_id=candidate_id,
            fake_seer_publicly_claimed=fake_seer_publicly_claimed,
        ))

    context = build_agent_context(
        engine, gs, candidate_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"speech_text": "", "action_trace": {}, "self_destruct": True}

    speech_text = getattr(action, "speech", "") or ""

    # Reject empty sheriff election speeches
    if not speech_text.strip() or len(speech_text.strip()) < 10:
        if uses_seer_protocol:
            speech_text = (
                f"我上警是因为我需要通过警徽流传递关键信息。"
                f"我的警徽流暂定先看{other_candidates[0] if other_candidates else '待定'}。"
                f"希望大家支持我当选警长。"
            )
        else:
            speech_text = (
                "我上警是想先给出自己的观察视角。"
                f"我会重点听{other_candidates[0] if other_candidates else '后置位'}的发言，"
                "看站边和逻辑是否前后一致。"
            )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action), "self_destruct": False}


_POST_GAME_KEEP = frozenset({"reflection_task", "game_outcome"})


def _strip_in_game_directives(context):
    """赛后反思:剥离赛内决策 directive,只留 allowlist。

    `_agent_reflection` 调 `build_agent_context(TaskType.REFLECTION)` 拿到的
    context.strategy_directive 仍装满赛内决策 directive(role_alerts /
    skill_tactical_advice / witch_poison_deterrent / must_address_alerts 等),
    反思指令 reflection_task 只是一个平级 key 被淹没,LLM 因此输出赛内决策
    (刀人计划 / 发言分析)而非赛后反思。

    此 helper 在 merge reflection_directive 之前清掉赛内 directive,最终
    strategy_directive == {reflection_task, game_outcome}。幂等:若本就无赛内
    directive 则为无害 no-op。
    """
    kept = {k: v for k, v in (context.strategy_directive or {}).items()
            if k in _POST_GAME_KEEP}
    return context.model_copy(update={"strategy_directive": kept})


def _agent_reflection(
    state: dict[str, Any],
    engine: Any,
    registry: Any,
    player_id: str,
) -> dict[str, Any]:
    """Post-game reflection: each player reviews their performance.

    Design doc §10.2: generates key judgments, mistakes, successful
    strategies, deception experienced, and improvement suggestions.

    Per [[feedback-reflection-role-specific]]: the reflection prompt must
    branch on role family (good / wolf / hybrid-defer) instead of a
    single generic prompt. Each branch asks role-specific questions and
    mandates a "保留的优点" section so cross-game learning preserves
    what worked, not just what failed.
    """
    agent = registry.get_agent(player_id)
    if agent is None:
        return {}

    gs: GameState = state["game_state"]
    player = gs.players.get(player_id)
    winner = gs.winning_faction or "?"

    try:
        # P0-RF1: pass TaskType.REFLECTION so speech_quality_phase
        # returns None and skips the public-speech 4-field check.
        # Reflection text is post-game review and has no stance /
        # suspicion_target / vote_leaning / evidence fields. Using
        # TaskType.SPEECH here triggered a retry loop that surfaced
        # as 8/12 reflection failures in the post-merge game trace.
        context = build_agent_context(
            engine, gs, player_id, TaskType.REFLECTION,
            legal_actions=[ActionType.SPEECH],
            restored_memory=state.get("restored_memory"),
            cognition_state_manager=state.get("cognition_state_manager"),
        )
        reflection_task = build_reflection_prompt(
            player=player,
            winner=winner,
            hybrid_master_faction=gs.hybrid_master_faction,
        )
        reflection_directive = {
            "reflection_task": reflection_task,
            "game_outcome": (
                f"胜利方是{'好人' if winner == 'good' else '狼人'}阵营。"
                f"你{'存活到' if (player and player.alive) else '在'}游戏结束。"
                f"你的身份是 {player.role if player else '?'}。"
            ),
        }
        context = _strip_in_game_directives(context)
        context = _merge_strategy_directive(context, reflection_directive)

        action, _retry_info = agent.act(context)
        # P0-RF2: scrub raw p\d+ tokens from the LLM-written reflection
        # before it lands in graph state and gets persisted to
        # ReflectionMemory. The template's 1-line PII hint is a
        # best-effort prompt; this post-processing is the authoritative
        # guard against cross-game ID leakage.
        from werewolf_agent.memory.store import _scrub_player_ids
        return {"reflection_text": _scrub_player_ids(getattr(action, "speech", "") or "")}
    except Exception:
        logger.warning("Reflection failed for %s", player_id, exc_info=True)
        return {"reflection_text": ""}
