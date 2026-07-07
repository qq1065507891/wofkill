# -*- coding: utf-8 -*-
"""
运行时 agent action pipeline。

作者: Mike
创建日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.runtime.agent_action_pipeline import agent_day_speech
    >>> agent_day_speech(...)
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import (
    ActionType,
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
    extract_first_balanced_json_object as _extract_first_balanced_json_object,  # noqa: F401
)
from werewolf_agent.runtime.last_words_directives import build_exile_last_words_strategy
from werewolf_agent.runtime.pk_speech_directives import build_pk_speech_strategy
from werewolf_agent.runtime.reflection_prompt import (
    GOOD_REFLECTION_TEMPLATE,
    WOLF_REFLECTION_TEMPLATE,
    build_reflection_prompt,
)
from werewolf_agent.runtime.sheriff_action_directives import (
    living_non_sheriff_ids,
)
from werewolf_agent.runtime.seer_night_directives import (
    build_badge_flow_next_targets,
    build_seer_legal_targets,
    build_seer_night_strategy_directive,
)
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS as AGENT_TIMEOUTS  # noqa: F401
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
    _SHERIFF_SPEECH_STYLE_OVERRIDES as _SHERIFF_SPEECH_STYLE_OVERRIDES,
    _TASK_STYLE_HINTS as _TASK_STYLE_HINTS,
    _get_persona_speech_style,
    _get_persona_task_style as _get_persona_task_style,
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
    build_wolf_night_directive as _build_wolf_night_directive,  # noqa: F401
    build_wolf_vote_directive as _build_wolf_vote_strategy,
)
from werewolf_agent.runtime.directives._shared import (
    build_sheriff_silent_directive as _build_sheriff_silent_directive,
)
from werewolf_agent.runtime.agent_action_audit import (
    VOTE_BASIS_GUIDANCE,  # noqa: F401
    _audit_context_kwargs,
    _inject_vote_basis_hint,
    _is_sheriff_silenced,
    _seer_credibility_audit_payload,
)
from werewolf_agent.runtime.agent_reflection_support import (
    _agent_reflection,  # noqa: F401
    _strip_in_game_directives,  # noqa: F401
)
from werewolf_agent.runtime.agent_registry import AgentRegistry, SimpleAgentRegistry  # noqa: F401
from werewolf_agent.runtime.wolf_kill_support import (
    _build_wolf_kill_directive,  # noqa: F401
    _single_wolf_vote,  # noqa: F401
)
from werewolf_agent.runtime.agent_sheriff_actions import (
    agent_sheriff_endorse,
    agent_sheriff_election_speech,
    agent_sheriff_pick_speech_order,
    agent_sheriff_register,
    agent_sheriff_vote,
    agent_sheriff_withdraw,
)
from werewolf_agent.runtime.agent_wolf_actions import (
    agent_wolf_consensus,
    agent_wolf_discussion,
    agent_wolf_team_plan,
)

logger = logging.getLogger(__name__)

# 兼容旧测试和调试入口：反思模板实现已移动到 runtime.reflection_prompt。
_build_reflection_prompt = build_reflection_prompt
_GOOD_REFLECTION_TEMPLATE = GOOD_REFLECTION_TEMPLATE
_WOLF_REFLECTION_TEMPLATE = WOLF_REFLECTION_TEMPLATE


# -- Backward-compatible re-exports from runtime.strategy (Task 2 extraction) --
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value as _estimate_witch_save_value,
    evaluate_seer_check_value as _evaluate_seer_check_value,
    evaluate_wolf_kill_target as _evaluate_wolf_kill_target,  # noqa: F401
    get_wolf_role_assignment as _get_wolf_role_assignment,  # noqa: F401
    has_publicly_claimed_seer as _has_publicly_claimed_seer,  # noqa: F401
)
from werewolf_agent.runtime.strategy.seer import (
    public_seer_claimants as _public_seer_claimants,
)  # noqa: F401


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
        engine,
        gs,
        witch_id,
        TaskType.NIGHT_ACTION,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_kill_target_id=wolf_kill_target_id,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
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
            from werewolf_agent.runtime.strategy.poison import (
                collect_witch_poison_candidates,
            )

            cands = collect_witch_poison_candidates(gs, witch_id)
        except Exception:
            cands = []
        alive = sum(1 for p in gs.players.values() if p.alive)
        witch_directive["witch_poison_candidates"] = (
            build_witch_poison_candidates_directive(
                cands,
                alive_count=alive,
            )
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
    poison_target_id = (
        action.target_id if action.action_type == ActionType.USE_POISON else None
    )

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
        engine,
        gs,
        seer_id,
        TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    seer_target_id = (
        action.target_id if action.action_type == ActionType.CHECK_ALIGNMENT else None
    )
    return {
        "seer_target_id": seer_target_id,
        "seer_action_trace": _action_trace_payload(action),
    }


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
        engine,
        gs,
        speaker_id,
        TaskType.DEFENSE_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        discussion_positions=state.get("discussion_positions"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
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
        engine,
        gs,
        speaker_id,
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        discussion_positions=state.get("discussion_positions"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
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
            gs,
            speaker_id,
            state.get("wolf_team_plan"),
        )
        strategy_directive.update(wolf_parts)
    elif player_role == "seer":
        # P0-G3223805846-3: pass the day's speech order so the seer directive
        # can enforce the "jump immediately when speaking late" rule.  The
        # order lives on RuntimeState (populated by free_discussion); fall
        # back to None when not yet materialised so the directive still
        # works in unit tests / early-day planning contexts.
        seer_speech_parts = _build_seer_day_speech_directive(
            gs,
            speaker_id,
            speech_order=state.get("speech_order"),
        )
        strategy_directive.update(seer_speech_parts)
    elif player_role == "hunter":
        strategy_directive["hunter_speech_directive"] = (
            _build_hunter_day_speech_directive(gs, speaker_id)
        )
    elif player_role == "hybrid":
        strategy_directive["hybrid_speech_directive"] = (
            _build_hybrid_day_speech_directive(gs, speaker_id)
        )
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
        alive_others = [
            pid for pid, p in gs.players.items() if p.alive and pid != speaker_id
        ]
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
                gs,
                sheriff_id=None,
                badge_state="torn",
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
        alive_others = [
            pid for pid, p in gs.players.items() if p.alive and pid != speaker_id
        ]
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
                speaker_id,
                player_role,
                claim_err,
            )
            alive_others = [
                pid for pid, p in gs.players.items() if p.alive and pid != speaker_id
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
        engine,
        gs,
        speaker_id,
        TaskType.PK_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    # Add prior tally to visible state
    if prior_tally:
        updated_visible = {
            **context.visible_world_state,
            "prior_vote_tally": prior_tally,
        }
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
            gs,
            voter_id,
            state.get("wolf_team_plan"),
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
            strategy_directive["seer_vote_strategy"] = (
                build_fallback_seer_vote_strategy()
            )
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
        engine,
        gs,
        voter_id,
        TaskType.VOTE,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
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
from werewolf_agent.runtime.strategy import (
    evaluate_hybrid_master_candidates as _evaluate_hybrid_master_candidates,
)


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
        engine,
        gs,
        hybrid_id,
        TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHOOSE_MASTER],
        legal_targets=candidates,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
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
        engine,
        gs,
        player_id,
        TaskType.LAST_WORDS,
        legal_actions=[ActionType.SPEECH],
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    alive_others = [
        pid for pid, player in gs.players.items() if player.alive and pid != player_id
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
        engine,
        gs,
        sheriff_id,
        TaskType.LAST_WORDS,
        legal_actions=[ActionType.BADGE_TRANSFER, ActionType.BADGE_TEAR],
        legal_targets=alive_others,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
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
from werewolf_agent.runtime.strategy import (
    evaluate_hunter_shot_target as _evaluate_hunter_shot_target,
)


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
    legal_targets = [
        pid for pid, p in gs.players.items() if p.alive and pid != hunter_id
    ]

    # Evaluate shot target value
    try:
        shot_assessment = _evaluate_hunter_shot_target(
            gs,
            hunter_id,
            legal_targets,
            death_reason,
        )
    except Exception:
        logger.warning("Hunter shot target evaluation failed", exc_info=True)
        shot_assessment = None

    strategy_directive = build_hunter_shot_directive(
        death_reason=death_reason,
        shot_assessment=shot_assessment,
    )

    context = build_agent_context(
        engine,
        gs,
        hunter_id,
        TaskType.HUNTER_SHOT,
        legal_actions=[ActionType.HUNTER_SHOT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    return build_hunter_shot_result(
        action_type=action.action_type,
        target_id=action.target_id,
        action_trace=_action_trace_payload(action),
    )


__all__ = [
    "agent_night_witch",
    "agent_night_seer",
    "agent_wolf_team_plan",
    "agent_wolf_consensus",
    "agent_wolf_discussion",
    "agent_defense_speech",
    "agent_day_speech",
    "agent_sheriff_pick_speech_order",
    "agent_sheriff_endorse",
    "agent_pk_speech",
    "agent_day_vote",
    "agent_hybrid_choose_master",
    "agent_exile_last_words",
    "agent_badge_decision",
    "agent_hunter_shot",
    "agent_sheriff_vote",
    "agent_sheriff_register",
    "agent_sheriff_withdraw",
    "agent_sheriff_election_speech",
]
