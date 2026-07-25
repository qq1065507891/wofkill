# -*- coding: utf-8 -*-
"""
把 GameState 组装成 PlayerAgent 决策所需的 AgentContext。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-25

使用示例:
    >>> from werewolf_agent.runtime.context import build_agent_context
    >>> build_agent_context(...)
"""


# 将 GameState 转换为 PlayerAgent 可用的 AgentContext。
# 作者: Mike
# 创建日期: 2025-01-15
# 修改日期: 2026-07-25
# 使用示例: 内部模块，无对外接口
# 从 agent_adapter.py 拆出，用于降低大型适配器的职责复杂度。
# 本模块负责：
# - 人设风格提示与画像。
# - RAG 提示注入。
# - 跨局记忆提示，包括画像、反思和认知矩阵。
# - 技能输出注入与工具定义。
# - 策略指令合并。
# - build_agent_context() 主入口。

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    TaskType,
)
from werewolf_agent.agents.discussion_summary import (
    discussion_summary_for_player,
    discussion_summary_text,
)
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.context_public_summary import (
    build_public_summary,
    build_recent_transcript,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.public_ledger import build_public_claim_text_ledger
from werewolf_agent.skills.registry import SkillRegistry as SkillRegistry  # noqa: F401

# Backward-compatible re-exports from runtime.directives package.
# Backward-compatible re-exports from runtime.strategy (Task 2 extraction).
from werewolf_agent.runtime.strategy import (
    build_witch_pressure_targets as _build_witch_pressure_targets,  # noqa: F401
    evaluate_death_cause_claims as _evaluate_death_cause_claims,
)
from werewolf_agent.runtime.context_action_trace import _action_trace_payload  # noqa: F401
from werewolf_agent.runtime.context_cross_game_memory import (
    build_cross_game_memory_hints,
)
from werewolf_agent.runtime.context_memory_hints import (
    HINT_BUDGET,  # noqa: F401
    REFLECTION_CARD_BUDGET,  # noqa: F401
    _cognition_matrix_hint,  # noqa: F401
    _evidence_id_ref,  # noqa: F401
    _profile_memory_hint,  # noqa: F401
    _reflection_memory_hints,  # noqa: F401
)
from werewolf_agent.runtime.context_persona import (
    _SHERIFF_SPEECH_STYLE_OVERRIDES,  # noqa: F401
    _SPEECH_STYLE_HINTS,  # noqa: F401
    _TASK_STYLE_HINTS,  # noqa: F401
    _get_persona_speech_style,  # noqa: F401
    _get_persona_task_style,  # noqa: F401
    _load_persona_profile,  # noqa: F401
)
from werewolf_agent.runtime.context_rag import (
    _extract_role_claim,  # noqa: F401
    _extract_suspects,  # noqa: F401
    _extract_trusts,  # noqa: F401
    _extract_vote_intent,  # noqa: F401
    _first_sentence,  # noqa: F401
    _inject_seed_rag_hints,
    _normalize_legal_actions_to_tags,  # noqa: F401
    _rag_phase_for_task,  # noqa: F401
)
from werewolf_agent.runtime.context_role_directives import (
    apply_role_strategy_context as _apply_role_strategy_context,
)
from werewolf_agent.runtime.context_skill_advice import (
    _inject_skill_output,
    _skill_advice_frame_to_prompt_dict,  # noqa: F401
    _skill_output_to_advice_frame,  # noqa: F401
)
# 兼容旧源码扫描测试：真实枚举值 "review_correct" 位于 context_skill_advice。
from werewolf_agent.runtime.context_strategy_directives import (
    cap_strategy_directive,  # R5 公开入口; 2026-07-21 build_agent_context 强制接通
    cap_strategy_directive as _cap_strategy_directive,  # noqa: F401  # 旧私有别名兼容
    _directive_size,  # noqa: F401
    _merge_strategy_directive,  # noqa: F401
    _ROUND_SPECIFIC_DROP_KEYS,  # noqa: F401
    _MAX_STRATEGY_DIRECTIVE_TOKENS,  # noqa: F401  # 历史兼容: 导入 _MAX_STRATEGY_DIRECTIVE_TOKENS 常量
)

logger = logging.getLogger(__name__)


def _public_world_evidence(
    gs: GameState,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], set[str]]:
    """使用认知层索引构建窄化的公开身份/阵营证据。"""
    from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex
    from werewolf_agent.cognition.visibility import VisibilityPolicy
    from werewolf_agent.cognition.world_state import extract_facts

    policy = VisibilityPolicy()
    index = PublicEvidenceIndex()
    for event_index, event in enumerate(gs.events):
        for fact in extract_facts(event, gs):
            if policy.compute_fact_visibility(fact, event_index).visibility != "public":
                continue
            prefix = "claim" if fact.fact_type in {
                "claimed_role", "claimed_good", "seer_check_claim"
            } else "event"
            evidence_id = f"{prefix}:{gs.game_id}:{event_index}"
            index.observe_assignment_reference(fact, evidence_id)
    return index.assignment_evidence(), index.assignment_evidence_ids()


def build_agent_context(
    engine: RuleEngine,
    gs: GameState,
    player_id: str,
    task_type: TaskType,
    *,
    legal_actions: list[ActionType] | None = None,
    legal_targets: list[str] | None = None,
    wolf_kill_target_id: str | None = None,
    wolf_team_plan: dict[str, Any] | None = None,
    rag_service: Any | None = None,
    restored_memory: Any | None = None,
    cognition_state_manager: Any | None = None,
    discussion_positions: dict[str, Any] | None = None,
    discussion_state: MutableMapping[str, Any] | None = None,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> AgentContext:
    """Build AgentContext for a player from current game state.

    Visibility rules:
    - Player only sees their own role.
    - No moderator_full, no other players' private state.
    - Wolf teammates visible only to wolves.
    - Seer sees own check results only.
    - Witch sees potion availability only.
    """
    # Lazy imports for functions that stayed in agent_adapter or other modules
    from werewolf_agent.runtime.private_memory import build_private_memory
    from werewolf_agent.runtime.visible_state import build_visible_player_state

    player = gs.players.get(player_id)
    if player is None:
        return AgentContext(agent_id=player_id, task_type=task_type)

    # P3-1: pass role+player_id so ``build_visible_player_state`` does
    # the role-specific injection (wolf_teammates / check_results /
    # antidote_available / master_id) instead of inlining here.  The
    # function now has a whitelist projection that strips any
    # accidentally-added private key — a defense-in-depth improvement
    # over the previous inline approach.
    visible: dict[str, Any] = build_visible_player_state(
        gs,
        role=player.role if player else None,
        player_id=player_id,
        wolf_team_plan=wolf_team_plan,
    )
    # PR2: REFLECTION (post-game review) must NOT carry the live board.
    # build_visible_player_state returns the in-progress view (alive
    # players, current day/night, role-specific private fields), which
    # makes the LLM act as an in-game analyst instead of a retrospective
    # reviewer. Swap in a retrospective summary for REFLECTION only;
    # SPEECH/VOTE/other task types keep the live board unchanged.
    if task_type == TaskType.REFLECTION:
        from werewolf_agent.runtime.visible_state import build_post_game_summary
        visible = build_post_game_summary(gs, player_id)
    # MEM-NEW-8: build_private_memory now returns a tuple
    # ``(memory, caveat)`` — the caveat is no longer a meta key in
    # the memory dict, so no ``pop()`` is needed. The schema is
    # uniform: memory values are category lists, caveat is a
    # top-level string.
    private_memory, private_memory_caveat = build_private_memory(gs, player_id)
    # PR2: REFLECTION 仅抑制 visible_world_state 里的 private_memory 字段
    # (避免 live 私有记忆污染回顾视角)。注意:private_memory_hints 仍传入
    # AgentContext 并经 prompt_builder._build_private_memory_hints 渲染为
    # 【辅助】section —— 这是已知旁路,反思时保留 viewer 自己的本局认知
    # 作为回顾参考;是否清空留待后续设计决策(本 PR 聚焦 visible_world_state)。
    if private_memory and task_type != TaskType.REFLECTION:
        visible["private_memory"] = private_memory
    private_memory_hints = private_memory or {}

    # P3-1: the static role-specific private fields (wolf_teammates /
    # check_results / antidote_available / poison_available / master_id)
    # are now injected inside ``build_visible_player_state(role=...)``
    # above, with a whitelist projection.  The inline role branches
    # that used to live here have been deleted — see the
    # defense-in-depth whitelist in ``visible_state.py``.
    #
    strategy_directive: dict[str, Any] = {}
    visible, strategy_directive = _apply_role_strategy_context(
        visible=visible,
        strategy_directive=strategy_directive,
        gs=gs,
        player_id=player_id,
        wolf_kill_target_id=wolf_kill_target_id,
    )

    transcript = build_recent_transcript(gs)
    public_summary = build_public_summary(gs)

    # ── 玩家自己的讨论摘要（私有策略记忆，不属于公开记录） ──
    summary_state: MutableMapping[str, Any] = (
        discussion_state
        if discussion_state is not None
        else {"discussion_positions": discussion_positions or {}}
    )
    own_summary = discussion_summary_text(
        discussion_summary_for_player(summary_state, player_id)
    )
    internal_discussion_summary = (
        f"【私有策略记忆·今日讨论总结 D{gs.day_number}】\n{own_summary}"
        if own_summary
        else ""
    )

    # Build contradiction alerts and belief state from world state
    ctx_alerts: list[dict[str, Any]] = []
    must_address: list[dict[str, Any]] = []
    belief_dict: dict[str, Any] = {}
    seer_credibility_summary: dict[str, Any] = {}
    possible_worlds_dict: dict[str, Any] = {}
    authoritative_world_identities: list[dict[str, Any]] = []
    simulation_predictions_dict: dict[str, Any] = {}
    possible_worlds_set = None
    public_evidence_ids: set[str] = set()
    world_state = None
    belief_state = None
    alerts: list[Any] = []

    try:
        from werewolf_agent.cognition.world_state import build_world_state
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        from werewolf_agent.cognition.belief import BeliefUpdater
        from werewolf_agent.cognition.claim_credibility import SeerClaimCredibilityEngine

        world_state = build_world_state(gs)

        # -- Belief update: who do I suspect / trust --
        updater = BeliefUpdater()
        belief_state = updater.initialize(list(gs.players.keys()), player_id)
        # 使用 VisibilityPolicy 过滤，仅传入该玩家可见的事实
        from werewolf_agent.cognition.visibility import VisibilityPolicy
        _vis_policy = VisibilityPolicy()
        _player_role = gs.players[player_id].role if player_id in gs.players else "villager"
        visible_facts = _vis_policy.filter_visible_facts(
            world_state, player_id, _player_role
        )
        credibility_engine = SeerClaimCredibilityEngine()
        belief_state = updater.update(
            belief_state, visible_facts, gs.day_number, credibility=credibility_engine,
        )
        seer_credibility_summary = credibility_engine.prompt_summary()

        # Build structured belief summary for agent prompt
        suspect_list = []
        trust_list = []
        for pid, b in belief_state.beliefs.items():
            if pid == player_id or pid not in gs.players or not gs.players[pid].alive:
                continue
            top_role, top_prob = b.top_role_guess()
            entry = {
                "player": pid,
                "faction_lean": b.faction_lean,
                "trust": round(b.trust, 2),
                "top_role_guess": top_role,
                "top_role_prob": round(top_prob, 2),
            }
            if b.faction_lean == "wolf_lean" or b.trust < 0.35:
                suspect_list.append(entry)
            elif b.faction_lean == "good_lean" or b.trust > 0.65:
                trust_list.append(entry)

        belief_dict = {
            "my_suspects": sorted(suspect_list, key=lambda x: x["trust"]),
            "my_trusted": sorted(trust_list, key=lambda x: -x["trust"]),
        }

        # -- Contradiction detection --
        contradiction_engine = ContradictionEngine(
            role_capacities=engine.ruleset.raw.get("role_distribution"),
        )
        alerts = contradiction_engine.detect(visible_facts, gs.day_number)

        for alert in alerts:
            alert_entry = {
                "alert_type": alert.alert_type,
                "player_id": alert.player_id,
                "priority": alert.priority,
                "description": alert.description,
                "evidence": list(alert.evidence),
            }
            ctx_alerts.append(alert_entry)

        for alert in ctx_alerts:
            if alert["priority"] == "low":
                continue
            players = [p for p in alert["player_id"].split(",") if p]
            must_address.append({
                "alert_type": alert["alert_type"],
                "players": players,
                "public_evidence": alert["description"],
                "required_response": ["question", "side_with", "park"],
                # NEW (v1.1.4 fallback-fix): priority passed through so the
                # speech_quality gate can weight high vs medium alerts
                # independently in validate_public_speech.
                "priority": alert["priority"],
            })

        if must_address:
            # Phase 1 self-audit (P1-1 revert): the legacy
            # ``strategy_directive["directive"] = "你必须在发言中回应..."``
            # text was deleted.  ``must_address_alerts`` already
            # conveys the imperative (the MUST sub-group framing
            # makes the binding explicit).  The duplicate natural-
            # language imperative was redundant.
            strategy_directive["must_address_alerts"] = must_address
    except Exception:
        logger.debug("Contradiction/belief building failed, skipping", exc_info=True)

    if cognition_state_manager is not None:
        try:
            prompt_belief_summary = cognition_state_manager.prompt_belief_summary(
                player_id,
                gs,
            )
            if prompt_belief_summary:
                belief_dict = prompt_belief_summary
        except Exception:
            logger.debug(
                "Live cognition manager summary failed for %s; using fallback",
                player_id,
                exc_info=True,
            )

    try:
        from werewolf_agent.cognition.worlds import PossibleWorldsEngine

        role_counts = {
            role: int(cfg.get("count", 0))
            for role, cfg in engine.ruleset.raw.get("roles", {}).items()
            if int(cfg.get("count", 0)) > 0
        }
        known_roles = {player_id: player.role}
        if player.role == "werewolf":
            known_roles.update({
                pid: p.role
                for pid, p in gs.players.items()
                if p.role == "werewolf"
            })
        if cognition_state_manager is not None and hasattr(
            cognition_state_manager, "public_world_evidence"
        ):
            assignment_evidence, public_evidence_ids = (
                cognition_state_manager.public_world_evidence(player_id)
            )
        else:
            assignment_evidence, public_evidence_ids = _public_world_evidence(gs)
        grounded_belief = {
            key: [
                {
                    **item,
                    "evidence_ids": [],
                }
                for item in belief_dict.get(key, [])
            ]
            for key in ("my_suspects", "my_trusted")
        }
        worlds = PossibleWorldsEngine().generate(
            viewer_id=player_id,
            viewer_role=player.role,
            player_ids=list(gs.players.keys()),
            role_counts=role_counts,
            belief_summary=grounded_belief,
            known_roles=known_roles,
            generated_at_event_index=len(gs.events),
            top_k=3,
            public_evidence_ids=public_evidence_ids,
            assignment_evidence=assignment_evidence,
        )
        if worlds.worlds:
            possible_worlds_set = worlds
            possible_worlds_dict = worlds.to_prompt_dict(max_assignments=4)
            authoritative_world_identities = worlds.to_audit_identity_proofs()
    except Exception:
        logger.debug("Possible-world generation failed for %s", player_id, exc_info=True)

    if possible_worlds_set is not None:
        try:
            from werewolf_agent.cognition.simulator import BoundedSimulator

            simulation = BoundedSimulator().simulate(
                viewer_id=player_id,
                possible_worlds=possible_worlds_set,
                alive_players=[
                    pid for pid, player_state in gs.players.items()
                    if player_state.alive
                ],
                day_number=gs.day_number,
                top_k=2,
            )
            simulation_predictions_dict = simulation.to_prompt_dict()
        except Exception:
            logger.debug("Simulation generation failed for %s", player_id, exc_info=True)

    if legal_actions is None:
        legal_actions = []
    if legal_targets is None:
        legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != player_id]

    # -- Skill-based tactical advice (pre-injection path; no tool exposure) --
    skill_analyses: dict[str, str] = {}
    skill_call_records: list[dict[str, Any]] = []
    try:
        try:
            strategy_directive, skill_analyses = _inject_skill_output(
                strategy_directive, gs, player_id,
                world_state, belief_state, alerts, task_type.value,
                legal_targets=legal_targets,
                wolf_team_plan=wolf_team_plan,
                skill_call_records=skill_call_records,
            )
        except TypeError as exc:
            # 兼容旧版/测试替身签名：新增审计记录参数不应让已有
            # skill 注入实现整体失效。
            if "skill_call_records" not in str(exc):
                raise
            strategy_directive, skill_analyses = _inject_skill_output(
                strategy_directive, gs, player_id,
                world_state, belief_state, alerts, task_type.value,
                legal_targets=legal_targets,
                wolf_team_plan=wolf_team_plan,
            )
    except Exception as exc:
        skill_call_records.append({
            "call_kind": "skill",
            "call_name": "skill_injection",
            "skill_name": "skill_injection",
            "status": "error",
            "success": False,
            "prompt_visible": False,
            "result_available_to_decision": False,
            "decision_usage": "not_available_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "input_summary": {
                "role": player.role,
                "phase": task_type.value,
                "task_type": task_type.value,
                "day": gs.day_number,
                "legal_target_count": len(legal_targets or []),
                "has_wolf_team_plan": bool(wolf_team_plan),
            },
        })
        logger.debug("Skill injection failed, skipping", exc_info=True)

    # -- Role state monitoring: inject role-specific critical/warning alerts --
    try:
        from werewolf_agent.cognition.role_monitor import RoleStateMonitor
        monitor = RoleStateMonitor()
        role_alerts = monitor.assess(gs, player_id, player.role, gs.phase)
        if role_alerts:
            strategy_directive["role_alerts"] = [
                {"alert_type": a.alert_type, "severity": a.severity, "message": a.message}
                for a in role_alerts
            ]
    except Exception:
        logger.warning("Role state monitoring failed, skipping", exc_info=True)

    # -- Death cause claim evaluation: does the player trust each claim? --
    # D-15: phase-gate the evaluation.  Pre-fix, the evaluator ran on
    # every context build, which meant night-action prompts (witch
    # decision, seer check, wolf kill) carried death-cause guidance
    # the model wasn't asking for and that bloated the prompt
    # unnecessarily.  The death-cause evaluation only makes sense in
    # the day-phase speech / vote / sheriff slots where the player is
    # actively weighing other players' public claims.
    if task_type in (
        TaskType.SPEECH,
        TaskType.PK_SPEECH,
        TaskType.SHERIFF_SPEECH,
        TaskType.VOTE,
        TaskType.DEFENSE_SPEECH,
    ):
        try:
            death_evaluations = _evaluate_death_cause_claims(
                gs, player_id, player.role,
                wolf_kill_target_id=wolf_kill_target_id,
            )
            if death_evaluations:
                strategy_directive["death_cause_evaluation"] = death_evaluations
        except Exception:
            logger.debug("Death cause evaluation failed, skipping", exc_info=True)

    # -- Cross-game memory: inject accumulated learning from previous games --
    cross_game_memory = build_cross_game_memory_hints(
        None,
        player_id=player_id,
        current_role=player.role,
    )
    if restored_memory is not None:
        try:
            cross_game_memory = build_cross_game_memory_hints(
                restored_memory,
                player_id=player_id,
                current_role=player.role,
            )
        except Exception:
            logger.debug("Failed to inject cross-game memory for %s", player_id, exc_info=True)

    context = AgentContext(
        agent_id=player_id,
        task_type=task_type,
        phase=gs.phase,
        day_number=gs.day_number,
        night_number=gs.night_number,
        own_role=player.role,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        public_summary=public_summary,
        internal_discussion_summary=internal_discussion_summary,
        public_claim_ledger=build_public_claim_text_ledger(gs),
        visible_world_state=visible,
        private_memory_hints=private_memory_hints,
        private_memory_caveat=private_memory_caveat,
        reflection_memory_hints=cross_game_memory.reflection_memory_hints,
        profile_memory_hint=cross_game_memory.profile_memory_hint,
        cognition_matrix_hint=cross_game_memory.cognition_matrix_hint,
        error_pattern_hint=cross_game_memory.error_pattern_hint,
        recent_transcript=transcript,
        contradiction_alerts=ctx_alerts,
        seer_credibility=seer_credibility_summary,
        belief_state=belief_dict,
        possible_worlds=possible_worlds_dict,
        authoritative_world_identities=authoritative_world_identities,
        public_world_evidence_ids=sorted(public_evidence_ids),
        simulation_predictions=simulation_predictions_dict,
        strategy_directive=cap_strategy_directive(strategy_directive),
        skill_analyses=skill_analyses,
        decision_identity=decision_identity,
        exposure_collector=exposure_collector,
        # NEW-S04-A: skill_analysis_hints is no longer populated. The
        # single source of truth is strategy_directive.skill_tactical_advice
        # (rendered inside the strategy_directive section). The old
        # dual-render path passed the same opaque dict to BOTH
        # skill_analyses AND skill_analysis_hints, doubling the token
        # budget. Now only the structured path remains.
        skill_analysis_hints={},
    )
    final_context = _inject_seed_rag_hints(
        context,
        ruleset_id=gs.ruleset_id,
        rag_service=rag_service,
        game_id=gs.game_id,
        n_alive=sum(1 for p in gs.players.values() if p.alive),
    )
    if decision_identity is not None and exposure_collector is not None:
        exposure_collector.record_rag(decision_identity, final_context.rag_hints)
        exposure_collector.record_reflection(
            decision_identity,
            final_context.reflection_memory_hints,
        )
        exposure_collector.record_skill(decision_identity, final_context.skill_analyses)
        exposure_collector.record_skill_tool_calls(decision_identity, skill_call_records)
        exposure_collector.record_prompt_injections(decision_identity, final_context)
    return final_context
