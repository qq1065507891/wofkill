"""Agent runtime adapter: converts GameState into AgentContext for PlayerAgent.

When an AgentRegistry is provided to the runtime graph, night/day nodes will
delegate decisions to PlayerAgent instances. Without a registry, deterministic
scripted fallback is used (preserving existing test behavior).

Context building and persona/style logic live in runtime/context.py.
Speech directives live in runtime/directives/.
Strategy evaluation lives in runtime/strategy/.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import (
    ActionType,
    FallbackAction,
    PlayerAction,
    TaskType,
)
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.timeline import phase_label

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
    _inject_skill_output,
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
    build_wolf_directive,  # back-compat shim (M3-3)
    build_wolf_night_directive as _build_wolf_night_directive,
    build_wolf_vote_directive as _build_wolf_vote_strategy,
)
from werewolf_agent.runtime.directives._shared import (
    build_sheriff_silent_directive as _build_sheriff_silent_directive,
    collect_death_order as _collect_death_order,
    collect_public_vote_history as _collect_public_vote_history,
)

logger = logging.getLogger(__name__)


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
    evaluate_death_cause_claims as _evaluate_death_cause_claims,
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


def _action_result_to_dict(
    action: PlayerAction | FallbackAction,
) -> dict[str, Any]:
    """Convert a PlayerAction or FallbackAction to runtime state fields."""
    return {
        "action_type": action.action_type.value,
        "target_id": action.target_id,
        "speech": getattr(action, "speech", ""),
        "reason": getattr(action, "reason", ""),
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

    # Build legal actions for witch
    legal_actions = [ActionType.NO_ACTION]
    legal_targets: list[str] = []
    if wolf_kill_target_id and not gs.antidote_used:
        witch_cfg = engine.ruleset.raw["roles"]["witch"]["abilities"]
        if wolf_kill_target_id != witch_id or witch_cfg["antidote"].get("can_self_save", False):
            legal_actions.append(ActionType.USE_ANTIDOTE)
            legal_targets.append(wolf_kill_target_id)
    if not gs.poison_used:
        legal_actions.append(ActionType.USE_POISON)
        legal_targets.extend([
            pid for pid, p in gs.players.items()
            if p.alive and pid != witch_id
        ])

    context = build_agent_context(
        engine, gs, witch_id, TaskType.NIGHT_ACTION,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_kill_target_id=wolf_kill_target_id,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )

    # Build witch strategy directive with clear action guidance
    # Build clear status + options
    potion_status = (
        f"当前药水状态：解药{'已用' if gs.antidote_used else '可用'}，"
        f"毒药{'已用' if gs.poison_used else '可用'}。"
    )
    if gs.antidote_used and not gs.poison_used:
        potion_status += "你只剩毒药，只能选择毒人或不用。"
    elif not gs.antidote_used and gs.poison_used:
        potion_status += "你只剩解药，只能选择救人或不用。"

    witch_directive: dict[str, Any] = {
        "witch_night_action": (
            f"你是女巫，现在是夜间行动阶段。{potion_status}\n你的选择：\n"
        ),
    }
    options = []
    can_self = True
    if wolf_kill_target_id and not gs.antidote_used and ActionType.USE_ANTIDOTE in legal_actions:
        can_self = wolf_kill_target_id != witch_id
        save_hint = f"（他被狼人杀害了）" if can_self else "（但是你不能自救！）"
        options.append(
            f"1) [强烈推荐] 使用解药救{wolf_kill_target_id}{save_hint} —— action_type='use_antidote', target_id='{wolf_kill_target_id}'"
        )
    if not gs.poison_used and ActionType.USE_POISON in legal_actions:
        options.append(
            "2) [推荐] 使用毒药毒杀某人 —— action_type='use_poison', target_id='目标玩家ID'"
        )
    no_action_label = "3) [不推荐] 不使用药水 —— action_type='no_action'"
    if not options:
        no_action_label = "1) 不使用药水（无可用行动）—— action_type='no_action'"
    options.append(no_action_label)
    witch_directive["witch_night_action"] += "\n".join(options)
    witch_directive["witch_night_action"] += "\n\n重要规则：不能在同一夜同时使用解药和毒药。"
    if not can_self:
        witch_directive["witch_night_action"] += "解药不能自救。"
    # Push the LLM away from no_action
    if wolf_kill_target_id and not gs.antidote_used:
        witch_directive["witch_night_action"] += (
            f"\n\n你应该优先使用解药救{wolf_kill_target_id}。"
            f"不救人的女巫等于白板平民——你的解药是最强大的好人技能，不用则浪费。"
        )
    elif not gs.poison_used:
        witch_directive["witch_night_action"] += (
            "\n\n你的毒药还在。你是目前最能直接消灭狼人的人。"
            "选择你最有把握的狼人目标——这会改变游戏走势。"
        )

    # Structured target value assessment — LLM reasons over data, not vague text
    save_value = _estimate_witch_save_value(gs, wolf_kill_target_id)
    witch_directive["save_value_assessment"] = save_value
    if save_value.get("actionable"):
        if save_value.get("public_info_available"):
            # N2+: explicit score + interpretation
            score = save_value.get("save_value_score", 0)
            interp = save_value.get("interpretation", "")
            signals = "、".join(save_value.get("signals", []))
            witch_directive["witch_strategy_hint"] = (
                f"被杀者价值评估：得分{score}分（信号：{signals}）。{interp}"
            )
        else:
            # N1: probability framework + trade-off
            pf = save_value.get("probability_framework", {})
            trade = save_value.get("trade_off", {})
            p_power = pf.get("p_power_role", 0)
            witch_directive["witch_strategy_hint"] = (
                f"首夜无公开信息。被杀者是神职的概率约{p_power:.0%}，是村民的概率约{pf.get('p_villager', 0):.0%}。"
                f"权衡：{trade.get('save_now', '')} | {trade.get('save_later', '')} | {trade.get('risk_no_save', '')}"
            )
    else:
        witch_directive["witch_strategy_hint"] = ""
    if not gs.poison_used:
        witch_directive["witch_strategy_hint"] += " 毒药可用时，也可以考虑不救而保留毒药用于验证可疑目标。"
        # P1-D5: unified `witch_poison_strategy` directive.  Pre-fix the
        # code emitted two separate keys (`witch_poison_threshold` and
        # `poison_urgency`) that could contradict each other; there was
        # also no `no_pressure` branch for early game.  Pick ONE branch
        # per game state and render the matching text.
        alive = sum(1 for p in gs.players.values() if p.alive)
        if alive <= 7:
            branch = "urgency_under_X_alive"
            text = (
                f"【紧急】场上仅存活{alive}人！你的毒药还没有使用！"
                f"好人阵营已经没有犹豫的空间——选择你怀疑度最高的目标用毒。"
                f"不用毒药很可能意味着好人永远失去主动权。"
            )
        elif alive <= 9:
            branch = "evidence_required_threshold"
            text = (
                f"场上存活{alive}人，解药已用，你每夜只有毒药和空过两个选项。"
                f"如果你有怀疑目标（即使证据不够硬），应积极考虑用毒——但需权衡误毒好人的风险。"
            )
        else:
            branch = "no_pressure_save_for_late"
            text = (
                "【毒药决策指引】毒药是好人阵营唯一的主动击杀手段。"
                "以下情况应优先使用毒药："
                "1) 可信预言家的明确查杀；2) 强票型证据（连续保狼、冲票、关键轮分票）；"
                "3) 对跳失败或身份逻辑明显破产；4) 场上存活人数减少，再不用毒药可能来不及。"
                "如果存在合理怀疑但证据不够硬，应权衡'不用毒药导致好人出局'vs'误毒好人'的风险。"
                "解药已用后，你每夜只剩毒药或空过——空过意味着好人失去一轮主动权。"
            )
        witch_directive["witch_poison_strategy"] = {
            "branch": branch,
            "alive_count": alive,
            "text": text,
        }

    # D-8 (2026-06-08 balance audit): 注入毒药候选目标排序列表。
    # 4 局真实游戏中女巫 4 次用毒 3 次毒好人(0 命中率),根因是 LLM 看不到结构化候选。
    # 给出按证据强度排序的 top 候选;无证据时给 no_action 提示,避免凭印象用毒。
    if not gs.poison_used:
        try:
            from werewolf_agent.runtime.strategy.poison import collect_witch_poison_candidates
            cands = collect_witch_poison_candidates(gs, witch_id)
        except Exception:
            cands = []
        alive = sum(1 for p in gs.players.values() if p.alive)
        if cands:
            cand_desc = "; ".join(
                f"{c['player_id']}({c['reason']})" for c in cands[:5]
            )
            witch_directive["witch_poison_candidates"] = (
                f"【毒药候选目标(按证据强度排序)】: {cand_desc}。"
                f"如果你要用毒,请优先从以上候选中选(排在前面的证据更强)。"
                f"如果你认为以上都不够硬,可选择 no_action 并在 reason 中说明理由。"
            )
        else:
            if alive > 9:
                no_action_hint = "【默认 no_action】当前公开信息不足,无明确高证据度狼目标。不要凭印象用毒。"
            elif alive <= 7:
                no_action_hint = (
                    f"【紧急但证据不足】存活 ≤ 7 但无结构化候选。"
                    f"从你的怀疑中选最高度目标(可在 reason 中写'基于 X 的发言'说明依据)。"
                )
            else:
                no_action_hint = (
                    "【证据不足】当前公开信息不足以构成用毒依据。"
                    "如果没有强烈怀疑,默认 no_action。"
                )
            witch_directive["witch_poison_candidates"] = no_action_hint

    witch_directive["witch_night_action"] += "speech字段留空（夜间行动不需要发言）。"

    # Special directive: witch is first-night wolf kill target
    if wolf_kill_target_id == witch_id and not gs.poison_used:
        witch_directive["first_night_killed"] = (
            "你是女巫，N1 / 首夜就被狼人杀害了！你即将死亡，无法自救。"
            "强烈建议使用毒药毒杀一名你怀疑是狼人的玩家。"
            "理由：1) 你已确认死亡，毒药留着没有用；"
            "2) 你的毒药命中可以帮好人阵营获取关键信息；"
            "3) 你可以在遗言中公布身份和毒药目标，为好人提供信息。"
        )

    # Add poison pressure targets if available
    poison_pressure = context.visible_world_state.get("poison_pressure_targets", [])
    if poison_pressure:
        pressure_desc = "; ".join(
            f"{p['player_id']}({p['pressure_type']}: {p['description']})"
            for p in poison_pressure
        )
        witch_directive["witch_pressure"] = f"存在毒药压力目标: {pressure_desc}"
        witch_directive["required_evaluation"] = "如果选择不用毒药，必须在reason中解释为什么不用。"

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

    legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != seer_id]
    counterclaiming_seers = _public_seer_claimants(gs) - {seer_id}
    if counterclaiming_seers:
        legal_targets = [pid for pid in legal_targets if pid not in counterclaiming_seers]

    # Exclude already-checked targets (no value in re-checking known alignments)
    # 预言家自身策略：seer_check 不含 seer_id，所有结果均属该预言家
    checked_ids: set[str] = set()
    for e in gs.events:
        if e.type == "seer_check":
            checked_ids.add(e.payload["target_id"])
    if checked_ids:
        legal_targets = [pid for pid in legal_targets if pid not in checked_ids]

    # Build seer strategy: follow badge flow plan from election speech
    badge_flow_next = None
    for e in gs.events:
        if e.type == "sheriff_speech" and e.payload.get("speaker") == seer_id:
            text = e.payload.get("text", "")
            mentioned = re.findall(r'p\d+', text)
            if mentioned:
                badge_flow_next = [pid for pid in mentioned if pid in legal_targets]
            break

    # Structured check-value assessment for each unchecked target
    check_value = _evaluate_seer_check_value(gs, seer_id, legal_targets)

    night_num = gs.night_number
    night_label = phase_label("night", night_num)
    if night_num == 1:
        seer_guidance = (
            f"{night_label} 验人策略：选择你最怀疑的人，或者按照你上警时承诺的警徽流首夜验人对象。"
            "如果上警时没有明确指定，优先验发言最少、最不透明的人。"
            "不要查验对跳预言家的玩家；对跳位应通过白天发言、票型和放逐解决，夜晚验人用于开新视角。"
        )
    else:
        seer_guidance = (
            f"{night_label} 验人策略：根据白天讨论中你最怀疑的人选择查验目标。"
            "优先验：1) 发言前后矛盾的人；2) 站边不明确的人；3) 被多人怀疑但你不确定的人。"
            "不要查验对跳预言家的玩家；对跳位应通过白天发言、票型和放逐解决，夜晚验人用于开新视角。"
        )

    strategy_directive = {
        "seer_night_check": (
            "你是预言家，现在是夜间验人阶段。你必须选择一名玩家查验其身份。"
            "验人结果（好人/狼人）将在明天白天得知。"
            f"\n\n{seer_guidance}"
            "\n\n注意：本局没有守卫，预言家无法被守护，必须谨慎选择。"
            "\n\n【重要】本局存在混血儿角色，你的验人技能对混血儿显示'好人'，"
            "但混血儿可能在狼人阵营（取决于其主人阵营）。验出'好人'不代表100%安全。"
            "speech字段留空（夜间行动不需要发言）。"
        ),
    }
    if check_value:
        strategy_directive["check_value_assessment"] = check_value
    if badge_flow_next:
        if counterclaiming_seers:
            badge_flow_next = [pid for pid in badge_flow_next if pid not in counterclaiming_seers]
    if badge_flow_next:
        strategy_directive["badge_flow_plan"] = (
            f"你在警上承诺的警徽流计划中提到的验人对象: {badge_flow_next}，"
            "请优先按此计划验人以保持信息传递的一致性。"
        )
    if counterclaiming_seers:
        strategy_directive["excluded_counterclaiming_seers"] = sorted(counterclaiming_seers)

    context = build_agent_context(
        engine, gs, seer_id, TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    seer_target_id = action.target_id if action.action_type == ActionType.CHECK_ALIGNMENT else None
    return {
        "seer_target_id": seer_target_id,
        "seer_action_trace": _action_trace_payload(action),
    }


def agent_wolf_consensus(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
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

    for wolf_id in wolves:
        vote = _single_wolf_vote(state, engine, registry, wolf_id)
        if vote is None:
            no_kill_count += 1
            continue
        if vote.get("action_trace"):
            action_traces[wolf_id] = vote["action_trace"]
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
                "action_traces": action_traces}
    return {"wolf_action": "no_kill", "wolf_kill_target_id": None,
            "wolf_action_reason": f"no_kill_majority({no_kill_count}/{total_kill + no_kill_count})",
            "action_traces": action_traces}


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

    # Collect all wolf discussion speeches already in this night's events
    wolf_ids = [pid for pid, p in gs.players.items() if p.alive and p.role == "werewolf"]
    prev_speeches: list[dict[str, str]] = []
    for e in gs.events:
        if e.type == "wolf_discussion" and e.payload.get("wolf_id") in wolf_ids:
            prev_speeches.append({
                "wolf_id": e.payload.get("wolf_id", ""),
                "round": str(e.payload.get("round", "")),
                "text": e.payload.get("text", ""),
            })

    # Build readable transcript of teammate speeches for prompt injection
    wolf_teammates = [
        pid for pid, p in gs.players.items()
        if p.alive and p.role == "werewolf" and pid != wolf_id
    ]
    teammate_speeches = [s for s in prev_speeches if s["wolf_id"] != wolf_id]
    transcript_lines = []
    for s in teammate_speeches[-12:]:  # Last 12 teammate speeches
        transcript_lines.append(f"[第{s['round']}轮 {s['wolf_id']}]: {s['text']}")
    teammate_transcript = "\n".join(transcript_lines)

    # Build discussion instruction based on round and teammate content
    has_teammate_input = bool(teammate_speeches)
    discussion_instruction = (
        "这是狼队密谈，只有狼人队友能看到。你必须以狼人视角发言，讨论狼队策略。"
        "禁止假装好人视角发言，禁止质疑或试探队友身份——你清楚知道谁是队友。"
        "必须发言，不能沉默。必须提出具体的击杀目标或战术建议。\n"
        "注意用词：被放逐或已死的队友是'队友'或'悍跳狼'，不要叫TA'预言家'。"
        "即使TA白天冒充了预言家，在狼队内部你们应该用真实身份称呼。\n"
        f"【身份约束】你的玩家ID是{wolf_id}。在发言中只能以{wolf_id}自称，"
        "绝对不能自称其他玩家的ID或使用别人的ID发言。"
    )
    if has_teammate_input:
        discussion_instruction += (
            "\n\n重要：你必须回应队友的发言！看看队友提出了什么建议，"
            "表示同意、反对或补充意见，形成真正的团队讨论，而不是自顾自发言。"
        )

    # N1: suggest role division among wolves
    if gs.night_number == 1 and not prev_speeches:
        discussion_instruction += (
            "\n\n【首夜角色分工建议】狼队可以分工配合：\n"
            "1) 悍跳位——白天假装预言家争夺警徽（建议由能言善辩的队友担任）\n"
            "2) 冲锋位——为悍跳队友强力站边，质疑真预言家\n"
            "3) 倒钩位——表面上站边真预言家，暗中破坏好人节奏\n"
            "4) 深水位——保持低调，活到最后为团队收尾\n"
            "讨论谁适合什么角色，但不一定每局都需要悍跳。"
            "如果真预言家查验理由薄弱，悍跳是很好的选择。"
        )

    strategy_directive = {
        "wolf_team_discussion": discussion_instruction,
        "round_focus": requirements.get("required", "讨论狼队策略。"),
        "wolf_teammates": wolf_teammates,
        "previous_discussion": prev_speeches[-8:],
    }
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
    )

    # Inject teammate transcript into recent_transcript for prompt visibility
    extra_transcript = []
    for s in teammate_speeches[-6:]:
        extra_transcript.append({
            "speaker": s["wolf_id"],
            "text": s["text"],
        })
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
        speech_text = (
            f"我是{wolf_id}，本轮讨论我认为应该刀{fallback_target}。"
            f"{requirements.get('required', '')}请大家发表意见。"
        )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_defense_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
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
        discussion_positions=state.get("discussion_positions"),
    )

    strategy_directive = context.strategy_directive or {}
    strategy_directive["defense_context"] = (
        "你正处于被质疑/被指控的状态，正在做防御性发言。\n"
        "防御性发言要点：\n"
        "1) 直接回应针对你的具体指控——不能含糊其辞\n"
        "2) 提供你当时发言/投票的合理解释（'我投TA是因为……'）\n"
        "3) 如果指控是误会，提供事实证据（'我可以查我的发言记录'）\n"
        "4) 不要泛泛地喊'我真是好人'——这没有信息量\n"
        "5) 反问指控者的逻辑漏洞（'你为什么认为我有狼面？'）\n"
        "6) 收尾时给出你希望被如何对待的建议（'请听我解释后再投票'）"
    )
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(strategy_directive, gs, speaker_id)

    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""

    # Fallback for empty defense speech
    if not speech_text.strip():
        speech_text = (
            f"我是{speaker_id}，我理解大家的质疑。让我解释一下："
            f"我当时的判断基于公开信息，可能不全面但绝不是恶意带节奏。"
            f"请大家听完我的解释后再做决定。"
        )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_day_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
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
        discussion_positions=state.get("discussion_positions"),
    )

    strategy_directive = context.strategy_directive or {}
    strategy_directive["anti_following_and_peace_night_rule"] = (
        "不要跟风复述已有指控；如果质疑女巫或预言家，必须给出独立证据并区分事实和推测。"
        "平安夜只代表公开无人死亡，不代表狼人没有刀人。"
        "质疑跳女巫玩家时，应询问是否用药、为什么暂不公开银水、以及发言是否前后矛盾。"
    )

    # Persona-based speech style for day discussion
    style_hint = ""
    ss = _get_persona_speech_style(agent)
    if ss and ss in _SPEECH_STYLE_HINTS:
        style_hint = f"\n- 你的发言风格：{_SPEECH_STYLE_HINTS[ss]}"

    strategy_directive["speech_originality"] = (
        "【发言原创性要求】\n"
        "- 禁止复述其他玩家已经说过的观点——你可以表示同意或反对，但必须补充自己的理由\n"
        "- 禁止使用模板化句式（如'我需要XX正面回应站边、票型'等重复套话）\n"
        "- 你的发言应该展现你独特的思考角度和分析能力\n"
        "- 如果前面已经有人分析了某个玩家，你应换个角度或分析不同的玩家\n"
        "- 【严禁编造任何玩家没有在上方transcript中明确说过的话】"
        "如果某玩家显示'未发表有效言论'或'沉默'，你必须认定该玩家没有做出任何声明、查验或验人报告"
        f"{style_hint}"
    )

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
        if _is_sheriff_silenced(gs, speaker_id):
            # Sheriff is alive with the active badge but is currently
            # muted (e.g., poisoned by witch, self-destructed on a prior
            # turn) and cannot speak.  The pre-fix code still told them
            # to 明确归票, which would have been a hallucination.
            #
            # P1-3 follow-up (Phase 1 self-audit): explicitly state the
            # sheriff STILL must submit a vote action — the silence is
            # on speech only, not on the vote.  Without this the LLM
            # had been observed to skip the round entirely.
            strategy_directive["sheriff_silent"] = (
                "本轮你无法发言，但仍需提交 vote action。"
                "若已提前指定归票目标，在 vote action 的 target_id 字段中给出；"
                "如未指定则由投票开放决定，speech 字段留空。"
            )
        else:
            strategy_directive["sheriff_vote_push"] = (
                "你是警长，你的发言需要归票：总结本轮讨论的关键信息点，"
                "明确表态你怀疑谁、要推谁，号召大家集中投票。"
                "警长归票是核心职责，不能含糊其辞。"
            )
            strategy_directive["sheriff_alive_others"] = alive_others

    # After badge tear → no sheriff for the rest of the game.  Every
    # player (not just the previous sheriff) must know there is no
    # 归票人 and that speech order is now random (design doc §警长规则).
    if gs.sheriff_id is None and gs.sheriff_badge_state == "torn":
        strategy_directive["sheriff_election_state"] = (
            "本局无警长；本轮发言顺序随机；无归票人。"
        )
        # P0-G3223805846-9: inject 归票 hint so players don't fall back
        # on "loudest voice wins".  Distinct key from `sheriff_silent`
        # (which is reserved for the silenced-but-alive sheriff case).
        strategy_directive.update(
            _build_sheriff_silent_directive(
                gs, sheriff_id=None, badge_state="torn",
            )
        )

    # Include sheriff election speeches as salience items for day 1 discussion
    sheriff_speeches = []
    for e in gs.events:
        if e.type == "sheriff_speech" and e.payload.get("text"):
            sheriff_speeches.append({
                "speaker": e.payload.get("speaker", ""),
                "text": e.payload.get("text", ""),
            })
    if sheriff_speeches:
        # Truncate long speeches to prevent copying/repeating
        speech_summaries = []
        for s in sheriff_speeches:
            snippet = s["text"][:120] + ("..." if len(s["text"]) > 120 else "")
            speech_summaries.append(f"  [{s['speaker']}]: {snippet}")
        strategy_directive["sheriff_election_record"] = (
            "以下是警上竞选环节各候选人发言的摘要：\n"
            + "\n".join(speech_summaries)
        )

    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"speech_text": "", "action_trace": {}, "self_destruct": True}

    speech_text = getattr(action, "speech", "") or ""

    # Reject empty day speeches — provide fallback
    if not speech_text.strip():
        alive_others = [pid for pid, p in gs.players.items() if p.alive and pid != speaker_id]
        target_hint = alive_others[0] if alive_others else ""
        speech_text = (
            f"我是{speaker_id}，我认为目前场上信息不够明确。"
            f"我关注{target_hint}的发言，需要更多信息来判断。"
        )

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
            speech_text = (
                f"我是{speaker_id}，目前信息不足，我需要先观察其他玩家的发言再做判断。"
                f"我会重点关注{target_hint}的站边和投票倾向。"
            )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action), "self_destruct": False}


def agent_sheriff_pick_speech_order(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
) -> list[str] | None:
    """Ask the sheriff agent to choose the first speaker. Returns full speech order or None."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_players = [pid for pid, p in gs.players.items() if p.alive and pid != sheriff_id]
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
    )
    strategy_directive = {
        "choose_speech_order": (
            "你是警长，需要选择发言顺序。请选择第一个发言的玩家（你将最后一个发言进行归票）。"
            "在speech字段中说明你的选择理由。"
        ),
        "alive_players": alive_players,
    }
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    first_speaker = action.target_id if action.action_type == ActionType.VOTE else None

    if first_speaker and first_speaker in alive_players:
        # Build order: first_speaker, then remaining in original order, sheriff last
        remaining = [pid for pid in alive_players if pid != first_speaker]
        return [first_speaker] + remaining + [sheriff_id]
    return None


def agent_sheriff_endorse(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
) -> dict[str, Any] | None:
    """Sheriff privately decides endorsement target via VOTE action.

    Returns dict with endorse_target / private_reason / action_trace
    (or None for scripted fallback when no agent is registered).
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_others = [
        pid for pid, p in gs.players.items()
        if p.alive and pid != sheriff_id
    ]

    strategy_directive = {
        "sheriff_endorse": (
            "你是警长。现在所有玩家已经发言完毕，即将开始放逐投票。"
            "作为警长，你需要归票——选择你认为应该被投票放逐的玩家。"
            "这是你的私人决策，你的内心理由不会让其他玩家看到。"
            "但你的归票目标会被法官公开宣布。"
        ),
        "legal_endorse_targets": alive_others,
    }

    context = build_agent_context(
        engine, gs, sheriff_id, TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=alive_others,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    target = action.target_id if action.action_type == ActionType.VOTE else None

    # Validate target is alive and not self
    if target and target in alive_others:
        return {
            "endorse_target": target,
            "private_reason": getattr(action, "reason", "") or "",
            "action_trace": _action_trace_payload(action),
        }
    return {
        "endorse_target": "",
        "private_reason": "",
        "action_trace": None,
    }


def agent_pk_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
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
    )
    # Add prior tally to visible state
    if prior_tally:
        updated_visible = {**context.visible_world_state, "prior_vote_tally": prior_tally}
        context = context.model_copy(update={"visible_world_state": updated_visible})

    # D-2: PK speech needs role-specific directives.  A tied candidate
    # must convince the room in 60s; their argument should be anchored
    # in the strongest role-specific evidence (seer checks, witch
    # private info, hunter last-words, wolf team plan, etc.).
    pk_strategy: dict[str, Any] = {
        "pk_urgent": (
            "你正处于PK发言阶段——平票候选人只有一次发言机会，必须在这一轮内说服足够多的人改投你。"
            "不要再'等下一轮'，不要再'观察'，直接亮出你最强的证据或分析。"
        ),
    }
    player_role = gs.players[speaker_id].role if speaker_id in gs.players else ""
    if player_role == "werewolf":
        # Wolves: keep cover, attack the rival, push for team target
        pk_strategy["wolf_pk_push"] = (
            "你是狼人，PK发言策略：\n"
            "1) 攻击对手的发言漏洞——TA的逻辑不完整、TA的站边前后矛盾\n"
            "2) 表现得像一个有分析能力的好人，不要替狼队说话\n"
            "3) 如果场上有队友的推人目标，借机把票型引导到目标玩家"
        )
    elif player_role == "seer":
        # Seer: anchor in check results
        try:
            check_results = []
            for e in gs.events:
                if e.type == "seer_check":
                    check_results.append({
                        "target": e.payload["target_id"],
                        "alignment": e.payload["alignment"],
                        "night": e.payload["night_number"],
                    })
            if check_results:
                pk_strategy["seer_pk_check_evidence"] = (
                    f"你是预言家，PK发言必须以你的查验结果为核心："
                    f"你已获得 {len(check_results)} 个查验结果。"
                    "在PK中直接报出最关键的一个查杀或金水，"
                    "告诉所有人'信我，我查了[玩家]是[好人/狼人]'，"
                    "让对跳预言家或你的对手无法在60秒内反驳。"
                )
        except Exception:
            logger.debug("Failed to build seer PK check evidence", exc_info=True)
    elif player_role == "witch":
        pk_strategy["witch_pk_evidence"] = (
            "你是女巫，PK发言策略：\n"
            "1) 不要轻易透露药水状态，但你需要给出可信的分析来赢得PK\n"
            "2) 引用场上具体的发言矛盾、票型异常来支撑你的判断\n"
            "3) 如果你救了某人（银水），可以暗示'我手里有信息'但不要明说"
        )
    elif player_role == "hunter":
        pk_strategy["hunter_pk_pressure"] = (
            "你是猎人，PK发言策略：\n"
            "1) 利用'我有枪'的威慑——明确说'我被放逐会开枪带走最可疑的人'\n"
            "2) 这会给狼队压力，让他们考虑放逐你的风险\n"
            "3) 但不要虚张声势说已经决定带谁"
        )
    elif player_role == "villager":
        pk_strategy["villager_pk_logic"] = (
            "你是普通村民，PK发言必须基于公开信息逻辑分析：\n"
            "1) 引用场上具体的发言矛盾、票型异常、警徽流\n"
            "2) 不要喊'我是好人'——这没有信息量\n"
            "3) 直接分析对手为什么更像狼，列出2-3个具体证据"
        )
    elif player_role == "idiot":
        pk_strategy["idiot_pk_caution"] = (
            "你是白痴（未翻牌或已翻牌），PK发言：\n"
            "1) 翻牌前的白痴不要暴露身份，专注逻辑分析\n"
            "2) 翻牌后的白痴可以大胆表达观点——你已免疫放逐"
        )
    elif player_role == "hybrid":
        # Hybrid PK: align with master if known
        master_id = gs.hybrid_master_id
        if master_id:
            pk_strategy["hybrid_pk_master_align"] = (
                f"你是混血儿，主人是{master_id}。"
                "PK发言要表现得像主人的判断方向——"
                "如果主人在场，分析与主人站边一致；"
                "但不要每轮都跟主人保持完全一致，那会暴露关系。"
            )
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

    # Pass consecutive no-exile info as strategy directive
    consecutive_no_exile = state.get("consecutive_no_exile_days", 0)
    strategy_directive: dict[str, Any] = {
        "require_vote_quality": True,
        "vote_structured_contract": {
            "seer_stance": ["trust", "distrust", "undecided", "no_claim"],
            "vote_basis": [
                "seer_check",
                "seer_siding",
                "speech_logic",
                "vote_pattern",
                "pressure_test",
                "anti_herd",
                "fallback",
            ],
        },
        "vote_silent": (
            "投票阶段不允许公开发言。speech字段必须留空。"
            "你只能内心选择要投谁，不能在投票时发表任何公开言论。"
            "请在reason字段中写下简短公开理由；同时在JSON中额外写"
            "seer_stance、vote_basis、standing_with_seer、suspect_reason、"
            "not_voting_reason、private_reason。"
            "这些字段是你的投票心理活动，只给主持人审计，不会公开给其他玩家。"
        ),
        "vote_strategy": (
            "投票原则：\n"
            "1) 有查杀走查杀：如果被信任的预言家查杀了某人，优先投查杀对象。"
            "2) 跟预言家走：听完发言后，根据你信任的预言家的归票方向投票。"
            "3) 警上单边预言家可信度高，警下跳预言家的可信度很低。"
            "4) 如果没有明确查杀，投发言最可疑、逻辑最不通的人。"
        ),
    }
    if voter_role in ("villager", "seer", "witch", "hunter", "idiot"):
        strategy_directive["good_vote_decision_guard"] = (
            "【好人投票决策纪律】先按硬信息优先级排序，再做选择："
            "1) 预言家查验/可信查杀；2) 已知死亡身份和死亡原因；"
            "3) 投票票型；4) 警徽流/警长归票；5) 发言怀疑。"
            "【出错成本】投票前必须比较：如果目标其实是预言家、女巫、猎人、白痴或关键金水，"
            "今天出错会不会直接导致屠神/屠民；若只有发言风格可疑而没有查验或强票型，"
            "不要轻易推出疑似神职或高价值好人。"
        )
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(strategy_directive, gs, voter_id)
    if not allow_abstain:
        parts = ["必须投票选出一名玩家放逐，不能弃票。"]
        if consecutive_no_exile > 0:
            parts.append(f"已经连续{consecutive_no_exile}天无人出局，必须做出决定。")
        strategy_directive["vote_pressure"] = " ".join(parts)
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
        strategy_directive["anti_herd"] = (
            "Do not mechanically follow a near-unanimous push unless it has "
            "concrete evidence such as checks, counterclaims, vote records, "
            "contradictions, or quoted speeches."
        )
    except Exception:
        logger.debug("Vote quality context build failed, skipping", exc_info=True)

    # Role-specific vote strategy (voter_role computed above for legal_targets filtering)
    if voter_role == "werewolf":
        wolf_vote_parts = _build_wolf_vote_strategy(
            gs, voter_id, state.get("wolf_team_plan"),
        )
        strategy_directive.update(wolf_vote_parts)
    elif voter_role == "hybrid" and gs.hybrid_master_id:
        strategy_directive["hybrid_vote_strategy"] = (
            f"你是混血儿，主人是 {gs.hybrid_master_id}。投票策略：\n"
            "1) 观察主人投谁——如果主人投了某人，考虑也投该方向（对主人阵营有利）\n"
            "2) 不要每轮都跟主人投一样的人，那会暴露你们的关系\n"
            "3) 如果主人投的人你确实认为可疑，正常投票即可\n"
            "4) 如果主人被投了——你需要判断主人被放逐是否对主人阵营不利，"
            "考虑是否投别人来稀释票数\n"
            "5) 绝对不要在投票理由中暴露你的混血儿身份"
        )
    elif voter_role == "seer":
        # D-3: seer voters must anchor their vote in check results and
        # surface any unreported check-kills in the vote reasoning.
        try:
            check_results = []
            for e in gs.events:
                if e.type == "seer_check":
                    check_results.append({
                        "target": e.payload["target_id"],
                        "alignment": e.payload["alignment"],
                        "night": e.payload["night_number"],
                    })
            wolf_checks = [c for c in check_results if c["alignment"] == "wolf"]
            good_checks = [c for c in check_results if c["alignment"] == "good"]
            wolf_list = "、".join(c["target"] for c in wolf_checks) or "（无）"
            good_list = "、".join(c["target"] for c in good_checks) or "（无）"
            strategy_directive["seer_vote_strategy"] = (
                "你是预言家，投票策略：\n"
                f"1) 你已查验出狼人: {wolf_list}——必须把票投给这些查杀对象中的某一个\n"
                f"2) 你已查验出好人: {good_list}——不要投这些人\n"
                "3) 如果场上多个人被查杀，优先投你最近查杀、警徽流计划中的下一个\n"
                "4) 如果场上没有查杀对象，引用票型/警徽流/发言矛盾\n"
                "5) 投票时公开重申你的查杀——这是预言家的核心职责"
            )
        except Exception:
            logger.debug("Failed to build seer vote strategy", exc_info=True)
            strategy_directive["seer_vote_strategy"] = (
                "你是预言家，投票时以你的查验结果为核心依据。"
            )
    elif voter_role == "witch":
        # D-3: witch voters have private potion info that informs
        # the vote (saved person, poisoned person).
        strategy_directive["witch_vote_strategy"] = (
            "你是女巫，投票策略：\n"
            "1) 你的解药目标（银水）是好人的强信号——给银水站台、帮其站边\n"
            "2) 你的毒药目标如果是狼人，那一票已经定局；如果是好人，提醒自己不要把票投给无辜者\n"
            "3) 不要在公开投票理由中提及药水使用细节（'我救了TA'/'我毒了TA'）\n"
            "4) 但你可以引用场上其他公开信息（发言矛盾、票型）来支撑你的投票"
        )
    elif voter_role == "hunter":
        # D-3: hunter voters must consider shot-after-exile implications
        # and avoid wasting the gun on a low-value target.
        strategy_directive["hunter_vote_strategy"] = (
            "你是猎人，投票策略：\n"
            "1) 投完票后可能被放逐——一旦你被放逐，你会开枪\n"
            "2) 投票时考虑：如果我被放逐，我最想带谁？把票投给最像狼的人\n"
            "3) 不要投给明显是好人的人——浪费你的枪\n"
            "4) 如果你不想暴露自己，宁可弃票或跟大多数人票"
        )
    elif voter_role in ("villager", "idiot"):
        seer_claimants = _public_seer_claimants(gs)
        # D-16: explicit single-seer branch when there is exactly one
        # public claimant (no counterclaim) — villagers should
        # default to trusting them unless evidence strongly disagrees.
        if len(seer_claimants) == 1:
            claimant = sorted(seer_claimants)[0]
            strategy_directive["villager_vote_strategy"] = (
                f"你是普通好人（无私有信息），投票策略必须基于公开信息独立判断：\n"
                f"1) 场上只有{claimant}单边跳预言家（无对跳预言家）——"
                "单边预言家的可信度较高，可以优先跟其查杀走\n"
                "2) 但即使是单边预言家，也要看TA的发言是否有验人逻辑链、是否遵守警徽流\n"
                "3) 不要无条件跟任何人的票——先用你自己的分析判断谁更像狼\n"
                "4) 关注票型数据：谁在保谁、谁在投谁——狼人倾向于抱团投票\n"
                "5) 如果场上没有查杀，投发言逻辑最混乱、站边最模糊的人\n"
                "6) 不要投自己——这没有任何价值"
            )
        else:
            seer_claimants = _public_seer_claimants(gs)
            strategy_directive["villager_vote_strategy"] = (
                "你是普通好人（无私有信息），投票策略必须基于公开信息独立判断：\n"
                "1) 先判断预言家真假：\n"
                "   - 单边预言家（无对跳）：可信度高，可以跟其查杀走\n"
                f"   - 对跳预言家 {sorted(seer_claimants) if seer_claimants else '（暂无）'}："
                "谁的验人逻辑链更完整？谁的发言有实质信息？谁在遵守警徽流？\n"
                "2) 不要无条件跟任何人的票——先用你自己的分析判断谁更像狼\n"
                "3) 关注票型数据：谁在保谁、谁在投谁——狼人倾向于抱团投票\n"
                "4) 如果场上没有查杀，投发言逻辑最混乱、站边最模糊的人\n"
                "5) 不要投自己——这没有任何价值"
            )

    # Pre-compute evidence-based fallback target for structured failure
    non_self_legal = [t for t in legal_targets if t != voter_id]
    if non_self_legal:
        try:
            from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target
            fb = choose_vote_fallback_target(gs, voter_id, non_self_legal)
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
    )
    if strategy_directive:
        context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    target = action.target_id if action.action_type == ActionType.VOTE else None
    # Fallback: if agent returned wrong action type but has legal targets,
    # pick an evidence-aware target rather than abstaining silently.
    if target is None and legal_targets:
        target = choose_vote_fallback_target(gs, voter_id, legal_targets)
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
) -> dict[str, Any] | None:
    """Ask hybrid agent to choose their master. Returns None if agent unavailable."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(hybrid_id)
    if agent is None:
        return None

    candidates = [pid for pid, p in gs.players.items() if p.alive and pid != hybrid_id]

    master_assessment = _evaluate_hybrid_master_candidates(gs, hybrid_id, candidates)

    strategy_directive = {
        "hybrid_master_choice": (
            "你是混血儿，N1 / 首夜需要选择一名玩家作为你的主人。"
            "你不知道主人的身份和阵营，但你将跟随主人的原始阵营获胜。"
            "如果主人是好人阵营，你跟好人赢；如果主人是狼人阵营，你跟狼人赢。"
            "选择后不能更改。speech字段留空。"
        ),
        "master_assessment": master_assessment,
    }

    context = build_agent_context(
        engine, gs, hybrid_id, TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHOOSE_MASTER],
        legal_targets=candidates,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    master_target_id = action.target_id if action.action_type == ActionType.CHOOSE_MASTER else None
    if master_target_id is None and candidates:
        master_target_id = candidates[0]

    return {"master_target_id": master_target_id}


def agent_exile_last_words(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    player_id: str,
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
    )
    strategy_directive = {
        "last_words": (
            f"你已被放逐出局，这是你的遗言。你已确认死亡，身份已公开（{player_role}）。"
            "遗言中你可以：\n"
            "- 预言家：交代你所有的验人结果和警徽流\n"
            "- 猎人：声明你可以开枪（如果被放逐而非毒杀）\n"
            "- 其他好人：表达你对场上局势的最终看法，给存活玩家建议\n"
            "- 狼人：做最后的表演，误导好人\n"
            "遗言必须简短有力。"
        ),
    }
    if player_role == "hunter":
        alive_others = [pid for pid, p in gs.players.items() if p.alive and pid != player_id]
        strategy_directive["hunter_last_words"] = (
            "你是猎人，被放逐出局。你有权开枪带走一名玩家。\n"
            "遗言中建议：\n"
            "1) 声明猎人身份和开枪意图\n"
            "2) 明确说出你要带走的目标：用'带走{玩家ID}'的格式（如'带走p03'）\n"
            "3) 解释你选择该目标的理由（发言矛盾、站边不明、被查杀等）\n"
            "4) 如果没有明确目标，可以声明'我选择不开枪'\n"
            f"当前存活玩家（不含你）: {alive_others}"
        )
    elif player_role == "hybrid":
        strategy_directive["hybrid_last_words"] = (
            "【严禁泄漏混血儿身份】你是混血儿，但你的身份不应该在遗言中暴露。\n"
            "遗言中你必须：\n"
            "1) 以普通好人视角发言，绝不提及'主人'、'混血儿'、'阵营选择'等概念\n"
            "2) 不要透露你的主人是谁，也不要暗示你与某位玩家有特殊关系\n"
            "3) 以普通村民身份表达对场上局势的看法和建议\n"
        )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_badge_decision(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
) -> dict[str, Any] | None:
    """Dying sheriff decides to transfer badge or tear it."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_others = [pid for pid, p in gs.players.items() if p.alive and pid != sheriff_id]
    context = build_agent_context(
        engine, gs, sheriff_id, TaskType.LAST_WORDS,
        legal_actions=[ActionType.BADGE_TRANSFER, ActionType.BADGE_TEAR],
        legal_targets=alive_others,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )
    player_role = gs.players[sheriff_id].role if sheriff_id in gs.players else ""
    role_hint = ""
    if player_role == "werewolf":
        role_hint = (
            "你是狼人警长，移交必须为狼队利益服务：\n"
            "   - 移交给狼队友：让狼队继续控制警徽和归票权。\n"
            "   - 移交给被狼队深度迷惑的好人：利用他替狼队带节奏。\n"
            "   - 撕毁：如果移交任何人都对狼队不利，撕掉不让好人拿到归票权。"
        )
    elif player_role == "seer":
        role_hint = (
            "你是预言家警长：移交给你验过的金水（被你验出好人的玩家）。"
            "让金水拿到警徽，继续传递你的验人信息。"
        )
    else:
        role_hint = (
            "你是好人警长：移交给你最信任的明好人，"
            "确保警徽不落入狼人手中。如果场上没有明确的明好人，可以撕毁。"
        )

    strategy_directive = {
        "badge_decision": (
            "你是即将离场的警长，必须决定警徽去向：\n"
            "1) 移交（BADGE_TRANSFER）：选择一名存活玩家作为新警长。\n"
            f"   {role_hint}\n"
            "2) 撕毁（BADGE_TEAR）：撕毁警徽，本局不再有警长。\n"
            "请根据你的身份和阵营做出最有利的决定。"
        ),
        "alive_players": alive_others,
    }
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    if action.action_type == ActionType.BADGE_TRANSFER and action.target_id:
        return {"badge_decision": "transfer", "badge_target_id": action.target_id}
    return {"badge_decision": "tear", "badge_target_id": None}


# Re-export from runtime.strategy (Task 2 extraction)
from werewolf_agent.runtime.strategy import evaluate_hunter_shot_target as _evaluate_hunter_shot_target


def agent_hunter_shot(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    hunter_id: str,
) -> str | None:
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

    death_label = {"wolf_kill": "被狼人袭击", "exile": "被投票放逐"}.get(
        death_reason, f"因{death_reason}"
    )
    # Determine if there are plausible wolf suspects
    has_suspects = (
        shot_assessment
        and shot_assessment.get("ranked_targets")
        and len(shot_assessment["ranked_targets"]) > 0
    )
    top_value = 0
    if has_suspects:
        top_value = int(shot_assessment["ranked_targets"][0].get("value", 0))
    if top_value >= 6:
        shoot_encouragement = "存在明确查杀、对跳或强公共证据目标，可以优先开枪带走。"
    elif top_value >= 3:
        shoot_encouragement = "存在一定公共证据目标；开枪前仍要比较出错成本，避免误伤好人。"
    else:
        shoot_encouragement = (
            "当前没有明确查杀、强票型或强对跳失败目标，优先选择不开枪（NO_ACTION），"
            "避免误伤好人；只有你能指出具体硬证据时才开枪。"
        )

    strategy_directive: dict[str, Any] = {
        "hunter_shot_directive": (
            f"你是猎人，{death_label}导致死亡。你现在可以开枪带走一名玩家。\n"
            "开枪是一次性的，但你的判断是场上最好的武器之一。\n"
            f"{shoot_encouragement}\n"
            "注意：本局没有守卫，如果你被女巫毒杀（而非被狼杀或放逐），你无法开枪。\n"
            "speech字段留空。"
        ),
    }
    if shot_assessment:
        strategy_directive["shot_value_assessment"] = shot_assessment

    context = build_agent_context(
        engine, gs, hunter_id, TaskType.HUNTER_SHOT,
        legal_actions=[ActionType.HUNTER_SHOT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    if action.action_type == ActionType.HUNTER_SHOT and action.target_id:
        return action.target_id
    return None


def agent_sheriff_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    voter_id: str,
    candidates: list[str],
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
    )

    # Wolf strategy for sheriff voting
    strategy_directive = context.strategy_directive or {}
    voter_role = gs.players[voter_id].role if voter_id in gs.players else ""
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(strategy_directive, gs, voter_id)
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
            "你是预言家，强烈建议上警！预言家几乎必须上警留警徽流，"
            "这是预言家的核心玩法——通过警徽流传递验人信息。"
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
    )
    context = _merge_strategy_directive(context, strategy_directive)

    try:
        action, retry_info = agent.act(context)
        if action.action_type == ActionType.SELF_DESTRUCT:
            return {"registered": False, "self_destruct": True}
        return {"registered": action.action_type == ActionType.SHERIFF_REGISTER, "self_destruct": False}
    except Exception:
        logger.warning("Sheriff registration failed for %s", player_id, exc_info=True)
        return {"registered": False, "self_destruct": False}


def agent_sheriff_withdraw(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
) -> dict[str, Any] | None:
    """Ask a sheriff candidate whether they want to withdraw.

    Returns dict with withdrawal result and self_destruct flag.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(candidate_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine, gs, candidate_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SHERIFF_WITHDRAW, ActionType.NO_ACTION],
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )

    try:
        action, retry_info = agent.act(context)
        if action.action_type == ActionType.SELF_DESTRUCT:
            return {"withdrew": False, "self_destruct": True}
        return {"withdrew": action.action_type == ActionType.SHERIFF_WITHDRAW, "self_destruct": False}
    except Exception:
        logger.warning("Sheriff withdrawal failed for %s", candidate_id, exc_info=True)
        return {"withdrew": False, "self_destruct": False}


def agent_sheriff_election_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
    all_candidates: list[str],
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

    # Badge flow is seer-exclusive. Only seer (or wolf claiming seer) should mention it.
    player_role = gs.players[candidate_id].role if candidate_id in gs.players else ""
    is_seer_or_claiming = player_role == "seer" or player_role == "werewolf"

    badge_flow_instruction = ""
    if is_seer_or_claiming:
        badge_flow_instruction = (
            "2) 你的警徽流：必须留两个晚上的验人对象！"
            "格式如'先验X，后验Y'。如果你验到好人，死后警徽给该好人；"
            "如果验到狼人，则不给警徽（撕徽或给之前验过的好人）。"
            "警徽流是预言家传递信息的核心机制，必须明确留出两夜验人计划。"
        )

    # Single-sided vs multi-seer context
    seer_count = sum(1 for c in all_candidates
                     if gs.players.get(c) and gs.players[c].role in ("seer", "werewolf"))
    seer_context = ""
    if seer_count >= 2:
        seer_context = (
            "场上有多人跳预言家，这是典型的悍跳局面。"
            "真预言家必须坚定立场，用逻辑和验人信息证明自己；"
            "悍跳预言家需要制造合理怀疑，攻击对方的逻辑漏洞。"
        )
    else:
        if player_role in ("seer", "werewolf"):
            seer_context = (
                "目前警上只有你跳预言家（单边预言家），你的可信度很高。"
                "要充分利用这一点，留下完整的警徽流，让好人信任你。"
            )

    # Check if previous candidates have already spoken (for response/analysis)
    prev_speeches = []
    for e in gs.events:
        if e.type == "sheriff_speech" and e.payload.get("text") and e.payload.get("speaker") != candidate_id:
            prev_speeches.append({
                "speaker": e.payload["speaker"],
                "text": e.payload["text"],
            })
    prev_speech_instruction = ""
    if prev_speeches:
        prev_summaries = []
        covered_topics = []
        for s in prev_speeches[-3:]:
            snippet = s["text"][:150] + ("..." if len(s["text"]) > 150 else "")
            prev_summaries.append(f"  [{s['speaker']}]: {snippet}")
            mentioned = re.findall(r'p\d{2}', s["text"])
            for m in mentioned[:3]:
                covered_topics.append(f"{s['speaker']}已分析过{m}")
        prev_texts = "\n".join(prev_summaries)
        covered_note = ""
        if covered_topics:
            covered_note = f"\n【已被覆盖的观点】{', '.join(covered_topics[:6])}。你必须分析不同的玩家或使用完全不同的推理路径。"
        prev_speech_instruction = (
            f"\n\n【前人发言摘要】在你之前已经有候选人发言了：\n"
            f"{prev_texts}"
            f"{covered_note}\n"
            "你可以反驳，也可以完全忽略前人走自己的分析路线。"
            "【严禁照搬/复述前人发言原文或结构】。"
        )
    else:
        prev_speech_instruction = (
            "\n\n你是本轮第一个发言的候选人。"
            "你只能基于目前场上的公开信息发言。"
            "【严禁编造/虚构尚未发言的候选人说过的话】——"
            "你不知道其他候选人会说什么，只能表达自己的立场和分析。"
        )

    # Persona-based speech style differentiation
    speech_style = _get_persona_speech_style(agent)
    task_style = _get_persona_task_style(agent, "sheriff_speech")

    merged_hints = {**_SPEECH_STYLE_HINTS, **_SHERIFF_SPEECH_STYLE_OVERRIDES}
    style_hint = merged_hints.get(speech_style, "从你自己的独特角度分析场上局势。")
    task_hint = _TASK_STYLE_HINTS.get(task_style, "")

    strategy_directive = {
        "sheriff_election_speech": (
            "你正在竞选警长，必须解释上警原因和你的初步判断。"
            "不能只说'我来上警'之类空洞的话，必须有实质内容。"
            "注意：只有预言家（或悍跳预言家）才能留警徽流，其他身份不要提警徽流。"
            "非预言家不要在警上冒充预言家抢警徽。\n"
            f"【你的发言风格】{style_hint}\n"
            f"{task_hint}\n"
            f"{badge_flow_instruction}"
            f"{seer_context}"
            f"{prev_speech_instruction}"
        ),
        "other_candidates": other_candidates,
        "anti_template": (
            "【禁止模板化】你的发言不能机械套用模板。以下句式会让你的发言"
            "被判定为无效：'我这轮先把视角压到XX身上'、'依据是XX最近发言：...'。"
            "你必须有自己独立的角度和分析逻辑。"
        ),
    }
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(strategy_directive, gs, candidate_id)

    # Role-specific speech differentiation
    if player_role == "idiot":
        strategy_directive["role_speech_hint"] = (
            "你是白痴，警上发言重点：观察所有人的发言逻辑，"
            "找出逻辑漏洞或矛盾点，展示你的分析能力。"
        )
    elif player_role == "hunter":
        strategy_directive["role_speech_hint"] = (
            "你是猎人，警上发言重点：关注谁在发言中暴露了信息不对称，"
            "谁的逻辑前后矛盾。你不需要暴露身份。"
        )
    elif player_role == "witch":
        strategy_directive["role_speech_hint"] = (
            "你是女巫，警上发言重点：基于你的夜间信息，"
            "引导讨论方向，但不要暴露你知道的具体信息。"
        )
    elif player_role == "villager":
        strategy_directive["role_speech_hint"] = (
            "你是村民，警上发言重点：用逻辑分析场上信息，"
            "找出预言家真假的判断依据，展示你作为好人的价值。"
        )
    elif player_role == "hybrid":
        strategy_directive["role_speech_hint"] = (
            "你是混血儿，警上发言重点：观察场上局势，"
            "在不确定主人阵营前保持中立分析。"
        )

    if player_role == "seer":
        strategy_directive["seer_verification_rationale"] = (
            "【查验理由要求】你每夜的查验目标必须有具体动机。"
            "禁止说'按顺序验'或'随便验的'。正确的说法示例："
            "'N1验p03是因为他发言内容展现了较强的逻辑分析能力，"
            "我需要确认他是好人核心还是狼人伪装'。"
            "查验理由应基于发言内容、投票行为等可观察信息，"
            "不要使用'警上/警下位置'等你在发言时可能记错的信息。"
            "如果没有特殊理由，可以说'首夜随机查验，选择了一个发言量较大的位置'。"
        )

    # Wolf: inject role-specific strategy from wolf_team_plan
    if player_role == "werewolf":
        wolf_plan = state.get("wolf_team_plan")
        wolf_day_directive = _build_wolf_day_speech_directive(gs, candidate_id, wolf_plan)
        # Merge wolf day directive into strategy_directive
        strategy_directive.update(wolf_day_directive)

        wolf_assignment = _get_wolf_role_assignment(wolf_plan, candidate_id)
        if wolf_assignment == "fake_seer":
            strategy_directive["wolf_sheriff_must_claim_seer"] = (
                "【强制执行】你是团队安排的悍跳预言家！你现在在警上竞选。"
                "你必须在这段发言中跳预言家，报出你的假验人结果和警徽流。"
                "格式参考：'我是预言家，昨晚我验了[玩家]，结果是[好人/查杀]，"
                "我的警徽流是先验[X]后验[Y]。' "
                "不要犹豫、不要含糊——你必须像真预言家一样坚定。"
                "你的假验人结果可以是：金水（假的好人结果）来拉拢人，"
                "或查杀（假的狼人结果）来推好人。选择你认为最优的策略。"
            )
        elif wolf_plan and wolf_plan.get("fake_seer"):
            fake_seer_id = wolf_plan["fake_seer"]
            if fake_seer_id != candidate_id and not _has_publicly_claimed_seer(gs, fake_seer_id):
                strategy_directive["wolf_no_reveal_seer"] = (
                    f"【严禁信息穿越】你的队友计划跳预言家但尚未在警上发言。"
                    "在你的警上发言中绝不能站边TA或透露TA会跳预言家。"
                    "你必须表现得像一个不知道谁是预言家的普通好人。"
                    "等TA自己发言后，在后续讨论中你才能像好人一样站边。"
                )

    context = build_agent_context(
        engine, gs, candidate_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"speech_text": "", "action_trace": {}, "self_destruct": True}

    speech_text = getattr(action, "speech", "") or ""

    # Reject empty sheriff election speeches
    if not speech_text.strip() or len(speech_text.strip()) < 10:
        if is_seer_or_claiming:
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
        context = build_agent_context(
            engine, gs, player_id, TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
        )
        reflection_task = _build_reflection_prompt(
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
        context = _merge_strategy_directive(context, reflection_directive)

        action, _retry_info = agent.act(context)
        return {"reflection_text": getattr(action, "speech", "") or ""}
    except Exception:
        logger.warning("Reflection failed for %s", player_id, exc_info=True)
        return {"reflection_text": ""}


_GOOD_ROLES = {"villager", "seer", "witch", "hunter", "idiot"}
_WOLF_ROLES = {"werewolf"}


def _build_reflection_prompt(
    player: Any,
    winner: str,
    hybrid_master_faction: str | None,
) -> str:
    """Build a role-family-specific reflection prompt.

    三族:
    - 好人 (villager / seer / witch / hunter / idiot / hybrid-master-good):
      反思投票/站边/信息缺失/神职执行
    - 狼人 (werewolf / hybrid-master-wolf):
      反思悍跳/暴露/角色分工
    - 混血儿 (master 未知 / 其他): 通用模板

    共同:末尾强制要求列出"本局保留的 1-2 个优点",避免只记错误。
    """
    role = (player.role if player else "") or ""
    faction_result = "胜" if (
        (role in _GOOD_ROLES and winner == "good")
        or (role == "werewolf" and winner == "werewolf")
        or (role == "hybrid" and (
            (hybrid_master_faction == "good" and winner == "good")
            or (hybrid_master_faction == "werewolf" and winner == "werewolf")
        ))
    ) else "负"

    if role in _GOOD_ROLES:
        return _GOOD_REFLECTION_TEMPLATE.format(faction_result=faction_result, role=role)
    if role == "werewolf":
        return _WOLF_REFLECTION_TEMPLATE.format(faction_result=faction_result)
    if role == "hybrid":
        if hybrid_master_faction == "good":
            return _GOOD_REFLECTION_TEMPLATE.format(faction_result=faction_result, role="hybrid(跟好人)")
        if hybrid_master_faction == "werewolf":
            return _WOLF_REFLECTION_TEMPLATE.format(faction_result=faction_result)
    return _GENERIC_REFLECTION_TEMPLATE.format(faction_result=faction_result, role=role)


_GOOD_REFLECTION_TEMPLATE = """你是{role},本局好人阵营{faction_result}。请按以下结构复盘:

【投票错误】本局你投过谁?有没有推错人?为什么站错边?
- 具体指出哪一天的投票决策有误,错投了谁,该投谁
- 分析站错边的根因(信息不足/被悍跳狼误导/被情绪带动)

【信息缺失】哪些关键信号被你忽略了?
- 预言家的查杀声明 / 悍跳狼的逻辑漏洞 / 票型异常(分票/跟票)
- 女巫的解药用错 / 毒药空过 / 白痴翻牌时机

【神职执行】(仅神职需要回答)
- 预言家:警徽流是否清晰?是否被首推?
- 女巫:解药救了谁?是否值得?毒药目标对了吗?
- 猎人:被放逐/夜杀时是否开枪?目标对了吗?
- 白痴:翻牌时机是否合适?

【保留的优点】本局你做对了什么?必须列出 1-2 个具体策略,下局复用:
- 例如:"N2 我用解药救了警长,后续警长归票带我们翻盘"
- 例如:"我在 D3 提前质疑悍跳狼的警徽流时间线,被采信了"

【PII 守卫 rag-hardening-4】反思文本中**不要写其他玩家的真实身份**(即使你已经推断出)。规则:
- 不要写"p03 是预言家因为查杀"或"p05 是狼因为他悍跳"这类具体身份断言
- 改用模糊指代:"某玩家", "被查杀的目标", "悍跳的狼"
- 原因:反思会跨局注入其他玩家的 LLM prompt,如果对方玩家 ID 在下局匹配到,会造成跨局信息泄漏
- 例外:可以写自己的身份/自己推断的逻辑,但**必须用模糊指代**

"""


_WOLF_REFLECTION_TEMPLATE = """你是狼人,本局狼队{faction_result}。请按以下结构复盘:

【悍跳分析】(如果有狼跳预言家)
- 悍跳发言为什么没人信?逻辑漏洞在哪?
- 验人口径是否前后矛盾?警徽流是否清晰?
- 真预言家对跳后,悍跳狼是否被迅速识别?

【暴露原因】狼队为什么被识破?
- 哪些发言/票型留下了痕迹(白天跟票太齐/发言风格雷同/悍跳失误)?
- 哪一局开始局势不可逆?转折点是什么?

【角色分工】深水/冲锋/倒钩的执行:
- 深水狼:是否成功藏到最后?有没有过早暴露?
- 冲锋狼:为悍跳狼站台是否有效?是否用力过猛?
- 倒钩狼:踩队友获取信任是否成功?

【保留的优点】本局你做对了什么?必须列出 1-2 个具体策略,下局复用:
- 例如:"我们 N1 空刀让好人视野混乱,第二天悍跳狼拿到警徽"
- 例如:"倒钩狼 D3 故意踩悍跳队友,后期反水一击致命"

【PII 守卫 rag-hardening-4】反思文本中**不要写本局好人的真实身份**。规则:
- 不要写"p03 是预言家被我查杀"或"p05 是女巫被我们毒了"这类具体身份断言
- 改用角色名/模糊指代:"预言家", "女巫", "被查杀的神职"
- 原因:反思会跨局注入其他玩家的 LLM prompt,即使本局你是狼,反思文本不应含具体身份(其他玩家跨局时可能匹配)
- 例外:可以写"我作为狼做了什么"这类自身视角

"""


_GENERIC_REFLECTION_TEMPLATE = """你是{role},本局{faction_result}。请复盘:
- 本局你做了哪些关键判断?哪些对?哪些错?
- 有没有被谁欺骗或误导?下局如何改进?
- 【保留的优点】本局你做对了什么?必须列出 1-2 个具体策略,下局复用。
"""
