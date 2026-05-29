"""Agent runtime adapter: converts GameState into AgentContext for PlayerAgent.

When an AgentRegistry is provided to the runtime graph, night/day nodes will
delegate decisions to PlayerAgent instances. Without a registry, deterministic
scripted fallback is used (preserving existing test behavior).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Protocol

_SPEECH_STYLE_HINTS = {
    "structured_logical": "用严谨的逻辑推理链分析场上信息，像法官一样条理清晰地展示判断依据。",
    "aggressive_short": "用简短犀利、一针见血的质疑制造压力，不需要长篇大论。",
    "moderate_calm": "用平和沉稳的语气，像旁观者一样冷静梳理场上各方的观点。",
    "emotional_intuitive": "用感性直觉的方式判断人，从'感觉不对'出发再找逻辑支撑。",
    "dramatic_theatrical": "用夸张、戏剧化的表达吸引注意力，善用比喻和反问。",
    "quiet_analytical": "不声不响地默默分析，发言内容重质不重量，专注关键细节。",
    "adaptable_flexible": "根据场上局势灵活调整发言策略，该激进时激进，该保守时保守。",
}

_SHERIFF_SPEECH_STYLE_OVERRIDES = {
    "aggressive_short": "用简短犀利、一针见血的质疑制造压力，不需要长篇大论，每句话都要有攻击性。",
    "moderate_calm": "用平和沉稳的语气，像旁观者一样冷静梳理场上各方的观点，指出其中的合理与矛盾。",
    "emotional_intuitive": "用感性直觉的方式判断人，从'感觉不对'出发再找逻辑支撑，可以适当表达情绪。",
    "dramatic_theatrical": "用夸张、戏剧化的表达吸引注意力，善用比喻和反问，让发言有记忆点。",
    "quiet_analytical": "不声不响地默默分析，发言内容重质不重量，专注于关键细节的挖掘。",
}

_TASK_STYLE_HINTS = {
    "authority_claim": "以领导者姿态出现，主动归纳场上信息，给出明确的方向性判断。",
    "forceful_claim": "用强烈的语气宣称自己的判断，对反对者直接施压。",
    "observation_first": "先全面观察再发言，重点分析别人的发言漏洞和信息差。",
    "mystery_hint": "暗示自己掌握关键信息但不直接亮底牌，制造悬念引导讨论方向。",
    "data_driven": "用事实和可验证的信息构建论证，避免空洞的定性判断。",
    "counterattack": "面对质疑时反击而不是解释，把压力转回给质疑者。",
}

_PERSONA_PROFILES_CACHE: dict[str, dict[str, Any]] | None = None
_PERSONA_PROFILES_LOCK: threading.Lock = threading.Lock()


def _load_persona_profile(persona_key: str) -> dict[str, Any]:
    global _PERSONA_PROFILES_CACHE
    with _PERSONA_PROFILES_LOCK:
        if _PERSONA_PROFILES_CACHE is None:
            _PERSONA_PROFILES_CACHE = {}
            try:
                from pathlib import Path
                import yaml
                p = Path(__file__).resolve().parent.parent.parent / "config" / "personas" / "jingcheng_style_prototypes.yaml"
                if p.exists():
                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    _PERSONA_PROFILES_CACHE = data.get("persona_profiles", {})
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to load persona profiles, all agents will use generic speech styles",
                    exc_info=True,
                )
    return _PERSONA_PROFILES_CACHE.get(persona_key, {})


def _get_persona_speech_style(agent: Any) -> str:
    if not agent or not getattr(agent, "persona_key", None):
        return ""
    profile = _load_persona_profile(agent.persona_key)
    return profile.get("base", {}).get("speech_style", "")


def _get_persona_task_style(agent: Any, task_key: str) -> str:
    if not agent or not getattr(agent, "persona_key", None):
        return ""
    profile = _load_persona_profile(agent.persona_key)
    ts = profile.get("task_styles", {})
    return ts.get(task_key, ts.get("speech", ""))


from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    PlayerAction,
    TaskType,
)
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target
from werewolf_agent.skills.registry import SkillRegistry
from werewolf_agent.skills.schemas import SkillInput, SkillName
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.timeline import (
    TIMELINE_ORDER_NOTE,
    current_phase_label,
    phase_label,
)
from werewolf_agent.runtime.private_memory import build_private_memory
from werewolf_agent.runtime.visible_state import build_visible_player_state

# Backward-compatible re-exports from runtime.directives package.
# These functions were extracted to keep agent_adapter.py focused on orchestration.
from werewolf_agent.runtime.directives import (
    build_hunter_directive as _build_hunter_day_speech_directive,
    build_hybrid_directive as _build_hybrid_day_speech_directive,
    build_idiot_directive as _build_idiot_day_speech_directive,
    build_seer_directive as _build_seer_day_speech_directive,
    build_villager_directive as _build_villager_day_speech_directive,
    build_wolf_directive as _build_wolf_day_speech_directive,
    build_wolf_vote_directive as _build_wolf_vote_strategy,
)
from werewolf_agent.runtime.directives._shared import (
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


def _rag_phase_for_task(task_type: TaskType, phase: str) -> str:
    if task_type in (TaskType.SHERIFF_SPEECH, TaskType.SHERIFF_REGISTRATION):
        return "sheriff_speech"
    if task_type == TaskType.WOLF_DISCUSSION:
        return "night_discussion"
    if task_type == TaskType.NIGHT_ACTION:
        return "night_action"
    if task_type in (TaskType.VOTE, TaskType.SPEECH, TaskType.PK_SPEECH):
        return "speech"
    if task_type == TaskType.DEFENSE_SPEECH:
        return "defense_speech"
    return phase or "general"


def _inject_seed_rag_hints(
    context: AgentContext,
    *,
    ruleset_id: str,
    rag_service: Any | None = None,
    game_id: str = "",
) -> AgentContext:
    if not context.own_role:
        return context

    try:
        phase = _rag_phase_for_task(context.task_type, context.phase)
        situation = " ".join([
            context.task_type.value,
            context.phase,
            " ".join(action.value for action in context.legal_actions),
        ]).strip()
        if rag_service is None:
            return context
        from werewolf_agent.rag.schemas import RAGQuery

        query = RAGQuery(
            role=context.own_role,
            phase=phase,
            situation=situation,
            ruleset_id=ruleset_id,
            max_results=3,
            viewer_role=context.own_role,
        )
        hits = rag_service.retrieve_live_hints(
            query,
            game_id=game_id,
            player_id=context.agent_id,
        )
        items = rag_service.hits_to_context_items(hits, max_items=3)
        if not items:
            return context
        existing = [
            item for item in context.rag_hints
            if item.get("type") != "rag_hit"
        ]
        return context.model_copy(update={"rag_hints": existing + items})
    except Exception:
        logger.debug("Seed RAG injection failed for %s", context.agent_id, exc_info=True)
        return context


def _profile_memory_hint(profile: Any, role_stats: dict[str, dict[str, int]]) -> dict[str, Any]:
    roles = [
        {"role": role, "games": stats["count"], "wins": stats["wins"]}
        for role, stats in sorted(role_stats.items())
    ]
    return {
        "games_played": profile.games_played,
        "logic": round(float(profile.logic), 2),
        "deception": round(float(profile.deception), 2),
        "credibility": round(float(profile.credibility), 2),
        "summary": (
            f"累计{profile.games_played}局 · "
            f"逻辑{profile.logic*10:.0f}/10 · "
            f"欺骗{profile.deception*10:.0f}/10 · "
            f"可信度{profile.credibility*10:.0f}/10"
        ),
        "roles": roles,
    }


def _reflection_memory_hints(reflections: list[Any], current_role: str, current_faction: str) -> list[dict[str, Any]]:
    def _ref_score(r: Any) -> tuple[int, Any]:
        priority = 0
        if r.role == current_role:
            priority = 2
        elif (r.role == "werewolf" and current_faction == "werewolf") or (
            r.role != "werewolf" and current_faction == "good"
        ):
            priority = 1
        return (-priority, str(r.entry_id))

    hints: list[dict[str, Any]] = []
    for ref in sorted(reflections, key=_ref_score)[:5]:
        hints.append({
            "role": ref.role,
            "result": "胜" if ref.faction_won else "负",
            "text": ref.text,
            "situation": ref.situation,
        })
    return hints


def _cognition_matrix_hint(restored_memory: Any, player_id: str) -> dict[str, Any]:
    get_matrix = getattr(restored_memory, "get_matrix", None)
    if not callable(get_matrix):
        return {}
    matrix = get_matrix(player_id)
    if matrix is None or not hasattr(matrix, "all_entries"):
        return {}

    suspects: list[dict[str, Any]] = []
    trusted: list[dict[str, Any]] = []
    for entry in matrix.all_entries():
        item = {
            "player": entry.player_id,
            "faction_read": entry.faction_read,
            "trust": round(float(entry.trust), 2),
            "key_evidence": list(getattr(entry, "key_evidence", []))[:3],
            "open_questions": list(getattr(entry, "open_questions", []))[:3],
        }
        if entry.faction_read == "wolf_lean" or float(entry.trust) < 0.35:
            suspects.append(item)
        elif entry.faction_read == "good_lean" or float(entry.trust) > 0.65:
            trusted.append(item)

    hint: dict[str, Any] = {}
    if suspects:
        hint["suspects"] = sorted(suspects, key=lambda x: x["trust"])[:5]
    if trusted:
        hint["trusted"] = sorted(trusted, key=lambda x: -x["trust"])[:5]
    return hint


def _action_trace_payload(action: Any) -> dict[str, Any] | None:
    trace = getattr(action, "trace", None)
    return trace.model_dump() if trace else None


# -- Backward-compatible re-exports from runtime.strategy (Task 2 extraction) --
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value as _estimate_witch_save_value,
    build_witch_pressure_targets as _build_witch_pressure_targets,
    evaluate_seer_check_value as _evaluate_seer_check_value,
    evaluate_death_cause_claims as _evaluate_death_cause_claims,
    evaluate_wolf_kill_target as _evaluate_wolf_kill_target,
    get_wolf_role_assignment as _get_wolf_role_assignment,
    has_publicly_claimed_seer as _has_publicly_claimed_seer,
)
from werewolf_agent.runtime.strategy.seer import public_seer_claimants as _public_seer_claimants  # noqa: F401


def _inject_skill_output(
    strategy_directive: dict[str, Any],
    gs: GameState,
    player_id: str,
    world_state: Any,
    belief_state: Any,
    contradiction_alerts: list[Any],
    phase: str,
    legal_targets: list[str] | None = None,
    wolf_team_plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Dispatch applicable skills once; inject non-tool advice, collect tool analyses.

    Returns (updated strategy_directive, tool_analyses).
    """
    player = gs.players.get(player_id)
    if not player or not player.alive:
        return strategy_directive, {}

    registry = SkillRegistry()
    skill_input = SkillInput(
        role=player.role,
        phase=phase,
        day=gs.day_number,
        game_state=gs,
        world_state=world_state,
        belief_state=belief_state,
        contradiction_alerts=contradiction_alerts,
        player_id=player_id,
        legal_targets=legal_targets or [],
        extra={"wolf_team_plan": wolf_team_plan} if wolf_team_plan else {},
    )

    outputs = registry.dispatch_for_role(player.role, phase, skill_input)

    # Filter skills that conflict with wolf team role assignment
    wolf_role = None
    if wolf_team_plan and player.role == "werewolf":
        for role_key in ("fake_seer", "pusher", "hooker", "deep_cover"):
            if wolf_team_plan.get(role_key) == player_id:
                wolf_role = role_key
                break

    parts: list[str] = []
    tool_analyses: dict[str, str] = {}

    for o in outputs:
        if not o.prompt_injectable or o.confidence < 0.4:
            continue
        # Collect tool skill outputs for on-demand LLM access
        if o.skill_name in _TOOL_SKILL_NAMES:
            tool_def = _SKILL_TOOL_DEFS.get(o.skill_name)
            if tool_def:
                tool_analyses[tool_def["name"]] = o.prompt_injectable
            continue
        # Skip bold_claim for non-fake_seer wolves
        if o.skill_name == "bold_claim" and wolf_role and wolf_role != "fake_seer":
            continue
        # Skip deep_hook for fake_seer/pusher wolves
        if o.skill_name == "deep_hook" and wolf_role and wolf_role in ("fake_seer", "pusher"):
            continue
        # Skip swing_vote for hooker wolves (conflicts with deep-hook mission)
        if o.skill_name == "swing_vote" and wolf_role == "hooker":
            continue
        parts.append(o.prompt_injectable)

    if parts:
        strategy_directive["skill_tactical_advice"] = "\n".join(parts)
    return strategy_directive, tool_analyses


# Skill names and definitions loaded from SKILL.md frontmatter.
def _resolve_tool_skills() -> 'tuple[set[str], dict[str, dict[str, Any]]]':
    try:
        from werewolf_agent.skills.werewolf_skills import _load_tool_skills as _lts
        return _lts()
    except Exception:
        return set(), {}


_TOOL_SKILL_NAMES: set[str]
_SKILL_TOOL_DEFS: dict[str, dict[str, Any]]
_TOOL_SKILL_NAMES, _SKILL_TOOL_DEFS = _resolve_tool_skills()


def _build_skill_tool_defs(role: str, phase: str) -> list[dict[str, Any]]:
    """Build LLM-callable skill tool definitions for applicable on-demand skills."""
    from werewolf_agent.skills.registry import SkillRegistry

    registry = SkillRegistry()
    return [
        _SKILL_TOOL_DEFS[s.name.value]
        for s in registry.all_skills()
        if s.name.value in _TOOL_SKILL_NAMES and s.is_applicable(role, phase)
    ]


def _merge_strategy_directive(
    context: Any,
    new_directive: dict[str, Any],
) -> Any:
    """Merge new directive into existing context strategy_directive, preserving skill_tactical_advice."""
    existing = context.strategy_directive or {}
    merged = {**existing, **new_directive}
    return context.model_copy(update={"strategy_directive": merged})


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
) -> AgentContext:
    """Build AgentContext for a player from current game state.

    Visibility rules:
    - Player only sees their own role.
    - No moderator_full, no other players' private state.
    - Wolf teammates visible only to wolves.
    - Seer sees own check results only.
    - Witch sees potion availability only.
    """
    player = gs.players.get(player_id)
    if player is None:
        return AgentContext(agent_id=player_id, task_type=task_type)

    # Build simplified visible state
    visible: dict[str, Any] = build_visible_player_state(gs)
    private_memory = build_private_memory(gs, player_id)
    if private_memory:
        visible["private_memory"] = private_memory
    private_memory_hints = private_memory or {}

    # Role-specific private info
    strategy_directive: dict[str, Any] = {}
    if player.role == "werewolf":
        visible["wolf_teammates"] = [
            pid for pid, p in gs.players.items()
            if p.alive and p.role == "werewolf" and pid != player_id
        ]
        if wolf_team_plan:
            visible["wolf_team_plan"] = wolf_team_plan
    elif player.role == "seer":
        # 预言家可见自己的验人结果（seer_check 不含 seer_id，所有结果均属预言家）
        check_results = []
        for e in gs.events:
            if e.type == "seer_check":
                check_results.append({
                    "target_id": e.payload["target_id"],
                    "alignment": e.payload["alignment"],
                    "night_number": e.payload["night_number"],
                })
        visible["check_results"] = check_results
    elif player.role == "witch":
        visible["antidote_available"] = not gs.antidote_used
        visible["poison_available"] = not gs.poison_used
        if not gs.poison_used and gs.phase == "day":
            alive = sum(1 for p in gs.players.values() if p.alive)
            if alive <= 8:
                strategy_directive["witch_poison_deterrent"] = (
                    "你的毒药还未使用。如果场上有人持续踩你、试图把你放逐出局，"
                    "你可以在发言中暗示自己有底牌——'我手里还有东西没用，不要太冲动'。"
                    "狼人听到这种暗示可能会退缩。但不要明报身份。"
                )
        if wolf_kill_target_id:
            visible["wolf_kill_target"] = wolf_kill_target_id
        # Poison pressure targets from public state
        if not gs.poison_used:
            pressure_targets = _build_witch_pressure_targets(gs)
            if pressure_targets:
                visible["poison_pressure_targets"] = pressure_targets
    elif player.role == "hybrid" and gs.hybrid_master_id:
        visible["master_id"] = gs.hybrid_master_id

    # Build recent transcript from public speech events
    # Include both day speeches and sheriff election speeches
    transcript: list[dict[str, Any]] = []
    for e in reversed(gs.events):
        if e.type in ("speech", "sheriff_speech") and len(transcript) < 8:
            transcript.insert(0, {
                "speaker": e.payload.get("speaker", ""),
                "text": e.payload.get("text", ""),
                "type": e.type,
            })

    # ── Build public summary (A: enriched events, B: smart truncation) ──
    # Priority: 1=critical(death/vote/seer), 2=secondary(PK/sheriff_no), 3=low(separators)
    SUMMARY_BUDGET = 2500

    summary_items: list[tuple[int, str]] = []

    for e in gs.events:
        if e.type == "day_announce":
            day = e.payload.get("day", "?")
            try:
                day_label = phase_label("day", int(day))
            except (TypeError, ValueError):
                day_label = f"D{day}"
            summary_items.append((3, f"\n===== {day_label} ====="))

        elif e.type == "judge_broadcast" and e.payload.get("visibility") == "public":
            phase = e.payload.get("phase", "")
            msg = e.payload.get("message", "")
            if phase == "death_announce":
                summary_items.append((1, f"[死讯] {msg}"))
            elif phase == "exile":
                summary_items.append((1, f"[法官] {msg}"))
            elif phase == "sheriff_elected":
                summary_items.append((1, f"[警长] {msg}"))
            elif phase in ("vote_tie_pk", "vote_second_tie"):
                summary_items.append((2, f"[法官] {msg}"))
            elif phase == "sheriff_no_election":
                summary_items.append((2, f"[警长] {msg}"))

        elif e.type == "vote_resolved":
            exiled = e.payload.get("exiled")
            reason = e.payload.get("reason", "")
            tied = e.payload.get("tied", [])
            weighted = e.payload.get("weighted_tally", {})
            day = e.payload.get("day_number", "?")
            if exiled:
                if weighted:
                    tally_str = "、".join(
                        f"{pid}={int(w)}票" for pid, w in
                        sorted(weighted.items(), key=lambda x: -x[1])[:5]
                    )
                    summary_items.append((1, f"[放逐] D{day} {exiled}被放逐 ({tally_str})"))
                else:
                    summary_items.append((1, f"[放逐] D{day} {exiled}被放逐"))
            elif reason == "second_tie_no_exile":
                summary_items.append((1, "[放逐] 二次平票，无人出局"))
            elif tied:
                summary_items.append((2, f"[放逐] 平票PK: {', '.join(tied)}"))

        elif e.type == "idiot_reveal":
            summary_items.append((1, f"[白痴] {e.payload.get('player_id', '?')} 亮牌"))

        elif e.type == "hunter_shot_public":
            summary_items.append((1, f"[枪声] 猎人带走了 {e.payload.get('target_id', '?')}"))

        elif e.type in ("speech", "sheriff_speech"):
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            # Extract public seer check claims from speech
            if any(kw in text for kw in ("验了", "查验", "查杀", "金水")):
                m = re.search(r'(?:第?(\d)夜|N(\d)).*?验[了过]?\s*(p\d+).*?(狼人|查杀|好人|金水)', text)
                if m:
                    night = m.group(1) or m.group(2)
                    target = m.group(3)
                    result_raw = m.group(4)
                    result_cn = {"狼人": "狼人", "查杀": "狼人", "好人": "好人", "金水": "好人"}.get(result_raw, result_raw)
                    summary_items.append((1, f"[验人] {speaker} 报 N{night} {target}={result_cn}"))
            # Extract death cause claims (poison / wolf-kill / saved)
            for pattern, label in [
                (r'(?:我|女巫).{0,4}(?:毒[杀了死]|撒毒).{0,4}(p\d+)', '自称毒杀'),
                (r'(p\d+).{0,6}(?:是|被)(?:女巫)?毒[杀了死]', '被指毒杀'),
                (r'(?:狼[刀杀人]|狼人[刀杀]).{0,4}(p\d+)|(p\d+).{0,4}(?:是|被)狼[刀杀了]', '被指狼刀'),
                (r'(?:我|女巫).{0,4}(?:救[了过]|用解药).{0,4}(p\d+)', '自称救了'),
                (r'(p\d+).{0,4}(?:是)?银水', '被指银水'),
            ]:
                for m in re.finditer(pattern, text):
                    target = m.group(1) or m.group(2)
                    if target:
                        summary_items.append((2, f"[死因] {speaker} 称 {target}{label}"))
                        break  # one claim per pattern per speech

    # ── Smart truncation by priority (B) ──
    total = sum(len(t) for _, t in summary_items)
    if total > SUMMARY_BUDGET:
        for drop_priority in (3, 2, 1):
            if total <= SUMMARY_BUDGET:
                break
            for i, (pri, text) in enumerate(summary_items):
                if pri == drop_priority and text:
                    total -= len(text)
                    summary_items[i] = (pri, "")
                    if total <= SUMMARY_BUDGET:
                        break
        summary_items = [(p, t) for p, t in summary_items if t]

    public_summary = "\n".join(text for _, text in summary_items)
    if public_summary:
        public_summary = f"{TIMELINE_ORDER_NOTE}\n{public_summary}"
    else:
        current_label = current_phase_label(
            gs.phase, day_number=gs.day_number, night_number=gs.night_number
        )
        public_summary = TIMELINE_ORDER_NOTE
        if current_label:
            public_summary = f"{public_summary}\n当前时间点：{current_label}"

    # Build contradiction alerts and belief state from world state
    ctx_alerts: list[dict[str, Any]] = []
    must_address: list[dict[str, Any]] = []
    belief_dict: dict[str, Any] = {}
    world_state = None
    belief_state = None
    alerts: list[Any] = []

    # Hybrid: when master is dead, provide faction-guidance
    if player.role == "hybrid" and gs.hybrid_master_id:
        master = gs.players.get(gs.hybrid_master_id)
        if master and not master.alive:
            faction_label = "好人" if gs.hybrid_master_faction == "good" else "狼人"
            strategy_directive["hybrid_master_dead"] = (
                f"你的主人{gs.hybrid_master_id}已死亡。"
                f"你现在以{faction_label}阵营身份继续。"
                f"你现在等同于村民——用分析而非技能帮助阵营。"
            )

    try:
        from werewolf_agent.cognition.world_state import build_world_state
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        from werewolf_agent.cognition.belief import BeliefUpdater

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
        belief_state = updater.update(belief_state, visible_facts, gs.day_number)

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
        contradiction_engine = ContradictionEngine()
        alerts = contradiction_engine.detect(world_state.facts, gs.day_number)

        for alert in alerts:
            if alert.priority == "high":
                alert_entry = {
                    "alert_type": alert.alert_type,
                    "player_id": alert.player_id,
                    "priority": alert.priority,
                    "description": alert.description,
                    "evidence": list(alert.evidence),
                }
                ctx_alerts.append(alert_entry)

        for alert in ctx_alerts:
            players = [p for p in alert["player_id"].split(",") if p]
            must_address.append({
                "alert_type": alert["alert_type"],
                "players": players,
                "public_evidence": alert["description"],
                "required_response": ["question", "side_with", "park"],
            })

        if must_address:
            strategy_directive["must_address_alerts"] = must_address
            strategy_directive["directive"] = "你必须在发言中回应以下矛盾：选择站队、质疑、或明确表示暂不判断。"
    except Exception:
        logger.debug("Contradiction/belief building failed, skipping", exc_info=True)

    if legal_actions is None:
        legal_actions = []
    if legal_targets is None:
        legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != player_id]

    # -- Skill-based tactical advice + on-demand tool analyses (single dispatch) --
    skill_tools: list[dict[str, Any]] = []
    skill_analyses: dict[str, str] = {}
    try:
        strategy_directive, skill_analyses = _inject_skill_output(
            strategy_directive, gs, player_id,
            world_state, belief_state, alerts, task_type.value,
            legal_targets=legal_targets,
            wolf_team_plan=wolf_team_plan,
        )
        skill_tools = _build_skill_tool_defs(player.role, task_type.value)
    except Exception:
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
    profile_memory_hint: dict[str, Any] = {}
    reflection_memory_hints: list[dict[str, Any]] = []
    cognition_matrix_hint: dict[str, Any] = {}
    if restored_memory is not None:
        try:
            profile = restored_memory.get_profile(player_id)
            if profile is not None and profile.games_played > 0:
                # Aggregate by role across all reflections for pattern summary
                role_stats: dict[str, dict[str, int]] = {}
                for ref in restored_memory.reflections_by_player(player_id):
                    r = ref.role or "?"
                    role_stats.setdefault(r, {"count": 0, "wins": 0})
                    role_stats[r]["count"] += 1
                    if ref.faction_won:
                        role_stats[r]["wins"] += 1
                profile_memory_hint = _profile_memory_hint(profile, role_stats)

                # Inject detailed reflections (self-evolution)
                all_refs = restored_memory.reflections_by_player(player_id)
                if all_refs:
                    current_role = player.role
                    current_faction = (
                        "werewolf" if current_role == "werewolf"
                        else "good" if current_role in ("villager", "seer", "witch", "hunter", "idiot")
                        else ("werewolf" if gs.hybrid_master_faction == "werewolf" else "good") if current_role == "hybrid"
                        else "good"
                    )
                    reflection_memory_hints = _reflection_memory_hints(
                        all_refs, current_role, current_faction
                    )
            cognition_matrix_hint = _cognition_matrix_hint(restored_memory, player_id)
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
        visible_world_state=visible,
        private_memory_hints=private_memory_hints,
        reflection_memory_hints=reflection_memory_hints,
        profile_memory_hint=profile_memory_hint,
        cognition_matrix_hint=cognition_matrix_hint,
        recent_transcript=transcript,
        contradiction_alerts=ctx_alerts,
        belief_state=belief_dict,
        strategy_directive=strategy_directive,
        skill_tools=skill_tools,
        skill_analyses=skill_analyses,
        skill_analysis_hints=skill_analyses,
    )
    return _inject_seed_rag_hints(
        context,
        ruleset_id=gs.ruleset_id,
        rag_service=rag_service,
        game_id=gs.game_id,
    )


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
        witch_directive["witch_poison_threshold"] = (
            "【毒药硬证据门槛】毒药是好人阵营最容易打出负收益的技能，不能只因为"
            "“发言像狼”“感觉可疑”“有人带节奏”就单独开毒。优先满足至少一项硬证据："
            "1) 可信预言家的明确查杀；2) 强票型证据（连续保狼、冲票、关键轮分票）；"
            "3) 对跳失败或身份逻辑明显破产；4) 多条公开记录互相印证。"
            "如果证据只停留在泛化怀疑，应在reason里说明为什么暂不单独开毒。"
        )
        alive = sum(1 for p in gs.players.values() if p.alive)
        if alive <= 9:
            witch_directive["poison_urgency"] = (
                f"场上存活{alive}人，你的毒药还未使用。"
                f"如果存在查杀、强票型或对跳失败目标，建议今晚撒毒；"
                f"如果只有泛化怀疑，仍应谨慎，避免盲毒好人。"
            )
        if alive <= 7:
            witch_directive["poison_urgency"] = (
                f"【紧急】场上仅存活{alive}人！你的毒药还没有使用！"
                f"若存在查杀、强票型或对跳失败等硬证据，今晚应优先用毒；"
                f"若仍只有泛化怀疑，必须在reason里说明暂不毒的风险和理由，避免最后一毒误伤好人。"
            )

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
    strategy_directive: dict[str, Any] = {
        "wolf_kill_instruction": (
            "你是狼人，现在是夜间击杀阶段。选择今晚要击杀的目标。\n"
            "击杀策略：优先击杀对狼队威胁最大的玩家（已跳预言家、持警徽、分析能力强的）。\n"
            "如果狼队讨论已确定目标，按讨论共识执行。\n"
            "speech字段留空（夜间行动不需要发言）。"
        ),
    }
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
        "即使TA白天冒充了预言家，在狼队内部你们应该用真实身份称呼。"
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
        "- 【严禁编造尚未发言的玩家说过的话】你只能引用上方transcript中确实记录的发言内容"
        f"{style_hint}"
    )

    # Role-specific speech constraints
    player_role = gs.players[speaker_id].role if speaker_id in gs.players else ""
    if player_role == "werewolf":
        wolf_parts = _build_wolf_day_speech_directive(
            gs, speaker_id, state.get("wolf_team_plan"),
        )
        strategy_directive.update(wolf_parts)
    elif player_role == "seer":
        seer_speech_parts = _build_seer_day_speech_directive(gs, speaker_id)
        strategy_directive.update(seer_speech_parts)
    elif player_role == "hunter":
        strategy_directive["hunter_speech_directive"] = _build_hunter_day_speech_directive(gs, speaker_id)
    elif player_role == "hybrid":
        strategy_directive["hybrid_speech_directive"] = _build_hybrid_day_speech_directive(gs, speaker_id)
    elif player_role == "witch":
        strategy_directive["witch_speech_constraint"] = (
            "你是女巫，你掌握的夜间信息（谁被刀、药水使用情况、救了谁、毒了谁）是你的核心优势。"
            "不要轻易暴露这些信息——一旦公开，狼人会知道你的药水状态并针对性调整策略。"
            "但在以下情况可以适度透露：1) 你即将死亡需要传递关键信息；"
            "2) 场上好人阵营信息严重不足，需要你站出来带队；"
            "3) 有人假冒女巫需要你自证身份。"
            "透露时也要衡量利弊，不要在第一天就全部交底。"
        )
    elif player_role == "idiot":
        strategy_directive.update(_build_idiot_day_speech_directive(gs, speaker_id))
    elif player_role == "villager":
        strategy_directive.update(_build_villager_day_speech_directive(gs, speaker_id))

    # Sheriff gets 归票 (vote push) directive
    if gs.sheriff_id == speaker_id and gs.sheriff_badge_state == "active":
        alive_others = [pid for pid, p in gs.players.items() if p.alive and pid != speaker_id]
        strategy_directive["sheriff_vote_push"] = (
            "你是警长，你的发言需要归票：总结本轮讨论的关键信息点，"
            "明确表态你怀疑谁、要推谁，号召大家集中投票。"
            "警长归票是核心职责，不能含糊其辞。"
        )
        strategy_directive["sheriff_alive_others"] = alive_others

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

    context = build_agent_context(
        engine, gs, sheriff_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SPEECH],
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

    # Use VOTE action to pick a target (first speaker)
    context = context.model_copy(update={
        "legal_actions": [ActionType.VOTE],
        "legal_targets": alive_players,
    })

    action, retry_info = agent.act(context)
    first_speaker = action.target_id if action.action_type == ActionType.VOTE else None

    if first_speaker and first_speaker in alive_players:
        # Build order: first_speaker, then remaining in original order, sheriff last
        remaining = [pid for pid in alive_players if pid != first_speaker]
        return [first_speaker] + remaining + [sheriff_id]
    return None


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
    elif voter_role in ("villager", "idiot"):
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
            "'N1验p03是因为他在警下靠前位置，我需要尽早确认他的身份以建立信息基点'。"
            "如果没有特殊理由，可以说'首夜随机查验，但我选择了发言量较大的位置'。"
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
    """
    agent = registry.get_agent(player_id)
    if agent is None:
        return {}

    from werewolf_agent.agents.schemas import TaskType

    gs = state["game_state"]
    player = gs.players.get(player_id)
    role = player.role if player else "?"
    winner = gs.winning_faction or "?"

    system_prompt = (
        "你是刚完成一局狼人杀的玩家，现在进行对局复盘。"
        f"你的身份是{role}，{'存活到' if (player and player.alive) else '在'}游戏结束。"
        f"胜利方是{'好人' if winner == 'good' else '狼人'}阵营。"
        "请反思本局表现：你做了哪些关键判断？哪些是对的？哪些是错的？"
        "有没有被谁欺骗或误导？下局如何改进？"
    )
    prompt = "请给出你的对局复盘。"

    try:
        action = agent.act(
            prompt=prompt,
            system_prompt=system_prompt,
            task_type=TaskType.SPEECH,
        )
        return {"reflection_text": getattr(action, "speech_text", "") or ""}
    except Exception:
        return {"reflection_text": ""}
