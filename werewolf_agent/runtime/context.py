"""Agent context builder: converts GameState into AgentContext for PlayerAgent.

Extracted from agent_adapter.py to decompose the god object.
This module owns:
- Persona style hints and profiles
- RAG hint injection
- Cross-game memory hints (profile, reflection, cognition matrix)
- Skill output injection and tool definitions
- Strategy directive merging
- The main build_agent_context() function
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from typing import Any

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    TaskType,
)
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.skills.registry import SkillRegistry
from werewolf_agent.skills.schemas import SkillInput, SkillName
from werewolf_agent.runtime.timeline import (
    TIMELINE_ORDER_NOTE,
    current_phase_label,
    phase_label,
)

# Backward-compatible re-exports from runtime.directives package.
from werewolf_agent.runtime.directives import (
    build_hunter_directive as _build_hunter_day_speech_directive,
    build_hybrid_directive as _build_hybrid_day_speech_directive,
    build_idiot_directive as _build_idiot_day_speech_directive,
    build_seer_directive as _build_seer_day_speech_directive,
    build_villager_directive as _build_villager_day_speech_directive,
    build_wolf_directive as _build_wolf_day_speech_directive,
)
# Backward-compatible re-exports from runtime.strategy (Task 2 extraction).
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value as _estimate_witch_save_value,
    build_witch_pressure_targets as _build_witch_pressure_targets,
    evaluate_seer_check_value as _evaluate_seer_check_value,
    evaluate_death_cause_claims as _evaluate_death_cause_claims,
    evaluate_wolf_kill_target as _evaluate_wolf_kill_target,
    get_wolf_role_assignment as _get_wolf_role_assignment,
    has_publicly_claimed_seer as _has_publicly_claimed_seer,
)

logger = logging.getLogger(__name__)

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


# P1-G6: RAG retrieval is wasted on REFLECTION (post-game review of
# the agent's own play — strategy hints are not actionable in that
# context) and on JUDGE_* tasks (moderator persona; strategy hints
# don't apply). Skipping them saves an unnecessary embed/rerank call
# and keeps the live prompt free of irrelevant cases.
_RAG_SKIPPED_TASK_TYPES: frozenset[TaskType] = frozenset({
    TaskType.REFLECTION,
    TaskType.JUDGE_PHASE,
    TaskType.JUDGE_DEATH,
    TaskType.JUDGE_VOTE_CALLING,
    TaskType.JUDGE_VOTE_TALLY,
    TaskType.JUDGE_SKILL_GUIDE,
    TaskType.JUDGE_SHERIFF,
    TaskType.JUDGE_EXILE,
})


def _inject_seed_rag_hints(
    context: AgentContext,
    *,
    ruleset_id: str,
    rag_service: Any | None = None,
    game_id: str = "",
    n_alive: int = 0,
) -> AgentContext:
    if not context.own_role:
        return context

    # P1-G6: skip RAG for non-player task types (reflection + judge).
    if context.task_type in _RAG_SKIPPED_TASK_TYPES:
        return context

    # P2-G11: rag_service is None is an expected configuration
    # (RAG disabled / not provisioned). No log noise, no anomaly
    # count. The retrieval code path simply doesn't run.
    if rag_service is None:
        return context

    try:
        phase = _rag_phase_for_task(context.task_type, context.phase)
        # P1-G7: the situation is a small semantic key=value blob, not
        # a raw space-joined concat. The retriever tokenizes on `=`
        # and weights the tag-overlap score on the value tokens, so
        # the role/phase/task/alive/actions values reach the scoring
        # path cleanly. The old format ('speech day vote speech')
        # carried no semantic structure and the rule-based retriever
        # essentially never matched.
        actions = [a.value for a in context.legal_actions]
        situation = (
            f"role={context.own_role} phase={context.phase} "
            f"task={context.task_type.value} alive={n_alive} "
            f"actions={actions}"
        )
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
        # P0-G1: the live prompt must only see title/summary/key_decisions,
        # never the audit-only fields (relevance, quality, source type,
        # visibility, display annotation). Audit data stays on the
        # ``RAGInjector.audit_log()`` side.
        items = rag_service.hits_to_prompt_lines(hits, max_items=3)
        if not items:
            return context
        existing = [
            item for item in context.rag_hints
            if item.get("type") != "rag_hit"
        ]
        return context.model_copy(update={"rag_hints": existing + items})
    except Exception:
        # P2-G11: rag_service.retrieve_live_hints() raised — this is
        # an anomaly (the service is provisioned but its retrieval
        # path crashed). Warn-level log so operators notice; bump
        # rag_anomaly_count on the returned context so metrics can
        # track repeated failures per game.
        logger.warning(
            "RAG retrieval anomaly for %s (game=%s): incrementing rag_anomaly_count",
            context.agent_id, game_id, exc_info=True,
        )
        return context.model_copy(
            update={"rag_anomaly_count": context.rag_anomaly_count + 1}
        )


# ---------------------------------------------------------------------------
# Speech position extraction helpers (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _extract_suspects(text: str) -> list[str]:
    suspects: list[str] = []
    for m in re.finditer(r"(?:怀疑|标狼|狼面|定狼|抗推|出)\s*(p\d{2})", text):
        pid = m.group(1)
        if pid not in suspects:
            suspects.append(pid)
    return suspects


def _extract_trusts(text: str) -> list[str]:
    trusts: list[str] = []
    for m in re.finditer(r"(?:相信|好人|保|银水|金水|认好)\s*(p\d{2})", text):
        pid = m.group(1)
        if pid not in trusts:
            trusts.append(pid)
    return trusts


def _extract_role_claim(text: str) -> str | None:
    m = re.search(r"(?:我是|跳|身份是|底牌是)\s*(预言家|女巫|猎人|白痴|平民|村民|混血儿)", text)
    return m.group(1) if m else None


def _extract_vote_intent(text: str) -> str | None:
    m = re.search(r"(?:归票|票投|出|投给|投票|上票)\s*(p\d{2})", text)
    return m.group(1) if m else None


def _first_sentence(text: str, max_len: int = 60) -> str:
    for sep in ("。", "！", "？", "\n"):
        idx = text.find(sep)
        if idx > 0:
            sentence = text[:idx + 1].strip()
            return sentence[:max_len]
    return text.strip()[:max_len]


# ---------------------------------------------------------------------------
# Cross-game memory hint builders
# ---------------------------------------------------------------------------

def _profile_memory_hint(
    profile: Any,
    role_stats: dict[str, dict[str, int]],
    current_role: str,
) -> dict[str, Any]:
    """Build the profile memory hint for the agent prompt.

    Renders rank description ("前 30%" / "中等" / "需要提升") instead of
    raw ability floats to avoid biasing LLM self-confidence. Only the
    current role's win-rate is exposed (other roles' stats are private).

    P0-M5: renders ALL 6 schema dims (logic, deception, leadership,
    credibility, learning_rate, risk_preference). For the inner
    traits (learning_rate, risk_preference) uses neutral phrasing
    ("你的学习速度处于中等") so the LLM does not anchor on a
    judgmental token like "需要提升" applied to a private trait.

    Rank bins (heuristic, against the 0.0–1.0 score range):
    - > 0.66  → "前 30%"  (top tier) — for the 4 public traits
    - > 0.33  → "中等"    (middle tier)
    - ≤ 0.33  → "需要提升" (needs improvement)

    Inner traits (learning_rate, risk_preference) get a parallel
    neutral ranker ("较高" / "中等" / "偏低") with phrasing that
    says "你的 X 处于 Y", never a critical "需要提升".
    """
    def _rank(score: float) -> str:
        if score > 0.66:
            return "前 30%"
        if score > 0.33:
            return "中等"
        return "需要提升"

    def _inner_rank(score: float) -> str:
        """Neutral rank for private traits; never sounds like a critique."""
        if score > 0.66:
            return "较高"
        if score > 0.33:
            return "中等"
        return "偏低"

    # Filter role stats to current role only; default to zero stats if
    # the player has never played this role before.
    stats = role_stats.get(current_role, {"count": 0, "wins": 0})
    win_rate_pct = (
        round(100 * stats["wins"] / stats["count"]) if stats["count"] > 0 else 0
    )
    # Use getattr so test fakes / partial profiles (e.g. M4-era FakeProfile
    # that only set logic/deception/credibility) still work. The schema
    # defaults (PlayerProfile dataclass) supply 0.5 for missing fields.
    # P2-M15: only the 4 public traits render into the hint. The inner
    # traits (learning_rate, risk_preference) are review/judge-only per
    # the M4 contract; they no longer drive a summary line.
    logic_rank = _rank(float(getattr(profile, "logic", 0.5)))
    deception_rank = _rank(float(getattr(profile, "deception", 0.5)))
    leadership_rank = _rank(float(getattr(profile, "leadership", 0.5)))
    credibility_rank = _rank(float(getattr(profile, "credibility", 0.5)))

    return {
        "games_played": profile.games_played,
        "current_role": current_role,
        "current_role_games": stats["count"],
        "current_role_win_rate_pct": win_rate_pct,
        "logic_rank": logic_rank,
        "deception_rank": deception_rank,
        "leadership_rank": leadership_rank,
        "credibility_rank": credibility_rank,
        "learning_rate_rank": _inner_rank(float(getattr(profile, "learning_rate", 0.5))),
        "risk_preference_rank": _inner_rank(float(getattr(profile, "risk_preference", 0.5))),
    }


def _reflection_memory_hints(reflections: list[Any], current_role: str, current_faction: str) -> list[dict[str, Any]]:
    # P1-M12: cap at 2 hints per role so the top 5 surface reflections
    # from multiple perspectives rather than 5 from the same role /
    # scenario. The 5-hint output budget is preserved; we only restrict
    # how many may come from a single role.
    MAX_PER_ROLE = 2
    HINT_BUDGET = 5

    def _ref_score(r: Any) -> tuple[int, str, str]:
        priority = 0
        if r.role == current_role:
            priority = 2
        elif (r.role == "werewolf" and current_faction == "werewolf") or (
            r.role != "werewolf" and current_faction == "good"
        ):
            priority = 1
        # Include game_id so ties are broken by game recency (newer first).
        # entry_id alone is unreliable because it's a composite
        # "reflection_{game_id}_{player_id}" string. Invert char codes so
        # YYYY-MM-DD values sort newest-first under ascending comparison.
        # Use getattr so reflection-like test doubles without game_id still work.
        game_id = getattr(r, "game_id", "") or ""
        neg_game_id = "".join(chr(0x10FFFF - ord(c)) for c in str(game_id))
        return (-priority, neg_game_id, str(r.entry_id))

    # Sort by priority (highest first), then by game recency (newest
    # first). Walk the sorted list and admit each reflection that fits
    # within the role cap and the total budget.
    role_counts: dict[str, int] = {}
    hints: list[dict[str, Any]] = []
    for ref in sorted(reflections, key=_ref_score):
        if len(hints) >= HINT_BUDGET:
            break
        role = getattr(ref, "role", "") or ""
        if role_counts.get(role, 0) >= MAX_PER_ROLE:
            continue
        role_counts[role] = role_counts.get(role, 0) + 1
        hints.append({
            "role": ref.role,
            "result": "胜" if ref.faction_won else "负",
            "text": ref.text,
            "situation": ref.situation,
        })
    return hints


def _evidence_id_ref(text: str) -> str:
    """Render a short, stable ID reference for an evidence/question text.

    P0-M9: the raw text is no longer surfaced in the prompt. We hash
    the text into a 10-char hex suffix and prefix it with the
    ``salience_items#`` tag. The viewer can still cross-reference
    the same item across turns because identical text always maps
    to the same id.
    """
    if not text:
        return "salience_items#empty"
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]
    return f"salience_items#{h}"


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
        # P0-M9: render key_evidence and open_questions as ID references,
        # never as full text. The summary stats (trust, faction_read)
        # are already derived from public facts via BeliefUpdater.
        key_evidence = [
            _evidence_id_ref(text)
            for text in list(getattr(entry, "key_evidence", []))[:3]
        ]
        open_questions = [
            _evidence_id_ref(text)
            for text in list(getattr(entry, "open_questions", []))[:3]
        ]
        item = {
            "player": entry.player_id,
            "faction_read": entry.faction_read,
            "trust": round(float(entry.trust), 2),
            "key_evidence": key_evidence,
            "open_questions": open_questions,
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
    task_type: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    """Dispatch applicable skills once; inject non-tool advice, collect tool analyses.

    Returns (updated strategy_directive, tool_analyses).

    P0-K2: `task_type` is forwarded to `dispatch_for_role` so the
    `applies_to_task_types` filter can refine the dispatch.
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
        # P1-K5: forward task_type so handlers can branch on it.
        task_type=task_type,
    )

    # P0-K2: pass task_type so the new `applies_to_task_types` filter works.
    outputs = registry.dispatch_for_role(
        player.role, phase, skill_input, task_type=task_type,
    )

    # Filter skills that conflict with wolf team role assignment
    wolf_role = None
    if wolf_team_plan and player.role == "werewolf":
        for role_key in ("fake_seer", "pusher", "hooker", "deep_cover"):
            if wolf_team_plan.get(role_key) == player_id:
                wolf_role = role_key
                break

    parts: list[str] = []

    # P1-K3: do NOT drop on `confidence < 0.4`. Low-confidence output is
    # often negative-signal advice ("don't do X", "avoid Y") that is
    # still useful. E.g., bold_claim emits `你不需要悍跳` with
    # confidence=0.3 when a teammate is already assigned as fake_seer.
    # Sort by confidence descending so the highest-confidence advice
    # appears first in the rendered prompt — within the same budget,
    # the LLM sees the best signal first; actionable low-confidence
    # advice (e.g. "your teammate already handles X") remains reachable.
    sortable: list[tuple[float, str]] = []

    for o in outputs:
        if not o.prompt_injectable:
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
        sortable.append((o.confidence, o.prompt_injectable))

    # Sort highest confidence first; stable for ties.
    sortable.sort(key=lambda x: -x[0])
    parts = [text for _, text in sortable]

    if parts:
        strategy_directive["skill_tactical_advice"] = "\n".join(parts)
    return strategy_directive, {}


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
    discussion_positions: dict[str, str] | None = None,
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

    # Build recent transcript: speeches + votes (up to 12 items for pattern analysis)
    transcript: list[dict[str, Any]] = []
    for e in reversed(gs.events):
        if e.type in ("speech", "sheriff_speech"):
            if len(transcript) < 10:
                transcript.insert(0, {
                    "speaker": e.payload.get("speaker", ""),
                    "text": e.payload.get("text", ""),
                    "type": e.type,
                })
        elif e.type == "vote_resolved" and len(transcript) < 12:
            votes_detail = e.payload.get("votes", [])
            if votes_detail:
                voter_lines = {v.get("voter", "?"): v.get("target", "弃票") for v in votes_detail}
                transcript.insert(0, {
                    "type": "vote_record",
                    "day": e.payload.get("day_number", "?"),
                    "result": e.payload.get("exiled") or "无人出局",
                    "votes": voter_lines,
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
            elif phase == "sheriff_registered":
                summary_items.append((1, f"[上警] {msg}"))
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
            votes_detail = e.payload.get("votes", [])
            if exiled:
                if weighted:
                    tally_str = "、".join(
                        f"{pid}={int(w)}票" for pid, w in
                        sorted(weighted.items(), key=lambda x: -x[1])[:5]
                    )
                    summary_items.append((1, f"[放逐] D{day} {exiled}被放逐 ({tally_str})"))
                else:
                    summary_items.append((1, f"[放逐] D{day} {exiled}被放逐"))
                # Per-voter breakdown for pattern analysis
                if votes_detail:
                    voter_lines = []
                    for v in votes_detail:
                        voter = v.get("voter", "?")
                        target = v.get("target", "弃票") if v.get("target") else "弃票"
                        voter_lines.append(f"{voter}→{target}")
                    summary_items.append((1, f"[投票] D{day}: {'，'.join(voter_lines)}"))
            elif reason == "second_tie_no_exile":
                summary_items.append((1, "[放逐] 二次平票，无人出局"))
                if votes_detail:
                    voter_lines = []
                    for v in votes_detail:
                        voter = v.get("voter", "?")
                        target = v.get("target", "弃票") if v.get("target") else "弃票"
                        voter_lines.append(f"{voter}→{target}")
                    summary_items.append((1, f"[投票] D{day}: {'，'.join(voter_lines)}"))
            elif tied:
                summary_items.append((2, f"[放逐] 平票PK: {', '.join(tied)}"))
                if votes_detail:
                    voter_lines = []
                    for v in votes_detail:
                        voter = v.get("voter", "?")
                        target = v.get("target", "弃票") if v.get("target") else "弃票"
                        voter_lines.append(f"{voter}→{target}")
                    summary_items.append((1, f"[投票] D{day}: {'，'.join(voter_lines)}"))

        elif e.type == "idiot_revealed":
            summary_items.append((1, f"[白痴] {e.payload.get('player_id', '?')} 亮牌"))

        elif e.type == "hunter_shot_public":
            hunter = e.payload.get("hunter_id", "?")
            target = e.payload.get("target_id", "?")
            summary_items.append((1, f"[枪声] 猎人{hunter}带走了{target}"))

        elif e.type in ("speech", "sheriff_speech"):
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            # Mark silent/no-content speeches explicitly to prevent LLM hallucination
            if "未发表有效言论" in text or not text.strip():
                summary_items.append((3, f"[沉默] {speaker} 未发表任何有效言论"))
                continue
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

    # ── Player's own speech summary (from LLM or deterministic fallback) ──
    own_summary = (discussion_positions or {}).get(player_id, "")
    if own_summary:
        public_summary += f"\n\n--- 你对今日讨论的总结 (D{gs.day_number}) ---\n{own_summary}"

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

    # -- Skill-based tactical advice (pre-injection path; no tool exposure) --
    skill_analyses: dict[str, str] = {}
    try:
        strategy_directive, skill_analyses = _inject_skill_output(
            strategy_directive, gs, player_id,
            world_state, belief_state, alerts, task_type.value,
            legal_targets=legal_targets,
            wolf_team_plan=wolf_team_plan,
        )
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
                profile_memory_hint = _profile_memory_hint(profile, role_stats, player.role)

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
        skill_analyses=skill_analyses,
        skill_analysis_hints=skill_analyses,
    )
    return _inject_seed_rag_hints(
        context,
        ruleset_id=gs.ruleset_id,
        rag_service=rag_service,
        game_id=gs.game_id,
        n_alive=sum(1 for p in gs.players.values() if p.alive),
    )
