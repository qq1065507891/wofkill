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
from collections import Counter
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
    build_wolf_day_directive as _build_wolf_day_speech_directive,
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
    "confident_fake_claim": "表达坚定、自信、带有权威感，即使受到质疑也维持完整叙事并主动反压。",
    "subtle_helpful": "表面温和协作，先补充细节和可验证信息，再含蓄地引导讨论方向。",
    "emotional_vivid": "允许鲜明情绪和生活化表达，但最终要落回一个可以核对的判断依据。",
    "humorous_distracting": "用轻松、机敏和适量反问制造记忆点，同时避免让玩笑取代有效分析。",
    "evidence_based": "优先引用公开事件、原话和票型，用复盘式表达区分事实、推测与结论。",
    "brief_pointed": "少说套话，只抓一个关键细节，短句表达明确判断和后续观察点。",
    "mostly_quiet_then_explosive": "平时克制简短，发现关键矛盾时集中输出完整证据链并给出强判断。",
    "structured_slotting": "按玩家位置或信息链分组盘点，比较各组关系后再给出嫌疑排序。",
    "adaptive_varied": "根据局势切换长短、强弱和分析角度，避免连续使用同一种句式。",
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


# G-R4-14: legal_action → RAG tag mapping. The previous code
# serialized ``[a.value for a in context.legal_actions]`` directly
# into the situation as a Python list repr, e.g.
# ``actions=['wolf_kill', 'sheriff_vote']``. Seed entry tags live
# in a different shape (``[werewolf, deep_hook, deception]``), so
# the retriever's tag-overlap scoring never surfaced a match.
#
# The mapping below is intentionally aligned with the seed tag
# vocabulary (``seer``, ``werewolf``, ``witch``, ``hunter``,
# ``sheriff``, ``speech``, ``deception``, ``seer_check``,
# ``witch_save``, ``witch_poison``, ``hunter_shot``, etc.). A
# legal action of e.g. ``WOLF_KILL`` contributes the tags
# ``werewolf`` (role) + ``wolf_kill`` (action); a
# ``SHERIFF_REGISTER`` contributes ``sheriff_register`` (action).
_LEGAL_ACTION_TAGS: dict[str, tuple[str, ...]] = {
    "vote": ("vote",),
    "wolf_kill": ("werewolf", "wolf_kill"),
    "wolf_no_kill": ("werewolf", "wolf_no_kill"),
    "use_antidote": ("witch", "witch_save", "antidote"),
    "use_poison": ("witch", "witch_poison", "poison"),
    "check_alignment": ("seer", "seer_check"),
    "choose_master": ("hybrid", "hybrid_master"),
    "hunter_shot": ("hunter", "hunter_shot"),
    "self_destruct": ("idiot", "idiot_reveal"),
    "sheriff_register": ("sheriff", "sheriff_register"),
    "sheriff_withdraw": ("sheriff", "sheriff_withdraw"),
    "sheriff_vote": ("sheriff", "sheriff_vote"),
    "badge_transfer": ("sheriff", "badge_transfer", "badge_flow"),
    "badge_tear": ("sheriff", "badge_tear"),
    "speech": ("speech",),
    "no_action": (),
}


def _normalize_legal_actions_to_tags(
    legal_actions: list[ActionType],
) -> str:
    """G-R4-14: serialize legal_actions as a deduplicated,
    space-joined tag string (no list repr, no quote chars) so the
    retriever's tag-overlap scoring can match against seed entry
    tag shapes. Unknown actions fall back to their raw value so a
    future ActionType addition does not silently drop the action
    from the situation.
    """
    seen: set[str] = set()
    out: list[str] = []
    for action in legal_actions:
        key = action.value if hasattr(action, "value") else str(action)
        for tag in _LEGAL_ACTION_TAGS.get(key, (key,)):
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
    return " ".join(out)


# P1-G6: RAG retrieval is wasted on REFLECTION (post-game review of
# the agent's own play — strategy hints are not actionable in that
# context) and on JUDGE_* tasks (moderator persona; strategy hints
# don't apply). Skipping them saves an unnecessary embed/rerank call
# and keeps the live prompt free of irrelevant cases.
#
# G-R4-08: LAST_WORDS is a deathbed speech — strategy hints are not
# actionable, and the task type falls through ``_rag_phase_for_task``
# to the raw game phase (day/night), which never matches any seed
# entry's ``phase`` value (seeds are tagged
# ``speech``/``night_action``/``night_discussion``/etc.). Skipping
# avoids a guaranteed-miss retrieval call.
_RAG_SKIPPED_TASK_TYPES: frozenset[TaskType] = frozenset({
    TaskType.REFLECTION,
    TaskType.LAST_WORDS,
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
        # G-R4-14: legal_actions are now normalized to a deduplicated
        # space-joined tag string (no Python list repr). The mapping
        # table in ``_LEGAL_ACTION_TAGS`` aligns the action values
        # with the seed entry tag vocabulary so the retriever's
        # tag-overlap scoring has a chance to surface a match.
        actions_tags = _normalize_legal_actions_to_tags(context.legal_actions)
        # G-R4-07: the situation carried a bare ``phase=`` key whose
        # value (day/night) collided with the query's own ``phase``
        # field (the task phase: speech / night_action / wolf_discussion).
        # The retriever tokenizes on ``=`` and couldn't tell them
        # apart. Renamed to ``game_phase=`` so the two fields are
        # unambiguous at the retriever's tag-overlap scoring step.
        # P2-12: when legal_actions is empty (or all actions map to
        # no tags), skip the ``actions=`` segment entirely.  Otherwise
        # the situation string ends with a trailing ``actions=`` and
        # the retriever's tag-overlap scoring picks up that empty
        # token as a stray feature.  Filter tokens are now exactly
        # the non-empty ones.
        situation_parts = [
            f"role={context.own_role}",
            f"game_phase={context.phase}",
            f"task={context.task_type.value}",
            f"alive={n_alive}",
        ]
        if actions_tags:
            situation_parts.append(f"actions={actions_tags}")
        situation = " ".join(situation_parts)
        # R18: build the RAGQuery through the RAGInjector helper so the
        # query defaults (ruleset_id, max_results) live in one place.
        # Adding a new default there now also flows through this path.
        # P2-6: import the live-prompt cap constant from the slim
        # renderer so the 3 sites (this max_results, the slim
        # max_items below, and prompt_builder.py's [:3] slices) all
        # share one source of truth.
        from werewolf_agent.rag.injector import RAGInjector
        from werewolf_agent.rag.prompt_renderer import RAG_LIVE_PROMPT_CAP

        query = RAGInjector.build_rag_query(
            role=context.own_role,
            phase=phase,
            situation=situation,
            ruleset_id=ruleset_id,
            max_results=RAG_LIVE_PROMPT_CAP,
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
        # P2-6: use the shared live-prompt cap constant.
        items = rag_service.hits_to_prompt_lines(hits, max_items=RAG_LIVE_PROMPT_CAP)
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

    # P2-7 removed the last caller of _inner_rank (the
    # learning_rate_rank / risk_preference_rank review-only fields
    # are no longer surfaced in the player-facing hint).  The
    # helper was kept around for one extra commit in case a future
    # change wanted to re-introduce inner-trait ranks; Phase 3
    # audit found no such caller, so removing it now.
    # Phase 3 (clean-1) dead code removal.

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

    def _confidence_label(games: int) -> str:
        # Phase 2 P2-8: surface the sample size as a Chinese label
        # so the LLM can distinguish 1-game 100% from 10-game 67%.
        # Without this label the LLM has been observed to over-trust
        # small-sample win rates (and abandon "I always lose this role"
        # when N=1 because the LLM misread the precision).
        if games == 0:
            return "无历史"
        if games < 3:
            return f"样本不足(仅{games}局)"
        if games < 10:
            return f"样本中等({games}局)"
        return f"样本充足({games}局)"

    return {
        "games_played": profile.games_played,
        "current_role": current_role,
        "current_role_games": stats["count"],
        "current_role_win_rate_pct": win_rate_pct,
        # Phase 2 P2-8: sample-size confidence label
        "win_rate_confidence": _confidence_label(stats["count"]),
        "logic_rank": logic_rank,
        "deception_rank": deception_rank,
        "leadership_rank": leadership_rank,
        "credibility_rank": credibility_rank,
        # Phase 2 P2-7: learning_rate_rank / risk_preference_rank
        # removed from the player-facing hint.  These are review /
        # judge-only fields per the M4 contract; the schema still
        # exposes them on ``PlayerProfile`` for review tooling but
        # the LLM no longer sees them mid-game.
    }


# M4-1 (2026-06-09): moved to module level so the prompt builder can
# import it as a single source of truth.  Previously this was a
# closure constant inside ``_reflection_memory_hints``; the prompt
# builder sliced with the hard-coded literal ``[:5]`` and silently
# dropped 3 hints from every LLM call.
#
# reflect-cross-2: budget 5 → 8. 4 局真实游戏分析显示 top-5 经常被
# 同一角色填满 (max 2 per role × 2~3 roles),跨角色学习受限。
# 8 hints × 2 per role = 覆盖 4 角色族,适合好人阵营 5 角色 + 狼 1 角色场景。
HINT_BUDGET = 8


def _reflection_memory_hints(reflections: list[Any], current_role: str, current_faction: str) -> list[dict[str, Any]]:
    # P1-M12: cap at 2 hints per role so the top hints surface reflections
    # from multiple perspectives rather than 5 from the same role /
    # scenario. The hint output budget is preserved; we only restrict
    # how many may come from a single role.
    MAX_PER_ROLE = 2

    def _ref_score(r: Any) -> tuple[int, int, str, str]:
        priority = 0
        if r.role == current_role:
            priority = 2
        elif (r.role == "werewolf" and current_faction == "werewolf") or (
            r.role != "werewolf" and current_faction == "good"
        ):
            priority = 1
        # reflect-cross-3: 胜局反思优先 (成功模式可复用)。
        # 同 priority 内,faction_won=True 排前 → LLM 优先看"做对的事"而非"做错的事"。
        won = 1 if getattr(r, "faction_won", False) else 0
        # Include game_id so ties are broken by game recency (newer first).
        # entry_id alone is unreliable because it's a composite
        # "reflection_{game_id}_{player_id}" string.
        #
        # Phase 2 P2-9: the previous chr-invert trick
        # (``"".join(chr(0x10FFFF - ord(c)) for c in str(game_id))``)
        # was brittle to game_id format variations — e.g.
        # ``g_2024-12-20`` vs ``g2024-12-20`` (with/without separator)
        # could rank out of order because the underscore vs no-
        # underscore changed the char-code inversion at that
        # position.  Replace with a parseable YYYY-MM-DD regex +
        # arithmetic invert that is robust to any prefix/separator.
        #
        # Use getattr so reflection-like test doubles without
        # game_id still work (empty string falls through to a
        # stable tiebreaker on entry_id).
        game_id = str(getattr(r, "game_id", "") or "")
        ts = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})", game_id)
        if ts is not None:
            yyyy, mm, dd = int(ts.group(1)), int(ts.group(2)), int(ts.group(3))
            # Invert each component so that newer dates sort first
            # under ascending comparison (Python's sort is stable).
            neg_game_id = f"{(9999 - yyyy):04d}-{(12 - mm):02d}-{(31 - dd):02d}"
        else:
            # No parseable date — fall back to entry_id stable sort.
            neg_game_id = ""
        return (-priority, -won, neg_game_id, str(r.entry_id))

    # Sort by priority (highest first), then by faction_won (winning first),
    # then by game recency (newest first). Walk the sorted list and admit
    # each reflection that fits within the role cap and the total budget.
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


# ---------------------------------------------------------------------------
# Error pattern aggregation (reflect-cross-1)
# ---------------------------------------------------------------------------

# Section header → category mapping. Templates in agent_adapter.py emit
# these headers; we parse them to derive error categories without an
# extra LLM call.
_REFLECTION_HEADER_CATEGORIES: dict[str, str] = {
    "【投票错误】": "vote_mistake",
    "【信息缺失】": "info_miss",
    "【神职执行】": "role_execution",
    "【悍跳分析】": "claim_failed",
    "【暴露原因】": "exposure",
    "【角色分工】": "role_execution",
    "【保留的优点】": "preserved_strength",
}


def _categorize_reflection_text(text: str) -> list[str]:
    """从反思文本中解析章节头,返回 category 列表。

    Section header regex: ``【...】`` 出现在文本中即视为该类目命中。
    同类目多次出现算 1 次 (去重),保证权重不被重复章节头放大。
    """
    if not text:
        return []
    cats: list[str] = []
    seen: set[str] = set()
    for header, cat in _REFLECTION_HEADER_CATEGORIES.items():
        if header in text and cat not in seen:
            cats.append(cat)
            seen.add(cat)
    return cats


def _compute_error_pattern(
    reflections: list[Any],
    current_role: str,
) -> dict[str, Any]:
    """聚合某 player 跨局反思,提取 top 错误模式 + 保留优点。

    返回 dict:
      - top_mistakes: list[(category, count)] 前 2 类错误 (按频次)
      - preserved_strength_count: int 含【保留的优点】段的反思数
      - total_reflections: int
      - same_role_reflections: int 与当前角色相同的反思数
      - dominant_mistake_ratio: float 最高频错误 / 总错误数 (0~1)
      - current_role: str

    用途:作为 error_pattern_hint 注入 LLM prompt,让 LLM 看到
    "你历史最常犯的错误是 X" 这种聚合信号,而不只是单条反思。
    """
    if not reflections:
        return {
            "top_mistakes": [],
            "preserved_strength_count": 0,
            "total_reflections": 0,
            "same_role_reflections": 0,
            "dominant_mistake_ratio": 0.0,
            "current_role": current_role,
        }

    mistake_counter: Counter = Counter()
    preserved_count = 0
    same_role_reflections = 0

    for r in reflections:
        r_text = getattr(r, "text", "") or ""
        cats = _categorize_reflection_text(r_text)
        if "preserved_strength" in cats:
            preserved_count += 1
            cats.remove("preserved_strength")
        if getattr(r, "role", "") == current_role:
            same_role_reflections += 1
        for c in cats:
            mistake_counter[c] += 1

    top_mistakes = mistake_counter.most_common(2)
    total_mistakes = sum(mistake_counter.values())
    dominant_ratio = (
        round(mistake_counter.most_common(1)[0][1] / total_mistakes, 2)
        if total_mistakes > 0 and mistake_counter
        else 0.0
    )

    return {
        "top_mistakes": top_mistakes,
        "preserved_strength_count": preserved_count,
        "total_reflections": len(reflections),
        "same_role_reflections": same_role_reflections,
        "dominant_mistake_ratio": dominant_ratio,
        "current_role": current_role,
    }


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
        # MEM-07: key_evidence items may be EvidenceItem or bare str.
        def _claim_str(item: Any) -> str:
            claim = getattr(item, "claim", None)
            return str(claim) if claim is not None else str(item)
        key_evidence = [
            _evidence_id_ref(_claim_str(text))
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
    task_type: str,
    legal_targets: list[str] | None = None,
    wolf_team_plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Inject skill advice into strategy_directive, return (directive, analyses).

    The positional order is role/.../legal_targets; ``task_type`` is
    a kwarg forwarded to ``SkillRegistry.dispatch_for_role`` so the
    ``applies_to_task_types`` filter (P0-K2) can refine the dispatch.
    """
    player = gs.players.get(player_id)
    if not player or not player.alive:
        return strategy_directive, {}

    registry = SkillRegistry()
    skill_input = SkillInput(
        role=player.role,
        phase=task_type,  # legacy kwarg, kept for backward compat
        day=gs.day_number,
        game_state=gs,
        world_state=world_state,
        belief_state=belief_state,
        contradiction_alerts=contradiction_alerts,
        player_id=player_id,
        legal_targets=legal_targets or [],
        extra={"wolf_team_plan": wolf_team_plan} if wolf_team_plan else {},
        task_type=task_type,
    )

    # S-05: dispatch_for_role receives the task_type value as both
    # `phase` (2nd positional — backward compat with the older API)
    # and `task_type` (kwarg — used by the P0-K2 precise filter).
    # NEW-S02-A: forward `gs` so the registry can resolve hybrid's
    # faction (S-02) — without `gs`, hybrid-with-wolf-master falls
    # back to GOOD and WOLF-faction skills are unreachable.
    outputs = registry.dispatch_for_role(
        player.role, task_type, skill_input, task_type=task_type, gs=gs,
    )

    # NEW-S16-A: the dead-code block that computed `wolf_role` (and
    # the `for role_key in ("fake_seer", "pusher", "hooker", ...)`
    # loop scanning wolf_team_plan) was removed. S-16 moved the
    # wolf-role skip into the handler (bold_claim, deep_hook,
    # swing_vote) — context.py must not re-implement it. The variable
    # was computed but never read, so it is removed entirely.

    parts: list[str] = []

    # P1-K3: do NOT drop on `confidence < 0.4`. Low-confidence output is
    # often negative-signal advice ("don't do X", "avoid Y") that is
    # still useful. E.g., bold_claim emits `你不需要悍跳` with
    # confidence=0.3 when a teammate is already assigned as fake_seer.
    # Sort by confidence descending so the highest-confidence advice
    # appears first in the rendered prompt — within the same budget,
    # the LLM sees the best signal first; actionable low-confidence
    # advice (e.g. "your teammate already handles X") remains reachable.
    # NEW-S04-B: each sortable entry carries the originating
    # SkillOutput object as the third element. Two different skills
    # can produce identical prompt_injectable strings (e.g. S-06
    # truncation); the previous prompt-based lookup returned the
    # first match for both, dropping the second skill. Identifying
    # the *specific* output lets us key the dedupe on object
    # identity (skill_name on the source output) instead of the
    # truncated prompt.
    sortable: list[tuple[float, str, Any]] = []

    # S-04: collect per-skill output keyed by skill_name. Each entry
    # is the skill's prompt_injectable (or empty string if the skill
    # didn't produce advice). The dict is non-empty whenever at
    # least one skill fires — the contract that downstream
    # `AgentContext.skill_analyses` depends on.
    skill_analyses: dict[str, str] = {}

    for o in outputs:
        if not o.prompt_injectable:
            # Still record the skill in the analyses dict so callers
            # can see which skills were considered but produced no
            # advice (confidence=0.0, empty prompt).
            skill_analyses.setdefault(o.skill_name, o.prompt_injectable or "")
            continue
        # S-16: wolf-role skip is the handler's responsibility, not
        # context.py's.  Handlers (bold_claim, deep_hook, swing_vote)
        # already emit role-neutral / low-confidence skip prompts for
        # wolves that aren't assigned to that skill.  Re-implementing
        # the skip here risks drift between the two copies.
        skill_analyses[o.skill_name] = o.prompt_injectable
        sortable.append((o.confidence, o.prompt_injectable, o))

    # Sort highest confidence first; stable for ties.
    sortable.sort(key=lambda x: -x[0])
    # S-07: skill_tactical_advice is a structured list of
    # {skill, advice, confidence} dicts — not an opaque joined
    # string.  Sibling directive keys (must_address_alerts,
    # role_alerts) are already structured lists; advice should
    # match that contract.  The prompt builder renders the list
    # into the user prompt block.
    structured: list[dict[str, Any]] = []
    seen: set[str] = set()
    # S-19: post-step — drop any advice entry that names a player_id
    # outside `legal_targets`.  Handlers may recommend players who are
    # now dead or otherwise unavailable; surfacing that advice would
    # confuse the LLM into an illegal action.
    legal_set = set(legal_targets or [])
    # NEW-S19-A: legal_targets typically excludes dead players (it
    # is the *action* target set — you can't vote for or shoot a
    # dead player). But for analysis skills (last_words, review) we
    # WANT to mention dead players (e.g. "p05的遗言：..."). Skip the
    # S-19 filter for skills whose `applicable_phases` includes
    # `last_words` or `review` — and use a widened analysis set
    # that includes all known players (alive + dead) for them.
    from werewolf_agent.skills.schemas import SkillName as _SkillName
    # Build a set of skill names whose applicable_phases includes
    # `last_words` or `review` — these are exempt from S-19.
    # NEW-R4-P2-8: the actual enum value for REVIEW_CORRECTION is
    # the real string `review_correct`. The underscored form
    # was dead code — never matched any enum value.
    _analysis_exempt_skills: set[str] = set()
    for _sn in _SkillName:
        if _sn.value in ("last_words", "review_correct"):
            # last_words + review_correct are exempt
            _analysis_exempt_skills.add(_sn.value)
    # NEW-S19-A: for the exempt skills, build a wider
    # `legal_targets_for_analysis` that includes dead players.
    # That way, if a non-exempt skill's advice mentions a dead
    # player by name (rare but possible), the S-19 check below can
    # still see them. For the exempt skills themselves, we simply
    # SKIP the S-19 check entirely.
    analysis_legal_set = set(legal_set)
    for _pid, _p in gs.players.items():
        analysis_legal_set.add(_pid)
    # Build the structured list from the skill_analyses dict (which
    # is keyed by skill name) plus the original outputs' confidence
    # and skill_name.  We iterate in confidence-sorted order so
    # high-confidence advice appears first.
    for conf, prompt, source_output in sortable:
        # NEW-S04-B: identify the *specific* output by reading
        # skill_name directly from the source object (no prompt
        # lookup — that was the source of the dedupe collision).
        skill_name = source_output.skill_name if source_output else ""
        # NEW-S04-B: key the `seen` dedupe by skill_name (not by
        # the full prompt string).
        if skill_name in seen:
            continue
        seen.add(skill_name)
        # S-19: filter entries that reference illegal targets.
        # NEW-S19-A: skip for analysis skills (last_words, review).
        if (
            legal_set
            and prompt
            and skill_name not in _analysis_exempt_skills
        ):
            # P2-10: widened regex to catch all player-ID variants
            # observed in skill prompts:
            #   - ``p05``        → \bp\d+\b (lowercase)
            #   - ``P10``        → \bP\d+\b (uppercase, single digit ok)
            #   - ``10号玩家``   → \d+号玩家
            #   - ``玩家 10``    → 玩家\s*\d+
            # Pre-fix the single ``p\d{2}`` regex missed 3 of 4 forms,
            # causing advice that mentioned dead players (e.g. via
            # Chinese-numbered references) to slip through S-19 and
            # be injected into the prompt.  Each variant is parsed
            # into a ``p\d+`` canonical form before the legal-set
            # check.
            import re as _re
            mentioned: set[str] = set()
            for m in _re.finditer(r"\b[pP](\d+)\b", prompt):
                mentioned.add(f"p{int(m.group(1)):02d}")
            for m in _re.finditer(r"(\d+)\s*号\s*玩家?", prompt):
                mentioned.add(f"p{int(m.group(1)):02d}")
            for m in _re.finditer(r"玩家\s*(\d+)", prompt):
                mentioned.add(f"p{int(m.group(1)):02d}")
            illegal = mentioned - legal_set
            if illegal:
                # Drop this entry — it recommends an illegal target.
                continue
        structured.append({
            "skill": skill_name,
            "advice": prompt,
            "confidence": conf,
        })
    if structured:
        strategy_directive["skill_tactical_advice"] = structured
    return strategy_directive, skill_analyses


def _merge_strategy_directive(
    context: Any,
    new_directive: dict[str, Any],
) -> Any:
    """Merge new directive into existing context strategy_directive, preserving skill_tactical_advice.

    D-9: enforce a hard cap on the rendered size of the directive
    dict.  The LLM context window is finite, and an unbounded
    ``strategy_directive`` has caused OOMs in long games where the
    accumulated round-specific blocks (vote pressure, day summaries,
    sheriff election records, etc.) grew past ~10k chars.

    The cap is approximate (we count the joined string length of
    ``str(v)`` for each value) and ``_MAX_STRATEGY_DIRECTIVE_TOKENS``
    is interpreted as "characters" to avoid pulling in a tokenizer
    dependency.  When the merged directive exceeds the cap, the
    oldest round-specific blocks are dropped first (by insertion
    order of the existing context).  Round-specific keys we
    recognize as droppable live in ``_ROUND_SPECIFIC_DROP_KEYS``;
    everything else is considered structural and is kept.
    """
    existing = context.strategy_directive or {}
    merged: dict[str, Any] = {**existing, **new_directive}
    merged = _cap_strategy_directive(merged)
    return context.model_copy(update={"strategy_directive": merged})


# D-9: hard cap on the strategy_directive payload, expressed as an
# approximate token count.  Conservative: 1 token ≈ 1.5 chars for
# Chinese / 4 chars for ASCII — we use a flat 2 chars/token estimate.
_MAX_STRATEGY_DIRECTIVE_TOKENS = 1500
# Round-specific blocks that can be dropped first when the cap is
# exceeded.  Ordering matters: earlier entries are dropped first.
_ROUND_SPECIFIC_DROP_KEYS: tuple[str, ...] = (
    "sheriff_election_record",
    "day_discussion_summary",
    "vote_pressure_context",
    "vote_pressure",
    "vote_history",
    "skill_tactical_advice",
    "role_alerts",
    "must_address_alerts",
    "death_cause_evaluation",
    "witch_death_cause_evaluations",
    "wolf_fake_seer_teammate",
    "wolf_teammate_exposed",
    "belief_state",
    "must_address",
)


def _directive_size(directive: dict[str, Any]) -> int:
    """Approximate token count for a strategy_directive payload."""
    total = 0
    for v in directive.values():
        try:
            total += len(str(v))
        except Exception:
            continue
    return total // 2  # rough chars→tokens


def _cap_strategy_directive(
    directive: dict[str, Any],
    cap_tokens: int = _MAX_STRATEGY_DIRECTIVE_TOKENS,
) -> dict[str, Any]:
    """Drop oldest round-specific blocks until the cap fits.

    Preserves the structural / role-critical keys (the role
    directive, the speech strategy text, the vote contract, etc.).
    """
    if _directive_size(directive) <= cap_tokens:
        return directive
    # Walk round-specific keys in drop-priority order; remove
    # them one at a time until the cap fits.
    for key in _ROUND_SPECIFIC_DROP_KEYS:
        if _directive_size(directive) <= cap_tokens:
            break
        if key in directive:
            directive = {k: v for k, v in directive.items() if k != key}
    return directive


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
    # MEM-NEW-8: build_private_memory now returns a tuple
    # ``(memory, caveat)`` — the caveat is no longer a meta key in
    # the memory dict, so no ``pop()`` is needed. The schema is
    # uniform: memory values are category lists, caveat is a
    # top-level string.
    private_memory, private_memory_caveat = build_private_memory(gs, player_id)
    if private_memory:
        visible["private_memory"] = private_memory
    private_memory_hints = private_memory or {}

    # P3-1: the static role-specific private fields (wolf_teammates /
    # check_results / antidote_available / poison_available / master_id)
    # are now injected inside ``build_visible_player_state(role=...)``
    # above, with a whitelist projection.  The inline role branches
    # that used to live here have been deleted — see the
    # defense-in-depth whitelist in ``visible_state.py``.
    #
    # The strategy_directive still needs role-specific entries that
    # carry imperative text (witch poison deterrent, etc.).  These
    # are NOT private state for the LLM; they are prompt-side
    # behavioral guidance.  Keep the witch poison_deterrent
    # branch here.
    strategy_directive: dict[str, Any] = {}
    if (
        player.role == "witch"
        and not gs.poison_used
        and gs.phase == "day"
    ):
        alive = sum(1 for p in gs.players.values() if p.alive)
        if alive <= 8:
            strategy_directive["witch_poison_deterrent"] = (
                "你的毒药还未使用。如果场上有人持续踩你、试图把你放逐出局，"
                "你可以在发言中暗示自己有底牌——'我手里还有东西没用，不要太冲动'。"
                "狼人听到这种暗示可能会退缩。但不要明报身份。"
            )
    # Per-night transient fields (still inline — they're not static
    # role state, they depend on a specific GameState.event payload
    # that context.py has access to via wolf_kill_target_id).  These
    # are private to the witch only and bypass the public-fields
    # whitelist in ``build_visible_player_state``; if a future change
    # wants to surface them through the slim path, the whitelist
    # there should be updated.
    if player.role == "witch" and wolf_kill_target_id:
        visible["wolf_kill_target"] = wolf_kill_target_id
    if player.role == "witch" and not gs.poison_used:
        pressure_targets = _build_witch_pressure_targets(gs)
        if pressure_targets:
            visible["poison_pressure_targets"] = pressure_targets

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

    # Hybrid knows the master id, but never the master's hidden faction.
    if player.role == "hybrid" and gs.hybrid_master_id:
        master = gs.players.get(gs.hybrid_master_id)
        if master and not master.alive:
            strategy_directive["hybrid_master_dead"] = (
                f"你的主人{gs.hybrid_master_id}已死亡。"
                "你的胜利绑定仍按主人的原始阵营结算，但你仍不知道主人的阵营。"
                "继续根据主人的公开行为和场上信息独立判断。"
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
        contradiction_engine = ContradictionEngine(
            role_capacities=engine.ruleset.raw.get("role_distribution"),
        )
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
            # Phase 1 self-audit (P1-1 revert): the legacy
            # ``strategy_directive["directive"] = "你必须在发言中回应..."``
            # text was deleted.  ``must_address_alerts`` already
            # conveys the imperative (the MUST sub-group framing
            # makes the binding explicit).  The duplicate natural-
            # language imperative was redundant.
            strategy_directive["must_address_alerts"] = must_address
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
    profile_memory_hint: dict[str, Any] = {}
    reflection_memory_hints: list[dict[str, Any]] = []
    cognition_matrix_hint: dict[str, Any] = {}
    error_pattern_hint: dict[str, Any] = {}
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
                    # MEM-NEW-3: use the canonical _player_faction
                    # helper instead of a duplicated ternary. The
                    # inline version was functionally correct for the
                    # current role set, but the two WILL drift if a
                    # new role is added or MemoryStore's role sets
                    # change. A single source of truth is much easier
                    # to keep aligned.
                    from werewolf_agent.memory.store import MemoryStore
                    current_faction = MemoryStore._player_faction(
                        current_role,
                        master_faction=None,
                    )
                    reflection_memory_hints = _reflection_memory_hints(
                        all_refs, current_role, current_faction
                    )
                    # reflect-cross-1: 跨局错误模式聚合 (top 2 错误类别 + 保留优点段)。
                    # 不调 LLM,纯 section header regex 解析 + 频率统计。
                    error_pattern_hint = _compute_error_pattern(
                        all_refs, current_role
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
        private_memory_caveat=private_memory_caveat,
        reflection_memory_hints=reflection_memory_hints,
        profile_memory_hint=profile_memory_hint,
        cognition_matrix_hint=cognition_matrix_hint,
        error_pattern_hint=error_pattern_hint,
        recent_transcript=transcript,
        contradiction_alerts=ctx_alerts,
        belief_state=belief_dict,
        strategy_directive=strategy_directive,
        skill_analyses=skill_analyses,
        # NEW-S04-A: skill_analysis_hints is no longer populated. The
        # single source of truth is strategy_directive.skill_tactical_advice
        # (rendered inside the strategy_directive section). The old
        # dual-render path passed the same opaque dict to BOTH
        # skill_analyses AND skill_analysis_hints, doubling the token
        # budget. Now only the structured path remains.
        skill_analysis_hints={},
    )
    return _inject_seed_rag_hints(
        context,
        ruleset_id=gs.ruleset_id,
        rag_service=rag_service,
        game_id=gs.game_id,
        n_alive=sum(1 for p in gs.players.values() if p.alive),
    )
