"""Agent runtime adapter: converts GameState into AgentContext for PlayerAgent.

When an AgentRegistry is provided to the runtime graph, night/day nodes will
delegate decisions to PlayerAgent instances. Without a registry, deterministic
scripted fallback is used (preserving existing test behavior).
"""

from __future__ import annotations

import json
import logging
import re
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


def _load_persona_profile(persona_key: str) -> dict[str, Any]:
    global _PERSONA_PROFILES_CACHE
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
from werewolf_agent.skills.schemas import SkillName
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.timeline import (
    TIMELINE_ORDER_NOTE,
    current_phase_label,
    phase_label,
)
from werewolf_agent.runtime.private_memory import build_private_memory
from werewolf_agent.runtime.visible_state import build_visible_player_state

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
            item for item in context.salience_items
            if item.get("type") != "rag_hit"
        ]
        return context.model_copy(update={"salience_items": existing + items})
    except Exception:
        logger.debug("Seed RAG injection failed for %s", context.agent_id, exc_info=True)
        return context


def _action_trace_payload(action: Any) -> dict[str, Any] | None:
    trace = getattr(action, "trace", None)
    return trace.model_dump() if trace else None


def _estimate_witch_save_value(
    gs: GameState,
    target_id: str | None,
) -> dict[str, Any]:
    """Estimate the strategic value of saving the wolf-kill target.

    Returns structured decision data — the LLM still makes the final call.
    On N1 there is no public info, so we provide a probability framework
    and explicit trade-offs instead of a numeric score.
    On N2+ we score the target based on observable behavior.
    """
    if target_id is None:
        return {"actionable": False, "reason": "no_wolf_kill"}

    non_wolf_alive = sum(
        1 for p in gs.players.values() if p.alive and p.role != "werewolf"
    )
    power_roles_alive = sum(
        1 for p in gs.players.values()
        if p.alive and p.role in ("seer", "hunter", "idiot", "hybrid")
    )

    # N1: no public information, use probability framework
    if gs.night_number == 1 and gs.day_number == 0:
        return {
            "actionable": True,
            "night": 1,
            "public_info_available": False,
            "probability_framework": {
                "p_seer": round(1 / max(non_wolf_alive, 1), 3),
                "p_power_role": round(power_roles_alive / max(non_wolf_alive, 1), 3),
                "p_villager": round(
                    max(non_wolf_alive - power_roles_alive, 0) / max(non_wolf_alive, 1), 3
                ),
            },
            "trade_off": {
                "save_now": "首夜盲救，被杀者有约{:.0f}%概率是神职，价值较高；但也有{:.0f}%概率是普通村民".format(
                    power_roles_alive / max(non_wolf_alive, 1) * 100,
                    max(non_wolf_alive - power_roles_alive, 0) / max(non_wolf_alive, 1) * 100,
                ),
                "save_later": "保留解药，在后续有信息（警长选举、预言家验人、发言分析）时精准救人，价值更高",
                "risk_no_save": "如果不救而恰好被杀者是预言家，好人方将失去最重要的信息源",
            },
        }

    # N2+: score target based on observable public behavior
    score = 0
    signals: list[str] = []

    # Was target the sheriff or sheriff candidate?
    if gs.sheriff_id == target_id and gs.sheriff_badge_state == "active":
        score += 8
        signals.append("is_sheriff")
    for e in gs.events:
        if e.type == "sheriff_registration" and e.payload.get("player_id") == target_id:
            score += 3
            signals.append("ran_for_sheriff")
            break

    # Did target claim seer or power role in public speech?
    for e in gs.events:
        if e.type != "speech" or e.payload.get("speaker") != target_id:
            continue
        text = e.payload.get("text", "")
        if "预言家" in text or "seer" in text.lower():
            score += 6
            signals.append("claimed_seer_in_speech")
            break
        if "猎人" in text or "白痴" in text:
            score += 2
            signals.append("claimed_power_role")

    # Did anyone else confirm target as good (seer check result)?
    for e in gs.events:
        if e.type == "speech":
            text = e.payload.get("text", "")
            if target_id in text and ("金水" in text or "好人" in text):
                score += 3
                signals.append("confirmed_good_by_seer_claim")
                break

    # How many speeches did target give? (active participants are usually power roles)
    speech_count = sum(
        1 for e in gs.events
        if e.type == "speech" and e.payload.get("speaker") == target_id
    )
    if speech_count >= 2:
        score += 1
        signals.append(f"active_speaker({speech_count}_speeches)")

    return {
        "actionable": True,
        "night": gs.night_number,
        "public_info_available": True,
        "target_id": target_id,
        "save_value_score": score,
        "signals": signals,
        "interpretation": (
            f"目标公开行为分析：得分{score}分。"
            + ("高价值目标，强烈建议救人。" if score >= 6 else
              "中等价值目标，需综合判断。" if score >= 3 else
              "低价值目标，可考虑保留解药。")
        ),
    }


def _build_witch_pressure_targets(gs: GameState) -> list[dict[str, Any]]:
    """Build poison pressure targets from public state.

    Pressure sources:
    - Unresolved seer black claim (查杀 in public speech)
    - Player contradicted claimed role
    """
    import re as _re

    targets: dict[str, dict[str, Any]] = {}

    # Extract from public speeches: black claims (查杀)
    for event in gs.events:
        if event.type == "speech":
            text = event.payload.get("text", "")
            # Look for 查杀 claims targeting a player (support various ID formats)
            # Pattern: "PLAYER查杀" or "PLAYER...查杀"
            black_match = _re.search(r"([a-zA-Z]+\d+).*?查杀", text)
            if not black_match:
                # Also try reverse: "查杀...PLAYER"
                black_match = _re.search(r"查杀.*?([a-zA-Z]+\d+)", text)
            if black_match:
                target_id = black_match.group(1)
                if target_id not in targets:
                    targets[target_id] = {
                        "player_id": target_id,
                        "pressure_type": "black_claim",
                        "source": event.payload.get("speaker", ""),
                        "description": f"被{event.payload.get('speaker', '?')}查杀",
                    }

    return list(targets.values())


def _public_seer_claimants(gs: GameState) -> set[str]:
    """Return public players who have claimed seer in speeches."""
    claimants: set[str] = set()
    seer_markers = (
        "我是预言家",
        "我跳预言家",
        "认预言家",
        "悍跳预言家",
        "claim seer",
        "claimed seer",
        "i am the seer",
    )
    for event in gs.events:
        if event.type not in ("speech", "sheriff_speech"):
            continue
        speaker = event.payload.get("speaker")
        if not speaker:
            continue
        claims = event.payload.get("claims") or []
        for claim in claims:
            if claim.get("type") == "role" and claim.get("value") == "seer":
                claimants.add(speaker)
                break
        else:
            text = str(event.payload.get("text", "")).lower()
            if any(marker in text for marker in seer_markers):
                claimants.add(speaker)
    return claimants


def _evaluate_seer_check_value(
    gs: GameState,
    seer_id: str,
    legal_targets: list[str],
) -> dict[str, Any] | None:
    """Score unchecked targets by information value for the seer."""
    if not legal_targets:
        return None

    scores: dict[str, dict[str, Any]] = {}
    for pid in legal_targets:
        sig: list[str] = []
        value = 0

        # High-value: was accused of being wolf in public speech
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            if speaker == seer_id:
                continue
            if pid in text and ("狼" in text or "可疑" in text or "查杀" in text):
                sig.append(f"public_suspect_by_{speaker}")
                value += 3
                break

        # High-value: claimed a power role — verify authenticity
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            claims = e.payload.get("claims") or []
            for claim in claims:
                if claim.get("type") == "role" and claim.get("value") in (
                    "seer", "witch", "hunter",
                ):
                    sig.append(f"claimed_{claim['value']}")
                    value += 5
                    break
            if "女巫" in text or "我是猎人" in text:
                sig.append("claimed_power_in_text")
                value += 4
            break

        # Medium: ran for sheriff (potential power role or wolf)
        for e in gs.events:
            if e.type == "sheriff_registration" and e.payload.get("player_id") == pid:
                sig.append("ran_for_sheriff")
                value += 2
                break

        # Medium: unclear stance — not clearly aligned with any seer claimant
        seer_claimants = _public_seer_claimants(gs)
        if seer_claimants:
            supported_a_seer = False
            for e in gs.events:
                if e.type not in ("speech", "sheriff_speech"):
                    continue
                if e.payload.get("speaker") != pid:
                    continue
                text = str(e.payload.get("text", ""))
                if any(sc in text for sc in seer_claimants):
                    supported_a_seer = True
                    break
            if not supported_a_seer:
                sig.append("unclear_stance")
                value += 3

        # Low: active speaker (more data available for LLM to judge)
        speech_count = sum(
            1 for e in gs.events
            if e.type in ("speech", "sheriff_speech")
            and e.payload.get("speaker") == pid
        )
        if speech_count == 0:
            sig.append("silent_player")
            value += 2

        scores[pid] = {"value": value, "signals": sig}

    # Sort by value descending
    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    return {
        "description": "未验玩家的信息价值评估（分数越高越值得验）",
        "ranked_targets": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "recommendation": (
            f"建议优先查验: {ranked[0][0]}（价值分={ranked[0][1]['value']}，"
            f"信号: {', '.join(ranked[0][1]['signals']) or '无特殊信号'}）"
            if ranked else "无可用验人目标"
        ),
    }


def _build_seer_day_speech_directive(
    gs: GameState,
    seer_id: str,
) -> dict[str, Any]:
    """Build structured day speech directives for the seer."""
    parts: dict[str, Any] = {}

    # Collect check results this seer has obtained
    # seer_check 事件不包含 seer_id，此函数仅由预言家调用，所有结果均属于该预言家
    check_results: list[dict[str, Any]] = []
    for e in gs.events:
        if e.type == "seer_check":
            check_results.append({
                "target": e.payload["target_id"],
                "alignment": e.payload["alignment"],
                "night": e.payload["night_number"],
            })

    # Determine which results have been publicly reported
    reported: set[str] = set()
    for e in gs.events:
        if e.type not in ("speech", "sheriff_speech"):
            continue
        if e.payload.get("speaker") != seer_id:
            continue
        text = str(e.payload.get("text", ""))
        for cr in check_results:
            if cr["target"] in text:
                reported.add(f"N{cr['night']}:{cr['target']}")

    unreported = [
        cr for cr in check_results
        if f"N{cr['night']}:{cr['target']}" not in reported
    ]

    # Counterclaim context
    counterclaiming_seers = _public_seer_claimants(gs) - {seer_id}

    # Build reporting guidance
    reporting_parts: list[str] = [
        "你是预言家。你的白天发言需要传递验人信息，带领好人阵营。核心原则：",
    ]

    if unreported:
        wolf_checks = [cr for cr in unreported if cr["alignment"] == "wolf"]
        good_checks = [cr for cr in unreported if cr["alignment"] == "good"]

        if wolf_checks:
            wc = wolf_checks[0]
            reporting_parts.append(
                f"【查杀未报】你在N{wc['night']}验出 {wc['target']} 是狼人，"
                "这个查杀必须在本轮发言中报出！查杀是你的最强武器。"
            )
        if good_checks:
            gc = good_checks[0]
            reporting_parts.append(
                f"【金水未报】你在N{gc['night']}验出 {gc['target']} 是好人。"
                "可以选择在发言中报出金水（增加好人阵营信息），"
                "但不必一次全部报出——保留部分验人信息可以作为后续发言的证据。"
            )

        parts["unreported_checks"] = [
            {"target": cr["target"], "alignment": cr["alignment"], "night": cr["night"]}
            for cr in unreported
        ]

    if counterclaiming_seers:
        reporting_parts.append(
            f"【对跳局面】有玩家对跳预言家: {sorted(counterclaiming_seers)}。"
            "你必须坚定立场，用你的验人信息和逻辑链证明自己才是真预言家："
            "1) 报出你的验人结果和验人逻辑链；"
            "2) 分析对跳预言家的发言漏洞；"
            "3) 强调你的警徽流是否被遵守。"
        )
    else:
        reporting_parts.append(
            "场上没有对跳预言家，你的身份可信度很高。"
            "集中传递验人信息，归票推狼。"
        )

    reporting_parts.append(
        "\n报验人的标准格式：'我在第X夜验了[玩家]，结果是[好人/狼人]。'"
    )
    reporting_parts.append(
        "注意：混血儿验出是'好人'，但可能属于狼人阵营，注意这个盲区。"
    )

    parts["seer_speech_directive"] = "\n".join(reporting_parts)

    # Include all check results for reference
    if check_results:
        parts["my_check_history"] = [
            {"target": cr["target"], "alignment": cr["alignment"],
             "night": cr["night"], "reported": f"N{cr['night']}:{cr['target']}" in reported}
            for cr in check_results
        ]

    return parts


def _build_hunter_day_speech_directive(
    gs: GameState,
    hunter_id: str,
) -> str:
    """Build day speech directive for the hunter."""
    # Check if hunter identity has been publicly revealed
    identity_exposed = False
    for e in gs.events:
        if e.type not in ("speech", "sheriff_speech"):
            continue
        text = str(e.payload.get("text", ""))
        speaker = e.payload.get("speaker", "")
        if speaker == hunter_id and ("猎人" in text or "我是猎人" in text):
            identity_exposed = True
            break
        # Someone else identified the hunter
        if hunter_id in text and "猎人" in text and speaker != hunter_id:
            identity_exposed = True
            break

    if identity_exposed:
        return (
            "你是猎人，且你的身份已经公开。\n"
            "身份公开后的策略：\n"
            "1) 利用'我有枪'的威慑力，给狼人施加压力\n"
            "2) 明确表达你的怀疑和站边，让狼人忌惮开枪带走他们\n"
            "3) 不要虚张声势说你会带走某人——如果你被毒杀将无法开枪\n"
            "4) 如果预言家已死，你可以主动承担信息整理和归票的职责"
        )

    return (
        "你是猎人，但你的身份尚未公开。\n"
        "猎人发言策略（核心：隐藏身份）：\n"
        "1) 不要暴露自己是猎人！狼人知道你是猎人后会避免刀你、改让女巫毒杀来禁枪\n"
        "2) 像普通村民一样发言，参与讨论、表达站边、分析逻辑\n"
        "3) 注意观察谁发言矛盾、站边模糊——这些是你未来可能的枪击目标\n"
        "4) 如果预言家验了你且报了金水，可以帮预言家站边增强好人阵营凝聚力\n"
        "5) 不需要刻意低调到完全沉默，正常参与讨论即可"
    )


def _build_hybrid_day_speech_directive(
    gs: GameState,
    hybrid_id: str,
) -> dict[str, Any]:
    """Build day speech directive for the hybrid."""
    parts: dict[str, Any] = {}

    # Core rules reminder
    parts["hybrid_speech_directive"] = (
        "你是混血儿。你的胜利条件是跟随主人的原始阵营获胜。"
        f"你的主人是 {gs.hybrid_master_id}，你不知道主人的身份和阵营。\n\n"
        "发言核心原则：\n"
        "1) 绝对不要暴露你的混血儿身份——一旦暴露，好人阵营会怀疑你（尤其如果主人是狼），"
        "狼人也会利用你\n"
        "2) 表现得像一个普通村民——参与讨论、表达站边、分析逻辑\n"
        "3) 观察你的主人的行为：主人站哪边、投谁、发什么言——"
        "如果主人帮好人阵营，你倾向好人；如果主人帮狼人，你要暗中配合\n"
        "4) 不要刻意跟随主人的每一个观点——那会暴露你们的关系\n"
        "5) 在不暴露身份的前提下，尽量确保你的投票方向对主人阵营有利"
    )

    # Master behavior analysis (if enough days have passed)
    if gs.day_number >= 2 and gs.hybrid_master_id:
        master_speeches: list[str] = []
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != gs.hybrid_master_id:
                continue
            master_speeches.append(str(e.payload.get("text", ""))[:100])

        if master_speeches:
            parts["master_behavior_summary"] = (
                f"主人 {gs.hybrid_master_id} 的历史发言摘要（前3条）：\n"
                + "\n".join(f"  - {s}" for s in master_speeches[:3])
                + "\n\n根据这些发言，判断主人更可能属于哪个阵营，调整你的站边方向。"
            )

    return parts


def _build_villager_day_speech_directive(
    gs: GameState,
    villager_id: str,
) -> dict[str, Any]:
    """Build day speech directive for villager/idiot — pure analysis, no private info."""
    parts: dict[str, Any] = {}

    # Collect public information for analysis
    seer_claimants = _public_seer_claimants(gs)
    vote_history = _collect_public_vote_history(gs)
    death_order = _collect_death_order(gs)

    # Build seer claim analysis if there are competing claims
    seer_analysis = ""
    if len(seer_claimants) >= 2:
        seer_analysis = (
            "\n\n【对跳预言家分析】场上有多个预言家声明，你需要独立判断：\n"
            "1) 验人逻辑链：谁的验人报告与死亡、投票数据吻合？\n"
            "2) 警徽流一致性：谁在遵守自己的警徽流承诺？\n"
            "3) 发言质量：谁的发言有实质信息，谁只是在泛泛而谈？\n"
            "4) 站边分析：谁在帮好人说话，谁在帮狼人打掩护？\n"
            f"对跳预言家: {sorted(seer_claimants)}"
        )
    elif len(seer_claimants) == 1:
        seer_analysis = (
            f"\n\n【单边预言家】场上只有一个预言家声明: {sorted(seer_claimants)}，"
            "单边预言家可信度较高，但仍需关注其验人逻辑是否合理。"
        )

    # Build vote pattern analysis
    vote_analysis = ""
    if vote_history:
        vote_analysis = "\n\n【投票数据参考】" + vote_history

    # Build death order analysis
    death_analysis = ""
    if death_order:
        death_analysis = "\n\n【死亡顺序】" + death_order

    role_label = "白痴" if gs.players.get(villager_id, None) and gs.players[villager_id].role == "idiot" else "普通村民"
    parts["villager_speech_directive"] = (
        f"你是{role_label}，没有夜间技能和私有信息，你的核心价值是逻辑分析能力。\n\n"
        "发言策略：\n"
        "1) 不要复述别人的观点——提出你自己的分析和判断\n"
        "2) 引用具体的发言内容和投票数据来支撑你的论点\n"
        "3) 如果你有独立的怀疑对象，说明理由；不要无证据跟风\n"
        "4) 不要冒充任何角色——你没有信息来支撑冒充\n"
        "5) 如果预言家已死或被怀疑，好人阵营需要你站出来做逻辑整理"
        f"{seer_analysis}{vote_analysis}{death_analysis}"
    )

    return parts


def _collect_public_vote_history(gs: GameState) -> str:
    """Collect public vote history for villager analysis."""
    lines: list[str] = []
    for e in gs.events:
        if e.type != "vote_resolved":
            continue
        exiled = e.payload.get("exiled")
        tied = e.payload.get("tied", [])
        votes = e.payload.get("votes", [])
        day = e.payload.get("day_number", "?")
        if exiled:
            # votes is a list of {"voter": ..., "target": ..., "reason": ...}
            supporters = [
                v.get("voter", "") for v in votes
                if isinstance(v, dict) and v.get("target") == exiled
            ]
            lines.append(f"D{day}: {exiled}被放逐（投TA的: {', '.join(supporters)}）")
        elif tied:
            lines.append(f"D{day}: 平票PK {', '.join(tied)}，无人出局")
    if not lines:
        return ""
    return "\n".join(lines)


def _collect_death_order(gs: GameState) -> str:
    """Collect public death order for villager analysis."""
    lines: list[str] = []
    for d in gs.deaths:
        reason_label = {"wolf_kill": "狼杀", "exile": "放逐", "witch_poison": "毒杀",
                        "hunter_shot": "猎人开枪"}.get(d.reason, d.reason)
        lines.append(f"{d.player_id}({reason_label})")
    if not lines:
        return ""
    return " → ".join(lines)


def _build_idiot_day_speech_directive(
    gs: GameState,
    idiot_id: str,
) -> dict[str, Any]:
    """Build day speech directive for the idiot — context-aware before/after reveal."""
    parts: dict[str, Any] = {}
    player = gs.players.get(idiot_id)
    revealed = player.revealed_idiot if player else False

    # Reuse villager analysis framework
    villager_parts = _build_villager_day_speech_directive(gs, idiot_id)
    # Extract the analysis sections appended to the villager directive
    villager_text = villager_parts.get("villager_speech_directive", "")
    # Everything after the core strategy is analysis data (seer claims, votes, deaths)
    analysis_sections = ""
    for marker in ("【对跳预言家分析】", "【单边预言家】", "【投票数据参考】", "【死亡顺序】"):
        idx = villager_text.find(marker)
        if idx != -1:
            analysis_sections += "\n" + villager_text[idx:]

    if revealed:
        parts["idiot_speech_directive"] = (
            "你是白痴，已经翻牌亮明身份。你当前状态：\n"
            "- 仍然存活，可以发言\n"
            "- 已经失去投票权（无法参与投票）\n"
            "- 免疫放逐（不会再被投出局）\n"
            "- 唯一的死法是被狼人夜间击杀\n\n"
            "亮牌后策略：\n"
            "1) 你不怕被投票，大胆发言传递你的分析和判断\n"
            "2) 整理场上的关键信息：预言家验人、投票数据、逻辑矛盾\n"
            "3) 明确表态你怀疑谁、信任谁——你不用担心被投\n"
            "4) 不要虚张声势说你有什么特殊信息——你只是普通好人\n"
            "5) 你的发言仍然需要逻辑和证据支撑，否则存活玩家不会采信"
            f"{analysis_sections}"
        )
    else:
        parts["idiot_speech_directive"] = (
            "你是白痴，但尚未翻牌。你的特殊规则：\n"
            "- 如果被投票放逐，你会翻牌自证身份并存活\n"
            "- 但翻牌后你会失去投票权，严重削弱好人阵营的力量\n"
            "- 翻牌后你唯一的死法是被狼人夜杀\n\n"
            "翻牌前策略（核心：避免被投）：\n"
            "1) 发言保持温和理性，不要太激进或攻击性太强\n"
            "2) 有理有据地表达观点，但避免成为焦点\n"
            "3) 不要站边太极端——容易被反推\n"
            "4) 不要冒充任何角色\n"
            "5) 如果有人攻击你，冷静回应而非激烈对抗"
            f"{analysis_sections}"
        )

    return parts


def _evaluate_wolf_kill_target(
    gs: GameState,
    wolf_id: str,
    legal_targets: list[str],
) -> dict[str, Any] | None:
    """Score potential kill targets by threat level for the wolf team."""
    if not legal_targets:
        return None

    alive_teammates = [
        w for w, p in gs.players.items()
        if p.alive and p.role == "werewolf" and w != wolf_id
    ]

    scores: dict[str, dict[str, Any]] = {}
    for pid in legal_targets:
        sig: list[str] = []
        value = 0

        # Claimed seer and produced wolf-check results — biggest threat
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            if "预言家" in text or "seer" in text.lower():
                sig.append("claimed_seer")
                value += 6
                break

        # 公开查杀声明中指向狼人 — 威胁评估基于公开信息
        seer_check_wolf_from_pid = False
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            _ws = build_world_state(gs)
            for f in _ws.facts_of_type("seer_check_claim"):
                if f.source_player == pid and ("wolf" in (f.value or "").lower() or "狼" in (f.value or "")):
                    seer_check_wolf_from_pid = True
                    sig.append("seer_check_wolf_reporter")
                    value += 10
                    break
        except Exception:
            logger.warning("Failed to check seer-check claims during suspect scoring", exc_info=True)
        if not seer_check_wolf_from_pid:
            # Check if player publicly reported a wolf-check in speech
            for e in gs.events:
                if e.type not in ("speech", "sheriff_speech"):
                    continue
                if e.payload.get("speaker") != pid:
                    continue
                text = str(e.payload.get("text", ""))
                if "查杀" in text or "验出狼" in text:
                    sig.append("publicly_reported_wolf_check")
                    value += 8
                    break

        # Is sheriff — leadership + vote bonus
        if gs.sheriff_id == pid and gs.sheriff_badge_state == "active":
            sig.append("is_sheriff")
            value += 8

        # Claimed power role (witch, hunter) — ability threat
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            if "女巫" in text or "猎人" in text:
                sig.append("claimed_power_role")
                value += 4
                break

        # Active analyst — speeches that pointed at wolves
        wolf_mentions = 0
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            for w in alive_teammates:
                if w in text and ("狼" in text or "可疑" in text):
                    wolf_mentions += 1
        if wolf_mentions >= 2:
            sig.append(f"analyst_accused_{wolf_mentions}_wolves")
            value += 5
        elif wolf_mentions == 1:
            sig.append("accused_teammate")
            value += 2

        # Ran for sheriff — potentially important role
        for e in gs.events:
            if e.type == "sheriff_registration" and e.payload.get("player_id") == pid:
                sig.append("ran_for_sheriff")
                value += 2
                break

        scores[pid] = {"value": value, "signals": sig}

    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    top_value = ranked[0][1]["value"] if ranked else 0

    return {
        "description": "击杀目标威胁评估（分数越高对狼队威胁越大，越应优先击杀）",
        "ranked_targets": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "recommendation": (
            f"建议击杀: {ranked[0][0]}（威胁分={ranked[0][1]['value']}，"
            f"信号: {', '.join(ranked[0][1]['signals']) or '无特殊信号'}）"
            if ranked and top_value > 0 else
            "无明确高威胁目标，可自由选择"
        ),
    }


def _get_wolf_role_assignment(
    wolf_team_plan: dict[str, Any] | None,
    wolf_id: str,
) -> str:
    """Determine this wolf's role assignment from the team plan."""
    if not wolf_team_plan:
        return "unassigned"
    for role in ("fake_seer", "pusher", "hooker", "deep_cover"):
        if wolf_team_plan.get(role) == wolf_id:
            return role
    return "unassigned"


def _has_publicly_claimed_seer(gs: GameState, player_id: str) -> bool:
    """Check if a player has publicly claimed seer in any speech event."""
    seer_keywords = ("预言家", "查杀", "金水", "验了", "查验")
    for e in gs.events:
        if e.type in ("sheriff_speech", "speech") and e.payload.get("speaker") == player_id:
            text = e.payload.get("text", "")
            if any(kw in text for kw in seer_keywords):
                return True
    return False


_WOLF_ROLE_STRATEGY = {
    "fake_seer": (
        "你是悍跳狼（假预言家）。白天发言策略：\n"
        "1) 跳预言家，报出你的'验人结果'。必须像真预言家一样有验人逻辑链\n"
        "2) 如果场上有真预言家跳了，你必须对跳——质疑真预言家的验人逻辑和警徽流\n"
        "3) 报验人的标准格式：'我在第X夜验了[玩家]，结果是[好人/狼人]'\n"
        "4) 你可以报一个好人做金水来拉拢，或者报一个好人查杀来推人\n"
        "5) 不要暴露你的队友——你'验出'的好人可以是你的队友（假金水）\n"
        "6) 如果真预言家验出了狼（你的队友），你必须质疑该验人结果的可信度\n"
        "7) 不要慌张——对跳预言家是正常游戏行为，保持自信和逻辑连贯"
    ),
    "pusher": (
        "你是冲锋狼。白天发言策略：\n"
        "1) 你是狼队的攻击手——主动带节奏、推人、制造混乱\n"
        "2) 集中火力推团队的白天目标（day_push_target），用具体的'证据'指控\n"
        "3) 可以质疑预言家的验人结果，为悍跳狼队友站台\n"
        "4) 发言要有攻击性但不要无脑——每个指控都需要'理由'\n"
        "5) 如果悍跳狼被质疑，你要主动为其辩护或转移话题\n"
        "6) 不要直接暴露和队友的配合关系——表现得像独立判断"
    ),
    "hooker": (
        "你是倒钩狼。白天发言策略：\n"
        "1) 核心任务：获取好人信任。你的价值在于'被信任后的背叛'\n"
        "2) 可以轻踩队友（质疑悍跳狼的验人、指出冲锋狼的漏洞）来换取信任\n"
        "3) 踩队友时必须用独立逻辑——'我觉得X的验人时间线不对'而不是'他是狼'\n"
        "4) 投票时可以跟好人走（投队友）来加深信任\n"
        "5) 不要太早暴露——N1/D1尽量低调，D2+再开始'独立分析'\n"
        "6) 关键时刻（4-5人残局）你可以突然跳出来带节奏推好人"
    ),
    "deep_cover": (
        "你是深水狼。白天发言策略：\n"
        "1) 核心任务：像普通村民一样存活到最后。完全隐藏身份\n"
        "2) 表现得像一个有分析能力的普通好人——参与讨论、表达站边、分析逻辑\n"
        "3) 不要太出色引人注目，也不要太沉默被怀疑\n"
        "4) 可以帮真预言家站边（如果真预言家已经暴露），增强你的好人面\n"
        "5) 不要主动为队友辩护——那会暴露你们的关系\n"
        "6) 如果队友被推，表现得'意外'并附和好人的推人逻辑\n"
        "7) 你的目标是活到最后阶段（3-4人残局），那时你的1票就能决定胜负"
    ),
    "unassigned": (
        "你是狼人，但没有特定角色分工。白天发言策略：\n"
        "1) 表现得像一个普通好人——参与讨论、表达站边\n"
        "2) 不要暴露队友，不要暴露自己\n"
        "3) 观察场上局势，配合队友的节奏\n"
        "4) 投票时注意不要和队友完全一致"
    ),
}


def _build_wolf_day_speech_directive(
    gs: GameState,
    wolf_id: str,
    wolf_team_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build day speech directive for a werewolf with role-aware strategy."""
    parts: dict[str, Any] = {}

    assignment = _get_wolf_role_assignment(wolf_team_plan, wolf_id)
    parts["wolf_speech_directive"] = _WOLF_ROLE_STRATEGY.get(
        assignment, _WOLF_ROLE_STRATEGY["unassigned"],
    )

    # Universal wolf speech constraints
    parts["wolf_universal_rules"] = (
        "你是狼人。以下规则对所有狼人角色通用：\n"
        "1) 绝对不要提到你的队友是狼人——队友是你的'好人朋友'\n"
        "2) 不要在发言中使用狼人视角的词汇（'我们狼人'、'刀了谁'等）\n"
        "3) 不要完美配合队友——好人间也有分歧，太过一致会暴露\n"
        "4) 如果有人指控你的队友，用独立逻辑回应而非本能保护\n"
        "5) 如果你被预言家验出狼人，你需要做出回应：质疑预言家身份、"
        "指出验人逻辑漏洞、或声称被冤枉\n"
        "6) 【严禁信息穿越】你不能使用你作为狼人的未来信息。"
        "如果某个队友计划跳预言家但还没发言，你不能提前站边或透露TA的身份。"
        "你必须表现得像一个不知道谁是预言家的普通好人，等TA发言后才能站边。"
    )

    # Day push target from team plan
    if wolf_team_plan:
        push_target = wolf_team_plan.get("day_push_target")
        if push_target and push_target in gs.players and gs.players[push_target].alive:
            parts["wolf_day_push_target"] = (
                f"狼队白天推人目标: {push_target}。在发言中引导其他玩家怀疑该目标，"
                "但不要直接说'投TA'——用分析和质疑的方式引导。"
            )

        # Inform about fake seer identity for coordination
        fake_seer = wolf_team_plan.get("fake_seer")
        if fake_seer and fake_seer != wolf_id:
            if _has_publicly_claimed_seer(gs, fake_seer):
                # Teammate has already spoken — coordinate normally
                parts["wolf_fake_seer_teammate"] = (
                    f"你的队友 {fake_seer} 是悍跳狼（假预言家），已公开跳预言家。"
                    "你的发言要配合TA的叙事——如果TA报了验人，你要像好人对真预言家一样回应。"
                )
                if assignment == "pusher":
                    parts["wolf_fake_seer_teammate"] += (
                        "主动为TA的验人结果站台、质疑对跳预言家。"
                    )
                elif assignment == "hooker":
                    parts["wolf_fake_seer_teammate"] += (
                        "你可以轻踩TA来获取信任，但不要太用力。"
                    )
                elif assignment == "deep_cover":
                    parts["wolf_fake_seer_teammate"] += (
                        "表现得像一个中立的好人来判断谁更像真预言家。"
                    )
            else:
                # Teammate hasn't claimed yet — strict anti-reveal constraint
                parts["wolf_fake_seer_teammate"] = (
                    f"【严禁信息穿越】你的队友计划悍跳预言家，但TA尚未在公开场合跳预言家。"
                    "在你的发言中绝不能：\n"
                    "- 站边TA的预言家身份（'我站边XX的预言家'之类）\n"
                    "- 透露TA会跳预言家\n"
                    "- 以任何方式暗示你已知道谁是预言家\n"
                    "你必须表现得像一个对场上信息不确定的普通好人。"
                    "等TA自己发言后，在后续的发言轮次中你才能像好人一样'分析站边'。"
                )

    # Counterclaim context: if a real seer has publicly checked a wolf teammate
    wolf_teammates = [
        pid for pid, p in gs.players.items()
        if p.alive and p.role == "werewolf" and pid != wolf_id
    ]
    teammate_checked = []
    # 使用公开查杀声明，不直接读取 seer_check 私有事件
    try:
        from werewolf_agent.cognition.world_state import build_world_state
        _ws = build_world_state(gs)
        for f in _ws.facts_of_type("seer_check_claim"):
            val = (f.value or "").lower()
            if ("wolf" in val or "狼" in (f.value or "")) and f.target_player in wolf_teammates:
                teammate_checked.append({
                    "target": f.target_player,
                    "seer": f.source_player,
                    "night": f.day or f.night if hasattr(f, 'night') else "",
                })
    except Exception:
        logger.warning("Failed to check teammate-exposure claims", exc_info=True)
    if teammate_checked:
        parts["wolf_teammate_exposed"] = (
            f"警告：你的队友被真预言家验出狼人了！"
            f"被验队友: {', '.join(t['target'] + '(被' + t['seer'] + '验出)' for t in teammate_checked)}。"
            "应对策略：质疑该预言家的身份和验人逻辑，或直接对跳。"
        )

    return parts


def _build_wolf_vote_strategy(
    gs: GameState,
    voter_id: str,
    wolf_team_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build vote strategy for a werewolf."""
    assignment = _get_wolf_role_assignment(wolf_team_plan, voter_id)
    parts: dict[str, Any] = {}

    # Base strategy
    parts["wolf_vote_strategy"] = (
        "你是狼人，你的投票目标是帮助狼队推走好人，同时隐藏身份。\n"
        "核心原则：\n"
        "1) 优先投狼队的推人目标\n"
        "2) 不要和所有队友投同一人——至少要有1-2人投不同目标\n"
        "3) 投票理由要像好人一样基于公开信息分析\n"
        "4) 不要在投票理由中暴露夜间信息\n"
        "5) 除非是倒钩策略需要，否则不要投你的狼人队友"
    )

    # Role-specific vote behavior
    if assignment == "hooker":
        parts["wolf_vote_role_hint"] = (
            "你是倒钩狼，投票策略：可以投你的狼人队友（如果有人推TA），"
            "这能增强你的好人面。但不要每轮都投队友。"
        )
    elif assignment == "deep_cover":
        parts["wolf_vote_role_hint"] = (
            "你是深水狼，投票策略：跟随主流好人票型投票，"
            "不要做那个唯一投不同目标的人。"
        )
    elif assignment == "fake_seer":
        parts["wolf_vote_role_hint"] = (
            "你是悍跳狼，投票策略：投票给对跳预言家或TA的支持者，"
            "强化你的'预言家'人设。"
        )
    elif assignment == "pusher":
        parts["wolf_vote_role_hint"] = (
            "你是冲锋狼，投票策略：带头投推人目标，"
            "在投票理由中用'分析'和'证据'来带动其他好人跟票。"
        )

    # Day push target
    if wolf_team_plan:
        push_target = wolf_team_plan.get("day_push_target")
        if push_target and push_target in gs.players and gs.players[push_target].alive:
            parts["wolf_vote_target"] = f"狼队推人目标: {push_target}"

    return parts


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
    from werewolf_agent.skills.registry import SkillRegistry
    from werewolf_agent.skills.schemas import SkillInput

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


# Skill names exposed as on-demand LLM tools (not injected as prompt text).
_TOOL_SKILL_NAMES: set[str] = {
    SkillName.WOLF_PIT_ANALYSIS.value,
    SkillName.FIND_POWER.value,
    SkillName.LAST_WORDS_ANALYSIS.value,
}

# Tool definitions for each on-demand skill (module-level constant).
_SKILL_TOOL_DEFS: dict[str, dict[str, Any]] = {
    SkillName.WOLF_PIT_ANALYSIS.value: {
        "name": "skill_analyze_wolf_pit",
        "description": (
            "分析当前狼坑（嫌疑区和排除区）。基于验人信息、投票模式、发言矛盾等给出完整分析。"
            "调用后会返回详细分析结果供你参考。"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    SkillName.FIND_POWER.value: {
        "name": "skill_find_power_roles",
        "description": (
            "分析场上哪些玩家可能是神职（预言家、女巫、猎人等）。"
            "基于发言、行为模式和已知信息推断。"
            "调用后返回推测结果供你参考。"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    SkillName.LAST_WORDS_ANALYSIS.value: {
        "name": "skill_analyze_last_words",
        "description": (
            "分析刚出局玩家的遗言。提取关键信息、判断可信度、与已知信息对比。"
            "调用后返回分析结果供你参考。"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
}


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

    # Role-specific private info
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

    # Build public summary: chronological timeline of key public events.
    # Strategy to stay within context budget (~2500 chars):
    #   - Structural events (deaths, votes, exiles): keep in full (compact)
    #   - Previous-day speech text is intentionally not replayed. Players keep
    #     only their own private_memory notes across days.
    #   - If total exceeds SUMMARY_BUDGET, drop oldest lines first
    SUMMARY_BUDGET = 2500
    summary_parts: list[str] = []
    for e in gs.events:
        if e.type == "day_announce":
            day = e.payload.get("day", "?")
            try:
                day_label = phase_label("day", int(day))
            except (TypeError, ValueError):
                day_label = f"D{day}"
            summary_parts.append(f"\n===== {day_label} =====")
        elif e.type == "judge_broadcast" and e.payload.get("visibility") == "public":
            msg = e.payload.get("message", "")
            phase = e.payload.get("phase", "")
            if phase in ("death_announce", "exile", "vote_result_announce",
                         "vote_tie_pk", "vote_second_tie",
                         "sheriff_elected", "sheriff_no_election"):
                summary_parts.append(f"[法官] {msg}")
        elif e.type == "vote_resolved":
            exiled = e.payload.get("exiled")
            reason = e.payload.get("reason", "")
            tied = e.payload.get("tied", [])
            if exiled:
                summary_parts.append(f"[投票结果] {exiled} 被放逐")
            elif reason == "second_tie_no_exile":
                summary_parts.append("[投票结果] 二次平票，无人出局")
            elif tied:
                summary_parts.append(f"[投票结果] 平票PK: {', '.join(tied)}")
        elif e.type == "idiot_reveal":
            summary_parts.append(f"[白痴亮牌] {e.payload.get('player_id', '?')} 亮出白痴身份")
        elif e.type == "hunter_shot_public":
            summary_parts.append(f"[猎人开枪] 猎人带走了 {e.payload.get('target_id', '?')}")

    # Trim from front to stay within budget
    total = sum(len(p) for p in summary_parts)
    while total > SUMMARY_BUDGET and len(summary_parts) > 1:
        dropped = summary_parts.pop(0)
        total -= len(dropped)
    public_summary = "\n".join(summary_parts) if summary_parts else ""
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
    strategy_directive: dict[str, Any] = {}
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
            strategy_directive = {
                "must_address_alerts": must_address,
                "directive": "你必须在发言中回应以下矛盾：选择站队、质疑、或明确表示暂不判断。",
            }
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
        logger.debug("Role state monitoring failed, skipping", exc_info=True)

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
        recent_transcript=transcript,
        contradiction_alerts=ctx_alerts,
        belief_state=belief_dict,
        strategy_directive=strategy_directive,
        skill_tools=skill_tools,
        skill_analyses=skill_analyses,
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
    if wolf_kill_target_id and not gs.antidote_used and ActionType.USE_ANTIDOTE in legal_actions:
        can_self = wolf_kill_target_id != witch_id
        save_hint = f"（他被狼人杀害了）" if can_self else "（但是你不能自救！）"
        options.append(
            f"1) 使用解药救{wolf_kill_target_id}{save_hint} —— action_type='use_antidote', target_id='{wolf_kill_target_id}'"
        )
    if not gs.poison_used and ActionType.USE_POISON in legal_actions:
        options.append(
            "2) 使用毒药毒杀某人 —— action_type='use_poison', target_id='目标玩家ID'"
        )
    options.append("3) 不使用药水 —— action_type='no_action'")
    witch_directive["witch_night_action"] += "\n".join(options)
    witch_directive["witch_night_action"] += (
        "\n\n重要规则：不能在同一夜同时使用解药和毒药。"
        "解药不能自救。"
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
        alive = sum(1 for p in gs.players.values() if p.alive)
        if alive <= 8:
            witch_directive["poison_urgency"] = (
                f"场上仅存活{alive}人。你的毒药还没有使用。"
                f"你必须认真考虑今晚撒毒——选择你最有把握的狼人目标。"
                f"如果你被刀或被投出局，毒药将浪费。"
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
        return {"wolf_action": "kill", "wolf_kill_target_id": action.target_id, "action_trace": action_trace}
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
            snippet = s["text"][:80] + ("..." if len(s["text"]) > 80 else "")
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
    )
    if strategy_directive:
        context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    target = action.target_id if action.action_type == ActionType.VOTE else None
    # Fallback: if agent returned wrong action type but has legal targets,
    # pick an evidence-aware target rather than abstaining silently.
    if target is None and legal_targets:
        from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target

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


def _evaluate_hybrid_master_candidates(
    gs: GameState,
    hybrid_id: str,
    candidates: list[str],
) -> dict[str, Any]:
    """Evaluate master candidates for the hybrid on N1.

    The hybrid doesn't know any player's role, so scoring is based on
    observable public signals from the sheriff election and speeches.
    """
    scores: dict[str, dict[str, Any]] = {}
    for pid in candidates:
        sig: list[str] = []
        value = 0

        # Registered for sheriff election — likely power role or confident player
        for e in gs.events:
            if e.type == "sheriff_registration" and e.payload.get("player_id") == pid:
                sig.append("ran_for_sheriff")
                value += 3
                break

        # Gave a substantive sheriff speech — engaged and likely experienced
        for e in gs.events:
            if e.type == "sheriff_speech" and e.payload.get("speaker") == pid:
                text = str(e.payload.get("text", ""))
                if len(text) > 50:
                    sig.append("substantive_sheriff_speech")
                    value += 2
                break

        # Claimed a power role in speech — could be real or fake, but either way influential
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            claims = e.payload.get("claims") or []
            for claim in claims:
                if claim.get("type") == "role":
                    sig.append(f"claimed_{claim['value']}")
                    value += 4
                    break
            break

        # Position-based: prefer players in middle positions (less likely to be
        # first-night wolf targets in a positional meta)
        scores[pid] = {"value": value, "signals": sig}

    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    total_candidates = len(candidates)
    # Probability breakdown for the hybrid
    god_count = 4  # seer + witch + hunter + idiot
    wolf_count = 4
    villager_count = 3

    return {
        "description": "主人候选评估（分数越高，玩家影响力越大，对混血儿越有价值）",
        "probability_framework": {
            "p_good_faction": f"~{(god_count + villager_count) / total_candidates:.0%}（7/11 好人阵营）",
            "p_wolf_faction": f"~{wolf_count / total_candidates:.0%}（4/11 狼人阵营）",
            "note": "你不知道主人阵营，选到好人和狼人的概率都有，策略需要灵活适应",
        },
        "ranked_candidates": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "strategy_guidance": (
            "选主人策略考量：\n"
            "1) 选择影响力大的玩家（上警、发言积极、声称神职）——无论主人是好人还是狼人，"
            "影响力大的主人意味着你的胜利条件更容易实现\n"
            "2) 避免选择自己——你不能选自己\n"
            "3) 不要过于纠结概率——7:4 的好人:狼人比例下，你更大概率是好人阵营，"
            "但游戏进程会告诉你主人的真正阵营\n"
            "4) 选定后无法更改，请在考虑后做出决定"
        ),
    }


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


def _evaluate_hunter_shot_target(
    gs: GameState,
    hunter_id: str,
    legal_targets: list[str],
    death_reason: str,
) -> dict[str, Any] | None:
    """Score potential shot targets by evidence strength for the hunter."""
    if not legal_targets:
        return None

    scores: dict[str, dict[str, Any]] = {}
    for pid in legal_targets:
        sig: list[str] = []
        value = 0

        # 公开查杀声明：狼人阵营是最强信号 (+10)
        # 使用 seer_check_claim 公开信息，不直接读取 seer_check 私有事件
        _wolf_check_found = False
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            _ws = build_world_state(gs)
            for f in _ws.facts_of_type("seer_check_claim"):
                val = (f.value or "").lower()
                if f.target_player == pid and ("wolf" in val or "狼" in (f.value or "")):
                    sig.append(f"seer_check_wolf_claim_{f.source_player}")
                    value += 10
                    _wolf_check_found = True
                    break
        except Exception:
            logger.warning("Failed to score wolf-kill target via seer claims", exc_info=True)

        # Counterclaiming seer: high-value target (+6)
        counterclaiming_seers = _public_seer_claimants(gs)
        if pid in counterclaiming_seers:
            sig.append("counterclaiming_seer")
            value += 6

        # Publicly accused of being wolf (+4)
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            if speaker == hunter_id or speaker == pid:
                continue
            if pid in text and ("狼" in text or "查杀" in text or "可疑" in text):
                sig.append(f"public_suspect_by_{speaker}")
                value += 4
                break

        # Voted to exile the hunter (+3)
        for e in gs.events:
            if e.type != "vote_resolved":
                continue
            voter_map = e.payload.get("votes", {})
            if voter_map.get(pid) == hunter_id:
                sig.append("voted_exile_hunter")
                value += 3
                break

        # Contradiction alerts (+3)
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            from werewolf_agent.cognition.contradiction import ContradictionEngine
            world_state = build_world_state(gs)
            engine = ContradictionEngine()
            alerts = engine.detect(world_state.facts, gs.day_number)
            for alert in alerts:
                if pid in str(alert):
                    sig.append(f"contradiction_{alert.alert_type}")
                    value += 3
                    break
        except Exception:
            logger.warning("Failed to score wolf-kill target via contradiction alerts", exc_info=True)

        # Claimed a power role (+2)
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            claims = e.payload.get("claims") or []
            for claim in claims:
                if claim.get("type") == "role" and claim.get("value") in (
                    "seer", "witch", "hunter",
                ):
                    sig.append(f"claimed_{claim['value']}")
                    value += 2
                    break
            break

        scores[pid] = {"value": value, "signals": sig}

    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    top_value = ranked[0][1]["value"] if ranked else 0
    has_seer_check_wolf = any(
        "seer_check_wolf" in s
        for _, d in ranked
        for s in d["signals"]
    )

    if has_seer_check_wolf:
        advisory = "有明确查杀目标，强烈建议开枪带走该玩家。"
    elif top_value >= 6:
        advisory = "有较高嫌疑目标，建议开枪。"
    elif top_value >= 3:
        advisory = "有一定嫌疑目标，可以开枪，但也可以选择不开枪。"
    else:
        advisory = "无明显高价值目标，建议不开枪（NO_ACTION），避免误伤好人。"

    return {
        "description": "猎人开枪目标价值评估（分数越高越值得开枪）",
        "death_reason": death_reason,
        "ranked_targets": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "recommendation": (
            f"建议开枪带走: {ranked[0][0]}（价值分={ranked[0][1]['value']}，"
            f"信号: {', '.join(ranked[0][1]['signals']) or '无特殊信号'}）"
            if ranked else "无可用开枪目标"
        ),
        "shoot_advisory": advisory,
    }


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
        shot_assessment = None

    death_label = {"wolf_kill": "被狼人袭击", "exile": "被投票放逐"}.get(
        death_reason, f"因{death_reason}"
    )
    strategy_directive: dict[str, Any] = {
        "hunter_shot_directive": (
            f"你是猎人，{death_label}导致死亡。你现在可以开枪带走一名玩家。\n"
            "开枪是一次性的：选错目标会帮狼人减少好人数量，必须谨慎。\n"
            "如果场上没有明确狼人目标，选择不开枪（NO_ACTION）比乱枪更好。\n"
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
) -> dict[str, Any] | bool:
    """Ask a player whether they want to register for sheriff election.

    Returns dict with registration result and self_destruct flag.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(player_id)
    if agent is None:
        return False

    player_role = gs.players[player_id].role if player_id in gs.players else ""
    # Build role-specific registration guidance
    if player_role == "seer":
        role_hint = (
            "你是预言家，强烈建议上警！预言家几乎必须上警留警徽流，"
            "这是预言家的核心玩法——通过警徽流传递验人信息。"
        )
    elif player_role == "werewolf":
        # Check if any wolf is already planning to claim seer
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
        rag_service=state.get("rag_service"),
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
) -> dict[str, Any] | bool:
    """Ask a sheriff candidate whether they want to withdraw.

    Returns dict with withdrawal result and self_destruct flag.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(candidate_id)
    if agent is None:
        return False

    context = build_agent_context(
        engine, gs, candidate_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SHERIFF_WITHDRAW, ActionType.NO_ACTION],
        rag_service=state.get("rag_service"),
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

    # Wolf anti-reveal: don't expose fake seer teammate before they speak
    if player_role == "werewolf":
        wolf_plan = state.get("wolf_team_plan")
        if wolf_plan and wolf_plan.get("fake_seer"):
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
        return {"reflection_text": action.speech_text or ""}
    except Exception:
        return {"reflection_text": ""}
