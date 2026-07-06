# -*- coding: utf-8 -*-
"""
提供 AgentContext 的 RAG 检索提示注入和公开发言摘要解析。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.context_rag import _rag_phase_for_task
"""

from __future__ import annotations

import logging
import re
from typing import Any

from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType

# 兼容旧监控和测试：RAG 注入原先在 runtime.context 内记录日志。
logger = logging.getLogger("werewolf_agent.runtime.context")


def _rag_phase_for_task(task_type: TaskType, phase: str) -> str:
    if task_type in (TaskType.SHERIFF_SPEECH, TaskType.SHERIFF_REGISTRATION):
        return "sheriff_speech"
    if task_type == TaskType.WOLF_DISCUSSION:
        return "night_discussion"
    if task_type == TaskType.NIGHT_ACTION:
        return "night_action"
    if task_type == TaskType.VOTE:
        return "vote"
    if task_type in (TaskType.SPEECH, TaskType.PK_SPEECH):
        return "speech"
    if task_type == TaskType.DEFENSE_SPEECH:
        return "defense_speech"
    return phase or "general"


_LEGAL_ACTION_TAGS: dict[str, tuple[str, ...]] = {
    "vote": ("vote",),
    "wolf_kill": ("werewolf", "wolf_kill"),
    "wolf_no_kill": ("werewolf", "wolf_no_kill"),
    "use_antidote": ("witch", "witch_save", "antidote"),
    "use_poison": ("witch", "witch_poison", "poison"),
    "check_alignment": ("seer", "seer_check"),
    "choose_master": ("hybrid", "hybrid_master"),
    "hunter_shot": ("hunter", "hunter_shot"),
    "self_destruct": ("werewolf", "self_destruct"),
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
    """把 legal_actions 转成去重后的标签字符串，供 RAG 检索匹配。"""

    seen: set[str] = set()
    out: list[str] = []
    for action in legal_actions:
        key = action.value if hasattr(action, "value") else str(action)
        for tag in _LEGAL_ACTION_TAGS.get(key, (key,)):
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
    return " ".join(out)


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
    if context.task_type in _RAG_SKIPPED_TASK_TYPES:
        return context
    if rag_service is None:
        return context

    try:
        phase = _rag_phase_for_task(context.task_type, context.phase)
        actions_tags = _normalize_legal_actions_to_tags(context.legal_actions)
        situation_parts = [
            f"role={context.own_role}",
            f"game_phase={context.phase}",
            f"task={context.task_type.value}",
            f"alive={n_alive}",
        ]
        if actions_tags:
            situation_parts.append(f"actions={actions_tags}")
        situation = " ".join(situation_parts)

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
        items = rag_service.hits_to_prompt_lines(hits, max_items=RAG_LIVE_PROMPT_CAP)
        if not items:
            return context
        existing = [
            item for item in context.rag_hints
            if item.get("type") != "rag_hit"
        ]
        return context.model_copy(update={"rag_hints": existing + items})
    except Exception:
        logger.warning(
            "RAG retrieval anomaly for %s (game=%s): incrementing rag_anomaly_count",
            context.agent_id, game_id, exc_info=True,
        )
        return context.model_copy(
            update={"rag_anomaly_count": context.rag_anomaly_count + 1}
        )


def _extract_suspects(text: str) -> list[str]:
    suspects: list[str] = []
    for match in re.finditer(r"(?:怀疑|标狼|狼面|定狼|抗推|出)\s*(p\d{2})", text):
        player_id = match.group(1)
        if player_id not in suspects:
            suspects.append(player_id)
    return suspects


def _extract_trusts(text: str) -> list[str]:
    trusts: list[str] = []
    for match in re.finditer(r"(?:相信|好人|保|银水|金水|认好)\s*(p\d{2})", text):
        player_id = match.group(1)
        if player_id not in trusts:
            trusts.append(player_id)
    return trusts


def _extract_role_claim(text: str) -> str | None:
    match = re.search(r"(?:我是|跳|身份是|底牌是)\s*(预言家|女巫|猎人|白痴|平民|村民|混血儿)", text)
    return match.group(1) if match else None


def _extract_vote_intent(text: str) -> str | None:
    match = re.search(r"(?:归票|票投|出|投给|投票|上票)\s*(p\d{2})", text)
    return match.group(1) if match else None


def _first_sentence(text: str, max_len: int = 60) -> str:
    for sep in ("。", "！", "？", "\n"):
        idx = text.find(sep)
        if idx > 0:
            sentence = text[:idx + 1].strip()
            return sentence[:max_len]
    return text.strip()[:max_len]
