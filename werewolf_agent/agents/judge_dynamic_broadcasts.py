# -*- coding: utf-8 -*-
"""
构建依赖 LLM 的法官动态公共播报。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> build_sheriff_result_broadcast(None, lambda p, t: (p, None), None, "torn").broadcast_type
    'sheriff_result'
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from werewolf_agent.agents.schemas import JudgeBroadcast
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.runtime.timeline import phase_label

logger = logging.getLogger(__name__)

PersonaInject = Callable[[str, str], tuple[str, str | None]]


def _generate_judge_text(
    model_router: ModelRouter,
    persona_inject: PersonaInject,
    *,
    prompt: str,
    task_type: str,
) -> str | None:
    """生成法官台词；空响应统一交给调用方 fallback。"""
    bounded_prompt, system_prompt = persona_inject(prompt, task_type)
    result = model_router.generate(
        agent_id="judge",
        task_type=task_type,
        prompt=bounded_prompt,
        system_prompt=system_prompt,
        jitter_seconds=(0.0, 0.0),
    )
    if result.text and result.text.strip():
        return result.text.strip()
    return None


def build_vote_calling_broadcast(
    model_router: ModelRouter | None,
    persona_inject: PersonaInject,
    voter_id: str,
    voter_name: str,
    candidates: list[str],
    position: int,
    total: int,
    day_number: int = 1,
    sheriff_weight: float = 1.0,
) -> JudgeBroadcast:
    """构建逐玩家唱票播报。"""
    label = phase_label("day", day_number)
    weight_note = f"（警长{sheriff_weight}票）" if sheriff_weight > 1.0 else ""
    fallback = f"请{voter_name}玩家投票{weight_note}，第{position}/{total}位"
    public_data = {"voter_id": voter_id}
    if model_router is not None:
        try:
            prompt = (
                f"你是狼人杀游戏的法官。现在是{label}投票阶段。\n"
                f"请用简洁的中文唱票，邀请第{position}/{total}位投票者{voter_name}投票。\n"
                f"{'该玩家是警长，拥有'+str(sheriff_weight)+'票。' if sheriff_weight > 1.0 else ''}"
                f"可投票目标：{', '.join(candidates) if candidates else '任意存活玩家'}。\n"
                f"只输出唱票台词，不要输出其他内容。"
            )
            message = _generate_judge_text(
                model_router,
                persona_inject,
                prompt=prompt,
                task_type="judge_vote_calling",
            )
            if message:
                return JudgeBroadcast(
                    broadcast_type="vote_calling",
                    message=message,
                    phase="vote",
                    day_number=day_number,
                    public_data=public_data,
                )
        except Exception:
            logger.warning("judge.broadcast_vote_calling failed", exc_info=True)
    return JudgeBroadcast(
        broadcast_type="vote_calling",
        message=fallback,
        phase="vote",
        day_number=day_number,
        public_data=public_data,
    )


def build_skill_guide_broadcast(
    model_router: ModelRouter | None,
    persona_inject: PersonaInject,
    role: str,
    player_id: str,
    player_name: str,
    available_actions: list[str],
    context_hints: dict[str, object] | None = None,
) -> JudgeBroadcast:
    """构建夜间技能引导播报。"""
    hints = context_hints or {}
    role_labels: dict[str, str] = {
        "witch": "女巫",
        "hunter": "猎人",
        "seer": "预言家",
        "idiot": "白痴",
        "hybrid": "混血儿",
        "werewolf": "狼人",
        "villager": "平民",
    }
    role_cn = role_labels.get(role, role)
    fallback = f"{role_cn}请睁眼。可用行动：{', '.join(available_actions)}。"
    public_data = {"role": role, "player_id": player_id}
    if model_router is not None:
        try:
            hints_text = ""
            if hints:
                hints_text = "当前信息：" + "；".join(
                    f"{key}: {value}" for key, value in hints.items()
                ) + "。\n"
            prompt = (
                f"你是狼人杀游戏的法官。{role_cn} {player_name} 睁眼。\n"
                f"{hints_text}"
                f"可用行动：{', '.join(available_actions)}。\n"
                f"请用叙事化的中文引导该玩家做出选择。不要替玩家做决定。\n"
                f"只输出引导台词，不要输出其他内容。"
            )
            message = _generate_judge_text(
                model_router,
                persona_inject,
                prompt=prompt,
                task_type="judge_skill_guide",
            )
            if message:
                return JudgeBroadcast(
                    broadcast_type="skill_guide",
                    message=message,
                    phase="night",
                    public_data=public_data,
                )
        except Exception:
            logger.warning("judge.guide_skill_use failed", exc_info=True)
    return JudgeBroadcast(
        broadcast_type="skill_guide",
        message=fallback,
        phase="night",
        public_data=public_data,
    )


def build_vote_tally_broadcast(
    model_router: ModelRouter | None,
    persona_inject: PersonaInject,
    tally: dict[str, float],
    player_names: dict[str, str],
    sheriff_id: str | None = None,
    sheriff_weight: float = 1.5,
    day_number: int = 1,
) -> JudgeBroadcast:
    """构建投票计票结果播报。"""
    label = phase_label("day", day_number)
    lines = []
    for pid, weight in sorted(tally.items(), key=lambda item: -item[1]):
        name = player_names.get(pid, pid)
        is_sheriff = pid == sheriff_id
        mark = f"（警长{sheriff_weight}票）" if is_sheriff else ""
        lines.append(f"  {name}: {weight}票{mark}")
    fallback = f"{label} 投票结果：\n" + "\n".join(lines) if lines else f"{label} 投票结束。"
    public_data: dict[str, str | int | float | bool] = {
        "tally_count": int(sum(tally.values())),
        "tally_top_id": max(tally.items(), key=lambda item: item[1])[0] if tally else "",
        "tally_top_votes": max(tally.values()) if tally else 0,
    }
    if model_router is not None:
        try:
            tally_text = "；".join(
                f"{player_names.get(pid, pid)} {weight}票"
                + (f"（警长{sheriff_weight}票）" if pid == sheriff_id else "")
                for pid, weight in sorted(tally.items(), key=lambda item: -item[1])
            )
            prompt = (
                f"你是狼人杀游戏的法官。{label} 投票结束。\n"
                f"得票情况：{tally_text}。\n"
                f"请用简洁的中文宣布投票结果。只输出宣布台词，不要输出其他内容。"
            )
            message = _generate_judge_text(
                model_router,
                persona_inject,
                prompt=prompt,
                task_type="judge_vote_tally",
            )
            if message:
                return JudgeBroadcast(
                    broadcast_type="vote_tally",
                    message=message,
                    phase="vote",
                    day_number=day_number,
                    public_data=public_data,
                )
        except Exception:
            logger.warning("judge.announce_vote_tally failed", exc_info=True)
    return JudgeBroadcast(
        broadcast_type="vote_tally",
        message=fallback,
        phase="vote",
        day_number=day_number,
        public_data=public_data,
    )


def build_exile_result_broadcast(
    model_router: ModelRouter | None,
    persona_inject: PersonaInject,
    exiled_player_id: str | None,
    exiled_player_name: str = "",
    reason: str = "",
    tied_player_ids: list[str] | None = None,
    day_number: int = 1,
) -> JudgeBroadcast:
    """构建放逐结果播报。"""
    label = phase_label("day", day_number)
    tied = tied_player_ids or []
    public_data: dict[str, str | int | float | bool] = {
        "exiled_player_id": exiled_player_id or "",
        "reason": reason,
        "tied_count": len(tied),
    }
    if exiled_player_id:
        name = exiled_player_name or exiled_player_id
        fallback = f"{label}：{name} 被放逐出局。"
    elif reason == "first_tie_pk":
        fallback = f"首次平票（{'、'.join(tied)}），进入PK发言。"
    elif reason == "second_tie_no_exile":
        fallback = f"再次平票，无人出局，进入夜晚。"
    else:
        fallback = f"{label} 投票结束。"

    if model_router is not None:
        try:
            prompt = _build_exile_prompt(
                label=label,
                exiled_player_id=exiled_player_id,
                exiled_player_name=exiled_player_name,
                reason=reason,
                tied=tied,
            )
            message = _generate_judge_text(
                model_router,
                persona_inject,
                prompt=prompt,
                task_type="judge_exile",
            )
            if message:
                return JudgeBroadcast(
                    broadcast_type="exile_result",
                    message=message,
                    phase="vote",
                    day_number=day_number,
                    public_data=public_data,
                )
        except Exception:
            logger.warning("judge.announce_exile_result failed", exc_info=True)
    return JudgeBroadcast(
        broadcast_type="exile_result",
        message=fallback,
        phase="vote",
        day_number=day_number,
        public_data=public_data,
    )


def _build_exile_prompt(
    *,
    label: str,
    exiled_player_id: str | None,
    exiled_player_name: str,
    reason: str,
    tied: list[str],
) -> str:
    """根据放逐分支构建带上下文的 LLM prompt。"""
    if exiled_player_id:
        name = exiled_player_name or exiled_player_id
        return (
            f"你是狼人杀游戏的法官。{label}，{name}被放逐出局。\n"
            f"请用简洁的中文宣布放逐结果。只输出宣布台词，不要输出其他内容。"
        )
    if reason == "first_tie_pk":
        return (
            f"你是狼人杀游戏的法官。{label}，投票出现平票（{'、'.join(tied)}），进入PK发言。\n"
            f"请用简洁的中文宣布平票结果。只输出宣布台词，不要输出其他内容。"
        )
    if reason == "second_tie_no_exile":
        return (
            f"你是狼人杀游戏的法官。{label}，再次平票，无人出局。\n"
            f"请用简洁的中文宣布结果。只输出宣布台词，不要输出其他内容。"
        )
    tied_str = f"平票玩家: {'、'.join(tied)}。" if tied else ""
    reason_str = f"（原因: {reason or '投票已结束'}）" if reason else "（原因: 投票已结束）"
    return (
        f"你是狼人杀游戏的法官。{label}，投票结束。"
        f"{reason_str}{tied_str}\n"
        f"请用简洁的中文宣布结果。只输出宣布台词，不要输出其他内容。"
    )


def build_sheriff_result_broadcast(
    model_router: ModelRouter | None,
    persona_inject: PersonaInject,
    sheriff_id: str | None,
    badge_state: str,
) -> JudgeBroadcast:
    """构建警长选举结果播报。"""
    if sheriff_id and badge_state == "active":
        fallback = f"选举结果：{sheriff_id} 当选警长。"
    elif badge_state == "torn":
        fallback = "警长撕掉了警徽，本局不再有警长。"
    else:
        fallback = "未产生警长。"
    public_data = {
        "sheriff_id": sheriff_id or "",
        "badge_state": badge_state,
    }

    if model_router is not None:
        try:
            outcome_str = (
                f"{sheriff_id} 当选警长" if sheriff_id and badge_state == "active"
                else ("警长撕徽" if badge_state == "torn" else "未产生警长")
            )
            prompt = (
                f"你是狼人杀游戏的法官。请宣布警长选举结果：{outcome_str}。\n"
                f"请用简洁的中文宣布结果。只输出宣布台词，不要输出其他内容。"
            )
            message = _generate_judge_text(
                model_router,
                persona_inject,
                prompt=prompt,
                task_type="judge_sheriff",
            )
            if message:
                return JudgeBroadcast(
                    broadcast_type="sheriff_result",
                    message=message,
                    phase="sheriff_election",
                    public_data=public_data,
                )
        except Exception:
            logger.warning("judge.broadcast_sheriff_result failed", exc_info=True)
    return JudgeBroadcast(
        broadcast_type="sheriff_result",
        message=fallback,
        phase="sheriff_election",
        public_data=public_data,
    )


__all__ = [
    "build_exile_result_broadcast",
    "build_sheriff_result_broadcast",
    "build_skill_guide_broadcast",
    "build_vote_calling_broadcast",
    "build_vote_tally_broadcast",
]
