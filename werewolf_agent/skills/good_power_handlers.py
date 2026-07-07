# -*- coding: utf-8 -*-
"""
实现找神和保护神职类技能 handler。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.skills.good_power_handlers import find_power_handler
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.skills.advice_frames import _cap_prompt_injectable
from werewolf_agent.skills.schemas import SkillDefinition, SkillInput, SkillName, SkillOutput
from werewolf_agent.skills.skill_context import _vote_targets_for_player
from werewolf_agent.skills.skill_handler_registry import register_handler

@register_handler(SkillName.FIND_POWER)
def find_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # NEW-R4-P2-7: when no game_state is provided, the handler
        # has no signals to analyze. The previous fallback used
        # abstract "系统性分析" advice that gave the LLM no
        # concrete next step. Replace with an explicit "wait" —
        # the dynamic branch is the real value-add; the fallback
        # is a placeholder.
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["分析发言信息量", "观察投票倾向", "识别保护行为"],
            confidence=0.5,
            reasoning="找神需要综合多个信号源进行推断",
            prompt_injectable=_cap_prompt_injectable(
                "找神建议：当前信息不足，等待关键发言出现后再下判断。"
                "重点关注信息量异常的玩家、保守的投票倾向，以及对特定玩家的保护行为。"
            ),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    # S-11: include `idiot` in power_roles. A revealed idiot is a
    # confirmed good player; protecting them keeps a vote-loss but
    # living role in play. Excluding idiot meant the post-exile idiot
    # (公开白露光后) was a free kill for wolves — the protection
    # skill was not flagging them as at-risk.
    power_roles = {"seer", "witch", "hunter", "idiot"}
    candidates: list[dict[str, Any]] = []

    if bs is not None:
        for pid, belief in bs.beliefs.items():
            # NEW-S19-D: skip dead players. A dead player with high
            # role probability would land in candidates and trip the
            # S-19 filter downstream. Mirror the wolf_pit belief-state
            # loop's alive guard.
            player = gs.players.get(pid)
            if not player or not player.alive:
                continue
            for role, prob in belief.role_probabilities.items():
                if role in power_roles and prob > 0.3:
                    candidates.append({
                        "player": pid,
                        "role": role,
                        "probability": prob,
                        "source": "belief",
                    })

    if ws is not None:
        for f in ws.facts_of_type("badge_flow_claim"):
            if f.source_player:
                candidates.append({
                    "player": f.source_player,
                    "role": "seer",
                    "probability": 0.6,
                    "source": "badge_flow",
                })

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in sorted(candidates, key=lambda x: x["probability"], reverse=True):
        key = f"{c['player']}_{c['role']}"
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if not unique:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="暂无足够信号推断神职",
            prompt_injectable=_cap_prompt_injectable("找神分析：当前信息不足，建议继续观察发言信息量和投票模式。"),
        )

    lines = []
    for c in unique[:3]:
        lines.append(f"{c['player']} 大概率是 {c['role']}（依据：{c['source']}，置信{c['probability']:.0%}）")
    prompt = f"找神分析：{'；'.join(lines)}"

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["分析发言信息量", "观察投票倾向", "识别保护行为"],
        confidence=0.5 + min(0.2, len(unique) * 0.05),
        reasoning=f"动态分析：识别到{len(unique)}个疑似神职信号",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )

@register_handler(SkillName.PROTECT_POWER)
def protect_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["暗示关键角色需要保护", "引导怀疑方向远离神职", "分散狼队注意力"],
            risk_alerts=["过度保护某个玩家反而暴露其身份"],
            confidence=0.5,
            reasoning="保护强神需要隐蔽的引导而非明显的保护行为",
            prompt_injectable=_cap_prompt_injectable("保护强神建议：如果推测某玩家是神职且被推，用'我觉得他的逻辑没问题'等方式引导怀疑方向远离，不要直接说'保护他'。"),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    # S-11: include `idiot` in power_roles (see comment above in
    # find_power_handler).  Revealed idiot is a confirmed good role
    # to protect from wolf night-kill.
    power_roles = {"seer", "witch", "hunter", "idiot"}
    at_risk: list[dict[str, Any]] = []
    # NEW-R4-P2-9: also collect *candidate* power roles — players
    # whose top_role_guess is in power_roles but who currently have
    # no vote/pressure. We need this list to give the LLM concrete
    # names when the at_risk set is empty (the previous fallback
    # was a circular "继续观察" with no actionable info).
    candidates: list[tuple[str, str, float]] = []  # (pid, role, prob)

    if bs is not None:
        for pid, belief in bs.beliefs.items():
            top_role, prob = belief.top_role_guess()
            if top_role in power_roles and prob > 0.3:
                votes_on = _vote_targets_for_player(ws, pid)
                # Also check social pressure: claimed_suspect against this player
                suspect_pressure = 0
                if ws is not None:
                    for f in ws.facts_of_type("claimed_suspect"):
                        if f.target_player == pid:
                            suspect_pressure += 1
                if votes_on or suspect_pressure > 0:
                    at_risk.append({
                        "player": pid,
                        "likely_role": top_role,
                        "votes": len(votes_on),
                        "suspect_claims": suspect_pressure,
                    })
                else:
                    candidates.append((pid, top_role, prob))

    risks = ["过度保护某个玩家反而暴露其身份"]

    if at_risk:
        target = at_risk[0]
        pressure_desc = f"{target['votes']}票"
        if target["suspect_claims"] > 0:
            pressure_desc += f"、{target['suspect_claims']}次被公开怀疑"
        prompt = (
            f"保护强神建议：疑似{target['likely_role']}的 {target['player']} "
            f"正被施压（{pressure_desc}）。"
            f"建议发言引导怀疑方向远离TA：提出其他嫌疑人、质疑推票逻辑。"
            f"注意保护要隐蔽，不要让狼队察觉你在保人。"
        )
        conf = 0.6
    else:
        # NEW-R4-P2-9: when no power role is currently under pressure,
        # the previous fallback said only "继续观察" — circular and
        # useless. List the concrete candidates the handler has
        # identified so the LLM knows WHO to keep an eye on, and
        # give a concrete next-step suggestion.
        if candidates:
            cand_str = "、".join(
                f"{pid}({role},置信{prob:.0%})"
                for pid, role, prob in candidates[:3]
            )
            prompt = (
                f"保护强神建议：场上暂无被推票压力的疑似神职，"
                f"但已识别以下候选需要持续关注：{cand_str}。"
                f"建议在发言中适度认可其逻辑（'我觉得X的分析有道理'），"
                f"建立'保护性'站边，同时避免直接公开其身份。"
            )
        else:
            prompt = (
                f"保护强神建议：当前未识别到高置信度疑似神职。"
                f"继续观察重点发言，"
                f"留意今晚死亡信息以缩小下一轮的神职候选范围。"
            )
        conf = 0.45

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["引导怀疑方向远离", "提出替代嫌疑人", "隐蔽保护"],
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据神职受压情况调整保护策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )
