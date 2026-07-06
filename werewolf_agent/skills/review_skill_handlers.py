# -*- coding: utf-8 -*-
"""
实现复盘纠错技能 handler。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.skills.review_skill_handlers import review_correction_handler
"""

from __future__ import annotations

from werewolf_agent.skills.advice_frames import _cap_prompt_injectable
from werewolf_agent.skills.schemas import SkillDefinition, SkillInput, SkillName, SkillOutput
from werewolf_agent.skills.skill_handler_registry import register_handler


@register_handler(SkillName.REVIEW_CORRECTION)
def review_correction_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["回顾关键判断点", "识别错误和原因", "总结改进方向"],
            confidence=0.7,
            reasoning="复盘纠错以事实为基础，系统性地回顾决策过程",
            prompt_injectable=_cap_prompt_injectable("复盘建议：回顾每个Day的站边选择和投票决策。找出判断失误的关键节点，分析误判原因（信息不足？逻辑链断裂？被误导？），总结改进方向。"),
        )
    # NEW-R4-P1-2: werewolves need wolf-specific review. Voting OUT
    # wolves would be betraying the team, so a "missed wolf" metric
    # inverts the wolf team's actual objective. Branch on role.
    if inp.role == "werewolf":
        return _review_correction_wolf(inp, skill)
    return _review_correction_good(inp, skill)

def _review_correction_wolf(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    """Werewolf-side review. Focus on team-coordination signals that
    actually matter to a wolf — bold-claim exposure, night-kill chain
    safety, and whether teammate vote patterns leaked information.

    NEW-R4-P1-2: this branch is goal-aligned with the wolf team.
    A wolf's day-vote "correctness" is NOT measured by how many
    wolves it voted out (that would be betrayal); it's measured by
    how well it executed 悍跳 / 归票 / 夜杀.
    """
    gs = inp.game_state
    ws = inp.world_state
    day = gs.day_number
    my_id = inp.player_id

    # Build the wolf team roster.
    wolf_ids = [pid for pid, p in gs.players.items() if p.role == "werewolf"]
    alive_wolves = [pid for pid, p in gs.players.items() if p.alive and p.role == "werewolf"]

    # Count deaths by cause.
    deaths_by_wolf = 0
    deaths_by_exile = 0
    exiled_players: list[str] = []
    wolf_killed_players: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("player_died"):
            reason = f.value or ""
            if "wolf" in reason:
                deaths_by_wolf += 1
                if f.target_player:
                    wolf_killed_players.append(f.target_player)
            elif "exile" in reason:
                deaths_by_exile += 1
                if f.target_player:
                    exiled_players.append(f.target_player)

    # Collect this wolf's day votes (and which ones hit good vs wolf).
    my_votes: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("vote"):
            if f.source_player == my_id and f.target_player:
                my_votes.append(f.target_player)
    wolf_votes_on_good = [t for t in my_votes if t not in wolf_ids]
    wolf_votes_on_wolf = [t for t in my_votes if t in wolf_ids]

    # Check for 悍跳 exposure: a wolf who publicly claimed seer is a
    # coordination risk; if the real seer hasn't been silenced, the
    # claimer will be cross-checked.
    bold_claim_facts: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("claimed_role"):
            if f.value == "seer" and f.source_player in wolf_ids:
                bold_claim_facts.append(f.source_player)

    # Build wolf-specific advice.
    parts: list[str] = []
    parts.append(
        f"狼队复盘：Day {day}，狼队{len(wolf_ids)}人，现存活{len(alive_wolves)}人。"
    )
    parts.append(f"狼队战绩：狼刀{deaths_by_wolf}人，被放逐{deaths_by_exile}人。")
    if wolf_killed_players:
        parts.append(f"狼刀成功：{'、'.join(wolf_killed_players[:6])}。")
    if exiled_players:
        parts.append(f"被放逐玩家：{'、'.join(exiled_players[:6])}。")

    if bold_claim_facts:
        parts.append(
            f"悍跳状态：{'、'.join(bold_claim_facts)}已跳预言家。"
            "检查悍跳是否被真预言家反咬、警徽流是否被识破。"
        )
    else:
        parts.append("悍跳状态：暂无跳预言家动作，可考虑D1/D2悍跳压制真预言家。")

    if my_votes:
        if wolf_votes_on_wolf:
            parts.append(
                f"警告：你有{len(wolf_votes_on_wolf)}票投给了队友（"
                f"{'、'.join(wolf_votes_on_wolf)}）"
                "——这是反向暴露，需要立即调整站边。"
            )
        if wolf_votes_on_good:
            parts.append(
                f"归票分析：你投出的{len(wolf_votes_on_good)}票（"
                f"{'、'.join(wolf_votes_on_good[:6])}）"
                "都归到好人阵营，符合狼队归票目标。"
            )

    # Detect "piled-on" targets: 2+ teammates voting the same person
    # on the same day → tells good side the wolf team is small.
    from collections import Counter
    day_targets: list[tuple[int, str]] = []
    if ws is not None:
        for wid in wolf_ids:
            if wid == my_id:
                continue
            for f in ws.facts_of_type("vote"):
                if f.source_player == wid and f.target_player and f.day is not None:
                    day_targets.append((f.day, f.target_player))
    if day_targets:
        per_day: dict[int, list[str]] = {}
        for d, t in day_targets:
            per_day.setdefault(d, []).append(t)
        piles: list[str] = []
        for d, targets in sorted(per_day.items()):
            counts = Counter(targets)
            for tgt, c in counts.items():
                if c >= 2:
                    piles.append(f"Day{d}的{tgt}（{c}票）")
        if piles:
            parts.append(
                "票型暴露：以下目标被多名队友同天归票（"
                f"{'、'.join(piles[:3])}"
                "），可能暴露狼队人数，需要分散归票。"
            )

    # Night-kill chain safety: did we kill power roles?
    if wolf_killed_players:
        power_killed = [
            tgt for tgt in wolf_killed_players
            if gs.players.get(tgt) and gs.players[tgt].role in (
                "seer", "witch", "hunter",
            )
        ]
        if power_killed:
            parts.append(
                f"夜刀成果：已成功击杀关键神职（{'、'.join(power_killed)}），"
                "夜杀链目标正确。"
            )
        else:
            parts.append(
                "夜刀成果：未击杀神职，下次夜杀优先考虑预言家/女巫/猎人。"
            )

    parts.append(
        "狼队改进方向：检查悍跳是否暴露、票型是否被识破、夜杀目标是否包含神职。"
    )

    conf = 0.75 if bold_claim_facts or wolf_votes_on_wolf else 0.65
    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["回顾狼队战绩", "识别悍跳/夜杀/票型问题", "总结狼队改进方向"],
        confidence=conf,
        reasoning="狼队动态复盘：基于悍跳状态、夜杀链、队友票型进行复盘",
        prompt_injectable=_cap_prompt_injectable("\n".join(parts)),
    )

def _review_correction_good(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    """Good-side (non-werewolf) review. Vote accuracy against the
    wolf set is the right metric here — voting OUT wolves is the
    good team's objective.
    """
    gs = inp.game_state
    ws = inp.world_state

    winner = gs.winning_faction or "unknown"
    day = gs.day_number
    my_id = inp.player_id

    # Count deaths by cause
    deaths_by_wolf = 0
    deaths_by_exile = 0
    exiled_players: list[str] = []
    wolf_killed_players: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("player_died"):
            reason = f.value or ""
            if "wolf" in reason:
                deaths_by_wolf += 1
                if f.target_player:
                    wolf_killed_players.append(f.target_player)
            elif "exile" in reason:
                deaths_by_exile += 1
                if f.target_player:
                    exiled_players.append(f.target_player)

    # Analyze my vote accuracy: did I vote for wolves or good players?
    my_votes: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("vote"):
            if f.source_player == my_id and f.target_player:
                my_votes.append(f.target_player)

    # Check seer check accuracy if applicable
    seer_checks: list[str] = []
    if ws is not None:
        for fact_type in ("seer_check_claim",):
            for f in ws.facts_of_type(fact_type):
                if f.source_player == my_id and f.target_player:
                    val = f.value or ""
                    seer_checks.append(f"{f.target_player}={val}")

    # Build analysis
    parts: list[str] = []
    parts.append(f"复盘分析：游戏进行到Day {day}，狼刀{deaths_by_wolf}人，放逐{deaths_by_exile}人。")
    if winner != "unknown":
        parts.append(f"获胜方：{winner}。")

    correct_votes = 0
    if my_votes:
        parts.append(f"你共投出{len(my_votes)}票，投票目标：{'、'.join(my_votes[:6])}。")
        # Check if any vote hit a wolf (cross-reference with actual roles)
        wolf_ids = {pid for pid, p in gs.players.items() if p.role == "werewolf"}
        correct_votes = sum(1 for t in my_votes if t in wolf_ids)
        if correct_votes > 0:
            parts.append(f"其中{correct_votes}票命中狼人，投票准确率{correct_votes / len(my_votes):.0%}。")
        else:
            parts.append("所有投票均未命中狼人，需要反思站边和判断逻辑。")

    if seer_checks:
        parts.append(f"验人记录：{'、'.join(seer_checks[:4])}。")

    if exiled_players:
        parts.append(f"被放逐玩家：{'、'.join(exiled_players[:6])}。")
    if wolf_killed_players:
        parts.append(f"被狼刀玩家：{'、'.join(wolf_killed_players[:6])}。")

    parts.append("改进方向：检查站边选择是否正确、是否被狼人发言误导、投票链是否暴露了信息。")

    conf = 0.7
    if my_votes and correct_votes == 0:
        conf = 0.8  # High confidence in review when votes were all wrong

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["回顾关键判断点", "识别错误和原因", "总结改进方向"],
        confidence=conf,
        reasoning="动态分析：基于投票准确率和事件时间线进行复盘",
        prompt_injectable=_cap_prompt_injectable("\n".join(parts)),
    )
