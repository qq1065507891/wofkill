"""Werewolf day-speech and vote-strategy directive builders."""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.core.models import GameState

logger = logging.getLogger(__name__)

_WOLF_ROLE_STRATEGY = {
    "fake_seer": (
        "你是悍跳狼（假预言家）。白天发言策略：\n"
        "1) 跳预言家，报出你的'验人结果'。必须像真预言家一样有验人逻辑链\n"
        "2) **关键规则**：预言家一夜只能查验 1 人，绝对不能声称同一夜查验多人\n"
        "3) 如果场上有真预言家跳了，你必须对跳——质疑真预言家的验人逻辑和警徽流\n"
        "4) 报验人的标准格式：'我在第X夜验了[玩家]，结果是[好人/狼人]'\n"
        "5) 你可以报一个好人做金水来拉拢，或者报一个好人查杀来推人\n"
        "6) 不要暴露你的队友——你'验出'的好人可以是你的队友（假金水）\n"
        "7) 如果真预言家验出了狼（你的队友），你必须质疑该验人结果的可信度\n"
        "8) 不要慌张——对跳预言家是正常游戏行为，保持自信和逻辑连贯\n"
        "9) **严禁在公开发言/投票理由中列举真实刀口 ID 列表**（如'刀口指向 p0X'、'昨夜倒下的是 p0X'）。"
        "质疑真预言家时只能使用模糊话术（'时间线对不上'、'逻辑说不通'），"
        "若必须提到死亡名单只能引用'昨夜死亡 X'的公开广播文字。"
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


def build_wolf_directive(
    gs: GameState,
    wolf_id: str,
    wolf_team_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build day speech directive for a werewolf with role-aware strategy."""
    from werewolf_agent.runtime.strategy import (
        get_wolf_role_assignment as _get_wolf_role_assignment,
        has_publicly_claimed_seer as _has_publicly_claimed_seer,
    )
    from werewolf_agent.runtime.strategy.seer import public_seer_claimants as _public_seer_claimants

    parts: dict[str, Any] = {}

    # P0-G3223805846-2: live seer claimants from day_speech events.
    # This avoids the N2 hallucination where wolves cited p07 as
    # the seer claimant (p07 was actually a villager who never
    # claimed seer).  Source from public events so the prompt
    # reflects what actually happened on day, not stale memory.
    claimants = _public_seer_claimants(gs)
    if claimants:
        parts["wolf_live_seer_claimants"] = (
            "【实时场上信息】以下玩家已公开跳预言家："
            + ", ".join(sorted(claimants))
            + "。\n讨论/制定狼队策略时，**只参考以上已公开跳预言家的玩家**。"
            "不要凭印象或记忆把'看起来像预言家'的玩家列入。"
        )

    assignment = _get_wolf_role_assignment(wolf_team_plan, wolf_id)
    parts["wolf_speech_directive"] = _WOLF_ROLE_STRATEGY.get(
        assignment, _WOLF_ROLE_STRATEGY["unassigned"],
    )

    # Universal wolf speech constraints
    is_fake_seer = (assignment == "fake_seer")
    parts["wolf_universal_rules"] = (
        "你是狼人。以下规则对所有狼人角色通用：\n"
        "1) 绝对不要提到你的队友是狼人——队友是你的'好人朋友'\n"
        "2) 不要在发言中使用狼人视角的词汇（'我们狼人'、'刀了谁'等）\n"
        "3) 不要完美配合队友——好人间也有分歧，太过一致会暴露\n"
        "4) 如果有人指控你的队友，用独立逻辑回应而非本能保护\n"
        "5) 如果你被预言家验出狼人，你需要做出回应：质疑预言家身份、"
        "指出验人逻辑漏洞、或声称被冤枉\n"
    )
    if is_fake_seer:
        parts["wolf_universal_rules"] += (
            "6) 你是今天负责跳预言家的悍跳狼，应尽快以预言家身份起跳，"
            "不要等待或犹豫。按你的角色策略执行。"
        )
    else:
        # Check if fake_seer has already publicly claimed — if so, coordination is allowed
        fake_seer = wolf_team_plan.get("fake_seer") if wolf_team_plan else None
        fake_seer_spoke = fake_seer and _has_publicly_claimed_seer(gs, fake_seer)
        if fake_seer_spoke:
            # D-10: substitute the teammate name via plain str.replace so
            # a leftover ``{fake_seer}`` placeholder can never leak
            # through.  str.format() is brittle here because the rest of
            # ``wolf_universal_rules`` already contains Chinese
            # punctuation that could collide with the format engine
            # if any other placeholder ever appears upstream.
            teammate_clause = (
                f"6) 你的队友{fake_seer}已经在公开场合跳了预言家。"
                "你可以像普通好人一样站边TA、引用TA的查验结果——"
                "这对其他好人来说是正常的信息接收行为，不会暴露你的身份。"
                "不要表现出比普通好人更了解TA的真实身份。"
            )
            parts["wolf_universal_rules"] += teammate_clause
        else:
            parts["wolf_universal_rules"] += (
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
        if fake_seer and fake_seer == wolf_id:
            # fake_seer itself: reinforce execution directive
            if not _has_publicly_claimed_seer(gs, fake_seer):
                parts["wolf_fake_seer_execution"] = (
                    "【执行指令】你是今天负责悍跳预言家的狼人。"
                    "你必须在本次发言中跳预言家，报出验人结果和警徽流。"
                    "不要犹豫，不要等待其他队友先发言。你就是那个该站出来的人。"
                )
            else:
                parts["wolf_fake_seer_execution"] = (
                    "你已经在公开场合跳了预言家。继续维护你的预言家身份，"
                    "对质疑做出有力回应，保持验人逻辑的一致性。"
                )
        elif fake_seer and fake_seer != wolf_id:
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
                # D-11: drop the dead ``f.night`` reference and the
                # defensive ``hasattr`` shim.  Fact rows in the
                # world_state schema carry a single ``day`` attribute
                # for the round reference; if it's missing we render
                # an empty string rather than crash.
                teammate_checked.append({
                    "target": f.target_player,
                    "seer": f.source_player,
                    "night": getattr(f, "day", "") or "",
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


def build_wolf_vote_directive(
    gs: GameState,
    voter_id: str,
    wolf_team_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build vote strategy for a werewolf."""
    from werewolf_agent.runtime.strategy import get_wolf_role_assignment as _get_wolf_role_assignment

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
