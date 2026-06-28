"""Idiot day-speech directive builder."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.directives.villager import build_villager_directive


def build_idiot_directive(
    gs: GameState,
    idiot_id: str,
) -> dict[str, Any]:
    """Build day speech directive for the idiot -- context-aware before/after reveal."""
    parts: dict[str, Any] = {}
    player = gs.players.get(idiot_id)
    revealed = player.revealed_idiot if player else False

    # Reuse villager analysis framework
    villager_parts = build_villager_directive(gs, idiot_id)
    # Extract the analysis sections appended to the villager directive
    villager_text = villager_parts.get("villager_speech_directive", "")
    # Everything after the core strategy is analysis data (seer claims, votes, deaths)
    analysis_sections = ""
    for marker in ("【对跳预言家分析】", "【单边预言家】", "【投票数据参考】", "【死亡顺序】"):
        idx = villager_text.find(marker)
        if idx != -1:
            analysis_sections += "\n" + villager_text[idx:]

    # Check if idiot was saved by witch (witch_private audit means witch used antidote)
    witch_saved_note = ""
    for e in gs.events:
        if e.type == "witch_decision_audit" and e.payload.get("action_taken") == "use_antidote":
            if e.payload.get("wolf_kill_target_id") == idiot_id:
                witch_saved_note = (
                    "\n\n【注意：你被女巫救了】首夜狼人刀了你，女巫用解药救了你。"
                    "女巫知道你的好身份。你可以在发言中暗示女巫的存在——"
                    "'我第一晚就知道自己是谁'——来让狼人猜忌，但不要直接报出女巫身份。"
                )
            break

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
            f"{witch_saved_note}{analysis_sections}"
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
            "5) 如果有人攻击你，冷静回应而非激烈对抗\n"
            "6) 不要盲从警长或近乎一边倒的归票；先说明你自己的证据链，"
            "避免被狼队包装成简单抗推位\n"
            "7) 翻牌能让你存活，但会失去投票权，不是免费收益；"
            "能不翻牌就不要用翻牌换低质量发言空间"
            f"{analysis_sections}"
        )

    return parts
