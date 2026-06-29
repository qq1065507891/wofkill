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
            "- 已经被放逐出局，身份公开证明为好人\n"
            "- 可以发表被放逐时的遗言\n"
            "- 遗言后不能再参与常规发言\n"
            "- 已经失去投票权，不能参与投票或接警徽\n\n"
            "遗言策略：\n"
            "1) 用最后一次发言整理关键公开信息：预言家验人、投票数据、逻辑矛盾\n"
            "2) 明确留下你最怀疑和最信任的位置，但必须给出证据\n"
            "3) 不要虚张声势说你有什么特殊信息——你只是已自证的好人\n"
            "4) 发言要短而清楚，帮助仍在场的好人继续推进"
            f"{witch_saved_note}{analysis_sections}"
        )
    else:
        parts["idiot_speech_directive"] = (
            "你是白痴，但尚未翻牌。你的特殊规则：\n"
            "- 如果被投票放逐，你会翻牌自证好人身份\n"
            "- 你可以发表遗言，随后出局\n"
            "- 出局后你不能再参与常规发言或投票\n\n"
            "翻牌前策略（核心：避免被投）：\n"
            "1) 发言保持温和理性，不要太激进或攻击性太强\n"
            "2) 有理有据地表达观点，但避免成为焦点\n"
            "3) 不要站边太极端——容易被反推\n"
            "4) 不要冒充任何角色\n"
            "5) 如果有人攻击你，冷静回应而非激烈对抗\n"
            "6) 不要盲从警长或近乎一边倒的归票；先说明你自己的证据链，"
            "避免被狼队包装成简单抗推位\n"
            "7) 翻牌能证明你是好人，但会让你出局，不是免费收益；"
            "能不翻牌就不要用翻牌换低质量发言空间"
            f"{analysis_sections}"
        )

    return parts
