"""Hybrid day-speech directive builder."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_hybrid_directive(
    gs: GameState,
    hybrid_id: str,
) -> dict[str, Any]:
    """Build day speech directive for the hybrid.

    The hybrid's day-speech prompt is faction-aware: once a master is
    chosen and ``gs.hybrid_master_faction`` is populated, this builder
    injects a faction-specific block so the hybrid's behavior aligns
    with its master's original faction.

    P0-I2: pre-fix the directive was identical for the wolf-master
    and good-master cases (both fell through to a neutral
    villager-style block).  Hybrid p04 in g_3528592081 (master=p01
    wolf) consequently voted like a villager instead of subtly
    supporting the wolf team.
    """
    parts: dict[str, Any] = {}

    # Core rules reminder (always present, faction-agnostic framing).
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

    # Faction-aware block (P0-I2).  Only present when the hybrid has
    # already chosen a master and the master's faction is known.
    master_id = gs.hybrid_master_id
    master_faction = gs.hybrid_master_faction
    if master_id and master_faction in ("werewolf", "good"):
        if master_faction == "werewolf":
            # Wolf-master case: nudge the hybrid to act as a hidden
            # ally of the wolf team without breaking cover.
            parts["hybrid_wolf_master_directive"] = (
                f"你的主人 {master_id} 属于狼人阵营。"
                "你的暗中任务：作为隐藏的狼队盟友，配合狼队的整体节奏。\n"
                "隐藏盟友发言原则：\n"
                "1) 绝对隐藏你的盟友关系——你表现得像独立判断的普通好人，"
                "不要让任何人（包括真正的狼队友）知道你选择了主人\n"
                "2) 在投票时，倾向与主人一致的方向，但不要每轮都跟——"
                "偶尔的偏离能维持你'独立分析'的好人形象\n"
                "3) 保护主人的身份：如果有人公开指控主人，用独立逻辑质疑，"
                "但不要过于用力（避免反推）\n"
                "4) 如果主人被推/被毒，不要在发言中表现出明显情绪波动\n"
                "5) 关键时刻（4-5人残局）可以突然带节奏推好人，配合狼队收尾\n"
                "6) 严禁使用狼人视角的词汇（'我们狼人'、'刀了谁'等）\n"
                "7) 【跟随规则 P0-G3223805846-7】投票时机：\n"
                "   a) 主人存活且未被公开质疑（PK / 多人冲票）→ 本轮投票跟随主人\n"
                "   b) 主人被 PK 或被 ≥3 人公开质疑 → 独立判断，但避免明显反向（投主队）\n"
                "   c) 主人已死亡 → 按主人原始阵营的好人逻辑行事（因为你的胜利条件跟着 master 阵营）"
            )
        else:
            # Good-master case: nudge the hybrid to actively help
            # the good team (e.g. by siding with the seer, calling
            # out wolf patterns, etc.).
            parts["hybrid_good_master_directive"] = (
                f"你的主人 {master_id} 属于好人阵营。"
                "你的暗中任务：作为隐藏的好人盟友，主动帮助好人阵营。\n"
                "好人盟友发言原则：\n"
                "1) 绝对隐藏你的盟友关系——你表现得像独立判断的普通好人，"
                "不要让任何人知道你选择了主人\n"
                "2) 主动为好人阵营提供逻辑分析——整理场上矛盾、质疑可疑票型、"
                "支持可信预言家\n"
                "3) 如果主人被狼队怀疑，公开为主人站台——"
                "用'独立分析'的视角支持TA，不要暴露你们的关系\n"
                "4) 投票时跟主人的方向一致，但保留独立判断的形象\n"
                "5) 不要主动暴露你的混血儿身份——保留到最后关键时刻可作为信息炸弹\n"
                "6) 你的胜利条件是好人阵营赢；用分析而非技能帮助阵营\n"
                "7) 【跟随规则 P0-G3223805846-7】投票时机：\n"
                "   a) 主人存活且未被公开质疑（PK / 多人冲票）→ 本轮投票跟随主人\n"
                "   b) 主人被 PK 或被 ≥3 人公开质疑 → 独立判断，但避免明显反向（投主队）\n"
                "   c) 主人已死亡 → 主人阵营的胜利条件不变；继续按 good logic 行事"
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
