"""Hybrid day-speech directive builder."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_hybrid_directive(
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
