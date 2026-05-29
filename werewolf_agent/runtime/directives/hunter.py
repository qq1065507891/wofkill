"""Hunter day-speech directive builder."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_hunter_directive(
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
