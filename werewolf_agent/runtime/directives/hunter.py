"""Hunter day-speech directive builder."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_hunter_directive(
    gs: GameState,
    hunter_id: str,
) -> dict[str, Any]:
    """Build day speech directive for the hunter.

    D-4: the return type is now ``dict[str, Any]`` for symmetry with the
    other role-directive builders in this package.  The directive text is
    exposed under the ``hunter_speech_directive`` key.

    Backward-compat: callers that index ``result["hunter_speech_directive"]``
    continue to work; the older "bare str" API was a one-off and is no
    longer used (the adapter now goes through ``_merge_strategy_directive``).
    """
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
        text = (
            "你是猎人，且你的身份已经公开。\n"
            "身份公开后的策略：\n"
            "1) 利用'我有枪'的威慑力，给狼人施加压力\n"
            "2) 明确表达你的怀疑和站边，让狼人忌惮开枪带走他们\n"
            "3) 不要虚张声势说你会带走某人——如果你被毒杀将无法开枪\n"
            "4) 如果预言家已死，你可以主动承担信息整理和归票的职责\n"
            "5) 【开枪前硬约束 P0-G3223805846-6】临死开枪必须基于"
            "**≥ 2 处独立公开证据**指向同一目标（预言家报查杀 + 至少 1 处"
            "公开站边/票型/发言矛盾），单点孤证时倾向 no_action。"
            "如果场上只剩 1-2 个疑似目标且证据都不足，宁可不开枪也不要乱开——"
            "乱开带走好人对好人阵营是灾难。\n"
            "6) 【遗言一致性】你的放逐遗言与开枪行为必须一致：\n"
            "   - 如果你明确说'我会带走 p0X'，系统会按 p0X 开枪\n"
            "   - 如果你说'我选择不开枪'或未指定目标，系统走 no_action\n"
            "   - 不要在遗言里说'必须开枪'但实际找不到目标——这种 prompt 会让系统误判"
        )
    else:
        text = (
            "你是猎人，但你的身份尚未公开。\n"
            "猎人发言策略（核心：隐藏身份）：\n"
            "1) 不要暴露自己是猎人！狼人知道你是猎人后会避免刀你、改让女巫毒杀来禁枪\n"
            "2) 像普通村民一样发言，参与讨论、表达站边、分析逻辑\n"
            "3) 注意观察谁发言矛盾、站边模糊——这些是你未来可能的枪击目标\n"
            "4) 如果预言家验了你且报了金水，可以帮预言家站边增强好人阵营凝聚力\n"
            "5) 不需要刻意低调到完全沉默，正常参与讨论即可\n"
            "6) 【开枪前硬约束 P0-G3223805846-6】临死开枪必须基于"
            "**≥ 2 处独立公开证据**指向同一目标，孤证时倾向 no_action。"
        )

    return {"hunter_speech_directive": text}
