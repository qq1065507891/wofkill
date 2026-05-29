"""Seer day-speech directive builder."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_seer_directive(
    gs: GameState,
    seer_id: str,
) -> dict[str, Any]:
    """Build structured day speech directives for the seer."""
    from werewolf_agent.runtime.strategy.seer import public_seer_claimants as _public_seer_claimants

    parts: dict[str, Any] = {}

    # Collect check results this seer has obtained
    # seer_check 事件不包含 seer_id，此函数仅由预言家调用，所有结果均属于该预言家
    check_results: list[dict[str, Any]] = []
    for e in gs.events:
        if e.type == "seer_check":
            check_results.append({
                "target": e.payload["target_id"],
                "alignment": e.payload["alignment"],
                "night": e.payload["night_number"],
            })

    # Determine which results have been publicly reported
    reported: set[str] = set()
    for e in gs.events:
        if e.type not in ("speech", "sheriff_speech"):
            continue
        if e.payload.get("speaker") != seer_id:
            continue
        text = str(e.payload.get("text", ""))
        for cr in check_results:
            if cr["target"] in text:
                reported.add(f"N{cr['night']}:{cr['target']}")

    unreported = [
        cr for cr in check_results
        if f"N{cr['night']}:{cr['target']}" not in reported
    ]

    # Counterclaim context
    counterclaiming_seers = _public_seer_claimants(gs) - {seer_id}

    # Build reporting guidance
    reporting_parts: list[str] = [
        "你是预言家。你的白天发言需要传递验人信息，带领好人阵营。核心原则：",
    ]

    if unreported:
        wolf_checks = [cr for cr in unreported if cr["alignment"] == "wolf"]
        good_checks = [cr for cr in unreported if cr["alignment"] == "good"]

        if wolf_checks:
            wc = wolf_checks[0]
            reporting_parts.append(
                f"【查杀未报】你在N{wc['night']}验出 {wc['target']} 是狼人，"
                "这个查杀必须在本轮发言中报出！查杀是你的最强武器。"
            )
        if good_checks:
            gc = good_checks[0]
            reporting_parts.append(
                f"【金水未报】你在N{gc['night']}验出 {gc['target']} 是好人。"
                "可以选择在发言中报出金水（增加好人阵营信息），"
                "但不必一次全部报出——保留部分验人信息可以作为后续发言的证据。"
            )

        parts["unreported_checks"] = [
            {"target": cr["target"], "alignment": cr["alignment"], "night": cr["night"]}
            for cr in unreported
        ]

    if counterclaiming_seers:
        reporting_parts.append(
            f"【对跳局面】有玩家对跳预言家: {sorted(counterclaiming_seers)}。"
            "你必须坚定立场，用你的验人信息和逻辑链证明自己才是真预言家："
            "1) 报出你的验人结果和验人逻辑链；"
            "2) 分析对跳预言家的发言漏洞；"
            "3) 强调你的警徽流是否被遵守。"
        )
    else:
        reporting_parts.append(
            "场上没有对跳预言家，你的身份可信度很高。"
            "集中传递验人信息，归票推狼。"
        )

    reporting_parts.append(
        "\n报验人的标准格式：'我在第X夜验了[玩家]，结果是[好人/狼人]。'"
    )
    reporting_parts.append(
        "注意：混血儿验出是'好人'，但可能属于狼人阵营，注意这个盲区。"
    )

    parts["seer_speech_directive"] = "\n".join(reporting_parts)

    # Include all check results for reference
    if check_results:
        parts["my_check_history"] = [
            {"target": cr["target"], "alignment": cr["alignment"],
             "night": cr["night"], "reported": f"N{cr['night']}:{cr['target']}" in reported}
            for cr in check_results
        ]

    return parts
