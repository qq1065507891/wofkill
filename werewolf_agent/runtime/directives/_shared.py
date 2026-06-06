"""Shared helper utilities used by multiple role directive builders."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def collect_public_vote_history(gs: GameState) -> str:
    """Collect public vote history for villager analysis."""
    lines: list[str] = []
    for e in gs.events:
        if e.type != "vote_resolved":
            continue
        exiled = e.payload.get("exiled")
        tied = e.payload.get("tied", [])
        votes = e.payload.get("votes", [])
        day = e.payload.get("day_number", "?")
        if exiled:
            # votes is a list of {"voter": ..., "target": ..., "reason": ...}
            supporters = [
                v.get("voter", "") for v in votes
                if isinstance(v, dict) and v.get("target") == exiled
            ]
            lines.append(f"D{day}: {exiled}被放逐（投TA的: {', '.join(supporters)}）")
        elif tied:
            lines.append(f"D{day}: 平票PK {', '.join(tied)}，无人出局")
    if not lines:
        return ""
    return "\n".join(lines)


def collect_death_order(gs: GameState) -> str:
    """Collect public death order for villager analysis.

    Only exile and hunter_shot reasons are public knowledge.
    wolf_kill and witch_poison are indistinguishable to players -- both are night deaths.
    """
    _public_reasons = {"exile": "放逐", "hunter_shot": "枪杀"}
    lines: list[str] = []
    for d in gs.deaths:
        label = _public_reasons.get(d.reason)
        if label:
            lines.append(f"{d.player_id}({label})")
        else:
            lines.append(d.player_id)
    if not lines:
        return ""
    return " → ".join(lines)


def build_sheriff_silent_directive(
    gs: GameState,
    sheriff_id: str | None,
    badge_state: str,
) -> dict[str, Any]:
    """Build the no-active-sheriff vote directive.

    P0-G3223805846-9: when the badge has been torn (or otherwise lost)
    there is no 归票人 (vote-pusher) in the game.  Without explicit
    guidance the LLM tends to fall back on personal whim, or to
    "follow the loudest voice", both of which are easy for the wolf
    team to exploit by simply being the loudest faction.

    This directive injects a 归票 hint that tells the model to:
      1) follow the publicly confirmed 查杀 side (jumped/fake-seer
         hunter) if one is on the table;
      2) otherwise follow a player who has demonstrated a clear
         站边 (side-taking) logic, not just loudness;
      3) avoid voting without evidence on D1/D2 — the first two
         days should be information-gathering.

    The dict key ``no_sheriff_vote_hint`` is intentionally distinct
    from the existing ``sheriff_silent`` key in ``agent_adapter.py``,
    which is reserved for the *silenced-but-alive* sheriff case.
    Naming the two directives differently prevents the two distinct
    no-归票 scenarios from being conflated by the LLM (and by the
    test suite guarding them).
    """
    parts: dict[str, Any] = {}
    if badge_state != "torn" or sheriff_id is not None:
        # Active sheriff or pre-game setup — this directive is a
        # no-op.  Callers should gate on the same condition used
        # in agent_adapter.py (``gs.sheriff_id is None and
        # gs.sheriff_badge_state == "torn"``).
        return parts
    parts["no_sheriff_vote_hint"] = (
        "【无警长归票提示 P0-G3223805846-9】本局警徽已流失，无警长归票人。"
        "投票建议：\n"
        "1) 如有公开查杀/悍跳预言家，**跟随查杀方**归票；\n"
        "2) 如无明确查杀，**跟随有明确站边逻辑的玩家**归票——"
        "依据其发言中的验人链/逻辑分析，而非单纯声音大；\n"
        "3) 避免在 D1/D2 无依据投人——首两日优先收集信息，"
        "避免被狼队带节奏投出好身份"
    )
    return parts
