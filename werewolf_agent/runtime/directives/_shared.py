"""Shared helper utilities used by multiple role directive builders."""

from __future__ import annotations

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
