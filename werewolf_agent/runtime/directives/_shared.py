"""Shared helper utilities used by multiple role directive builders."""

from __future__ import annotations

import logging
import re
from typing import Any

from werewolf_agent.core.models import GameState

logger = logging.getLogger(__name__)

# M3-2: strict regex for resolution_batch "day_N" form.  ``fullmatch``
# means the entire string must be exactly ``day_<digits>`` -- this
# rejects "day_", "day_BAD", "day_1_extra", "night_4", and "" without
# any try/except plumbing.  See review I-3.
_DAY_BATCH_RE = re.compile(r"day_(\d+)")


def collect_public_vote_history(
    gs: GameState,
    current_day: int | None = None,
) -> str:
    """Collect public vote history for villager analysis.

    M3-2: optional ``current_day`` filter.  When supplied, only
    events with ``payload.day_number <= current_day`` are
    included.  Default ``None`` (no filter) preserves the
    pre-fix behavior for back-compat callers.  By day 5 the
    LLM was reading all 5 days of vote history and losing
    focus on the current game state; this filter lets
    directives cap the helper to "up to current day" without
    changing the helper signature.
    """
    lines: list[str] = []
    for e in gs.events:
        if e.type != "vote_resolved":
            continue
        day = e.payload.get("day_number", "?")
        if (
            current_day is not None
            and isinstance(day, int)
            and day > current_day
        ):
            continue
        exiled = e.payload.get("exiled")
        tied = e.payload.get("tied", [])
        votes = e.payload.get("votes", [])
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


def collect_death_order(
    gs: GameState,
    current_day: int | None = None,
) -> str:
    """Collect public death order for villager analysis.

    Only exile and hunter_shot reasons are public knowledge.
    wolf_kill and witch_poison are indistinguishable to players -- both are night deaths.

    M3-2: optional ``current_day`` filter.  When supplied, only
    deaths whose ``resolution_batch`` parses to ``day_N`` with
    ``N <= current_day`` are included.  Night deaths are never
    public (the ``_public_reasons`` map excludes them) and are
    therefore unaffected by the filter; the filter is purely a
    "show only the day deaths up through this day" cap.
    """
    _public_reasons = {"exile": "放逐", "hunter_shot": "枪杀"}
    lines: list[str] = []
    for d in gs.deaths:
        if current_day is not None:
            batch = d.resolution_batch or ""
            # resolution_batch format: "day_N" or "night_N".
            # ``_DAY_BATCH_RE.fullmatch`` only matches the strict
            # ``day_<digits>`` form; anything else (empty string,
            # ``night_4``, ``day_BAD``, ``day_1_extra``) is treated
            # as malformed and the death is kept defensively.
            m = _DAY_BATCH_RE.fullmatch(batch)
            if m and int(m.group(1)) > current_day:
                continue
            if not m and batch.startswith("day_"):
                # Batch LOOKS like a day batch but didn't match the
                # strict ``day_<digits>`` form -- this is a real
                # engine regression signal (e.g. ``day_BAD``,
                # ``day_1_extra``).  ``night_4`` and ``""`` are
                # structurally valid and are not logged.  See review
                # I-1.
                logger.warning(
                    "collect_death_order: malformed resolution_batch %r, keeping death %s",
                    d.resolution_batch,
                    d.player_id,
                )
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
        "1) 如有公开查杀或预言家对跳，先核验预言家可信度、验人链和前后逻辑，"
        "再决定是否投查杀对象；\n"
        "2) 如无可信查杀，基于发言矛盾、票型和站边链独立归票；\n"
        "3) 在证据接近时说明取舍，避免无理由跟票。"
    )
    return parts
