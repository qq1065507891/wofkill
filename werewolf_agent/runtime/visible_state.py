"""Shared player-visible runtime state builders."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.public_ledger import build_public_ledger
from werewolf_agent.runtime.timeline import (
    TIMELINE_ORDER_NOTE,
    build_timeline_facts,
    current_phase_label,
    phase_label,
)

# P3-1: defense-in-depth whitelist.  Pre-fix, ``build_visible_player_state``
# returned only public fields but the role-specific injection
# (``wolf_teammates`` / ``check_results`` / ``antidote_available`` /
# ``wolf_kill_target`` / ``master_id`` / etc.) lived inline in
# ``context.py:build_agent_context`` (lines ~914-948).  Any future
# caller that bypassed ``build_agent_context`` and called
# ``build_visible_player_state`` directly would silently miss the
# role-specific fields — AND any caller that called it and then
# added its own role-specific fields had to remember to filter them
# before passing the dict to a renderer.  Now: the role-specific
# injection lives INSIDE ``build_visible_player_state(role=...)``,
# with a strict default whitelist that drops every key not in
# ``_VISIBLE_PLAYER_PUBLIC_FIELDS``.  The default-whitelist projection
# is a last line of defense: even if a future change adds a private
# key to the returned dict, the whitelist strips it before render.
_VISIBLE_PLAYER_PUBLIC_FIELDS: frozenset[str] = frozenset({
    "phase", "day", "night", "phase_label",
    "timeline_note", "timeline_facts",
    "alive_players", "dead_players",
    "sheriff_id", "badge_state", "sheriff_candidates",
    "public_ledger",
})


def build_visible_player_state(
    game_state: GameState,
    role: str | None = None,
    *,
    player_id: str | None = None,
    wolf_team_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build player-visible state with role-aware private fields.

    Without a ``role``, returns ONLY public fields (whitelisted).
    With a ``role``, additionally returns role-specific private
    fields (e.g. wolf sees ``wolf_teammates``, seer sees
    ``check_results``, witch sees ``antidote_available``).  The
    whitelist projection at the end of the function guarantees
    no private key leaks even if a future change accidentally
    adds one to the returned dict.

    P3-1: the role-specific injection used to live inline in
    ``context.py:build_agent_context`` (~lines 914-948), which was
    a security fragility (any future caller could skip it).  This
    fix consolidates role injection into this single function and
    enforces the public/private split with a whitelist projection.
    """
    deaths = list(game_state.deaths)
    # Only reveal deaths after the judge has publicly announced them.
    # During the sheriff election on day 1, deaths are already recorded in
    # game_state.deaths but have NOT been announced yet — players must not
    # see them prematurely (e.g. in their election speeches).
    death_announced = any(
        e.type == "judge_broadcast" and e.payload.get("phase") == "death_announce"
        for e in game_state.events
    )
    if not death_announced:
        # Keep only deaths that were announced in a prior day (exile, hunter shot).
        # Night deaths are never visible until the first death_announce broadcast.
        deaths = [d for d in deaths if d.timing != "night"]

    visible: dict[str, Any] = {
        "phase": game_state.phase,
        "day": game_state.day_number,
        "night": game_state.night_number,
        "phase_label": current_phase_label(
            game_state.phase,
            day_number=game_state.day_number,
            night_number=game_state.night_number,
        ),
        "timeline_note": TIMELINE_ORDER_NOTE,
        "timeline_facts": build_timeline_facts(
            game_state.phase,
            day_number=game_state.day_number,
            night_number=game_state.night_number,
        ),
        "alive_players": [
            pid for pid, player in game_state.players.items() if player.alive
        ],
        "dead_players": [
            {"id": death.player_id, "reason": death.reason}
            for death in deaths
        ],
        "sheriff_id": _effective_sheriff_id(game_state),
        "badge_state": _effective_badge_state(game_state),
        "sheriff_candidates": list(game_state.sheriff_candidates),
        "public_ledger": _compact_public_ledger(build_public_ledger(game_state)),
    }

    # P3-1: role-specific private fields.  All produce role-gated
    # private info; the caller must pass the matching role+id.
    if role == "werewolf" and player_id:
        visible["wolf_teammates"] = [
            pid for pid, p in game_state.players.items()
            if p.alive and p.role == "werewolf" and pid != player_id
        ]
        if wolf_team_plan:
            visible["wolf_team_plan"] = wolf_team_plan
    elif role == "seer":
        # Seer's own check results (seer_check events have no seer_id;
        # all results are the player's own).
        check_results = []
        for e in game_state.events:
            if e.type == "seer_check":
                check_results.append({
                    "target_id": e.payload["target_id"],
                    "alignment": e.payload["alignment"],
                    "night_number": e.payload["night_number"],
                })
        visible["check_results"] = check_results
    elif role == "witch":
        visible["antidote_available"] = not game_state.antidote_used
        visible["poison_available"] = not game_state.poison_used
    elif role == "hybrid" and player_id:
        visible["master_id"] = game_state.hybrid_master_id

    # P3-1: enforce whitelist projection.  Any key not in the public
    # set is dropped (it must have been added by a future change that
    # forgot to update the whitelist).  The role-specific keys added
    # above (wolf_teammates, check_results, etc.) are NOT in the
    # whitelist — they are passed through to the renderer separately
    # by context.py and never rendered through the slim
    # ``_build_visible_state`` filter.  This last-line defense guards
    # against any future addition that silently adds a private key.
    return {k: v for k, v in visible.items() if k in _VISIBLE_PLAYER_PUBLIC_FIELDS or k in {
        "wolf_teammates", "wolf_team_plan", "check_results",
        "antidote_available", "poison_available", "master_id",
    }}


def _effective_sheriff_id(game_state: GameState) -> str | None:
    """Return sheriff_id only if the sheriff is still alive.

    When the sheriff has died but badge transfer has not yet executed,
    game_state still holds the dead player's id — hide it so agents
    don't reference a dead player as '在场'.
    """
    sid = game_state.sheriff_id
    if not sid:
        return None
    player = game_state.players.get(sid)
    if player and not player.alive:
        return None
    return sid


def _effective_badge_state(game_state: GameState) -> str | None:
    """Return badge state consistent with _effective_sheriff_id."""
    if _effective_sheriff_id(game_state) is None:
        return None
    return game_state.sheriff_badge_state


def _compact_public_ledger(
    ledger: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {key: value for key, value in ledger.items() if value}


def build_post_game_summary(
    game_state: GameState,
    player_id: str,
) -> dict[str, Any]:
    """Build a *post-game review* visible-world-state for REFLECTION.

    PR2: ``build_visible_player_state`` returns the live in-game board
    (alive players, current day/night, role-specific private fields). That
    is correct for SPEECH/VOTE but wrong for post-game reflection: the
    LLM sees a "live analyst" board and writes in-game decisions instead
    of a retrospective. This builder returns a *retrospective* summary:

    - viewer's own role + faction + whether they survived to the end;
    - the winning faction (``game_state.winning_faction``);
    - the final death/exile order (public result only — no identities);
    - viewer's own action timeline (extracted from ``game_state.events``:
      their votes and, if the viewer is the seer, seer checks), chronological.
      Witch poison / hunter shot / speech 暂未提取(事件 payload 缺 actor
      归属,无法稳定判定是否为 viewer 自己的行动),留后续改进。

    Visibility-safe: only public results + the viewer's OWN actions.
    Never includes other players' private identities, wolf teammates, or
    anyone else's private actions. Returns an empty dict if the viewer
    is unknown.
    """
    viewer = game_state.players.get(player_id)
    if viewer is None:
        return {}

    deaths = [
        {
            "player_id": d.player_id,
            "reason": d.reason,
            "timing": d.timing,
            "batch": d.resolution_batch,
        }
        for d in game_state.deaths
    ]

    timeline = _extract_viewer_action_timeline(game_state, player_id)

    return {
        "game_phase": "post_game",
        "winning_faction": game_state.winning_faction,
        "viewer_role": viewer.role,
        "viewer_faction": viewer.faction,
        "viewer_survived": viewer.alive,
        "deaths": deaths,
        "my_action_timeline": timeline,
    }


def _extract_viewer_action_timeline(
    game_state: GameState,
    player_id: str,
) -> list[dict[str, Any]]:
    """Extract the viewer's OWN actions from events, chronological.

    Only public/own-action event types are captured — this is the
    retrospective 'what did I do' view, never another player's private
    action. Returns events in the order they appear in
    ``game_state.events`` (which is append-order = chronological).
    """
    timeline: list[dict[str, Any]] = []
    viewer = game_state.players.get(player_id)
    viewer_role = viewer.role if viewer else None
    for e in game_state.events:
        p = e.payload or {}
        if e.type == "vote_resolved":
            # votes is a list of {voter, target, reason}. Skip abstain /
            # tie entries where target is None/empty — there is nothing
            # actionable to record.
            for vote in p.get("votes", []) or []:
                if (isinstance(vote, dict)
                        and vote.get("voter") == player_id
                        and vote.get("target")):
                    timeline.append({
                        "kind": "vote",
                        "day": p.get("day_number"),
                        "target": vote.get("target"),
                    })
        elif e.type == "seer_check" and viewer_role == "seer":
            # seer_check events carry NO seer_id (rule_engine H-5: omitted
            # on purpose to avoid leaking seer identity). Mirror the live
            # path (visible_state.py role=="seer" gate) — only the seer
            # viewer collects check results. A non-seer viewer (villager /
            # wolf) MUST NOT see another player's seer checks, or it leaks
            # the checked player's true alignment.
            timeline.append({
                "kind": "seer_check",
                "night": p.get("night_number"),
                "target": p.get("target_id"),
                "alignment": p.get("alignment"),
            })
    return timeline


def build_public_summary(game_state: GameState) -> str:
    """Build a compact phase summary for contexts without event replay."""
    parts: list[str] = [TIMELINE_ORDER_NOTE]
    if game_state.day_number > 0:
        parts.append(phase_label("day", game_state.day_number))
    if game_state.night_number > 0:
        parts.append(phase_label("night", game_state.night_number))
    alive = sum(1 for player in game_state.players.values() if player.alive)
    parts.append(f"存活 {alive} 人")
    # Use _effective_sheriff_id so a dead sheriff is hidden (the raw
    # game_state.sheriff_id may still point at a dead player if badge
    # transfer has not yet executed — design doc §visibility).
    eff_sheriff = _effective_sheriff_id(game_state)
    if eff_sheriff:
        parts.append(f"警长: {eff_sheriff}")
    return "。".join(parts) + "。"
