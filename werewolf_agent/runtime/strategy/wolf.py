"""Wolf strategy evaluation functions."""
from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.cognition.world_state import _has_self_seer_context
from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.strategy._shared import (
    speech_is_negated as _speech_is_negated,
)

logger = logging.getLogger(__name__)


def _evaluate_wolf_kill_target_impl(
    gs: GameState,
    wolf_id: str,
    legal_targets: list[str],
) -> dict[str, Any] | None:
    """Score potential kill targets by threat level for the wolf team.

    P-v3: this is the body of the original ``evaluate_wolf_kill_target``,
    split out so the public entry point can wrap it with a manual
    dict-cache and dedupe the double-call from
    ``_build_wolf_kill_directive`` and ``_single_wolf_vote``.
    """
    if not legal_targets:
        return None

    alive_teammates = [
        w for w, p in gs.players.items()
        if p.alive and p.role == "werewolf" and w != wolf_id
    ]

    scores: dict[str, dict[str, Any]] = {}
    for pid in legal_targets:
        sig: list[str] = []
        value = 0

        # Claimed seer and produced wolf-check results — biggest threat
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            if "预言家" in text or "seer" in text.lower():
                # D-6: a denial ("我不是预言家") must not count as a
                # seer claim.  Without negation detection, wolves
                # could feign denial in a sub-speech and still get
                # flagged as the top kill priority.
                if _speech_is_negated(text):
                    continue
                sig.append("claimed_seer")
                value += 6
                break

        # 公开查杀声明中指向狼人 — 威胁评估基于公开信息
        seer_check_wolf_from_pid = False
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            _ws = build_world_state(gs)
            for f in _ws.facts_of_type("seer_check_claim"):
                if f.source_player == pid and ("wolf" in (f.value or "").lower() or "狼" in (f.value or "")):
                    seer_check_wolf_from_pid = True
                    sig.append("seer_check_wolf_reporter")
                    value += 10
                    break
        except Exception:
            logger.warning("Failed to check seer-check claims during suspect scoring", exc_info=True)
        if not seer_check_wolf_from_pid:
            # Check if player publicly reported a wolf-check in speech
            for e in gs.events:
                if e.type not in ("speech", "sheriff_speech"):
                    continue
                if e.payload.get("speaker") != pid:
                    continue
                text = str(e.payload.get("text", ""))
                if "查杀" in text or "验出狼" in text:
                    sig.append("publicly_reported_wolf_check")
                    value += 8
                    break

        # Is sheriff — leadership + vote bonus
        if gs.sheriff_id == pid and gs.sheriff_badge_state == "active":
            sig.append("is_sheriff")
            value += 8

        # Claimed power role (witch, hunter) — ability threat
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            if "女巫" in text or "猎人" in text:
                sig.append("claimed_power_role")
                value += 4
                break

        # Active analyst — speeches that pointed at wolves
        wolf_mentions = 0
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            text = str(e.payload.get("text", ""))
            for w in alive_teammates:
                if w in text and ("狼" in text or "可疑" in text):
                    wolf_mentions += 1
        if wolf_mentions >= 2:
            sig.append(f"analyst_accused_{wolf_mentions}_wolves")
            value += 5
        elif wolf_mentions == 1:
            sig.append("accused_teammate")
            value += 2

        # Ran for sheriff — potentially important role
        for e in gs.events:
            if e.type == "sheriff_registration" and e.payload.get("player_id") == pid:
                sig.append("ran_for_sheriff")
                value += 2
                break

        scores[pid] = {"value": value, "signals": sig}

    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    top_value = ranked[0][1]["value"] if ranked else 0

    return {
        "description": "击杀目标威胁评估（分数越高对狼队威胁越大，越应优先击杀）",
        "ranked_targets": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "recommendation": (
            f"建议击杀: {ranked[0][0]}（威胁分={ranked[0][1]['value']}，"
            f"信号: {', '.join(ranked[0][1]['signals']) or '无特殊信号'}）"
            if ranked and top_value > 0 else
            "无明确高威胁目标，可自由选择"
        ),
    }


# P-v3: skill-layer memoization for evaluate_wolf_kill_target.
#
# Why a manual dict (not functools.lru_cache):
#   GameState is a frozen dataclass with dict/list fields (players, events,
#   votes, …) which are unhashable.  lru_cache hashes all positional and
#   keyword args to build its key, so any attempt to pass gs through it
#   raises ``TypeError: unhashable type: 'dict'`` at the first call.
#   A manual dict keyed only on the hashable tuple
#   ``(game_id, night_number, wolf_id, legal_targets)`` is both correct
#   and O(1).
#
# Cache key:
#   ``(game_id, night_number, wolf_id, tuple(legal_targets))`` —
#   excludes ``gs`` (which is the whole point: gs.events alone can be
#   hundreds of speech events).  ``legal_targets`` is included so two
#   callers passing different subsets (e.g. a test passing 4 of 8
#   alive non-wolves) get independent cache slots.  In production both
#   call sites pass the same full list, so they collapse onto the same
#   entry.
#
# Lifecycle:
#   Unbounded by design — V1 games have at most 4 wolves × ~10 nights
#   × 2 distinct legal_targets (night-time and post-death snapshots),
#   i.e. fewer than 100 entries.  ``clear_kill_value_cache`` is the
#   public reset hook for tests and game boundaries.
_KILL_VALUE_CACHE: dict[tuple, dict[str, Any] | None] = {}


def clear_kill_value_cache() -> None:
    """P-v3: test helper to reset the kill-value-assessment cache.

    Call this between games (or between tests) so a stale cached value
    for one game_id/night_number does not leak into a subsequent game.
    """
    _KILL_VALUE_CACHE.clear()


def evaluate_wolf_kill_target(
    gs: GameState,
    wolf_id: str,
    legal_targets: list[str],
) -> dict[str, Any] | None:
    """Score potential kill targets by threat level for the wolf team.

    P-v3: this public entry is memoized.  The cache key is
    ``(game_id, night_number, wolf_id, tuple(legal_targets))`` so
    repeated calls for the same wolf in the same night (e.g. once
    from ``_build_wolf_kill_directive`` and once from
    ``_single_wolf_vote``) collapse to a single impl invocation.
    The cost goes from ``2 × 4 × N_nights`` to ``4 × N_nights`` per
    game, and the O(N) inner loop is no longer paid twice.
    """
    key = (gs.game_id, gs.night_number, wolf_id, tuple(legal_targets))
    cached = _KILL_VALUE_CACHE.get(key)
    if cached is None and key not in _KILL_VALUE_CACHE:
        cached = _evaluate_wolf_kill_target_impl(gs, wolf_id, legal_targets)
        _KILL_VALUE_CACHE[key] = cached
    return cached


def get_wolf_role_assignment(
    wolf_team_plan: dict[str, Any] | None,
    wolf_id: str,
) -> str:
    """Determine this wolf's role assignment from the team plan."""
    if not wolf_team_plan:
        return "unassigned"
    for role in ("fake_seer", "pusher", "hooker", "deep_cover"):
        if wolf_team_plan.get(role) == wolf_id:
            return role
    return "unassigned"


def has_publicly_claimed_seer(gs: GameState, player_id: str) -> bool:
    """Check if a player has publicly claimed seer in any speech event.

    D-6: negated claims ("我不是预言家" / "否认我是seer" / "我没查验")
    are NOT counted as public claims.  Pre-fix the keyword-substring
    matcher would treat "我不是预言家" as a positive claim, polluting
    the wolf kill priority list with players who explicitly denied
    the role.
    """
    for e in gs.events:
        if e.type in ("sheriff_speech", "speech") and e.payload.get("speaker") == player_id:
            text = e.payload.get("text", "")
            if _speech_is_negated(text):
                continue
            if _has_self_seer_context(text):
                return True
    return False
