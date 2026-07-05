# -*- coding: utf-8 -*-
"""Sheriff-related policy helpers: election, badge transfer, speech order.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
All functions are pure (read GameState, return values). No side effects.
No natural language, no LLM calls -- pure rule logic.
"""

from __future__ import annotations

import random as _random
import re

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime._stable_seed import _stable_seed


# ---------------------------------------------------------------------------
# Deterministic seed helper is now shared (P-U3).
# Imported from the leaf module ``werewolf_agent.runtime._stable_seed``
# (which ``nodes._shared`` also re-exports) so both sites see the
# same function object.  The leaf module is used because importing
# from ``nodes._shared`` would trigger the
# ``sheriff_policy → nodes.__init__ → day → sheriff_policy`` circular
# import chain.
# ---------------------------------------------------------------------------


def eligible_sheriff_voters(
    gs: GameState,
    candidates: list[str],
    withdrew: list[str] | None = None,
) -> list[str]:
    """Return alive players who did NOT go on sheriff (and didn't just withdraw).

    Players who registered as candidates and then withdrew are still NOT
    eligible voters.  Only players who never went on sheriff can vote.
    """
    withdrew_set = set(withdrew or [])
    # Anyone who was ever a candidate (even if they withdrew) is excluded
    ever_on_sheriff = set(candidates) | withdrew_set
    return [
        pid
        for pid, p in gs.players.items()
        if p.alive and pid not in ever_on_sheriff
    ]


def filter_sheriff_votes_to_eligible(
    gs: GameState,
    votes: dict[str, str],
    *,
    candidates: list[str],
    withdrew: list[str] | None = None,
) -> dict[str, str]:
    """Keep only sheriff-election votes cast by off-sheriff alive players."""
    eligible = set(eligible_sheriff_voters(gs, candidates, withdrew=withdrew))
    return {
        voter_id: target_id
        for voter_id, target_id in votes.items()
        if voter_id in eligible and target_id in candidates
    }


def is_all_players_on_sheriff(gs: GameState, candidates: list[str]) -> bool:
    """True if every alive player is a sheriff candidate."""
    candidate_set = set(candidates)
    for pid, p in gs.players.items():
        if p.alive and pid not in candidate_set:
            return False
    return True


def resolve_no_vote_sheriff_reason(
    gs: GameState,
    candidates: list[str],
    voters: list[str],
) -> str:
    """Return the reason for no sheriff election.

    Returns one of: "all_players_on_sheriff", "no_candidates", "vote_tie"
    """
    if not candidates:
        return "no_candidates"
    if is_all_players_on_sheriff(gs, candidates):
        return "all_players_on_sheriff"
    return "vote_tie"


# ---------------------------------------------------------------------------
# Speech order
# ---------------------------------------------------------------------------

def choose_no_sheriff_speech_order(gs: GameState, seed: int | None = None) -> list[str]:
    """Judge selects deterministic random speech order when no sheriff exists.

    Returns ordered list of alive player IDs.
    """
    alive = [pid for pid, p in gs.players.items() if p.alive]
    if not alive:
        return []
    if seed is None:
        seed = _stable_seed(gs.game_id, "speech_order", gs.day_number)
    rng = _random.Random(seed)
    order = list(alive)
    rng.shuffle(order)
    return order


def choose_sheriff_led_speech_order(
    gs: GameState,
    sheriff_id: str,
    focus_players: list[str] | None = None,
    direction: str = "clockwise",
) -> list[str]:
    """Sheriff controls speech order: focus players early, sheriff last for vote-pushing.

    Focus players = counterclaim seers, black-claimed players, etc.
    Remaining players by seat order.
    Sheriff always last for summary/vote-pushing.
    """
    alive = [pid for pid, p in gs.players.items() if p.alive]
    focus = [p for p in (focus_players or []) if p in alive and p != sheriff_id]
    others = [p for p in alive if p not in focus and p != sheriff_id]

    # Sort others by seat order (p01, p02, ...)
    others.sort()

    # Direction handling for non-focus players
    if direction == "counterclockwise":
        others.reverse()

    # Final order: focus players first, then others, sheriff last
    order = focus + others
    if sheriff_id in alive:
        order.append(sheriff_id)
    return order


# ---------------------------------------------------------------------------
# Sheriff candidate filter
# ---------------------------------------------------------------------------

_TEMPLATE_PATTERNS = [
    r"我这轮先把视角压到",
    r"依据是.*最近发言",
]

def filter_sheriff_candidates(
    gs: GameState,
    candidates: list[str],
    speeches: dict[str, str] | None = None,
) -> list[str]:
    """Remove candidates whose sheriff speech is too short or matches templates."""
    speeches = speeches or {}
    kept: list[str] = []
    for c in candidates:
        text = speeches.get(c, "")
        if len(text) < 30:
            continue
        if any(re.search(p, text) for p in _TEMPLATE_PATTERNS):
            continue
        kept.append(c)
    return kept
