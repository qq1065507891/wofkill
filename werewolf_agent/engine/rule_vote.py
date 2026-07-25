# -*- coding: utf-8 -*-
"""
RuleEngine 的日间放逐投票 helper。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-25

使用示例:
    >>> from werewolf_agent.engine.rule_vote import resolve_vote
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Any

from werewolf_agent.core.models import GameState, VoteResult


def _stable_seed_val(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0xFFFFFFFF


def sheriff_vote_weight(raw: dict[str, Any], *, base_vote_weight: int) -> int:
    sheriff_weight = float(raw.get("sheriff", {}).get("vote_weight", 1.5))
    return round(sheriff_weight * base_vote_weight)


def vote_weight(
    raw: dict[str, Any],
    state: GameState,
    voter_id: str,
    *,
    base_vote_weight: int,
) -> int:
    if voter_id == state.sheriff_id and state.sheriff_badge_state == "active":
        return sheriff_vote_weight(raw, base_vote_weight=base_vote_weight)
    return base_vote_weight


def accepted_votes(
    raw: dict[str, Any],
    state: GameState,
    *,
    votes: dict[str, str],
    revote: bool,
    legal_targets: set[str] | None = None,
    pk_candidates: list[str] | None = None,
) -> dict[str, str]:
    """按当前规则返回实际进入计票的投票。"""
    resolved_legal_targets = _resolved_legal_targets(
        state,
        legal_targets=legal_targets,
        revote=revote,
        pk_candidates=pk_candidates,
    )
    vote_cfg = raw["day_flow"]["vote"]
    normalized: dict[str, str] = {}
    for voter_id, target_id in votes.items():
        voter = state.players.get(voter_id)
        if voter is None or not voter.alive or not voter.vote_enabled:
            continue
        if not vote_cfg.get("allow_self_vote", False) and target_id == voter_id:
            continue
        if target_id not in resolved_legal_targets:
            continue
        normalized[voter_id] = target_id
    return normalized


def resolve_vote(
    raw: dict[str, Any],
    state: GameState,
    *,
    votes: dict[str, str],
    revote: bool,
    base_vote_weight: int,
    legal_targets: set[str] | None = None,
    vote_weight_fn: Callable[[GameState, str], int] | None = None,
    anti_stall_tie_break_fn: Callable[
        [GameState, list[str], str | None, dict[str, str]],
        VoteResult,
    ] | None = None,
    consecutive_no_exile_days: int = 0,
    pk_candidates: list[str] | None = None,
    rng_seed: str | None = None,
) -> VoteResult:
    tally: dict[str, int] = {}
    resolved_legal_targets = _resolved_legal_targets(
        state,
        legal_targets=legal_targets,
        revote=revote,
        pk_candidates=pk_candidates,
    )
    normalized_votes = accepted_votes(
        raw,
        state,
        votes=votes,
        revote=revote,
        legal_targets=legal_targets,
        pk_candidates=pk_candidates,
    )
    for voter_id, target_id in normalized_votes.items():
        if vote_weight_fn is None:
            weight = vote_weight(raw, state, voter_id, base_vote_weight=base_vote_weight)
        else:
            weight = vote_weight_fn(state, voter_id)
        tally[target_id] = tally.get(target_id, 0) + weight

    cfg = raw["day_flow"]["vote"]

    if not tally:
        if revote:
            return _resolve_empty_revote_tally(
                cfg,
                legal_targets=resolved_legal_targets,
                consecutive_no_exile_days=consecutive_no_exile_days,
                pk_candidates=pk_candidates,
                rng_seed=rng_seed,
            )
        return VoteResult(exiled_player_id=None, next_phase="pk_speech", reason="first_tie_pk")

    max_votes = max(tally.values())
    top = [pid for pid, count in tally.items() if count == max_votes]

    if len(top) > 1:
        if not revote:
            return VoteResult(
                exiled_player_id=None,
                next_phase="pk_speech",
                reason="first_tie_pk",
                tied_player_ids=top,
            )
        if cfg["second_tie_policy"] == "no_exile_then_night":
            pace_cfg = cfg.get("simulation_pace", {})
            max_allowed = pace_cfg.get("max_consecutive_no_exile_days", 0)
            if (
                pace_cfg.get("enabled", False)
                and max_allowed > 0
                and consecutive_no_exile_days >= max_allowed
            ):
                if anti_stall_tie_break_fn is not None:
                    return anti_stall_tie_break_fn(
                        state,
                        top,
                        rng_seed,
                        votes,
                    )
                return anti_stall_tie_break(
                    state,
                    top,
                    rng_seed,
                    votes,
                )
            return VoteResult(exiled_player_id=None, next_phase="night", reason="second_tie_no_exile")

    return VoteResult(exiled_player_id=top[0], next_phase="resolve_exile", reason="majority")


def anti_stall_tie_break(
    state: GameState,
    tied: list[str],
    rng_seed: str | None,
    votes: dict[str, str],
) -> VoteResult:
    """重复平票时优先按警长投票破局，否则使用稳定随机种子。"""
    if state.sheriff_id and state.sheriff_badge_state == "active":
        sheriff_vote = votes.get(state.sheriff_id)
        if sheriff_vote and sheriff_vote in tied:
            return VoteResult(
                exiled_player_id=sheriff_vote,
                next_phase="resolve_exile",
                reason="anti_stall_tie_break",
            )

    seed_val = _stable_seed_val(rng_seed or "anti-stall", *tied)
    rng = random.Random(seed_val)
    chosen = rng.choice(tied)
    return VoteResult(
        exiled_player_id=chosen,
        next_phase="resolve_exile",
        reason="anti_stall_tie_break",
    )


def _legal_exile_targets(state: GameState) -> set[str]:
    return {
        pid
        for pid, player in state.players.items()
        if player.alive and not player.exile_immune
    }


def _resolved_legal_targets(
    state: GameState,
    *,
    legal_targets: set[str] | None,
    revote: bool,
    pk_candidates: list[str] | None,
) -> set[str]:
    resolved = (
        set(legal_targets)
        if legal_targets is not None
        else _legal_exile_targets(state)
    )
    if revote and pk_candidates:
        resolved &= set(pk_candidates)
    return resolved


def _resolve_empty_revote_tally(
    cfg: dict[str, Any],
    *,
    legal_targets: set[str],
    consecutive_no_exile_days: int,
    pk_candidates: list[str] | None,
    rng_seed: str | None,
) -> VoteResult:
    pace_cfg = cfg.get("simulation_pace", {})
    max_allowed = pace_cfg.get("max_consecutive_no_exile_days", 0)
    if (
        pace_cfg.get("enabled", False)
        and max_allowed > 0
        and consecutive_no_exile_days >= max_allowed
    ):
        candidates = [
            pid for pid in (pk_candidates or [])
            if pid in legal_targets
        ]
        if candidates:
            seed_val = _stable_seed_val(rng_seed or "anti-stall-empty", *candidates)
            chosen = random.Random(seed_val).choice(candidates)
            return VoteResult(
                exiled_player_id=chosen,
                next_phase="resolve_exile",
                reason="anti_stall_empty_tally",
            )
    return VoteResult(exiled_player_id=None, next_phase="night", reason="second_tie_no_exile")


__all__ = [
    "accepted_votes",
    "anti_stall_tie_break",
    "resolve_vote",
    "sheriff_vote_weight",
    "vote_weight",
]
