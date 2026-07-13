# -*- coding: utf-8 -*-
"""
功能描述：在角色数量约束下，枚举并评分可能的身份分配方案，
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-13
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class PossibleWorld:
    world_id: str
    probability: float
    roles: dict[str, str]
    score_breakdown: dict[str, float] = field(default_factory=dict)
    supporting_evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PossibleWorldSet:
    viewer_id: str
    generated_at_event_index: int
    worlds: list[PossibleWorld]
    marginal_role_probs: dict[str, dict[str, float]]
    public_evidence_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """按规范化身份分配合并重复世界并重新归一化概率。"""
        merged: dict[str, PossibleWorld] = {}
        for world in self.worlds:
            canonical = _canonical_assignment(world.roles)
            world_id = _world_id(canonical)
            previous = merged.get(world_id)
            if previous is None:
                merged[world_id] = PossibleWorld(
                    world_id=world_id,
                    probability=world.probability,
                    roles=dict(sorted(world.roles.items())),
                    score_breakdown=world.score_breakdown,
                    supporting_evidence=list(dict.fromkeys(world.supporting_evidence)),
                    contradictions=world.contradictions,
                )
            else:
                merged[world_id] = PossibleWorld(
                    world_id=world_id,
                    probability=previous.probability + world.probability,
                    roles=previous.roles,
                    score_breakdown=previous.score_breakdown,
                    supporting_evidence=list(dict.fromkeys(
                        previous.supporting_evidence + world.supporting_evidence
                    )),
                    contradictions=previous.contradictions,
                )
        total = sum(max(0.0, world.probability) for world in merged.values()) or 1.0
        normalized = [
            PossibleWorld(
                world_id=world.world_id,
                probability=max(0.0, world.probability) / total,
                roles=world.roles,
                score_breakdown=world.score_breakdown,
                supporting_evidence=world.supporting_evidence,
                contradictions=world.contradictions,
            )
            for world in merged.values()
        ]
        normalized.sort(key=lambda world: (-world.probability, world.world_id))
        object.__setattr__(self, "worlds", normalized)

    def promptable_worlds(self) -> list[PossibleWorld]:
        """仅返回由唯一且已知公开证据支撑的世界。"""
        return [
            world for world in self.worlds
            if world.supporting_evidence
            and len(world.roles) == len(set(world.roles))
            and all(ref in self.public_evidence_ids for ref in world.supporting_evidence)
        ]

    def to_prompt_dict(self, max_assignments: int = 4) -> dict[str, Any]:
        promptable = self.promptable_worlds()
        return {
            "type": "possible_worlds",
            "top_worlds": [
                {
                    "label": world.world_id,
                    "probability": round(world.probability, 3),
                    "key_assignments": {
                        pid: role
                        for pid, role in sorted(world.roles.items())[:max_assignments]
                        if pid != self.viewer_id
                    },
                    "why": list(world.supporting_evidence[:3]),
                    "watch_for": list(world.contradictions[:3]),
                }
                for world in promptable[:3]
            ],
            "summary": (
                "Evidence-insufficient: no role-specific hypothesis is available."
                if not promptable else "Public-evidence-supported hypotheses only."
            ),
            "warning": "These are hypotheses from visible evidence, not ground truth.",
        }


class PossibleWorldsEngine:
    """Generate bounded hidden-role hypotheses under role-count constraints."""

    def generate(
        self,
        *,
        viewer_id: str,
        viewer_role: str,
        player_ids: list[str],
        role_counts: dict[str, int],
        belief_summary: dict[str, Any] | None = None,
        known_roles: dict[str, str] | None = None,
        generated_at_event_index: int = 0,
        top_k: int = 3,
        max_candidates: int = 500,
        public_evidence_ids: set[str] | None = None,
    ) -> PossibleWorldSet:
        fixed_roles = dict(known_roles or {})
        fixed_roles[viewer_id] = viewer_role
        candidates = self._enumerate_candidates(
            player_ids=player_ids,
            role_counts=role_counts,
            fixed_roles=fixed_roles,
            max_candidates=max_candidates,
        )
        scored = [
            self._score_roles(roles, belief_summary or {}, public_evidence_ids or set())
            for roles in candidates
        ]
        scored.sort(key=lambda item: (-item[0], sorted(item[1].items())))
        selected = scored[:max(1, top_k)]
        total = sum(score for score, _, _, _, _ in selected) or 1.0
        worlds = [
            PossibleWorld(
                world_id="",
                probability=score / total,
                roles=roles,
                score_breakdown=breakdown,
                supporting_evidence=evidence,
                contradictions=contradictions,
            )
            for score, roles, breakdown, evidence, contradictions in selected
        ]
        return PossibleWorldSet(
            viewer_id=viewer_id,
            generated_at_event_index=generated_at_event_index,
            worlds=worlds,
            marginal_role_probs=self._marginals(worlds, player_ids),
            public_evidence_ids=set(public_evidence_ids or set()),
        )

    def _enumerate_candidates(
        self,
        *,
        player_ids: list[str],
        role_counts: dict[str, int],
        fixed_roles: dict[str, str],
        max_candidates: int,
    ) -> list[dict[str, str]]:
        remaining_counts = dict(role_counts)
        for role in fixed_roles.values():
            if role in remaining_counts:
                remaining_counts[role] -= 1
        if any(count < 0 for count in remaining_counts.values()):
            return []
        remaining_players = [pid for pid in player_ids if pid not in fixed_roles]
        candidates: list[dict[str, str]] = []

        def backtrack(index: int, current: dict[str, str]) -> None:
            if len(candidates) >= max_candidates:
                return
            if index >= len(remaining_players):
                if all(count == 0 for count in remaining_counts.values()):
                    roles = dict(fixed_roles)
                    roles.update(current)
                    candidates.append(roles)
                return
            player_id = remaining_players[index]
            for role in sorted(remaining_counts):
                if remaining_counts[role] <= 0:
                    continue
                remaining_counts[role] -= 1
                current[player_id] = role
                backtrack(index + 1, current)
                current.pop(player_id, None)
                remaining_counts[role] += 1

        backtrack(0, {})
        return candidates

    def _score_roles(
        self,
        roles: dict[str, str],
        belief_summary: dict[str, Any],
        public_evidence_ids: set[str],
    ) -> tuple[float, dict[str, str], dict[str, float], list[str], list[str]]:
        score = 1.0
        evidence: list[str] = []
        contradictions: list[str] = []
        belief_score = 0.0
        for item in belief_summary.get("my_suspects", []):
            player_id = str(item.get("player", ""))
            guessed_role = str(item.get("top_role_guess", ""))
            prob = _float(item.get("top_role_prob"), 0.5)
            if not player_id or not guessed_role:
                continue
            if roles.get(player_id) == guessed_role:
                score += prob
                belief_score += prob
                evidence.extend(
                    ref for ref in item.get("evidence_ids", [])
                    if ref in public_evidence_ids
                )
            else:
                penalty = prob * 0.5
                score -= penalty
                belief_score -= penalty
                contradictions.append(f"{player_id}->{guessed_role} not assigned here")
            trust = _float(item.get("trust"), 0.5)
            if roles.get(player_id) == "werewolf" and trust < 0.5:
                boost = 0.5 - trust
                score += boost
                belief_score += boost
        for item in belief_summary.get("my_trusted", []):
            player_id = str(item.get("player", ""))
            trust = _float(item.get("trust"), 0.5)
            if roles.get(player_id) != "werewolf" and trust > 0.5:
                boost = trust - 0.5
                score += boost
                belief_score += boost
        score = max(score, 0.01)
        return (
            score,
            roles,
            {"belief": round(belief_score, 3), "base": 1.0},
            evidence,
            contradictions,
        )

    @staticmethod
    def _marginals(
        worlds: list[PossibleWorld],
        player_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        marginals: dict[str, dict[str, float]] = {pid: {} for pid in player_ids}
        for world in worlds:
            for pid, role in world.roles.items():
                marginals.setdefault(pid, {})
                marginals[pid][role] = marginals[pid].get(role, 0.0) + world.probability
        return {
            pid: {role: round(prob, 3) for role, prob in sorted(probs.items())}
            for pid, probs in marginals.items()
        }


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_assignment(roles: dict[str, str]) -> bytes:
    """只序列化规范化 player-role 对，避免输入顺序影响身份。"""
    pairs = [[str(player), str(role)] for player, role in sorted(roles.items())]
    return json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _world_id(canonical: bytes) -> str:
    return f"world_{hashlib.sha256(canonical).hexdigest()[:16]}"
