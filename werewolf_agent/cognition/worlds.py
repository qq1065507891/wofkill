"""Constrained possible-world hypotheses for hidden-role inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
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

    def to_prompt_dict(self, max_assignments: int = 4) -> dict[str, Any]:
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
                for world in self.worlds[:3]
            ],
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
            self._score_roles(roles, belief_summary or {})
            for roles in candidates
        ]
        scored.sort(key=lambda item: (-item[0], sorted(item[1].items())))
        selected = scored[:max(1, top_k)]
        total = sum(score for score, _, _, _, _ in selected) or 1.0
        worlds = [
            PossibleWorld(
                world_id=f"World {idx}",
                probability=score / total,
                roles=roles,
                score_breakdown=breakdown,
                supporting_evidence=evidence,
                contradictions=contradictions,
            )
            for idx, (score, roles, breakdown, evidence, contradictions)
            in zip(count(1), selected)
        ]
        return PossibleWorldSet(
            viewer_id=viewer_id,
            generated_at_event_index=generated_at_event_index,
            worlds=worlds,
            marginal_role_probs=self._marginals(worlds, player_ids),
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
                evidence.append(f"{player_id}->{guessed_role} aligns with belief")
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
