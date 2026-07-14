# -*- coding: utf-8 -*-
"""
功能描述：在角色数量约束下，枚举并评分可能的身份分配方案，
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-14
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata


@dataclass(frozen=True)
class PossibleWorld:
    world_id: str
    probability: float
    roles: Mapping[str, str]
    score_breakdown: Mapping[str, float] = field(default_factory=dict)
    supporting_evidence: tuple[str, ...] = field(default_factory=tuple)
    contradictions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """在值对象边界做防御性转换，避免世界 ID 生成后被外部篡改。"""
        object.__setattr__(self, "roles", _deep_freeze(self.roles))
        object.__setattr__(self, "score_breakdown", _deep_freeze(self.score_breakdown))
        object.__setattr__(self, "supporting_evidence", _deep_freeze(self.supporting_evidence))
        object.__setattr__(self, "contradictions", _deep_freeze(self.contradictions))


@dataclass(frozen=True)
class PossibleWorldSet:
    viewer_id: str
    generated_at_event_index: int
    worlds: tuple[PossibleWorld, ...]
    marginal_role_probs: Mapping[str, Mapping[str, float]]
    public_evidence_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """按规范化身份分配合并重复世界并重新归一化概率。"""
        raw_worlds = tuple(self.worlds)
        probabilities = [float(world.probability) for world in raw_worlds]
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("world probability must be finite and non-negative")
        max_probability = max(probabilities, default=0.0)
        merged: dict[bytes, PossibleWorld] = {}
        for world, probability in zip(raw_worlds, probabilities, strict=True):
            scaled_probability = (
                probability / max_probability if max_probability > 0.0 else 0.0
            )
            roles = _normalize_assignment(world.roles)
            canonical = _canonical_assignment(roles)
            previous = merged.get(canonical)
            if previous is None:
                merged[canonical] = PossibleWorld(
                    world_id="",
                    probability=scaled_probability,
                    roles=roles,
                    score_breakdown=world.score_breakdown,
                    supporting_evidence=list(dict.fromkeys(world.supporting_evidence)),
                    contradictions=world.contradictions,
                )
            else:
                merged[canonical] = PossibleWorld(
                    world_id="",
                    probability=previous.probability + scaled_probability,
                    roles=previous.roles,
                    score_breakdown=previous.score_breakdown,
                    supporting_evidence=list(dict.fromkeys(
                        previous.supporting_evidence + world.supporting_evidence
                    )),
                    contradictions=previous.contradictions,
                )
        total = math.fsum(world.probability for world in merged.values())
        uniform_probability = 1.0 / len(merged) if merged and total == 0.0 else 0.0
        base_ids = {canonical: _world_id(canonical) for canonical in merged}
        id_counts: dict[str, int] = {}
        for world_id in base_ids.values():
            id_counts[world_id] = id_counts.get(world_id, 0) + 1
        normalized = [
            PossibleWorld(
                world_id=(
                    base_ids[canonical]
                    if id_counts[base_ids[canonical]] == 1
                    else f"{base_ids[canonical]}_{canonical.hex()}"
                ),
                probability=(world.probability / total if total > 0.0 else uniform_probability),
                roles=world.roles,
                score_breakdown=world.score_breakdown,
                supporting_evidence=world.supporting_evidence,
                contradictions=world.contradictions,
            )
            for canonical, world in merged.items()
        ]
        normalized.sort(key=lambda world: (-world.probability, world.world_id))
        object.__setattr__(self, "viewer_id", _normalize_text(self.viewer_id))
        object.__setattr__(self, "worlds", tuple(normalized))
        object.__setattr__(self, "public_evidence_ids", frozenset(self.public_evidence_ids))
        object.__setattr__(self, "marginal_role_probs", _immutable_marginals(normalized))

    def promptable_worlds(self) -> tuple[PossibleWorld, ...]:
        """仅返回由唯一且已知公开证据支撑的世界。"""
        return tuple(
            world for world in self.worlds
            if world.supporting_evidence
            and len(world.roles) == len(set(world.roles))
            and all(ref in self.public_evidence_ids for ref in world.supporting_evidence)
        )

    def to_prompt_dict(self, max_assignments: int = 4) -> dict[str, Any]:
        promptable = self.promptable_worlds()
        result = {
            "type": "possible_worlds",
            "top_worlds": [
                {
                    "label": world.world_id,
                    "probability": round(world.probability, 3),
                    "key_assignments": {
                        pid: _deep_thaw(role)
                        for pid, role in [
                            item for item in sorted(world.roles.items())
                            if item[0] != self.viewer_id
                        ][:max_assignments]
                    },
                    "why": _deep_thaw(world.supporting_evidence[:3]),
                    "watch_for": _deep_thaw(world.contradictions[:3]),
                }
                for world in promptable[:3]
            ],
            "summary": (
                "Evidence-insufficient: no role-specific hypothesis is available."
                if not promptable else "Public-evidence-supported hypotheses only."
            ),
            "warning": "These are hypotheses from visible evidence, not ground truth.",
        }
        if not promptable:
            result["faction_hypothesis"] = {
                "good": 0.5,
                "werewolf": 0.5,
                "basis": "uniform evidence-insufficient prior",
            }
        return result

    def to_audit_identity_proofs(self) -> list[dict[str, Any]]:
        """导出完整 assignment 身份证明；调用方必须只写入 moderator 审计。"""
        return [
            {
                "world_id": world.world_id,
                "canonical_assignment": [
                    [player, _deep_thaw(role)]
                    for player, role in sorted(world.roles.items())
                ],
            }
            for world in self.promptable_worlds()[:3]
        ]


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
        assignment_evidence: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
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
            self._score_roles(
                roles,
                belief_summary or {},
                public_evidence_ids or set(),
                assignment_evidence or {},
            )
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
        assignment_evidence: Mapping[str, Mapping[str, tuple[str, ...]]],
    ) -> tuple[float, dict[str, str], dict[str, float], list[str], list[str]]:
        score = 1.0
        evidence: list[str] = []
        contradictions: list[str] = []
        belief_score = 0.0
        for player_id, role in roles.items():
            concepts = assignment_evidence.get(player_id, {})
            matching_refs = list(concepts.get(f"role:{role}", ()))
            faction = "werewolf" if role == "werewolf" else "good"
            matching_refs.extend(concepts.get(f"faction:{faction}", ()))
            evidence.extend(matching_refs)
            score += 0.25 * len(matching_refs)
            belief_score += 0.25 * len(matching_refs)
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
        evidence = list(dict.fromkeys(
            ref for ref in evidence if ref in public_evidence_ids
        ))
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


def _deep_freeze(value: Any) -> Any:
    """递归复制并冻结 JSON-like 容器，切断调用方持有的可变引用。"""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    """将冻结快照递归还原为可 JSON 序列化的普通容器。"""
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return [_deep_thaw(item) for item in sorted(value, key=repr)]
    return value


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def _normalize_assignment(roles: Mapping[Any, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_player, raw_role in roles.items():
        player = _normalize_text(raw_player)
        role = _normalize_text(raw_role)
        if player in normalized:
            raise ValueError(f"duplicate player after normalization: {player}")
        normalized[player] = role
    return dict(sorted(normalized.items()))


def _immutable_marginals(
    worlds: list[PossibleWorld],
) -> Mapping[str, Mapping[str, float]]:
    marginals: dict[str, dict[str, float]] = {}
    for world in worlds:
        for player, role in world.roles.items():
            role_probs = marginals.setdefault(player, {})
            role_probs[role] = role_probs.get(role, 0.0) + world.probability
    return MappingProxyType({
        player: MappingProxyType({
            role: round(probability, 12)
            for role, probability in sorted(role_probs.items())
        })
        for player, role_probs in sorted(marginals.items())
    })


def _canonical_assignment(roles: Mapping[str, str]) -> bytes:
    """只序列化规范化 player-role 对，避免输入顺序影响身份。"""
    pairs = [[player, role] for player, role in sorted(roles.items())]
    return json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _world_id(canonical: bytes) -> str:
    return f"world_{hashlib.sha256(canonical).hexdigest()}"
