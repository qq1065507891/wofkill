# -*- coding: utf-8 -*-
"""
重建公开证据索引并验证可能世界的唯一性与证据覆盖。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

import json
import hashlib
import unicodedata
from collections import Counter
from typing import Any, Mapping

from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import extract_facts
from werewolf_agent.core.models import GameEvent, GameState

_FIXED_ROLE_FACTIONS = {
    "villager": "good",
    "seer": "good",
    "witch": "good",
    "hunter": "good",
    "idiot": "good",
    "werewolf": "werewolf",
}


def _public_assignment_support_before(
    game: dict[str, Any],
    before_index: int,
) -> dict[str, set[tuple[str, str]]]:
    """按运行时同一结构化事实语义重建 prompt 前的 assignment 支撑。"""
    game_id = str(game.get("game_id") or "")
    if not game_id:
        return {}
    state = GameState(game_id=game_id)
    policy = VisibilityPolicy()
    evidence = PublicEvidenceIndex()
    for index, raw_event in enumerate(game.get("events", [])[:before_index]):
        if not isinstance(raw_event, Mapping):
            continue
        event_type = raw_event.get("type")
        payload = raw_event.get("payload")
        if not isinstance(event_type, str) or not isinstance(payload, Mapping):
            continue
        event = GameEvent(type=event_type, payload=payload)
        for fact in extract_facts(event, state):
            if policy.compute_fact_visibility(fact, index).visibility != "public":
                continue
            prefix = "claim" if fact.fact_type in {
                "claimed_role", "claimed_good", "seer_check_claim"
            } else "event"
            evidence.observe_assignment_reference(
                fact, f"{prefix}:{game_id}:{index}"
            )
    by_ref: dict[str, set[tuple[str, str]]] = {}
    for player, concepts in evidence.assignment_evidence().items():
        for concept, refs in concepts.items():
            for ref in refs:
                by_ref.setdefault(ref, set()).add((player, concept))
    return by_ref


def _valid_unique_world_count(
    group: list[dict[str, Any]],
    identity_proofs: list[dict[str, Any]] | None = None,
    expected_roles: dict[str, str] | None = None,
) -> int:
    """同一 prompt 内要求非空唯一 ID 与规范化 assignment 同时唯一。"""
    if expected_roles is None:
        return 0
    authoritative = _authoritative_world_assignments(
        identity_proofs or [], expected_roles=expected_roles
    )
    identifiers = [_world_identifier(world) for world in group]
    assignments = [
        _validated_prompt_assignment(world, authoritative) for world in group
    ]
    id_counts = Counter(identifier for identifier in identifiers if identifier)
    assignment_counts = Counter(
        assignment for assignment in assignments if assignment is not None
    )
    return sum(
        bool(identifier)
        and assignment is not None
        and id_counts[identifier] == 1
        and assignment_counts[assignment] == 1
        for identifier, assignment in zip(identifiers, assignments, strict=True)
    )


def _world_identifier(world: dict[str, Any]) -> str:
    raw = world.get("world_id") or world.get("label")
    return unicodedata.normalize("NFKC", str(raw or "")).strip().casefold()


def _canonical_world_assignment(world: dict[str, Any]) -> str | None:
    assignments = world.get("key_assignments")
    if not isinstance(assignments, Mapping) or not assignments:
        return None
    normalized: dict[str, str] = {}
    for raw_player, raw_role in assignments.items():
        player = unicodedata.normalize("NFKC", str(raw_player)).strip().casefold()
        role = unicodedata.normalize("NFKC", str(raw_role)).strip().casefold()
        if not player or not role or player in normalized:
            return None
        normalized[player] = role
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _authoritative_world_assignments(
    proofs: list[dict[str, Any]],
    *,
    expected_roles: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """重算完整 assignment 的 SHA256；无效、重复证明一律剔除。"""
    candidates: list[tuple[str, dict[str, str], bytes]] = []
    for proof in proofs:
        if not isinstance(proof, Mapping):
            continue
        pairs = proof.get("canonical_assignment")
        if not isinstance(pairs, (list, tuple)) or not pairs:
            continue
        normalized: dict[str, str] = {}
        valid = True
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                valid = False
                break
            player = unicodedata.normalize("NFKC", str(pair[0])).strip().casefold()
            role = unicodedata.normalize("NFKC", str(pair[1])).strip().casefold()
            if not player or not role or player in normalized:
                valid = False
                break
            normalized[player] = role
        if not valid:
            continue
        if expected_roles is not None:
            expected_players = set(expected_roles)
            if set(normalized) != expected_players:
                continue
            if Counter(normalized.values()) != Counter(expected_roles.values()):
                continue
        canonical = json.dumps(
            [[player, role] for player, role in sorted(normalized.items())],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_id = f"world_{hashlib.sha256(canonical).hexdigest()}"
        supplied_id = unicodedata.normalize(
            "NFKC", str(proof.get("world_id") or "")
        ).strip().casefold()
        if supplied_id != expected_id:
            continue
        candidates.append((supplied_id, normalized, canonical))
    id_counts = Counter(item[0] for item in candidates)
    assignment_counts = Counter(item[2] for item in candidates)
    return {
        world_id: assignment
        for world_id, assignment, canonical in candidates
        if id_counts[world_id] == 1 and assignment_counts[canonical] == 1
    }


def _validated_prompt_assignment(
    world: dict[str, Any],
    authoritative: dict[str, dict[str, str]],
) -> str | None:
    world_id = _world_identifier(world)
    full = authoritative.get(world_id)
    prompt = _canonical_world_assignment(world)
    if full is None or prompt is None:
        return None
    prompt_assignment = json.loads(prompt)
    if any(full.get(player) != role for player, role in prompt_assignment.items()):
        return None
    return json.dumps(full, ensure_ascii=False, sort_keys=True)


def _support_matches_world(
    concepts: set[tuple[str, str]],
    world: dict[str, Any],
) -> bool:
    """一个 why 引用必须语义上支撑该世界中的至少一个身份分配。"""
    assignments = world.get("key_assignments")
    if not isinstance(assignments, Mapping):
        return False
    for raw_player, raw_role in assignments.items():
        player = unicodedata.normalize("NFKC", str(raw_player)).strip().casefold()
        role = unicodedata.normalize("NFKC", str(raw_role)).strip().casefold()
        if (player, f"role:{role}") in concepts:
            return True
        faction = _FIXED_ROLE_FACTIONS.get(role)
        if faction is not None and (player, f"faction:{faction}") in concepts:
            return True
    return False


def _current_game_public_evidence_refs(
    game: dict[str, Any],
    refs: list[Any],
) -> set[str]:
    """仅接受 runtime 权威索引中属于当前 game_id 的规范引用。"""
    game_id = str(game.get("game_id") or "")
    if not game_id:
        return set()
    return {
        ref for ref in refs
        if isinstance(ref, str)
        and (ref.startswith(f"event:{game_id}:") or ref.startswith(f"claim:{game_id}:"))
    }


# 公开名称供新模块使用；旧 facade 继续暴露原有下划线名称。
support_matches_world = _support_matches_world
