# -*- coding: utf-8 -*-
"""
游戏 API 认知视图数据与锁定配置快照 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.api.routes.game_cognition_views import _build_locked_config_snapshot
    >>> _build_locked_config_snapshot(req, project_root)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from werewolf_agent.api.schemas import CreateGameRequest
from werewolf_agent.core.models import GameState


def _build_cognition_data_for_viewer(
    state: GameState, viewer_id: str,
) -> dict[str, dict[str, Any]]:
    """为 cognitive-diff 视图构造观察者可见的信念数据。"""
    try:
        from werewolf_agent.cognition.belief import BeliefUpdater
        from werewolf_agent.cognition.visibility import VisibilityPolicy
        from werewolf_agent.cognition.world_state import build_world_state
    except Exception:
        return {}

    try:
        world_state = build_world_state(state)
    except Exception:
        return {}

    role_names = [
        "villager", "seer", "witch", "hunter", "idiot", "werewolf", "hybrid",
    ]
    updater = BeliefUpdater(all_role_names=role_names)
    belief_state = updater.initialize(list(state.players.keys()), viewer_id)

    try:
        viewer_role = state.players[viewer_id].role if viewer_id in state.players else "villager"
        vis_policy = VisibilityPolicy()
        visible_facts = vis_policy.filter_visible_facts(world_state, viewer_id, viewer_role)
        belief_state = updater.update(belief_state, visible_facts, state.day_number)
    except Exception:
        # 可见性计算失败时保留初始化信念，避免视图接口整体失败。
        pass

    cognition_data: dict[str, dict[str, Any]] = {}
    for pid, b in belief_state.beliefs.items():
        guessed_role, guessed_confidence = b.top_role_guess()
        faction_read = b.faction_lean if b.faction_lean != "unknown" else "unknown"
        cognition_data[pid] = {
            "guessed_role": guessed_role,
            "guessed_confidence": float(guessed_confidence),
            "faction_read": faction_read,
            "trust": float(b.trust),
            "key_evidence": list(b.open_questions),
            "belief_changes": [],
        }
    return cognition_data


def _build_locked_config_snapshot(req: CreateGameRequest, project_root: Path) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", req.ruleset_id):
        raise HTTPException(400, f"Invalid ruleset_id: {req.ruleset_id}")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", req.profile_pack_id):
        raise HTTPException(400, f"Invalid profile_pack_id: {req.profile_pack_id}")
    seed = req.seed if req.seed is not None else 0
    ruleset_path = project_root / "config" / "rulesets" / f"{req.ruleset_id}.yaml"
    ruleset_content = ruleset_path.read_text(encoding="utf-8") if ruleset_path.exists() else req.ruleset_id
    return {
        "ruleset_id": req.ruleset_id,
        "ruleset_version": "runtime-current",
        "ruleset_hash": hashlib.sha256(ruleset_content.encode("utf-8")).hexdigest(),
        "profile_pack_id": req.profile_pack_id,
        "profile_pack_version": "runtime-current",
        "profile_pack_hash": hashlib.sha256((
            (project_root / "config" / "persona_packs" / f"{req.profile_pack_id}.yaml")
            .read_text(encoding="utf-8")
            if (project_root / "config" / "persona_packs" / f"{req.profile_pack_id}.yaml").exists()
            else req.profile_pack_id
        ).encode("utf-8")).hexdigest(),
        "model_config_hash": "",
        "persona_adapter_version": 1,
        "rag_config_hash": "",
        "engine_version": "1.0",
        "random_seed": seed,
        "agent_behavior_seed": seed,
        "speech_order_seed": seed,
        "experience_mode": req.experience_mode,
        "human_seat": req.human_seat,
        "share_code": req.share_code,
    }


__all__ = [
    "_build_cognition_data_for_viewer",
    "_build_locked_config_snapshot",
]
