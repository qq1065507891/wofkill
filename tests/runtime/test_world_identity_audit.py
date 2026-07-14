# -*- coding: utf-8 -*-
"""
验证可能世界完整身份只进入 moderator-only 审计，不进入玩家 prompt。

作者: Project contributors
创建日期: 2026-07-14
"""

from __future__ import annotations

from werewolf_agent.agents.prompt_schemas import AgentContext, TaskType
from werewolf_agent.cognition.worlds import PossibleWorld, PossibleWorldSet
from werewolf_agent.runtime.world_model_audit import build_world_model_audit_from_context


def test_world_set_exports_authoritative_full_identity_proofs_for_audit_only() -> None:
    worlds = PossibleWorldSet(
        viewer_id="p01",
        generated_at_event_index=3,
        worlds=(PossibleWorld(
            world_id="ignored",
            probability=1.0,
            roles={"p01": "seer", "p02": "werewolf", "p03": "villager"},
            supporting_evidence=("claim:g1:0",),
        ),),
        marginal_role_probs={},
        public_evidence_ids=frozenset({"claim:g1:0"}),
    )
    prompt = worlds.to_prompt_dict(max_assignments=1)
    proofs = worlds.to_audit_identity_proofs()
    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        possible_worlds=prompt,
        authoritative_world_identities=proofs,
    )

    dumped = context.model_dump()
    audit = build_world_model_audit_from_context(context)

    assert "authoritative_world_identities" not in dumped
    assert len(prompt["top_worlds"][0]["key_assignments"]) == 1
    assert audit["authoritative_world_identities"] == proofs
    assert audit["authoritative_world_identities"][0]["canonical_assignment"] == [
        ["p01", "seer"], ["p02", "werewolf"], ["p03", "villager"],
    ]
