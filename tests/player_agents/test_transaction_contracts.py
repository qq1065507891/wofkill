# -*- coding: utf-8 -*-
"""
验证自主玩家 CommitTurn 请求、结果、审计和 outbox 契约。

作者: Project contributors
创建日期: 2026-07-29
"""

import json
import re

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.proposals import SpeechProposalEnvelope
from werewolf_agent.player_agents.contracts.records import PublicSpeechRecord
from werewolf_agent.player_agents.contracts.transactions import (
    CommitTurnRequest,
    CriticalAuditRecord,
    EventCandidate,
    ProjectionOutboxRecord,
)
from werewolf_agent.storage.autonomous_commit import request_hash

HASH = "a" * 64


def _proposal_payload(*, turn_id: str = "turn-1") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "turn_id": turn_id,
        "player_id": "p01",
        "window_id": "speech-d1-p01",
        "window_version": 1,
        "base_revision": 0,
        "view_fingerprint": HASH,
        "body": {
            "kind": "speech",
            "objective": "declare_vote_position",
            "moves": [
                {
                    "move_id": "m1",
                    "move_type": "alignment_read",
                    "modality": "suspected",
                    "evidence_refs": ["public-3"],
                    "target_id": "p03",
                    "alignment": "wolf",
                    "strength": "leaning",
                },
                {
                    "move_id": "m2",
                    "move_type": "vote_position",
                    "modality": "asserted",
                    "evidence_refs": ["public-3"],
                    "target_id": "p03",
                    "commitment": "provisional",
                },
            ],
            "response_record_refs": [],
            "delivery_plan": {
                "tone": "firm",
                "length_class": "standard",
                "address_style": "room",
                "move_order": ["m1", "m2"],
                "emphasis_move_ids": ["m2"],
                "connector_ids": ["because"],
            },
        },
    }


def _request(
    *,
    turn_id: str = "turn-1",
    game_id: str = "g1",
    base_revision: int = 0,
    rule_result: dict[str, object] | None = None,
    audit_ids: tuple[str, ...] = ("audit-1",),
    outbox_ids: tuple[str, ...] = ("outbox-1",),
    event_type: str = "speech_submitted",
    public_record: PublicSpeechRecord | None = None,
) -> CommitTurnRequest:
    return CommitTurnRequest(
        game_id=game_id,
        turn_id=turn_id,
        idempotency_key=f"{turn_id}:submit",
        base_game_revision=base_revision,
        proposal=SpeechProposalEnvelope.model_validate_json(
            json.dumps(_proposal_payload(turn_id=turn_id)),
        ),
        rule_result=rule_result or {"accepted": True},
        event=EventCandidate(type=event_type, payload={"turn_id": turn_id}),
        public_record=public_record,
        critical_audit_records=tuple(
            CriticalAuditRecord(audit_id=audit_id, kind="proposal_accepted")
            for audit_id in audit_ids
        ),
        projection_outbox_records=tuple(
            ProjectionOutboxRecord(outbox_id=outbox_id, kind="workspace_refresh")
            for outbox_id in outbox_ids
        ),
    )


def test_commit_request_binds_turn_and_rejects_extra_fields() -> None:
    request = _request()
    assert request.proposal.turn_id == request.turn_id
    with pytest.raises(ValidationError):
        CommitTurnRequest.model_validate({**request.model_dump(), "unexpected": True})


def test_commit_request_rejects_duplicate_audit_and_outbox_ids() -> None:
    with pytest.raises(ValidationError, match="audit IDs must not contain duplicates"):
        _request(audit_ids=("audit-1", "audit-1"))
    with pytest.raises(ValidationError, match="outbox IDs must not contain duplicates"):
        _request(outbox_ids=("outbox-1", "outbox-1"))


def test_request_hash_is_order_independent_for_json_object_keys() -> None:
    left = _request(rule_result={"accepted": True, "reason": "ok"})
    right = _request(rule_result={"reason": "ok", "accepted": True})
    assert request_hash(left) == request_hash(right)
    assert re.fullmatch(r"[0-9a-f]{64}", request_hash(left))


def test_transaction_json_payloads_are_deeply_immutable() -> None:
    request = _request(rule_result={"accepted": True, "details": ["safe"]})

    with pytest.raises(TypeError):
        request.rule_result["accepted"] = False  # type: ignore[index]
    assert request.rule_result["details"] == ("safe",)
