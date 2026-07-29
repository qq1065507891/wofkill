# -*- coding: utf-8 -*-
"""
验证昼间发言动作联合、引用关系、交付计划和终端信封。

作者: Project contributors
创建日期: 2026-07-29
"""

import json

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.proposals import (
    SpeechProposalEnvelope,
)

HASH = "a" * 64


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "turn_id": "turn-1",
        "player_id": "p01",
        "window_id": "speech-d1-p01",
        "window_version": 1,
        "base_revision": 4,
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


def test_speech_envelope_parses_discriminated_moves() -> None:
    proposal = SpeechProposalEnvelope.model_validate_json(json.dumps(_payload()))
    assert proposal.body.moves[0].move_type == "alignment_read"
    assert proposal.body.delivery_plan.move_order == ("m1", "m2")


def test_speech_rejects_extra_fields_and_bad_move_order() -> None:
    payload = _payload()
    payload["body"]["moves"][0]["reasoning"] = "private thought"  # type: ignore[index]
    with pytest.raises(ValidationError):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))

    payload = _payload()
    payload["body"]["delivery_plan"]["move_order"] = ["m1"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="move_order must contain every move ID"):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))


def test_speech_rejects_duplicate_and_cyclic_move_references() -> None:
    payload = _payload()
    payload["body"]["moves"][1]["move_id"] = "m1"  # type: ignore[index]
    with pytest.raises(ValidationError, match="move IDs must not contain duplicates"):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))

    payload = _payload()
    payload["body"]["moves"] = [  # type: ignore[index]
        {
            "move_id": "m1",
            "move_type": "public_evidence_citation",
            "modality": "asserted",
            "evidence_refs": ["public-3"],
            "relation": "supports",
            "subject_ids": ["p03"],
            "supports_move_ids": ["m2"],
        },
        {
            "move_id": "m2",
            "move_type": "public_evidence_citation",
            "modality": "asserted",
            "evidence_refs": ["public-4"],
            "relation": "supports",
            "subject_ids": ["p03"],
            "supports_move_ids": ["m1"],
        },
    ]
    with pytest.raises(ValidationError, match="move references must be acyclic"):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))


def test_speech_rejects_duplicate_response_record_refs() -> None:
    payload = _payload()
    payload["body"]["response_record_refs"] = [  # type: ignore[index]
        "public-3",
        "public-3",
    ]
    with pytest.raises(
        ValidationError,
        match="response_record_refs must not contain duplicates",
    ):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))
