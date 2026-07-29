# -*- coding: utf-8 -*-
"""
验证一次性私密披露授权、公共语义记录和渲染记录契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.disclosure import DisclosureGrant
from werewolf_agent.player_agents.contracts.records import (
    PublicSpeechRecord,
    RecordOrigin,
    RenderedUtterance,
)
from werewolf_agent.player_agents.contracts.speech import (
    Alignment,
    AlignmentRead,
    Modality,
    Strength,
    VoteCommitment,
    VotePosition,
)

HASH = "a" * 64


def _moves() -> tuple[AlignmentRead, VotePosition]:
    return (
        AlignmentRead(
            move_id="m1",
            move_type="alignment_read",
            modality=Modality.SUSPECTED,
            evidence_refs=("public-3",),
            target_id="p03",
            alignment=Alignment.WOLF,
            strength=Strength.LEANING,
        ),
        VotePosition(
            move_id="m2",
            move_type="vote_position",
            modality=Modality.ASSERTED,
            evidence_refs=("public-3",),
            target_id="p03",
            commitment=VoteCommitment.PROVISIONAL,
        ),
    )


def _record_payload() -> dict[str, object]:
    return {
        "record_id": "speech-5",
        "schema_version": "1.0.0",
        "game_id": "game-1",
        "turn_id": "turn-1",
        "actor_id": "p01",
        "day": 1,
        "phase": "day_discussion",
        "committed_revision": 5,
        "normalized_moves": _moves(),
        "source_evidence_refs": ("public-3",),
        "disclosure_grant_refs": (),
        "origin": RecordOrigin.MODEL_SUBMISSION,
        "renderer_contract_version": "speech-renderer-1",
        "rendered_utterance_hash": HASH,
    }


def test_disclosure_grant_requires_aware_expiry_and_exact_fact_hash() -> None:
    grant = DisclosureGrant(
        grant_id="grant-1",
        actor_id="p01",
        turn_id="turn-1",
        window_id="speech-d1-p01",
        game_revision=4,
        fact_kind="alignment_check",
        fact_record_id="seer-check-1",
        fact_hash=HASH,
        target_id="p03",
        timing_ref="night-1",
        expires_at=datetime(2026, 7, 29, 2, tzinfo=timezone.utc),
    )
    assert grant.fact_hash == HASH

    naive_expiry = datetime(2026, 7, 29, 2, tzinfo=timezone.utc).replace(tzinfo=None)
    with pytest.raises(ValidationError, match="timezone-aware"):
        DisclosureGrant.model_validate(
            {
                **grant.model_dump(),
                "expires_at": naive_expiry,
            }
        )
    with pytest.raises(ValidationError):
        DisclosureGrant.model_validate({**grant.model_dump(), "fact_hash": "not-a-hash"})
    with pytest.raises(ValidationError):
        DisclosureGrant.model_validate({**grant.model_dump(), "game_revision": -1})


def test_record_origin_has_only_the_three_contract_values() -> None:
    assert {origin.value for origin in RecordOrigin} == {
        "model_submission",
        "repaired_submission",
        "neutral_terminal_fallback",
    }


def test_public_record_keeps_semantics_separate_from_rendered_text() -> None:
    record = PublicSpeechRecord.model_validate(_record_payload())
    rendered = RenderedUtterance(
        record_id=record.record_id,
        sentence_plan_version="1.0.0",
        renderer_version="speech-renderer-1",
        text="我目前偏向投 p03。",
        content_hash=HASH,
        fallback_status="none",
    )

    assert "text" not in PublicSpeechRecord.model_fields
    assert record.normalized_moves == _moves()
    assert rendered.record_id == record.record_id


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("schema_version", "2.0.0"),
        ("day", -1),
        ("committed_revision", 0),
        ("normalized_moves", ()),
        ("normalized_moves", _moves() * 5),
        ("source_evidence_refs", ("public-3", "public-3")),
        ("disclosure_grant_refs", ("grant-1", "grant-1")),
    ],
)
def test_public_record_rejects_invalid_bounds_and_duplicate_refs(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        PublicSpeechRecord.model_validate({**_record_payload(), field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("text", ""),
        ("text", "发" * 2001),
        ("content_hash", "not-a-hash"),
        ("fallback_status", "unknown"),
    ],
)
def test_rendered_utterance_rejects_out_of_contract_values(
    field_name: str,
    value: object,
) -> None:
    payload = {
        "record_id": "speech-5",
        "sentence_plan_version": "1.0.0",
        "renderer_version": "speech-renderer-1",
        "text": "我目前偏向投 p03。",
        "content_hash": HASH,
        "fallback_status": "template_fallback",
    }
    with pytest.raises(ValidationError):
        RenderedUtterance.model_validate({**payload, field_name: value})
