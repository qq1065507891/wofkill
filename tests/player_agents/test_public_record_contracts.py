# -*- coding: utf-8 -*-
"""
验证一次性私密披露授权、公共语义记录和渲染记录契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents import contracts
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
    PrivateFactKind,
    PrivateResultDisclosure,
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


def _private_disclosure() -> PrivateResultDisclosure:
    return PrivateResultDisclosure(
        move_id="private-m1",
        move_type="private_result_disclosure",
        modality=Modality.ASSERTED,
        evidence_refs=("public-3",),
        fact_kind=PrivateFactKind.ALIGNMENT_CHECK,
        fact_ref="seer-check-1",
        disclosure_grant_id="grant-1",
        timing_ref="night-1",
        result_value_id="wolf",
        target_id="p03",
    )


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


def test_public_record_rejects_duplicate_move_ids() -> None:
    record = PublicSpeechRecord.model_validate(_record_payload())
    duplicate_move = record.normalized_moves[0].model_copy(
        update={"move_id": record.normalized_moves[1].move_id}
    )
    with pytest.raises(ValidationError, match="move IDs must not contain duplicates"):
        PublicSpeechRecord.model_validate(
            {
                **record.model_dump(),
                "normalized_moves": (duplicate_move, record.normalized_moves[1]),
            }
        )


def test_public_record_requires_every_move_evidence_ref() -> None:
    record = PublicSpeechRecord.model_validate(_record_payload())
    with pytest.raises(ValidationError, match="every move evidence ref"):
        record.model_copy(update={"source_evidence_refs": ()})

    supplemented = record.model_copy(
        update={"source_evidence_refs": ("public-3", "host-verified-4")}
    )
    assert supplemented.source_evidence_refs == ("public-3", "host-verified-4")


@pytest.mark.parametrize("disclosure_refs", [(), ("grant-1", "grant-2")])
def test_public_record_requires_exact_private_disclosure_grants(
    disclosure_refs: tuple[str, ...],
) -> None:
    payload = {
        **_record_payload(),
        "normalized_moves": (_private_disclosure(),),
        "disclosure_grant_refs": disclosure_refs,
    }
    with pytest.raises(ValidationError, match="disclosure_grant_refs must match"):
        PublicSpeechRecord.model_validate(payload)


def test_public_record_accepts_exact_private_disclosure_grant_set() -> None:
    record = PublicSpeechRecord.model_validate(
        {
            **_record_payload(),
            "normalized_moves": (_private_disclosure(),),
            "disclosure_grant_refs": ("grant-1",),
        }
    )
    assert record.disclosure_grant_refs == ("grant-1",)


def test_public_contract_facade_exports_record_contracts() -> None:
    assert contracts.DisclosureGrant is DisclosureGrant
    assert contracts.PublicSpeechRecord is PublicSpeechRecord
    assert contracts.RecordOrigin is RecordOrigin
    assert contracts.RenderedUtterance is RenderedUtterance
    assert {
        "DisclosureGrant",
        "PublicSpeechRecord",
        "RecordOrigin",
        "RenderedUtterance",
    } <= set(contracts.__all__)


def test_public_record_round_trips_nested_json() -> None:
    record = PublicSpeechRecord.model_validate(_record_payload())
    assert PublicSpeechRecord.model_validate_json(record.model_dump_json()) == record


def test_public_record_is_frozen_and_model_copy_revalidates() -> None:
    record = PublicSpeechRecord.model_validate(_record_payload())
    with pytest.raises(ValidationError, match="frozen"):
        record.day = 2
    with pytest.raises(ValidationError):
        record.model_copy(update={"day": -1})


def test_rendered_utterance_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError, match="non-whitespace"):
        RenderedUtterance(
            record_id="speech-5",
            sentence_plan_version="1.0.0",
            renderer_version="speech-renderer-1",
            text=" \n\t",
            content_hash=HASH,
            fallback_status="none",
        )

    original_text = "  我目前偏向投 p03。  "
    rendered = RenderedUtterance(
        record_id="speech-5",
        sentence_plan_version="1.0.0",
        renderer_version="speech-renderer-1",
        text=original_text,
        content_hash=HASH,
        fallback_status="none",
    )
    assert rendered.text == original_text


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
