# -*- coding: utf-8 -*-
"""
验证规范发言 schema 与仓库快照、严格结构及内容哈希完全一致。

作者: Project contributors
创建日期: 2026-07-29
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from werewolf_agent.player_agents.contracts.schema_catalog import (
    SCHEMA_VERSION,
    speech_proposal_schema,
    speech_proposal_schema_hash,
)

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "player_agents"
    / "speech_proposal_schema_v1.json"
)
EXPECTED_MOVE_TYPES = {
    "alignment_read",
    "conditional_commitment",
    "player_comparison",
    "private_result_disclosure",
    "public_evidence_citation",
    "question",
    "response",
    "retraction",
    "role_claim",
    "uncertainty",
    "vote_position",
}


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_speech_schema_matches_checked_in_fixture_and_hash() -> None:
    fixture_text = FIXTURE.read_text(encoding="utf-8")
    expected = json.loads(fixture_text)
    actual = speech_proposal_schema()

    assert actual == expected
    assert fixture_text == json.dumps(
        expected,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    assert speech_proposal_schema_hash() == expected["x-wofkill-content-hash"]
    assert len(expected["x-wofkill-content-hash"]) == 64


def test_speech_schema_hash_excludes_only_its_own_field() -> None:
    schema_without_hash = speech_proposal_schema()
    content_hash = schema_without_hash.pop("x-wofkill-content-hash")

    assert hashlib.sha256(_canonical_bytes(schema_without_hash)).hexdigest() == content_hash


def test_speech_schema_pins_version_and_canonical_contract_shape() -> None:
    schema = speech_proposal_schema()
    speech_body = schema["$defs"]["SpeechProposalBody"]
    move_items = speech_body["properties"]["moves"]["items"]
    discriminator = move_items["discriminator"]

    assert SCHEMA_VERSION == "1.0.0"
    assert schema["$id"] == "urn:wofkill:speech-proposal:1.0.0"
    assert schema["x-wofkill-schema-version"] == SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert speech_body["additionalProperties"] is False
    assert discriminator["propertyName"] == "move_type"
    assert set(discriminator["mapping"]) == EXPECTED_MOVE_TYPES
    assert len(move_items["oneOf"]) == len(EXPECTED_MOVE_TYPES)
    assert {
        branch["$ref"] for branch in move_items["oneOf"]
    } == set(discriminator["mapping"].values())
