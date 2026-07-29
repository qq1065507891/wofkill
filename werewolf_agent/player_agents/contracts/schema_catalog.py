# -*- coding: utf-8 -*-
"""
生成并哈希供 provider adapter 使用的规范自主玩家提案 JSON Schema。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from werewolf_agent.player_agents.contracts.proposals import SpeechProposalEnvelope

SCHEMA_VERSION = "1.0.0"


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _schema_without_hash() -> dict[str, Any]:
    schema = SpeechProposalEnvelope.model_json_schema()
    schema["$id"] = f"urn:wofkill:speech-proposal:{SCHEMA_VERSION}"
    schema["x-wofkill-schema-version"] = SCHEMA_VERSION
    return schema


def speech_proposal_schema_hash() -> str:
    return hashlib.sha256(_canonical_bytes(_schema_without_hash())).hexdigest()


def speech_proposal_schema() -> dict[str, Any]:
    schema = _schema_without_hash()
    schema["x-wofkill-content-hash"] = speech_proposal_schema_hash()
    return schema
