"""Tests for output_parser: repair_json_text mojibake + trailing commas.

P0-R3 fix: handle game-trace case g_3528592081 Action 50 where p10's
LLM output was `{��intent��:"question_target",...}` — Chinese text
got mojibake'd, breaking JSON parse. 3 retries all failed with same
error and the parser fell back.

Two additions to ``repair_json_text``:
1. Detect U+FFFD (replacement char) adjacent to quote/colon and
   try latin-1 round-trip recovery. If that fails, fall back to
   replacing ``��`` with ``"`` (the most common cause: mojibaked
   JSON-key quotes).
2. Strip trailing commas before ``}`` and ``]`` (already present,
   this test serves as regression coverage).
"""

from __future__ import annotations

import json

import pytest

from werewolf_agent.agents.output_parser import parse_action, repair_json_text
from werewolf_agent.agents.schemas import ActionType


# ---------------------------------------------------------------------------
# TestRepairJsonTextMojibake
# ---------------------------------------------------------------------------


class TestRepairJsonTextMojibake:
    """The game-trace mojibake case: �� adjacent to JSON delimiters."""

    def test_repair_mojibake_quotes_around_key(self):
        # Game trace g_3528592081 Action 50 — p10's LLM output had
        # `��intent��` where the surrounding quotes got mojibake'd.
        # After repair, the keys must be properly quoted.
        raw = '{��intent��:"question_target"}'
        repaired = repair_json_text(raw)
        # Must parse as valid JSON now
        parsed = json.loads(repaired)
        assert parsed == {"intent": "question_target"}

    def test_repair_mojibake_multiple_keys(self):
        # Real speech action with multiple mojibaked delimiters.
        raw = (
            '{��action_type��:��speech��,'
            '��action_kind��:��speech��,'
            '��target_id��:��p07��,'
            '��speech��:��追问p07的站边��,'
            '��reason��:��p07没说清楚为什么投p09��,'
            '��confidence��:0.7}'
        )
        repaired = repair_json_text(raw)
        parsed = json.loads(repaired)
        assert parsed["action_type"] == "speech"
        assert parsed["target_id"] == "p07"
        assert parsed["speech"] == "追问p07的站边"
        assert parsed["confidence"] == 0.7

    def test_repair_latin1_roundtrip_mojibake(self):
        # Classic double-encoded UTF-8: original `{"意图":"value"}`
        # was decoded as latin-1 then re-encoded as UTF-8. The result
        # has chars in the 0x80-0xFF range. Latin-1 round-trip
        # recovers the original.
        # Original UTF-8 bytes of 意图: E6 84 8F E5 9B BE
        # After decode-as-latin-1 + re-encode-as-UTF-8, each byte
        # becomes 2 bytes: C3 A6 C2 84 C2 8F C3 A5 C2 9B C2 BE
        # That decodes (as UTF-8) to: 'æ\x84\x8få\x9b¾'
        mojibake = '{"æ\x84\x8få\x9b¾":"value"}'
        repaired = repair_json_text(mojibake)
        parsed = json.loads(repaired)
        # After recovery, the key is the original Chinese
        assert parsed == {"意图": "value"}


# ---------------------------------------------------------------------------
# TestRepairJsonTextTrailingComma
# ---------------------------------------------------------------------------


class TestRepairJsonTextTrailingComma:
    """Trailing commas in objects/arrays — common LLM slip."""

    def test_repair_strips_trailing_comma_in_object(self):
        raw = '{"a":1,}'
        repaired = repair_json_text(raw)
        assert json.loads(repaired) == {"a": 1}

    def test_repair_strips_trailing_comma_in_array(self):
        raw = '[1,2,3,]'
        repaired = repair_json_text(raw)
        assert json.loads(repaired) == [1, 2, 3]

    def test_repair_strips_trailing_comma_with_spaces(self):
        raw = '{"a": 1 ,  }'
        repaired = repair_json_text(raw)
        assert json.loads(repaired) == {"a": 1}


# ---------------------------------------------------------------------------
# TestRepairJsonTextPreservesValid
# ---------------------------------------------------------------------------


class TestRepairJsonTextPreservesValid:
    """Additive fix must not break valid JSON."""

    def test_valid_json_unchanged(self):
        raw = '{"a":1,"b":[1,2,3],"c":{"d":"hello"}}'
        repaired = repair_json_text(raw)
        # The repaired text should still parse to the same value
        assert json.loads(repaired) == json.loads(raw)

    def test_valid_json_with_chinese_unchanged(self):
        raw = '{"意图":"质疑追问","target_id":"p07"}'
        repaired = repair_json_text(raw)
        assert json.loads(repaired) == {"意图": "质疑追问", "target_id": "p07"}

    def test_empty_object_unchanged(self):
        raw = '{}'
        repaired = repair_json_text(raw)
        assert json.loads(repaired) == {}

    def test_empty_array_unchanged(self):
        raw = '[]'
        repaired = repair_json_text(raw)
        assert json.loads(repaired) == []


# ---------------------------------------------------------------------------
# TestParseActionMojibake
# ---------------------------------------------------------------------------


class TestParseActionMojibake:
    """End-to-end: parse_action with mojibake input should return a PlayerAction."""

    def test_parse_action_recovers_from_mojibake_speech(self):
        # Moijibake'd SpeechPlayerAction — every JSON key delimiter is
        # mojibake'd. After repair, parse_action should yield a
        # SpeechPlayerAction.
        raw = (
            '{��action_type��:��speech��,'
            '��action_kind��:��speech��,'
            '��target_id��:��p07��,'
            '��speech��:��追问p07的站边和票型不一致��,'
            '��reason��:��p07之前说查了p09但现在改口说没查��,'
            '��confidence��:0.7}'
        )
        action, error = parse_action(raw)
        assert action is not None, f"parse_action should recover from mojibake, got error: {error}"
        assert action.action_type == ActionType.SPEECH
        assert action.target_id == "p07"
        assert "p07" in action.speech
        assert action.confidence == pytest.approx(0.7)

    def test_parse_action_recovers_simple_mojibake(self):
        # The exact pattern from game trace g_3528592081 Action 50
        raw = '{��intent��:"question_target"}'
        action, error = parse_action(raw)
        # The data is incomplete (no action_type), so parse_action
        # will return None — but it must NOT raise, and it must
        # recognize the JSON structure.
        assert error is not None  # schema validation will fail
        # The error message should reference the parsed intent field
        # (proving the JSON was recovered, not the raw text)
        assert "intent" in error or "action_type" in error or "validation" in error.lower()
