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

D4-6 (P2): ``clean_reason`` filter set is too small — extends it from
4 placeholders to 15, and logs a warning when filtering.
"""

from __future__ import annotations

import json
import logging

import pytest

from werewolf_agent.agents.output_parser import (
    clean_reason,
    extract_json_object_candidates,
    parse_action,
    repair_json_text,
)
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


# ---------------------------------------------------------------------------
# D4-6 (P2): clean_reason filter set is too small
# ---------------------------------------------------------------------------
#
# The old filter set was only 4 placeholders: {"未说明", "无", "none",
# "null"}. Real LLM output (game trace g_3528592081 and friends) shows
# the LLM producing ~15 distinct placeholder strings for the `reason`
# / `suspect_reason` / `private_reason` fields. Anything that survives
# the parser gets logged into the audit trail and surfaced in the
# dashboard, polluting downstream review.
#
# Fix: extend the placeholder set to 15 (8 Chinese + 4 English + 3
# punctuation) and log a warning (at WARNING level) from
# `clean_reason` / `sanitize_optional_private_fields` when a real
# reason is replaced with the empty string. The log is for ops/audit
# visibility — it doesn't change the return value.


# 8 Chinese + 4 English + 3 punctuation = 15 common placeholders.
COMMON_REASON_PLACEHOLDERS: list[str] = [
    # Chinese (8) — what real Chinese LLM fills in for "no reason"
    "未说明",
    "无",
    "未知",
    "不清楚",
    "暂无",
    "未填",
    "无理由",
    "没办法",
    # English (4) — the LLM sometimes flips to English on long context
    "none",
    "null",
    "N/A",
    "n/a",
    # Punctuation (3) — what happens when the LLM gives up mid-thought
    "-",
    "?",
    "...",
]


class TestCleanReasonFiltersCommonPlaceholders:
    """D4-6: clean_reason must drop the 15 common placeholders, not just 4."""

    @pytest.mark.parametrize("placeholder", COMMON_REASON_PLACEHOLDERS)
    def test_clean_reason_filters_placeholder(self, placeholder: str) -> None:
        """Every entry in the 15-placeholder set must be cleaned to ''.

        The old code only filtered 4 entries (未说明/无/none/null). A
        change that limits the filter set to those 4 (e.g., a refactor
        that mistakenly drops the 11 new entries) would let
        placeholder text leak into the audit log. The parametrization
        enforces all 15 in one go.
        """
        assert clean_reason(placeholder) == "", (
            f"D4-6: clean_reason must filter placeholder {placeholder!r} "
            "to empty string (audit log would otherwise show a junk "
            "reason and confuse downstream review)"
        )

    @pytest.mark.parametrize("placeholder", COMMON_REASON_PLACEHOLDERS)
    def test_clean_reason_filters_with_whitespace(self, placeholder: str) -> None:
        """Whitespace padding around placeholders must also be cleaned.

        Real LLM output is rarely clean — the placeholder often comes
        back as ``" 未填 "`` or ``"N/A\n"``. The stripped-and-matched
        behavior must still drop the value.
        """
        assert clean_reason(f"  {placeholder}  \n") == "", (
            f"D4-6: clean_reason must filter {placeholder!r} even when "
            "wrapped in whitespace (LLM output is rarely trimmed)"
        )

    @pytest.mark.parametrize(
        "value",
        [
            "I suspect p07 because they voted against the seer",
            "p09's defense speech was inconsistent with their day-2 claim",
            "信任p05的查杀结果",
        ],
    )
    def test_clean_reason_preserves_real_reasons(self, value: str) -> None:
        """Real reasons must NOT be filtered — regression guard.

        The previous parametrize checks that placeholders are dropped.
        This one confirms the filter doesn't over-match and accidentally
        drop legitimate reasoning text.
        """
        assert clean_reason(value) == value, (
            f"D4-6: clean_reason must preserve real reason {value!r} "
            "(the placeholder filter is exact-match, not substring)"
        )

    def test_clean_reason_logs_warning_on_placeholder(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A WARNING must be emitted when a placeholder is filtered.

        Ops needs a signal that the LLM is filling the reason field
        with garbage. A silent filter would lose the signal entirely.
        The caplog captures the log record so the test doesn't depend
        on the global logger config.
        """
        with caplog.at_level(logging.WARNING, logger="werewolf_agent.agents.output_parser"):
            result = clean_reason("未填")
        assert result == "", "D4-6: placeholder should be filtered to ''"
        # At least one WARNING was emitted on the output_parser logger.
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and r.name == "werewolf_agent.agents.output_parser"
        ]
        assert warning_records, (
            "D4-6: clean_reason must log a WARNING on the output_parser "
            "logger when a placeholder is filtered (ops needs the signal)"
        )


# ---------------------------------------------------------------------------
# D4-7 (P2): extract_json_object_candidates must reject non-action JSON
# ---------------------------------------------------------------------------
#
# The old code returned ALL balanced JSON objects from the model text if
# none had an `action_type` field:
#
#     action_candidates = [
#         c for c in candidates
#         if '"action_type"' in c or "'action_type'" in c
#     ]
#     return action_candidates or candidates  # ← falls back to all
#
# Real LLM thought chains contain objects like
# ``{"analysis": "I think p07 is suspicious because..."}`` — balanced
# JSON, no `action_type`. The fallback then treats that as an action
# candidate. The downstream ``action_from_data`` either errors (good)
# or, worse, accepts it if ``analysis`` happens to look like a known
# field name.
#
# Fix: require the first key to be a known ``ActionType`` value OR the
# object to carry an ``action_type`` field. Otherwise raise
# ``no_action_type_found`` so the caller can fall back to its
# "no valid action" path. The thought chain never re-enters the action
# pipeline as a candidate.


class TestExtractJsonRejectsAnalysisObject:
    """D4-7: extract_json_object_candidates rejects objects without action_type."""

    def test_extract_json_rejects_analysis_object(self) -> None:
        """The canonical thought-chain JSON must not be returned.

        Real LLM output starts with a thought/analysis block. The
        parser should not mistake that for an action candidate.
        """
        text = (
            "Let me think about this.\n"
            '{"analysis": "p07 is suspicious because of their vote pattern"}\n'
            "OK now I'll act on it."
        )
        # Either: the function raises ``no_action_type_found`` ...
        # ... OR: it returns an empty list.
        # The task spec says "raise" — we accept either form, but the
        # thought-chain object must not appear in the result.
        try:
            result = extract_json_object_candidates(text)
        except ValueError as exc:
            assert "no_action_type_found" in str(exc).lower() or "no_action" in str(exc).lower(), (
                f"D4-7: if extract_json_object_candidates raises, it must "
                f"signal 'no_action_type_found' (got: {exc!r})"
            )
            result = []
        assert result == [], (
            "D4-7: extract_json_object_candidates must reject the "
            f"thought-chain object. Got: {result!r}"
        )

    def test_extract_json_rejects_thinking_object(self) -> None:
        """Same as analysis — ``thinking`` is another common LLM key."""
        text = '{"thinking": "I should vote p07 because..."}'
        try:
            result = extract_json_object_candidates(text)
        except ValueError:
            result = []
        assert result == [], (
            f"D4-7: extract_json_object_candidates must reject {{thinking:...}}. "
            f"Got: {result!r}"
        )

    def test_extract_json_rejects_meta_object(self) -> None:
        """Random LLM metadata must not be returned as a candidate."""
        text = '{"step": 1, "thought": "narrowing down", "next": "check p07"}'
        try:
            result = extract_json_object_candidates(text)
        except ValueError:
            result = []
        assert result == [], (
            f"D4-7: extract_json_object_candidates must reject step/thought meta. "
            f"Got: {result!r}"
        )

    def test_extract_json_accepts_object_with_action_type(self) -> None:
        """Sanity: a JSON object with action_type is still returned."""
        text = '{"action_type": "vote", "target_id": "p07", "reason": "suspicious"}'
        result = extract_json_object_candidates(text)
        assert len(result) == 1, (
            f"D4-7: a valid action JSON must be returned as a candidate. "
            f"Got: {result!r}"
        )
        assert "action_type" in result[0]
        assert "vote" in result[0]

    def test_extract_json_accepts_object_with_action_type_after_thought(self) -> None:
        """A real LLM reply interleaves thought and action.

        The action JSON must be returned; the thought JSON must be
        rejected. The test verifies both at once.
        """
        text = (
            '{"analysis": "thinking about p07..."}\n'
            "Here's my action:\n"
            '{"action_type": "speech", "speech": "I am the seer", '
            '"target_id": "p03", "reason": "standing with seer"}'
        )
        try:
            result = extract_json_object_candidates(text)
        except ValueError:
            pytest.fail("D4-7: must not raise when at least one action JSON is present")
        # The action JSON must be in the result; the thought chain must not.
        assert any("action_type" in c and "speech" in c for c in result), (
            f"D4-7: action JSON must be returned when present. Got: {result!r}"
        )
        assert not any("analysis" in c for c in result), (
            f"D4-7: thought-chain JSON must be filtered out. Got: {result!r}"
        )
