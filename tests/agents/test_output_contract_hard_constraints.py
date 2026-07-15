# -*- coding: utf-8 -*-
"""Tests for ``PromptSystemMixin._build_output_contract`` JSON hard constraints (v1.1.4 fallback-fix Part D.1).

In the 4 game logs captured on/after 2026-07-14, ``truncated_json``,
``schema_validation``, and ``parse_error`` together accounted for 6/86
fallbacks (7%) — the smallest of the 6 error-code buckets but still
non-zero.  These three all share a common root: the LLM did not see,
at the system-prompt level, a hard statement of the JSON shape
contract, and only learned about it via the retry hint *after* the
first failure.

The fix pins the contract text into ``_build_output_contract`` so the
LLM sees it on attempt 1, not just attempt 2+.  This test re-reads
the renderer to confirm the contract carries five specific invariants.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from werewolf_agent.agents.prompt_system import PromptSystemMixin


def _make_mixin() -> PromptSystemMixin:
    """Return a minimally-initialised PromptSystemMixin instance.

    The mixin needs ``self.context``, ``self.player_name``, and
    ``self._USER_SECTION_SPECS`` attribute access.  We hand-roll a
    one-off object so we don't need a real ``AgentContext`` /
    ``PlayerPromptBuilder`` wiring to test the static-ish content.
    """
    obj = PromptSystemMixin.__new__(PromptSystemMixin)
    obj.context = MagicMock()
    obj.player_name = "p_tester"
    return obj


def test_output_contract_starts_with_structured_output_marker() -> None:
    """The original 【结构化输出】 prefix is the leading MUST marker.
    If somebody deletes the prefix the LLM loses the contract signal.
    """
    body = _make_mixin()._build_output_contract()
    assert body.startswith("【结构化输出】")


def test_output_contract_contains_json_hard_constraints() -> None:
    """NEW (v1.1.4 Part D.1): the JSON hard-constraints block must
    appear after the original structured-output paragraph.  We check
    the prefix marker that introduces it so future refactors don't
    silently drop it.
    """
    body = _make_mixin()._build_output_contract()
    assert "JSON 形式硬约束 / MUST" in body, (
        "fallback-fix Part D.1 missing — _build_output_contract "
        "must include the 【JSON 形式硬约束 / MUST】 block"
    )


def test_output_contract_lists_all_five_json_hard_constraints() -> None:
    """The 5 JSON invariants:
      (1) JSON must end with `}` (no truncation, no markdown fence)
      (2) JSON char count ≤ 4000 (speech) / ≤ 800 (vote/wolf/night)
      (3) ``speech`` required, ``reason`` required, ``confidence`` ∈ [0,1]
      (4) no extra fields (Pydantic extra='forbid' rejects)
      (5) ``target_id`` must be in legal set or null
    """
    body = _make_mixin()._build_output_contract()

    # (1) ending with `}` and no markdown fence
    assert "`}`" in body
    assert "markdown" in body.lower() or "Markdown" in body
    # (2) char count caps
    assert "4000" in body and "800" in body
    # (3) field requirements
    assert "speech" in body and "reason" in body and "confidence" in body
    assert "[0,1]" in body
    # (4) extra='forbid'
    assert "extra" in body and "forbid" in body
    # (5) target_id legality
    assert "target_id" in body
    assert "null" in body


def test_output_contract_includes_failure_modes() -> None:
    """The contract must name the failure modes it prevents — this is
    what makes it read like a *constraint* to the LLM rather than
    a tip.  Each rule names the error_code we'd otherwise see.
    """
    body = _make_mixin()._build_output_contract()
    for mode in ("truncated_json", "schema_validation", "parse_error"):
        assert mode in body, f"missing failure mode name: {mode}"


def test_output_contract_mentions_must_tier() -> None:
    body = _make_mixin()._build_output_contract()
    # The contract must label itself as MUST-tier so the LLM treats it
    # as binding.  In _build_output_contract the single 【JSON 形式硬约束
    # / MUST】marker carries this signal — the original 【结构化输出】
    # paragraph is the contract preamble (no explicit MUST label).
    assert "MUST" in body
    assert body.count("MUST") >= 1
