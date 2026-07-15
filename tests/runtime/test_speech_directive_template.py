# -*- coding: utf-8 -*-
"""Tests for the speech-quality + speech-consistency hard-constraint helpers.

NEW (v1.1.4 fallback-fix, Parts A.2 + B.2):

Two MUST-text constants live in ``runtime/directives/_shared.py`` and
are exposed via ``build_speech_quality_hard_constraints`` /
``build_speech_consistency_hard_constraints``.  These pin the
contract the LLM sees in the system prompt — without it the
previous observed rate (49/86 = 57% ``speech_quality``,
30/86 = 35% ``semantic_claim_retention``) wouldn't budge.

We test three things:
  1. The two builders return the expected key + non-empty content;
  2. The strings carry the 【硬约束 / MUST】 prefix so the LLM
     recognises them as binding;
  3. The content lists concrete, non-filler directives — empty
     would defeat the purpose.
"""

from __future__ import annotations

import re

from werewolf_agent.runtime.directives._shared import (
    build_speech_consistency_hard_constraints,
    build_speech_quality_hard_constraints,
)


def test_quality_constraints_key_and_content_present() -> None:
    out = build_speech_quality_hard_constraints()
    assert "speech_quality_constraints" in out
    body = out["speech_quality_constraints"]
    assert isinstance(body, str)
    assert len(body) > 200, "constraint text should be substantive, not a stub"


def test_quality_constraints_carry_hard_prefix() -> None:
    """The 【硬约束 / MUST】 prefix must be the first marker — the LLM
    only treats it as binding when framed as MUST.  If somebody
    rewrites this prefix to e.g. 【参考】 the LLM will treat it as
    a tip and the speech_quality fallback rate will return.
    """
    body = build_speech_quality_hard_constraints()["speech_quality_constraints"]
    # The leading bracket label is what tells the LLM this is MUST.
    assert body.startswith("【"), f"body must start with 【: {body[:30]!r}"
    assert "硬约束" in body or "MUST" in body


def test_quality_constraints_lists_required_components() -> None:
    """The 6 numbered MUST items cover stance / suspicion_target /
    vote_leaning / evidence / role-claim (PK) / no-filler."""
    body = build_speech_quality_hard_constraints()["speech_quality_constraints"]
    # Each numbered item 1..6 should appear as "1)" / "2)" ...
    for marker in ("1)", "2)", "3)", "4)", "5)", "6)"):
        assert marker in body, f"missing numbered item {marker}"
    # Concrete component keywords:
    for keyword in ("身份立场", "怀疑对象", "投票倾向", "公开依据", "角色声明"):
        assert keyword in body, f"missing required component hint: {keyword}"


def test_quality_constraints_explicitly_calls_out_filler() -> None:
    """Empty `先听一下` / `信息不足` openings are explicitly forbidden.
    This rule was added in response to the historically-dominant
    filler pattern: 49/86 = 57% of the post-2026-07-14 fallbacks
    flagged the LLM starting with `先听[后下]` or `信息不足`."""
    body = build_speech_quality_hard_constraints()["speech_quality_constraints"]
    assert "先听" in body
    assert "再观察" in body or "信息不足" in body


def test_consistency_constraints_key_and_content_present() -> None:
    out = build_speech_consistency_hard_constraints()
    assert "speech_consistency_constraints" in out
    body = out["speech_consistency_constraints"]
    assert isinstance(body, str)
    assert len(body) >= 100


def test_consistency_constraints_lock_target_id() -> None:
    """The single most important rule: re-writing must keep the same
    target_id.  30/86 fallbacks were ``semantic_claim_retention``
    triggered by the LLM shifting attacks when retry-hinted to
    rephrase.  This is the rule that pins it.
    """
    body = build_speech_consistency_hard_constraints()["speech_consistency_constraints"]
    assert "target_id" in body
    assert "不变" in body
    # The actual contract phrase is "不得新增事实声明" — accept either form.
    assert "不得新增事实" in body or "不新增事实" in body


def test_consistency_constraints_forbid_public_record_claim() -> None:
    """Inference-as-fact is the historical ``public_record_grounding``
    miss in ``validate_public_speech``.  The constraint must redirect
    to writing "我推测/我质疑" so the LLM knows it's a leaky pattern.
    """
    body = build_speech_consistency_hard_constraints()["speech_consistency_constraints"]
    assert "公开记录" in body
    assert "我推测" in body or "我质疑" in body


def test_both_constraints_have_six_or_more_numbered_items() -> None:
    """If a future refactor compresses the lists, the LLM has less
    guard-rail detail.  Pin at least 4 numbered items per side.
    """
    quality = build_speech_quality_hard_constraints()["speech_quality_constraints"]
    consistency = build_speech_consistency_hard_constraints()["speech_consistency_constraints"]
    quality_items = len(re.findall(r"\d\)", quality))
    consistency_items = len(re.findall(r"\d\)", consistency))
    assert quality_items >= 4, f"quality only has {quality_items} items"
    assert consistency_items >= 4, f"consistency only has {consistency_items} items"
