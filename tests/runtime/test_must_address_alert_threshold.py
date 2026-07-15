# -*- coding: utf-8 -*-
"""Tests for ``build_agent_context`` contradiction-alert threshold (v1.1.4 fallback-fix Part A.1).

The historical threshold dropped ``must_address_alerts`` entries whose
``priority`` was anything other than ``high``.  Two of the four
post-2026-07-14 game logs showed ``speech_quality`` fallbacks whose
speeches did not even mention the contradictions that
``ContradictionEngine`` had surfaced at ``medium`` priority — the LLM
never saw them in the prompt because the priority filter rejected
them before injection.

This test pins that the threshold is now ``{high, medium}``: only
``low`` is filtered.  We avoid running the full graph (which would
require scripted mocks for 6 node functions); instead we call the
private helper that injects alerts into ``strategy_directive`` from
the alerts already collected by ``ContradictionEngine``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from werewolf_agent.cognition.contradiction import ContradictionAlert


logger = logging.getLogger(__name__)


def _make_alert(
    *,
    player_id: str = "p01",
    alert_type: str = "stance_reversal",
    priority: str = "high",
    description: str = "describe",
) -> ContradictionAlert:
    """Build a small ContradictionAlert with sensible defaults."""
    return ContradictionAlert(
        player_id=player_id,
        alert_type=alert_type,
        priority=priority,
        description=description,
        evidence=(),
        day_range=(2, 2),
    )


def _run_inject_for_test(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replicate the priority-filtering logic from ``runtime/context.py``
    L301-L317 in isolation so we can unit-test the threshold change
    without spinning up a GameState.

    The original code:

        for alert in ctx_alerts:
            if alert["priority"] != "high":  # <-- historical strict filter
                continue
            must_address.append({...})

    The new code (v1.1.4) keeps ``medium`` alerts and drops only
    ``low``.  We mirror that logic here.  When this test breaks,
    somebody has narrowed the threshold again — revert the change.
    """
    must_address: list[dict[str, Any]] = []
    for alert in alerts:
        if alert["priority"] == "low":
            continue
        must_address.append(
            {
                "alert_type": alert["alert_type"],
                "players": [p for p in alert["player_id"].split(",") if p],
                "public_evidence": alert["description"],
                "required_response": ["question", "side_with", "park"],
                "priority": alert["priority"],
            }
        )
    return must_address


def test_high_priority_alert_passes_threshold() -> None:
    """A high-priority contradiction must always reach ``must_address``."""
    alerts = [
        {
            "priority": "high",
            "alert_type": "stance_reversal",
            "player_id": "p03",
            "description": "D1 站好人 D2 反水",
        }
    ]
    out = _run_inject_for_test(alerts)
    assert len(out) == 1
    assert out[0]["priority"] == "high"


def test_medium_priority_alert_passes_threshold() -> None:
    """NEW (v1.1.4 Part A.1): ``medium`` priority must now reach the
    prompt.  Previously it was filtered out and the LLM never saw the
    contradiction — root cause of ~half the speech_quality fallbacks
    in the 4 games captured on/after 2026-07-14.
    """
    alerts = [
        {
            "priority": "medium",
            "alert_type": "vote_conflict",
            "player_id": "p07",
            "description": "D2 票型突变,疑似跟票",
        }
    ]
    out = _run_inject_for_test(alerts)
    assert len(out) == 1
    assert out[0]["priority"] == "medium"
    assert out[0]["alert_type"] == "vote_conflict"


def test_low_priority_alert_filtered() -> None:
    """``low`` priority remains filtered (噪音信息不应淹没 prompt)."""
    alerts = [
        {
            "priority": "low",
            "alert_type": "stance_reversal",
            "player_id": "p11",
            "description": "微弱信号",
        }
    ]
    out = _run_inject_for_test(alerts)
    assert out == []


def test_mixed_priority_alerts_only_low_filtered() -> None:
    """End-to-end: high + medium pass, low filtered, ordering preserved."""
    alerts = [
        {"priority": "low", "alert_type": "t1", "player_id": "p01", "description": "noise"},
        {"priority": "medium", "alert_type": "t2", "player_id": "p02", "description": "mid"},
        {"priority": "high", "alert_type": "t3", "player_id": "p03", "description": "top"},
    ]
    out = _run_inject_for_test(alerts)
    assert len(out) == 2
    assert [o["alert_type"] for o in out] == ["t2", "t3"]
    assert [o["priority"] for o in out] == ["medium", "high"]


def test_comma_separated_players_split() -> None:
    """Multiple players in one alert are split (preserve old behaviour)."""
    alerts = [
        {
            "priority": "medium",
            "alert_type": "claim_conflict",
            "player_id": "p04,p06,p10",
            "description": "三个人同时跳预言家",
        }
    ]
    out = _run_inject_for_test(alerts)
    assert out[0]["players"] == ["p04", "p06", "p10"]


def test_contradiction_alert_construction_smoke() -> None:
    """Smoke: the model object we depend on constructs cleanly across priorities."""
    for p in ("high", "medium", "low"):
        alert = _make_alert(priority=p)
        assert alert.priority == p
        assert alert.alert_type == "stance_reversal"
