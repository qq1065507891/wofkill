# -*- coding: utf-8 -*-
"""
验证语义修复说话者归属与否定关系进入最终验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-15
"""

from __future__ import annotations


def _game(*, speaker_preserved: object, negation_preserved: object) -> dict[str, object]:
    semantic = {
        "repairable": True,
        "success": True,
        "target_preserved": True,
        "speaker_attribution_preserved": speaker_preserved,
        "negation_preserved": negation_preserved,
        "introduced_claim_count": 0,
        "verified_claim_count": 1,
        "retained_verified_claim_count": 1,
        "generic_template_used": False,
        "fallback_kind": "no_fallback",
    }
    identity = {
        "trace_id": "trace-1",
        "game_id": "g1",
        "action_index": 1,
        "task_type": "speech",
    }
    return {
        "game_id": "g1",
        "players": {"p01": {"role": "villager"}},
        "events": [
            {
                "type": "semantic_repair_audit",
                "payload": {**semantic, **identity},
            },
            {
                "type": "action_trace_audit",
                "payload": {
                    **identity,
                    "action_trace": {"semantic_repair_audit": semantic},
                },
            },
        ],
    }


def test_acceptance_reports_speaker_and_negation_preservation_rates() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(speaker_preserved=True, negation_preserved=True),
    ])

    assert metrics["semantic_repair_metrics_supported"] is True
    assert metrics["semantic_repair_speaker_attribution_preservation_rate"] == 1.0
    assert metrics["semantic_repair_negation_preservation_rate"] == 1.0


def test_acceptance_fails_closed_when_semantic_invariant_is_missing() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(speaker_preserved=None, negation_preserved=True),
    ])

    assert metrics["semantic_repair_metrics_supported"] is False
    assert metrics["semantic_repair_speaker_attribution_preservation_rate"] is None
    assert metrics["semantic_repair_negation_preservation_rate"] is None
