# -*- coding: utf-8 -*-
"""
验证最终验收指标不会因缺失审计证据而产生假绿。

作者: Project contributors
创建日期: 2026-07-14
"""

from __future__ import annotations

import hashlib
import json


def _world_id(assignments: dict[str, str]) -> str:
    pairs = [[player, role] for player, role in sorted(assignments.items())]
    canonical = json.dumps(
        pairs, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return f"world_{hashlib.sha256(canonical).hexdigest()}"


def test_critical_trace_without_attempts_is_invalid_and_enters_denominators() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    metrics = compute_decision_execution_metrics([{"events": [
        {"type": "action_trace_audit", "payload": {
            "task_type": "vote",
            "action_trace": {},
        }},
        {"type": "action_trace_audit", "payload": {
            "task_type": "wolf_team_plan",
            "action_trace": {"execution_attempts": []},
        }},
    ]}])

    assert metrics["decision_execution_metrics_supported"] is True
    assert metrics["decision_count"] == 2
    assert metrics["decision_execution_invalid_sequence_count"] == 2
    assert metrics["critical_task_reasoning_request_count"] == 2
    assert metrics["critical_task_reasoning_requested_count"] == 0
    assert metrics["critical_task_reasoning_request_coverage"] == 0.0
    assert metrics["critical_task_reasoning_status_request_count"] == 2
    assert metrics["critical_task_reasoning_status_explicit_rate"] == 0.0


def test_world_identity_requires_full_assignment_hash_and_prompt_subset() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    full = {"p01": "seer", "p02": "werewolf", "p03": "villager"}
    identity = _world_id(full)
    good_world = {
        "label": identity,
        "key_assignments": {"p02": "werewolf"},
        "why": ["claim:g1:0"],
    }
    forged_world = {
        "label": _world_id({"p02": "werewolf"}),
        "key_assignments": {"p02": "werewolf"},
        "why": ["claim:g1:0"],
    }
    audit = {
        "possible_worlds": {"top_worlds": [good_world, forged_world]},
        "authoritative_world_identities": [{
            "world_id": identity,
            "canonical_assignment": [[player, role] for player, role in sorted(full.items())],
        }],
        "public_evidence_ids": ["claim:g1:0"],
    }
    game = {"game_id": "g1", "players": {
        player: {"role": role} for player, role in full.items()
    }, "events": [
        {"type": "speech", "payload": {
            "speaker": "p02", "text": "我是狼人。",
            "claims": [{"type": "role", "value": "werewolf"}],
        }},
        {"type": "action_trace_audit", "payload": {
            "action_trace": {"world_model_audit": audit},
        }},
    ]}

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["possible_world_total_count"] == 2
    assert metrics["possible_world_unique_count"] == 1
    assert metrics["possible_world_unique_rate"] == 0.5


def test_world_identity_rejects_prompt_assignment_outside_authoritative_world() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    full = {"p01": "seer", "p02": "werewolf"}
    identity = _world_id(full)
    game = {"game_id": "g1", "events": [{
        "type": "action_trace_audit", "payload": {"action_trace": {
            "world_model_audit": {
                "possible_worlds": {"top_worlds": [{
                    "label": identity,
                    "key_assignments": {"p02": "seer"},
                    "why": [],
                }]},
                "authoritative_world_identities": [{
                    "world_id": identity,
                    "canonical_assignment": [["p01", "seer"], ["p02", "werewolf"]],
                }],
            },
        }},
    }]}

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["possible_world_unique_count"] == 0


def test_power_evidence_denominator_comes_from_real_damage_and_missing_trace_fails() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    complete = {
        "target_id": "p02",
        "target_evidence": {
            "selected_score": 5, "selected_signals": ["seer_check_wolf"],
        },
        "target_comparison": {
            "selected_score": 5, "selected_signals": ["seer_check_wolf"],
            "alternative_target": "p03", "alternative_score": 1,
            "alternative_signals": ["weak_suspicion"],
            "comparison_basis": "ranked public evidence score",
        },
        "friendly_fire_risk": {
            "status": "assessed", "targets": [], "basis": "无公开误伤信号",
        },
        "retain_option": {
            "action": "no_action", "available": True,
            "required": False, "reason": "可以保留技能",
        },
        "alternative_comparison": {
            "legal_alternatives": ["p03"],
            "no_legal_alternative": False,
            "alternative_target": "p03",
        },
    }
    game = {"game_id": "g-power-evidence", "players": {
        "p01": {"role": "hunter"}, "p02": {"role": "werewolf"},
        "p03": {"role": "villager"},
    }, "deaths": [
        {"player_id": "p02", "reason": "hunter_shot", "source_player_id": "p01"},
        {"player_id": "p03", "reason": "witch_poison", "source_player_id": None},
    ], "events": [
        {"type": "action_trace_audit", "payload": {
            "player_id": "p01", "task_type": "hunter_shot",
            "action_trace": {
                "final_action_type": "hunter_shot",
                "power_role_evidence": complete,
            },
        }},
        {"type": "player_died", "payload": {
            "player_id": "p02", "reason": "hunter_shot", "source_player_id": "p01",
        }},
        {"type": "player_died", "payload": {
            "player_id": "p03", "reason": "witch_poison",
        }},
    ]}

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["power_role_damage_decision_count"] == 2
    assert metrics["power_role_evidence_complete_count"] == 1
    assert metrics["power_role_evidence_completeness_rate"] == 0.5


def test_power_evidence_rejects_selected_target_as_alternative() -> None:
    from werewolf_agent.evaluation.acceptance_audit import _power_role_evidence_complete

    evidence = {
        "target_id": "p02",
        "friendly_fire_risk": {
            "status": "assessed", "targets": [], "basis": "无公开误伤信号",
        },
        "retain_option": {
            "action": "no_action", "available": True,
            "required": False, "reason": "可以保留技能",
        },
        "alternative_comparison": {
            "legal_alternatives": ["p02"],
            "no_legal_alternative": False,
            "alternative_target": "p02",
        },
    }

    assert _power_role_evidence_complete(evidence) is False


def test_completed_reflection_requires_exact_player_set_and_count() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    game = {
        "winning_faction": "good",
        "players": {"p01": {"role": "seer"}, "p02": {"role": "werewolf"}},
        "events": [
            {"type": "reflection_complete", "payload": {
                "player_count": 1,
                "entries": [{"player_id": "p01", "verification": {
                    "decision_id": "reflection:g1:p01", "verified_lessons": [],
                    "rejected_fact_count": 0, "rejected_lesson_count": 0,
                }}],
            }},
            {"type": "reflection_persistence_audit", "payload": {
                "expected_entry_count": 0, "persistence_complete": True, "entries": [],
            }},
        ],
    }

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_completed_game_with_players_cannot_use_empty_reflection_as_success() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    game = {
        "winning_faction": "werewolf",
        "players": {"p01": {"role": "werewolf"}},
        "events": [
            {"type": "reflection_complete", "payload": {
                "player_count": 0, "entries": [],
            }},
            {"type": "reflection_persistence_audit", "payload": {
                "expected_entry_count": 0, "persistence_complete": True, "entries": [],
            }},
        ],
    }

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_semantic_repair_requires_one_identity_linked_event_per_eligible_trace() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    semantic = {
        "repairable": True, "success": True, "target_preserved": True,
        "speaker_attribution_preserved": True, "negation_preserved": True,
        "introduced_claim_count": 0, "verified_claim_count": 0,
        "retained_verified_claim_count": 0, "generic_template_used": False,
        "fallback_kind": "no_fallback", "trace_id": "trace-1",
        "game_id": "g1", "action_index": 1, "task_type": "speech",
    }
    trace_event = {"type": "action_trace_audit", "payload": {
        "trace_id": "trace-1", "game_id": "g1", "action_index": 1,
        "task_type": "speech", "action_trace": {"semantic_repair_audit": {
            key: value for key, value in semantic.items()
            if key not in {"trace_id", "game_id", "action_index", "task_type"}
        }},
    }}

    missing = compute_acceptance_audit_metrics([{"game_id": "g1", "events": [trace_event]}])
    duplicate = compute_acceptance_audit_metrics([{"game_id": "g1", "events": [
        {"type": "semantic_repair_audit", "payload": semantic},
        {"type": "semantic_repair_audit", "payload": semantic},
        trace_event,
    ]}])
    exact = compute_acceptance_audit_metrics([{"game_id": "g1", "events": [
        {"type": "semantic_repair_audit", "payload": semantic}, trace_event,
    ]}])

    assert missing["semantic_repair_metrics_supported"] is False
    assert duplicate["semantic_repair_metrics_supported"] is False
    assert exact["semantic_repair_metrics_supported"] is True
    assert exact["semantic_repair_eligible_count"] == 1


def test_verified_claim_retention_requires_every_verified_claim() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    semantic = {
        "repairable": True, "success": True, "target_preserved": True,
        "speaker_attribution_preserved": True, "negation_preserved": True,
        "introduced_claim_count": 0, "verified_claim_count": 2,
        "retained_verified_claim_count": 1, "generic_template_used": False,
        "fallback_kind": "no_fallback", "trace_id": "trace-1",
        "game_id": "g1", "action_index": 1, "task_type": "speech",
    }
    trace_semantic = {
        key: value for key, value in semantic.items()
        if key not in {"trace_id", "game_id", "action_index", "task_type"}
    }
    metrics = compute_acceptance_audit_metrics([{"game_id": "g1", "events": [
        {"type": "semantic_repair_audit", "payload": semantic},
        {"type": "action_trace_audit", "payload": {
            "trace_id": "trace-1", "game_id": "g1", "action_index": 1,
            "task_type": "speech",
            "action_trace": {"semantic_repair_audit": trace_semantic},
        }},
    ]}])

    assert metrics["semantic_repair_verified_claim_retained_count"] == 0
    assert metrics["semantic_repair_verified_claim_retention_rate"] == 0.0


def test_verified_claim_retention_rejects_count_above_verified_total() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    semantic = {
        "repairable": True, "success": True, "target_preserved": True,
        "speaker_attribution_preserved": True, "negation_preserved": True,
        "introduced_claim_count": 0, "verified_claim_count": 1,
        "retained_verified_claim_count": 2, "generic_template_used": False,
        "fallback_kind": "no_fallback", "trace_id": "trace-1",
        "game_id": "g1", "action_index": 1, "task_type": "speech",
    }
    trace_semantic = {
        key: value for key, value in semantic.items()
        if key not in {"trace_id", "game_id", "action_index", "task_type"}
    }
    metrics = compute_acceptance_audit_metrics([{"game_id": "g1", "events": [
        {"type": "semantic_repair_audit", "payload": semantic},
        {"type": "action_trace_audit", "payload": {
            "trace_id": "trace-1", "game_id": "g1", "action_index": 1,
            "task_type": "speech",
            "action_trace": {"semantic_repair_audit": trace_semantic},
        }},
    ]}])

    assert metrics["semantic_repair_verified_claim_retention_metrics_supported"] is False
    assert metrics["semantic_repair_verified_claim_retention_rate"] is None


def _completed_reflection_game(
    persistence_events: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "game_id": "g1", "winning_faction": "good",
        "players": {"p01": {"role": "seer"}},
        "events": [
            {"type": "reflection_complete", "payload": {
                "player_count": 1,
                "entries": [{"player_id": "p01", "verification": {
                    "decision_id": "reflection:g1:p01",
                    "verified_lessons": [{
                        "lesson_id": "l1", "abstraction": "先核验公开票型",
                    }],
                    "rejected_fact_count": 0, "rejected_lesson_count": 0,
                }}],
            }},
            *persistence_events,
        ],
    }


def _persistence_event(
    *, expected: int = 1, entries: list[dict[str, object]] | None = None,
    rollback_complete: bool = True,
) -> dict[str, object]:
    rows = entries if entries is not None else [{
        "player_id": "p01", "decision_id": "reflection:g1:p01",
        "entry_id": "reflection_g1_p01", "row_found": True,
        "persistence_complete": True, "persisted_rejected_fact_count": 0,
    }]
    return {"type": "reflection_persistence_audit", "payload": {
        "expected_entry_count": expected, "persistence_complete": True,
        "rollback_complete": rollback_complete, "entries": rows,
    }}


def test_reflection_rejects_multiple_post_reflection_transactions() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([_completed_reflection_game([
        _persistence_event(), _persistence_event(),
    ])])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_reflection_rejects_transaction_count_mismatch_and_duplicate_rows() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    row = _persistence_event()["payload"]["entries"][0]
    mismatched = compute_acceptance_audit_metrics([_completed_reflection_game([
        _persistence_event(expected=999),
    ])])
    duplicated = compute_acceptance_audit_metrics([_completed_reflection_game([
        _persistence_event(expected=2, entries=[row, dict(row)]),
    ])])

    assert mismatched["reflection_audited_game_count"] == 0
    assert duplicated["reflection_audited_game_count"] == 0


def test_reflection_requires_successful_rollback_and_matching_decision() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    wrong_decision_row = {
        "player_id": "p01", "decision_id": "reflection:g1:other",
        "entry_id": "reflection_g1_p01", "row_found": True,
        "persistence_complete": True, "persisted_rejected_fact_count": 0,
    }
    rollback_failed = compute_acceptance_audit_metrics([_completed_reflection_game([
        _persistence_event(rollback_complete=False),
    ])])
    wrong_decision = compute_acceptance_audit_metrics([_completed_reflection_game([
        _persistence_event(entries=[wrong_decision_row]),
    ])])

    assert rollback_failed["reflection_audited_game_count"] == 0
    assert wrong_decision["reflection_audited_game_count"] == 0


def test_critical_action_event_without_trace_object_is_invalid() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    metrics = compute_decision_execution_metrics([{"events": [
        {"type": "action_trace_audit", "payload": {"task_type": "vote"}},
        {"type": "action_trace_audit", "payload": {
            "task_type": "hunter_shot", "action_trace": "corrupt",
        }},
    ]}])

    assert metrics["decision_count"] == 2
    assert metrics["decision_execution_invalid_sequence_count"] == 2
    assert metrics["critical_task_reasoning_request_count"] == 2
    assert metrics["critical_task_reasoning_request_coverage"] == 0.0


def test_world_identity_rejects_partial_self_hash_against_game_roster() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    partial = {"p01": "seer", "p02": "werewolf"}
    identity = _world_id(partial)
    game = {
        "game_id": "g1",
        "players": {
            "p01": {"role": "seer"}, "p02": {"role": "werewolf"},
            "p03": {"role": "hybrid"},
        },
        "events": [{"type": "action_trace_audit", "payload": {
            "action_trace": {"world_model_audit": {
                "possible_worlds": {"top_worlds": [{
                    "label": identity, "key_assignments": {"p02": "werewolf"},
                    "why": [],
                }]},
                "authoritative_world_identities": [{
                    "world_id": identity,
                    "canonical_assignment": [[player, role] for player, role in partial.items()],
                }],
            }},
        }}],
    }

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["possible_world_unique_count"] == 0


def test_power_damage_sources_must_match_before_evidence_can_pass() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    evidence = {
        "target_id": "p02",
        "target_evidence": {
            "selected_score": 5, "selected_signals": ["seer_check_wolf"],
        },
        "target_comparison": {
            "selected_score": 5, "selected_signals": ["seer_check_wolf"],
            "alternative_target": "p03", "alternative_score": 1,
            "alternative_signals": ["weak_suspicion"],
            "comparison_basis": "ranked public evidence score",
        },
        "friendly_fire_risk": {
            "status": "assessed", "targets": [], "basis": "无误伤信号",
        },
        "retain_option": {
            "action": "no_action", "available": True,
            "required": False, "reason": "可以保留技能",
        },
        "alternative_comparison": {
            "legal_alternatives": ["p03"], "no_legal_alternative": False,
            "alternative_target": "p03",
        },
    }
    game = {
        "game_id": "g-power-source-reconcile",
        "players": {
            "p01": {"role": "hunter"},
            "p02": {"role": "werewolf"},
            "p03": {"role": "villager"},
        },
        "deaths": [{
            "player_id": "p02", "reason": "hunter_shot", "source_player_id": "p01",
        }],
        "events": [
            {"type": "action_trace_audit", "payload": {
                "player_id": "p01", "action_trace": {
                    "final_action_type": "hunter_shot", "power_role_evidence": evidence,
                },
            }},
            {"type": "player_died", "payload": {
                "player_id": "p02", "reason": "hunter_shot", "source_player_id": "p01",
            }},
            {"type": "player_died", "payload": {
                "player_id": "p03", "reason": "hunter_shot", "source_player_id": "p01",
            }},
        ],
    }

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["power_role_evidence_metrics_supported"] is True
    assert metrics["power_role_evidence_completeness_rate"] == 0.0


def test_power_evidence_requires_selected_and_alternative_comparison_details() -> None:
    from werewolf_agent.evaluation.acceptance_audit import _power_role_evidence_complete

    legacy = {
        "target_id": "p02",
        "friendly_fire_risk": {
            "status": "assessed", "targets": [], "basis": "无误伤信号",
        },
        "retain_option": {
            "action": "no_action", "available": True,
            "required": False, "reason": "可以保留技能",
        },
        "alternative_comparison": {
            "legal_alternatives": ["p03"], "no_legal_alternative": False,
            "alternative_target": "p03",
        },
    }

    assert _power_role_evidence_complete(legacy) is False
