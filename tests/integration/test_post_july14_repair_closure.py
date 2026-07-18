# -*- coding: utf-8 -*-
"""
验证 7 月 14 日后全部修复在真实跨模块路径上形成闭环。

作者: Project contributors
创建日期: 2026-07-18

使用示例:
    >>> python -m pytest tests/integration/test_post_july14_repair_closure.py -q
"""

from __future__ import annotations

import json
from dataclasses import replace

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.acceptance_audit import (
    compute_acceptance_audit_metrics,
)
from werewolf_agent.evaluation.game_projection import project_acceptance_game
from werewolf_agent.runtime.decision_outcomes import (
    normalize_decision_execution_trace,
)
from werewolf_agent.runtime.event_metadata import new_game_event
from werewolf_agent.runtime.exposure_audit import (
    is_safe_public_skill_resolution_payload,
)
from werewolf_agent.runtime.nodes.night_resolution import resolve_night
from werewolf_agent.runtime.nodes.wolf_consensus import wolf_consensus
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_validated_wolf_target_stance,
)
from werewolf_agent.runtime.wolf_no_kill_policy import NoKillPolicy


def _engine() -> RuleEngine:
    """加载与生产运行相同的标准规则集。"""
    return RuleEngine.from_yaml(
        "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"
    )


def _players(*, three_wolves: bool = True) -> dict[str, PlayerState]:
    """构造只包含闭环场景所需角色的最小玩家表。"""
    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "v2": PlayerState(id="v2", role="seer"),
        "witch": PlayerState(id="witch", role="witch"),
    }
    if three_wolves:
        players.update({
            "w2": PlayerState(id="w2", role="werewolf"),
            "w3": PlayerState(id="w3", role="werewolf"),
        })
    return players


def _append_stance(
    game_state: GameState,
    *,
    wolf_id: str,
    target_id: str,
    priority: str = "primary",
) -> GameState:
    """通过生产校验器追加一条权威、私有、V2 结构化立场。"""
    round_number = len([
        event for event in game_state.events if event.type == "wolf_discussion"
    ]) + 1
    payload = {
        "wolf_id": wolf_id,
        "round": round_number,
        "night_number": game_state.night_number,
        "text": "",
        "visibility": "werewolf_team_only",
    }
    source = new_game_event(game_state, "wolf_discussion", payload)
    stance = build_validated_wolf_target_stance(
        game_state,
        source,
        wolf_id=wolf_id,
        round_number=round_number,
        raw_stance={
            "target_id": target_id,
            "stance": "support",
            "priority": priority,
        },
    )
    event = replace(
        source,
        payload={**payload, "target_stance": stance.model_dump()},
    )
    return replace(game_state, events=[*game_state.events, event])


def test_majority_kill_saved_by_witch_is_skill_cancellation_not_no_kill() -> None:
    """三狼 2:1 共识的主刀被救时，平安夜必须归于技能抵消。"""
    game_state = GameState(
        game_id="closure-majority-antidote",
        players=_players(),
        phase="night",
        night_number=1,
    )
    game_state = _append_stance(game_state, wolf_id="w1", target_id="v1")
    game_state = _append_stance(game_state, wolf_id="w2", target_id="v1")
    game_state = _append_stance(game_state, wolf_id="w3", target_id="v2")

    selected = wolf_consensus({"game_state": game_state})
    resolved = resolve_night({
        **selected,
        "engine": _engine(),
        "use_antidote": True,
        "poison_target_id": None,
    })
    final_state = resolved["game_state"]

    assert selected["wolf_kill_target_id"] == "v1"
    assert final_state.players["v1"].alive is True
    assert any(event.type == "wolf_kill_selected" for event in final_state.events)
    assert any(event.type == "witch_antidote_used" for event in final_state.events)
    assert not any(event.type.startswith("wolf_no_kill") for event in final_state.events)
    assert not any(death.player_id == "v1" for death in final_state.deaths)


def test_single_wolf_primary_executes_without_tie() -> None:
    """只剩单狼时，其合法主刀立场就是权威选择，不制造伪平票。"""
    game_state = GameState(
        game_id="closure-single-wolf",
        players=_players(three_wolves=False),
        phase="night",
        night_number=2,
    )
    game_state = _append_stance(game_state, wolf_id="w1", target_id="v2")

    result = wolf_consensus({"game_state": game_state})

    assert result["wolf_kill_target_id"] == "v2"
    selected = [
        event for event in result["game_state"].events
        if event.type == "wolf_kill_selected"
    ]
    assert selected[-1].payload["plan_key"] == "night_kill_primary"
    assert not any(
        event.type.startswith("wolf_no_kill")
        for event in result["game_state"].events
    )


def test_invalid_primary_uses_independent_majority_backup() -> None:
    """主刀结算前死亡时，只执行独立达到多数的合法备刀。"""
    players = _players()
    game_state = GameState(
        game_id="closure-backup",
        players=players,
        phase="night",
        night_number=1,
    )
    for wolf_id in ("w1", "w2"):
        game_state = _append_stance(
            game_state, wolf_id=wolf_id, target_id="v1", priority="primary"
        )
        game_state = _append_stance(
            game_state, wolf_id=wolf_id, target_id="v2", priority="backup"
        )
    dead_primary_players = dict(game_state.players)
    dead_primary_players["v1"] = replace(
        dead_primary_players["v1"], alive=False
    )
    game_state = replace(game_state, players=dead_primary_players)

    result = wolf_consensus({"game_state": game_state})

    assert result["wolf_kill_target_id"] == "v2"
    selected = [
        event for event in result["game_state"].events
        if event.type == "wolf_kill_selected"
    ]
    assert selected[-1].payload["plan_key"] == "night_kill_backup"


def test_third_pre_resolution_no_kill_recovers_deterministically() -> None:
    """前两夜不同来源空刀后，第三夜按统一策略确定性恢复。"""
    prior_events = [
        GameEvent(
            type="wolf_no_kill_timeout",
            payload={
                "night_number": 1,
                "reason": "provider_unavailable",
                "no_kill_decision": {
                    "reason_code": "provider_unavailable",
                    "consecutive_pre_resolution_no_kill_count": 1,
                    "forced_recovery_applied": False,
                    "recovered_target_id": None,
                },
            },
        ),
        GameEvent(
            type="wolf_no_kill_declared",
            payload={
                "night_number": 2,
                "reason": "strategic_abstain",
                "no_kill_decision": {
                    "reason_code": "strategic_abstain",
                    "consecutive_pre_resolution_no_kill_count": 2,
                    "forced_recovery_applied": False,
                    "recovered_target_id": None,
                },
            },
        ),
    ]
    game_state = GameState(
        game_id="closure-recovery",
        players=_players(three_wolves=False),
        phase="night",
        night_number=3,
        events=prior_events,
    )

    result = NoKillPolicy(
        max_consecutive_pre_resolution_no_kill=2
    ).resolve(
        game_state,
        reason_code="true_tie",
        primary_positive_support={"v1": 1, "v2": 1},
        backup_positive_support={"v1": 2, "v2": 1},
    )

    assert result["wolf_kill_target_id"] == "v1"
    recovery, selected = result["game_state"].events[-2:]
    assert recovery.type == "wolf_kill_forced_recovery"
    assert recovery.payload["original_reasons"] == [
        "provider_unavailable", "strategic_abstain", "true_tie"
    ]
    assert selected.type == "wolf_kill_selected"
    assert selected.payload["reason"] == "forced_recovery"


def test_graph_recursion_abort_persists_minimal_json(tmp_path) -> None:
    """图递归上限必须成为带 V2 终态事件的最小化 aborted JSON。"""
    from langgraph.errors import GraphRecursionError

    from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig

    runner = GameRunner(GameRunnerConfig(
        seed=71415,
        emergency_artifact_dir=tmp_path,
        enable_default_rag_service=False,
    ))

    class RecursiveGraph:
        """只用于触发生产执行器的递归异常边界。"""

        def stream(self, *_args, **_kwargs):
            raise GraphRecursionError("closure-limit")

    runner._graph = RecursiveGraph()
    result = runner.run()
    artifact = tmp_path / f"emergency_abort_{runner.game_id}.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert result.status == "aborted"
    assert result.termination_reason == "graph_recursion_limit"
    terminal = [event for event in result.events if event.type == "game_aborted"]
    assert len(terminal) == 1
    assert terminal[0].schema_version == "2"
    assert payload["status"] == "aborted"
    assert payload["termination_reason"] == "graph_recursion_limit"
    assert set(payload) == {
        "game_id", "status", "termination_reason", "last_node", "phase",
        "day_number", "night_number", "step", "exception_type",
        "occurred_at",
    }


def _reflection_game(*, valid: bool) -> dict[str, object]:
    """构造有效或零有效条目的已结束反思事务。"""
    events: list[dict[str, object]]
    if valid:
        events = [
            {
                "type": "reflection_complete",
                "payload": {
                    "status": "complete",
                    "player_count": 1,
                    "valid_entry_count": 1,
                    "failure_count": 0,
                    "entries": [{
                        "player_id": "p01",
                        "decision_id": "reflection:closure-reflection:p01",
                        "transaction_state": "persisted",
                        "entry_id": "reflection_closure-reflection_p01",
                        "verification": {
                            "status": "verified",
                            "decision_id": "reflection:closure-reflection:p01",
                            "verified_claim_ids": ["claim-p01"],
                            "rejected_claim_ids": [],
                            "verified_lessons": [{
                                "lesson_id": "lesson-1",
                                "abstraction": "先核验公开票型",
                            }],
                            "rejected_fact_count": 0,
                            "rejected_lesson_count": 0,
                        },
                    }],
                },
            },
            {
                "type": "reflection_persistence_audit",
                "payload": {
                    "status": "complete",
                    "expected_entry_count": 1,
                    "persistence_complete": True,
                    "rollback_complete": True,
                    "entries": [{
                        "player_id": "p01",
                        "decision_id": "reflection:closure-reflection:p01",
                        "verified_claim_ids": ["claim-p01"],
                        "entry_id": "reflection_closure-reflection_p01",
                        "row_found": True,
                        "persistence_complete": True,
                        "persisted_rejected_fact_count": 0,
                    }],
                },
            },
        ]
    else:
        events = [{
            "type": "reflection_no_valid_entries",
            "payload": {
                "expected_entry_count": 1,
                "valid_entry_count": 0,
                "persistence_complete": False,
            },
        }]
    return {
        "game_id": "closure-reflection",
        "status": "finished",
        "winning_faction": "good",
        "players": {"p01": {"id": "p01", "role": "seer"}},
        "events": events,
    }


def test_final_quality_distinguishes_valid_and_invalid_reflection() -> None:
    """终局 quality 只认可完整持久化事务，零有效条目绝不算成功。"""
    valid = compute_acceptance_audit_metrics([_reflection_game(valid=True)])
    invalid = compute_acceptance_audit_metrics([_reflection_game(valid=False)])

    assert valid["reflection_completed_game_count"] == 1
    assert valid["reflection_audited_game_count"] == 1
    assert valid["reflection_contamination_metrics_supported"] is True
    assert invalid["reflection_completed_game_count"] == 1
    assert invalid["reflection_audited_game_count"] == 0
    assert invalid["reflection_contamination_metrics_supported"] is False
    assert invalid["reflection_persisted_rejected_fact_count"] is None


def test_v1_compatibility_marks_normalized_trace_and_unsupported_projection() -> None:
    """V1 只读兼容必须显式标记归一化或缺失证据，不伪装成原生 V2。"""
    legacy_trace = {
        "execution_attempts": [{
            "opaque_request_id": "legacy-request",
            "ordinal": 1,
            "provider": "legacy-provider",
            "model": "legacy-model",
            "route_kind": "primary",
            "root_cause": "none",
            "attempt_outcome": "attempt_success",
            "requested_reasoning_level": "high",
            "normalized_reasoning_status": "requested_unconfirmed",
            "reasoning_token_count": 0,
            "evidence_kind": "none",
        }],
    }
    normalized = normalize_decision_execution_trace(legacy_trace)
    projection = project_acceptance_game({
        "game_id": "legacy-closure",
        "winning_faction": "good",
        "events": [{"type": "legacy_event", "payload": {}}],
    })

    assert normalized["normalized_from_schema_version"] == "1"
    assert "normalized_from_schema_version" not in legacy_trace
    assert projection.supported is False
    assert projection.unsupported_reason == "missing_players"
    assert projection.to_mapping()["_acceptance_projection_supported"] is False


def test_closure_public_skill_payload_has_zero_sensitive_fields() -> None:
    """公开技能结果只保留白名单字段，私有理由与身份真值泄漏为零。"""
    public_payload = {
        "actor_id": "hunter",
        "target_id": "w1",
        "public_result": "shot_resolved",
    }

    assert is_safe_public_skill_resolution_payload(public_payload) is True
    for sensitive_field in (
        "reason", "alignment", "candidate_ids", "private_reason", "role"
    ):
        assert sensitive_field not in public_payload
