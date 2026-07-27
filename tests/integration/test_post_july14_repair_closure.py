# -*- coding: utf-8 -*-
"""
验证 7 月 14 日后全部修复在真实跨模块路径上形成闭环。

作者: Project contributors
创建日期: 2026-07-18
修改日期: 2026-07-27

使用示例:
    >>> python -m pytest tests/integration/test_post_july14_repair_closure.py -q
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
import inspect

import pytest

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.acceptance_audit import (
    compute_acceptance_audit_metrics,
)
from werewolf_agent.evaluation.game_projection import (
    normalize_acceptance_games,
    project_acceptance_game,
)
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.decision_outcomes import (
    normalize_decision_execution_trace,
)
from werewolf_agent.runtime.event_metadata import (
    new_game_event,
    validate_v2_event_log_identity,
)
from werewolf_agent.runtime.exposure_audit import (
    is_safe_public_skill_resolution_payload,
)
from werewolf_agent.runtime.nodes import summary
from werewolf_agent.runtime.nodes.night_resolution import resolve_night
from werewolf_agent.runtime.nodes.skills import resolve_self_destruct_node
from werewolf_agent.runtime.nodes.wolf_consensus import wolf_consensus
from werewolf_agent.runtime.skill_opportunity_events import (
    append_self_destruct_opportunity,
    append_self_destruct_selected,
)
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_validated_wolf_target_stance,
)


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
    """通过生产校验器追加权威、私有、V2 结构化立场。"""
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
    """三狼多数主刀被救时，审计只能归因为技能抵消且不得执行备刀。"""
    game_state = GameState(
        game_id="closure-majority-antidote",
        players=_players(),
        phase="night",
        night_number=1,
    )
    game_state = _append_stance(game_state, wolf_id="w1", target_id="v1")
    game_state = _append_stance(game_state, wolf_id="w2", target_id="v1")
    game_state = _append_stance(game_state, wolf_id="w3", target_id="v2")
    game_state = _append_stance(
        game_state, wolf_id="w1", target_id="v2", priority="backup"
    )
    game_state = _append_stance(
        game_state, wolf_id="w2", target_id="v2", priority="backup"
    )
    game_state = _append_stance(
        game_state, wolf_id="w3", target_id="v1", priority="backup"
    )

    selected = wolf_consensus({"game_state": game_state})
    resolved = resolve_night({
        **selected,
        "engine": _engine(),
        "use_antidote": True,
        "poison_target_id": None,
    })
    final_state = resolved["game_state"]
    audit_counts = Counter(event.type for event in final_state.events)

    assert selected["wolf_kill_target_id"] == "v1"
    assert final_state.players["v1"].alive is True
    assert not any(death.player_id == "v1" for death in final_state.deaths)
    assert audit_counts["wolf_kill_selected"] == 1
    assert audit_counts["witch_antidote_used"] == 1
    assert audit_counts["wolf_kill_forced_recovery"] == 0
    assert not any(
        event.type.startswith("wolf_no_kill") for event in final_state.events
    )


def test_single_wolf_primary_executes_without_tie() -> None:
    """单狼同时给出合法主备立场时，主刀必须优先执行。"""
    game_state = GameState(
        game_id="closure-single-wolf",
        players=_players(three_wolves=False),
        phase="night",
        night_number=2,
    )
    game_state = _append_stance(game_state, wolf_id="w1", target_id="v2")
    game_state = _append_stance(
        game_state, wolf_id="w1", target_id="v1", priority="backup"
    )

    result = wolf_consensus({"game_state": game_state})

    assert result["wolf_kill_target_id"] == "v2"
    selected = [
        event for event in result["game_state"].events
        if event.type == "wolf_kill_selected"
    ]
    assert selected[-1].payload["plan_key"] == "night_kill_primary"
    assert selected[-1].payload["target_id"] != "v1"
    assert not any(
        event.type.startswith("wolf_no_kill")
        for event in result["game_state"].events
    )


def test_invalid_primary_uses_independent_majority_backup() -> None:
    """主刀结算前死亡时，只执行独立达到多数的合法备刀。"""
    game_state = GameState(
        game_id="closure-backup",
        players=_players(),
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
    """两条真实空刀路径后，第三条真实共识路径确定性恢复。"""
    game_state = GameState(
        game_id="closure-recovery",
        players=_players(),
        phase="night",
        night_number=1,
    )
    from werewolf_agent.runtime.nodes.wolf_discussion import wolf_team_plan_node

    provider_failure = wolf_team_plan_node({
        "game_state": game_state,
        "engine": _engine(),
    })
    first = wolf_consensus({
        "game_state": provider_failure["game_state"],
        "engine": _engine(),
    })
    second = wolf_consensus({
        "game_state": replace(first["game_state"], night_number=2),
        "engine": _engine(),
        "wolf_action": "no_kill",
        "wolf_action_reason": "strategic test route",
    })
    third_state = replace(second["game_state"], night_number=3)
    third_state = _append_stance(third_state, wolf_id="w1", target_id="v1")
    third_state = _append_stance(third_state, wolf_id="w2", target_id="v2")
    result = wolf_consensus({"game_state": third_state, "engine": _engine()})

    no_kill_events = [
        event for event in result["game_state"].events
        if event.type in {"wolf_no_kill_timeout", "wolf_no_kill_declared"}
    ]
    recovery = next(
        event for event in result["game_state"].events
        if event.type == "wolf_kill_forced_recovery"
    )
    selected = [
        event for event in result["game_state"].events
        if event.type == "wolf_kill_selected"
    ][-1]

    assert result["wolf_kill_target_id"] == "v1"
    assert [event.type for event in no_kill_events] == [
        "wolf_no_kill_timeout", "wolf_no_kill_declared"
    ]
    assert recovery.payload["original_reasons"] == [
        "provider_unavailable", "strategic_abstain", "true_tie"
    ]
    assert selected.payload["reason"] == "forced_recovery"


def test_reasoning_claim_cannot_override_structured_support_quorum() -> None:
    """计划自由文本声称全票时，结构化支持者不足仍必须空刀。"""
    game_state = GameState(
        game_id="closure-authoritative-support",
        players=_players(),
        phase="night",
        night_number=1,
    )
    game_state = _append_stance(game_state, wolf_id="w1", target_id="v1")
    claimed_plan = new_game_event(
        game_state,
        "wolf_team_plan",
        {
            "night_number": 1,
            "night_kill_primary": "v1",
            "reasoning": "三名狼人已经一致同意",
            "consensus_method": "llm",
        },
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )
    game_state = replace(game_state, events=[*game_state.events, claimed_plan])

    result = wolf_consensus({
        "game_state": game_state,
        "wolf_team_plan": claimed_plan.payload,
    })

    assert result["wolf_kill_target_id"] is None
    assert not any(
        event.type == "wolf_kill_selected"
        for event in result["game_state"].events
    )
    no_kill = next(
        event for event in result["game_state"].events
        if event.type == "wolf_no_kill_timeout"
    )
    assert no_kill.payload["reason"] == "insufficient_quorum"
    assert no_kill.payload["supporters"] == {"v1": ["w1"]}


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


def test_running_wolf_discussion_checkpoint_aborts_at_step_limit(tmp_path) -> None:
    """运行中 wolf_discussion 检查点达到 200 步时形成 step-limit 终态。"""
    from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig

    runner = GameRunner(GameRunnerConfig(
        seed=71416,
        emergency_artifact_dir=tmp_path,
        enable_default_rag_service=False,
    ))
    runner._state = GameState(
        game_id=runner.game_id,
        players=_players(),
        phase="wolf_discussion",
        status="running",
        night_number=1,
    )
    runner._step_count = 200
    runner._graph = type(
        "ExhaustedGraph",
        (),
        {"stream": lambda *_args, **_kwargs: iter(())},
    )()

    result = runner.run(max_steps=200)
    payload = json.loads(
        (tmp_path / f"emergency_abort_{runner.game_id}.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.status == "aborted"
    assert result.termination_reason == "step_limit"
    assert result.phase == "wolf_discussion"
    assert payload["step"] == 200
    assert payload["phase"] == "wolf_discussion"


def test_provider_failure_no_kill_event_has_complete_v2_audit_identity() -> None:
    """真实空刀和合法落刀节点都具有完整、稳定的 V2 审计身份。"""
    game_state = GameState(
        game_id="closure-v2-no-kill",
        players=_players(),
        phase="night",
        night_number=1,
    )
    from werewolf_agent.runtime.nodes.wolf_discussion import wolf_team_plan_node

    provider_failure = wolf_team_plan_node({
        "game_state": game_state,
        "engine": _engine(),
    })
    result = wolf_consensus({
        "game_state": provider_failure["game_state"],
        "engine": _engine(),
    })
    events = result["game_state"].events

    validate_v2_event_log_identity(game_state.game_id, events)
    no_kill = next(event for event in events if event.type == "wolf_no_kill_timeout")
    assert no_kill.event_id == (
        f"{game_state.game_id}:e{no_kill.sequence_number:06d}"
    )
    assert no_kill.game_id == game_state.game_id
    assert no_kill.schema_version == "2"
    assert no_kill.occurred_at is not None
    assert no_kill.visibility is EventVisibility.WEREWOLF_TEAM_ONLY
    assert no_kill.trace_id == DecisionIdentity(
        game_id=game_state.game_id,
        player_id="werewolf_team",
        phase="wolf_consensus",
        day_number=game_state.day_number,
        night_number=game_state.night_number,
        task_type="wolf_no_kill_timeout",
        action_index=0,
    ).trace_id()

    selected_state = replace(game_state, game_id="closure-v2-selected")
    selected_result = wolf_consensus({
        "game_state": selected_state,
        "wolf_action": "kill",
        "wolf_kill_target_id": "v1",
    })
    selected_events = selected_result["game_state"].events
    validate_v2_event_log_identity(selected_state.game_id, selected_events)
    selected = next(
        event for event in selected_events
        if event.type == "wolf_kill_selected"
    )
    assert selected.visibility is EventVisibility.WEREWOLF_TEAM_ONLY
    assert selected.trace_id == DecisionIdentity(
        game_id=selected_state.game_id,
        player_id="werewolf_team",
        phase="wolf_consensus",
        day_number=selected_state.day_number,
        night_number=selected_state.night_number,
        task_type="wolf_kill_selected:explicit_state",
        action_index=0,
    ).trace_id()


def _run_reflection_transaction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    valid: bool,
):
    """从真实终局节点运行可重复的反思生成、事务汇总与持久化审计。"""
    from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
    from werewolf_agent.storage.memory_store import InMemoryGameRepository
    from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

    repository = InMemoryGameRepository()
    runner = GameRunner(GameRunnerConfig(
        seed=71417 if valid else 71418,
        repository=repository,
        memory_coordinator=PersistentMemoryCoordinator(repository),
        enable_default_rag_service=False,
    ))
    terminal = GameState(
        game_id=runner.game_id,
        phase="finished",
        status="finished",
        winning_faction="good",
        players={"p01": PlayerState(id="p01", role="seer")},
    )
    if valid:
        decision_id = f"reflection:{runner.game_id}:p01"
        dispatch_result = {"reflection_verification": {
            "status": "verified",
            "decision_id": decision_id,
            "verified_claim_ids": ["claim-p01"],
            "rejected_claim_ids": [],
            "verified_lessons": [{
                "lesson_id": "lesson-p01",
                "abstraction": "先核验公开票型",
            }],
            "rejected_fact_count": 0,
            "rejected_lesson_count": 0,
        }}
    else:
        dispatch_result = None
    monkeypatch.setattr(
        summary,
        "_dispatch_agent",
        lambda *_args, **_kwargs: dispatch_result,
    )

    class ReflectionLifecycleGraph:
        """让公共 run 消费真实 reflection 节点输出并自行完成终局保存。"""

        @staticmethod
        def stream(initial_state, _config):
            assert initial_state["game_state"].game_id == runner.game_id
            reflected = summary.reflection({
                "game_state": terminal,
                "engine": None,
                "agent_call_delay_ms": -1,
            })
            yield {"reflection": reflected}

    runner._graph = ReflectionLifecycleGraph()
    result = runner.run()

    assert result is runner.state
    assert runner.finished is True
    return runner


def test_reflection_closure_probe_uses_public_runner_lifecycle() -> None:
    """N11 探针不得私写 runner 状态或绕过 run 的终局持久化流程。"""
    source = inspect.getsource(_run_reflection_transaction)

    assert "runner.run(" in source
    for private_bypass in (
        "runner._state", "runner._process_chunk", "runner._save_memory_snapshot",
    ):
        assert private_bypass not in source


def test_final_quality_distinguishes_valid_and_invalid_reflection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """真实反思事务决定最终 quality，且保存值逐字段等于离线重算值。"""
    from scripts.run_real_game import compute_game_quality_score, save_game_log

    valid_runner = _run_reflection_transaction(monkeypatch, valid=True)
    invalid_runner = _run_reflection_transaction(monkeypatch, valid=False)
    valid = compute_acceptance_audit_metrics([valid_runner.state])
    invalid = compute_acceptance_audit_metrics([invalid_runner.state])

    assert valid["reflection_completed_game_count"] == 1
    assert valid["reflection_audited_game_count"] == 1
    assert valid["reflection_contamination_metrics_supported"] is True
    assert invalid["reflection_completed_game_count"] == 1
    assert invalid["reflection_audited_game_count"] == 0
    assert invalid["reflection_contamination_metrics_supported"] is False
    assert invalid["reflection_persisted_rejected_fact_count"] is None

    for runner in (valid_runner, invalid_runner):
        projection = project_acceptance_game(
            runner.state, steps=runner.step_count
        )
        saved_quality = compute_game_quality_score(projection)
        path = save_game_log(
            runner,
            elapsed=0.1,
            projection=projection,
            quality_score=saved_quality,
            output_dir=tmp_path,
        )
        complete_json = json.loads(path.read_text(encoding="utf-8"))
        offline_quality = compute_game_quality_score(
            project_acceptance_game(complete_json)
        )
        assert complete_json["quality_score"] == saved_quality
        assert offline_quality.keys() == saved_quality.keys()
        mismatches = {
            field: {
                "saved": saved_quality[field],
                "offline": offline_quality[field],
            }
            for field in saved_quality
            if offline_quality[field] != saved_quality[field]
        }
        assert not mismatches, mismatches
        if runner is invalid_runner:
            assert offline_quality["reflection_audited_game_count"] == 0
            assert (
                offline_quality["reflection_contamination_metrics_supported"]
                is False
            )
            assert offline_quality[
                "reflection_contamination_metrics_unsupported_reason"
            ] == "reflection_no_valid_entries"


def test_v1_compatibility_marks_normalized_trace_and_unsupported_projection() -> None:
    """同一份 V1 日志经兼容投影后同时给出 normalized 与 unsupported。"""
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
    legacy_log = {
        "game_id": "legacy-closure",
        "winning_faction": "good",
        "events": [{
            "type": "action_trace_audit",
            "payload": {
                "task_type": "speech",
                "action_trace": legacy_trace,
            },
        }],
    }
    compatibility_games = normalize_acceptance_games([legacy_log])
    projected_trace = compatibility_games[0]["events"][0]["payload"][
        "action_trace"
    ]
    normalized = normalize_decision_execution_trace(projected_trace)
    projection = project_acceptance_game(compatibility_games[0])
    metrics = compute_acceptance_audit_metrics(compatibility_games)

    assert normalized["normalized_from_schema_version"] == "1"
    assert "normalized_from_schema_version" not in legacy_trace
    assert projection.supported is False
    assert projection.unsupported_reason == "missing_players"
    assert projection.to_mapping()["_acceptance_projection_supported"] is False
    assert metrics["decision_count"] == 1
    assert metrics["attempt_retry_consistency_error_count"] == 0
    assert metrics["acceptance_projection_unsupported_reason"] == "missing_players"


def test_closure_public_skill_payload_has_zero_sensitive_fields() -> None:
    """检查生产节点生成的公开事件，而不是手写安全字典。"""
    game_state = GameState(
        game_id="closure-public-skill",
        players=_players(),
        phase="day",
        day_number=1,
    )
    game_state, offered = append_self_destruct_opportunity(
        game_state,
        actor_id="w1",
        day_number=1,
        opportunity_phase="day_discussion",
    )
    game_state, selected = append_self_destruct_selected(
        game_state,
        actor_id="w1",
        day_number=1,
        opportunity_phase="day_discussion",
    )
    assert offered is True and selected is True

    result = resolve_self_destruct_node({
        "game_state": game_state,
        "engine": _engine(),
        "self_destruct_wolf_id": "w1",
    })
    majority_state = GameState(
        game_id="closure-public-majority",
        players=_players(),
        phase="night",
        night_number=1,
    )
    for wolf_id, target_id in (
        ("w1", "v1"), ("w2", "v1"), ("w3", "v2")
    ):
        majority_state = _append_stance(
            majority_state, wolf_id=wolf_id, target_id=target_id
        )
    majority_selected = wolf_consensus({"game_state": majority_state})
    majority_result = resolve_night({
        **majority_selected,
        "engine": _engine(),
        "use_antidote": True,
        "poison_target_id": None,
    })
    from werewolf_agent.runtime.nodes.wolf_discussion import wolf_team_plan_node

    failure_plan = wolf_team_plan_node({
        "game_state": GameState(
            game_id="closure-public-provider-failure",
            players=_players(),
            phase="night",
            night_number=1,
        ),
        "engine": _engine(),
    })
    provider_failure_result = wolf_consensus({
        "game_state": failure_plan["game_state"],
        "engine": _engine(),
    })
    produced_states = (
        result["game_state"],
        majority_result["game_state"],
        provider_failure_result["game_state"],
    )
    public_events = [
        event for produced_state in produced_states
        for event in produced_state.events
        if event.visibility is EventVisibility.PUBLIC
    ]
    skill_events = [
        event for event in public_events
        if event.type == "self_destruct_resolved"
    ]

    assert len(skill_events) == 1
    assert skill_events[0].payload["day_number"] == 1
    assert is_safe_public_skill_resolution_payload(
        skill_events[0].payload
    ) is True
    serialized_public = json.dumps(
        [event.payload for event in public_events], ensure_ascii=False
    )
    for sensitive_field in (
        '"alignment"', '"candidate_ids"', '"private_reason"', '"role"'
    ):
        assert sensitive_field not in serialized_public
