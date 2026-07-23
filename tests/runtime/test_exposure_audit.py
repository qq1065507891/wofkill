# -*- coding: utf-8 -*-
"""
验证运行时模块曝光审计事件的采集、脱敏与决策身份关联。

作者: Project contributors
修改日期: 2026-07-23
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime import exposure_audit
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime import agent_adapter
from werewolf_agent.runtime.nodes._shared import _allocate_decision_identity
from werewolf_agent.runtime.nodes import day as day_nodes
from werewolf_agent.runtime.nodes import night as night_nodes
from werewolf_agent.runtime.nodes import sheriff as sheriff_nodes
from werewolf_agent.runtime.nodes import sheriff_pk as sheriff_pk_nodes
from werewolf_agent.runtime.nodes import skills as skill_nodes


def _identity() -> DecisionIdentity:
    return DecisionIdentity("g1", "p01", "vote", 2, 1, "vote", 4)


def _provider_payload_hmac(provider: Any, *, game_id: str) -> str:
    collector = ModuleExposureAuditCollector(prompt_proof_key_provider=provider)
    collector.record_provider_persona_prompt_proof(
        DecisionIdentity(game_id, "p01", "vote", 1, 1, "vote", 1),
        b"stable final system",
        "",
        "initial",
        attempt_ordinal=1,
        provider="test",
        model="test",
        final_system_location="messages",
        final_system_message_index=0,
        provider_payload_bytes=b"stable provider payload",
    )
    return collector.flush_events()[0].payload["proof"]["provider_payload_hmac_sha256"]


def test_runtime_node_collectors_always_receive_state_prompt_proof_key_provider() -> None:
    nodes_dir = Path(day_nodes.__file__).parent
    missing_provider: list[str] = []
    for source_path in nodes_dir.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            if not (
                isinstance(call.func, ast.Name)
                and call.func.id == "ModuleExposureAuditCollector"
            ):
                continue
            if not any(
                keyword.arg == "prompt_proof_key_provider"
                for keyword in call.keywords
            ):
                missing_provider.append(f"{source_path.name}:{call.lineno}")

    assert missing_provider == []


def test_same_game_id_uses_one_runner_key_but_distinct_runners_use_distinct_keys() -> None:
    from werewolf_agent.runtime.game_runner import GameRunner
    from werewolf_agent.runtime.game_runner_config import GameRunnerConfig

    config = GameRunnerConfig(
        game_id="same-game-id",
        enable_default_rag_service=False,
    )
    first_runner = GameRunner(config)
    second_runner = GameRunner(config)
    first_state = first_runner._build_runtime_state()
    second_state = second_runner._build_runtime_state()

    first_provider = first_state["prompt_proof_key_provider"]
    second_provider = second_state["prompt_proof_key_provider"]
    assert first_provider is first_runner._prompt_proof_key_provider
    assert second_provider is second_runner._prompt_proof_key_provider
    assert first_provider is not second_provider

    first_proof = _provider_payload_hmac(first_provider, game_id="same-game-id")
    assert first_proof == _provider_payload_hmac(first_provider, game_id="same-game-id")
    assert first_proof != _provider_payload_hmac(second_provider, game_id="same-game-id")


def test_collector_builds_rag_reflection_skill_persona_events() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_rag(_identity(), [{"entry_id": "rag1", "rank": 1, "title": "safe"}])
    collector.record_reflection(_identity(), [{"entry_id": "ref1", "rank": 1}])
    collector.record_skill(_identity(), {"vote_analysis": "push p02"})
    collector.record_persona(_identity(), {"profile_id": "aggressive", "effective_params": {"risk": 0.8}})

    events = collector.flush_events()

    assert [event.type for event in events] == [
        "rag_exposure_audit",
        "reflection_exposure_audit",
        "skill_exposure_audit",
        "persona_exposure_audit",
    ]
    assert all(event.payload["trace_id"] == "g1:p01:vote:D2:N1:vote:4" for event in events)
    assert all(event.payload["visibility"] == "moderator_only" for event in events)


def test_v2_stamping_preserves_private_visibility_without_copying_payload() -> None:
    from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
    from werewolf_agent.runtime.event_metadata import stamp_new_events

    collector = ModuleExposureAuditCollector()
    collector.record_skill(_identity(), {"vote_analysis": "push p02"})
    original = collector.flush_events()[0]
    expected_payload = dict(original.payload)
    expected_payload.pop("visibility")

    event = stamp_new_events("g1", [], [original])[0]

    assert event_visibility(event) is EventVisibility.MODERATOR_ONLY
    assert "visibility" not in event.payload
    assert event.payload == expected_payload


def test_v2_event_id_contains_no_player_identity_truth() -> None:
    from werewolf_agent.runtime.event_metadata import stamp_new_events

    collector = ModuleExposureAuditCollector()
    collector.record_skill(_identity(), {"vote_analysis": "push p02"})

    event = stamp_new_events("g1", [], collector.flush_events())[0]

    assert event.event_id == "g1:e000000"
    assert _identity().player_id not in event.event_id


def test_collector_strips_forbidden_private_fields() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_rag(_identity(), [{"entry_id": "x", "target_role": "werewolf"}])
    payload = collector.flush_events()[0].payload
    assert "target_role" not in str(payload)
    assert "werewolf" not in str(payload)


def test_exposure_sanitizer_always_denies_private_wolf_stance_fields() -> None:
    """即使未来误扩 allowlist，曝光审计也不能带出狼身份或 stance。"""
    private_payload = {
        "wolf_id": "wolf1",
        "target_stance": {"stance": "propose", "target_id": "villager1"},
        "source_event_id": "g1:e000001",
    }

    assert exposure_audit._sanitize_allowed(
        private_payload,
        frozenset(private_payload),
    ) == {}


def test_sanitized_empty_rag_and_reflection_do_not_emit_events() -> None:
    collector = ModuleExposureAuditCollector()

    collector.record_rag(_identity(), [{"target_role": "werewolf"}])
    collector.record_reflection(_identity(), [{"private_note": "unsafe"}])

    assert collector.flush_events() == []


def test_dict_skill_analyses_include_default_advice_type() -> None:
    collector = ModuleExposureAuditCollector()

    collector.record_skill(_identity(), {"vote_analysis": "push p02"})

    events = collector.flush_events()
    assert events[0].payload["analyses"] == [
        {
            "skill_name": "vote_analysis",
            "rank": 1,
            "prompt_visible": True,
            "summary_hash": events[0].payload["analyses"][0]["summary_hash"],
            "advice_type": "tactical",
        }
    ]


def test_collector_records_detailed_skill_tool_call_rows() -> None:
    collector = ModuleExposureAuditCollector()

    collector.record_skill_tool_calls(
        _identity(),
        [
            {
                "call_kind": "skill",
                "call_name": "push_vote",
                "skill_name": "push_vote",
                "status": "success",
                "success": True,
                "input_summary": {
                    "role": "villager",
                    "phase": "vote",
                    "task_type": "vote",
                    "legal_target_count": 3,
                    "private_role": "werewolf",
                },
                "output_summary": {
                    "confidence": 0.82,
                    "has_prompt_injectable": True,
                    "private_notes": "must not leak",
                },
                "result_available_to_decision": True,
                "decision_usage": "prompt_injected",
            }
        ],
    )

    events = collector.flush_events()

    assert [event.type for event in events] == ["skill_tool_call_audit"]
    payload = events[0].payload
    assert payload["trace_id"] == "g1:p01:vote:D2:N1:vote:4"
    row = payload["calls"][0]
    assert row["call_kind"] == "skill"
    assert row["call_name"] == "push_vote"
    assert row["status"] == "success"
    assert row["success"] is True
    assert row["input_summary"] == {
        "role": "villager",
        "phase": "vote",
        "task_type": "vote",
        "legal_target_count": 3,
    }
    assert row["output_summary"] == {
        "confidence": 0.82,
        "has_prompt_injectable": True,
    }
    assert "werewolf" not in str(payload)
    assert "private_notes" not in str(payload)


def test_action_audit_emits_model_tool_call_monitor_event() -> None:
    from werewolf_agent.runtime.nodes.action_audit import _action_audit_events

    collector = ModuleExposureAuditCollector()
    action_trace = {
        "tool_call_required": True,
        "tool_call_received": False,
        "tool_call_name": "submit_player_action",
        "parse_success": False,
        "fallback_reason": "fallback: retries exhausted",
        "retry_count": 3,
        "structured_failure_reason": "missing_tool_call",
        "structured_failure_stage": "protocol",
        "structured_output_mode": "native_tool",
    }

    events = _action_audit_events(
        state={},
        player_id="p01",
        phase="vote",
        action_trace=action_trace,
        decision_identity=_identity(),
        exposure_collector=collector,
        day_number=2,
        night_number=1,
    )

    monitor = [event for event in events if event.type == "skill_tool_call_audit"]
    assert monitor
    row = monitor[0].payload["calls"][0]
    assert row["call_kind"] == "tool"
    assert row["call_name"] == "submit_player_action"
    assert row["required"] is True
    assert row["received"] is False
    assert row["status"] == "missing"
    assert row["fallback_triggered"] is True
    assert row["decision_usage"] == "not_used_fallback"


def test_action_tool_call_monitor_marks_received_parse_failure_as_failed() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_action_tool_call(
        _identity(),
        {
            "tool_call_required": True,
            "tool_call_received": True,
            "tool_call_name": "submit_player_action",
            "parse_success": False,
            "retry_count": 1,
            "structured_failure_reason": "invalid_tool_arguments",
            "structured_failure_stage": "parse",
        },
    )

    events = collector.flush_events()
    row = events[0].payload["calls"][0]
    assert row["received"] is True
    assert row["status"] == "parse_failed"
    assert row["success"] is False
    assert row["decision_usage"] == "not_used_parse_failed"


def test_action_tool_call_monitor_fail_closes_unsafe_decision_trace_values() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_action_tool_call(
        _identity(),
        {
            "tool_call_required": True,
            "tool_call_received": False,
            "generated_by": "prompt: player p01 is werewolf",
            "terminal_failure_code": "identity_role_werewolf_p01",
        },
    )

    row = collector.flush_events()[0].payload["calls"][0]

    assert row["generated_by"] == "unknown"
    assert row["terminal_failure_code"] == "unknown"
    assert row["value_sanitization"] == [
        "generated_by_invalid",
        "terminal_failure_code_invalid",
    ]
    assert "werewolf" not in str(row)
    assert "p01" not in str(row)


def test_action_tool_call_monitor_fail_closes_terminal_fallback_metadata() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_action_tool_call(
        _identity(),
        {
            "tool_call_required": True,
            "tool_call_received": False,
            "generated_by": "terminal_fallback",
            "terminal_failure_code": "schema_validation",
            "original_failure_code": "private_role_werewolf_p01",
            "failure_stage": "private_role_werewolf_p01",
            "fallback_kind": "private_role_werewolf_p01",
        },
    )

    row = collector.flush_events()[0].payload["calls"][0]

    assert row["terminal_failure_code"] == "schema_validation"
    assert row["original_failure_code"] == "unknown"
    assert row["failure_stage"] == "unknown"
    assert row["fallback_kind"] == "unknown"
    assert "werewolf" not in str(row)
    assert "p01" not in str(row)


@pytest.mark.parametrize(
    "malicious_value",
    [
        {"generated_by": "nested_private_secret"},
        ["nested_private_secret", {"terminal_failure_code": "nested_private_secret"}],
        7,
        True,
        None,
    ],
)
def test_action_tool_call_monitor_rejects_non_string_decision_trace_values(
    malicious_value: object,
) -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_action_tool_call(
        _identity(),
        {
            "tool_call_required": True,
            "tool_call_received": False,
            "generated_by": malicious_value,
            "terminal_failure_code": malicious_value,
        },
    )

    row = collector.flush_events()[0].payload["calls"][0]

    assert row["generated_by"] == "unknown"
    assert row["terminal_failure_code"] == "unknown"
    assert row["value_sanitization"] == [
        "generated_by_invalid",
        "terminal_failure_code_invalid",
    ]
    assert "nested_private_secret" not in str(row)


def test_collector_records_prompt_injection_rows_without_raw_content() -> None:
    collector = ModuleExposureAuditCollector()
    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="vote",
        day_number=2,
        public_summary="p02 claimed seer; hidden raw text must be hashed only",
        visible_world_state={"alive_players": ["p01", "p02"], "private_role": "werewolf"},
        strategy_directive={"must_address_alerts": [{"alert_type": "claim_conflict"}]},
        skill_analyses={"push_vote": "raw tactical advice should not appear"},
    )

    collector.record_prompt_injections(_identity(), context)

    events = collector.flush_events()
    assert [event.type for event in events] == ["prompt_injection_audit"]
    rows = events[0].payload["injections"]
    by_field = {row["field_path"]: row for row in rows}
    assert by_field["public_summary"]["module_name"] == "public_summary"
    assert by_field["public_summary"]["injected"] is True
    assert by_field["public_summary"]["char_count"] > 0
    assert by_field["public_summary"]["content_hash"]
    assert by_field["strategy_directive"]["item_count"] == 1
    assert by_field["skill_analyses"]["decision_usage"] == "prompt_context_available"
    assert "hidden raw text" not in str(events[0].payload)
    assert "raw tactical advice" not in str(events[0].payload)
    assert "private_role" not in str(events[0].payload)
    assert "werewolf" not in str(events[0].payload)


def test_persona_exposure_can_be_recorded_after_prompt_visible_attachment() -> None:
    agent = PlayerAgent(
        agent_id="p01",
        model_router=None,  # type: ignore[arg-type]
        persona_key=None,
        persona_router=None,
    )
    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        persona_snapshot={
            "profile_id": "bold_pretender",
            "personality": "bold_deceiver",
            "speech_style": "confident_fake_claim",
            "task_style": "fake_authority",
            "effective_params": {"deception_skill": 0.91, "logic_skill": 0.55},
        },
    )
    collector = ModuleExposureAuditCollector()

    attached = agent._attach_persona_snapshot(context)
    collector.record_persona(_identity(), attached.persona_snapshot)

    events = collector.flush_events()
    assert [event.type for event in events] == ["persona_exposure_audit"]
    snapshot = events[0].payload["snapshot"]
    assert snapshot["profile_id"] == "bold_pretender"
    assert snapshot["sanitized"] is True
    assert "effective_params" not in snapshot
    assert "deception_skill" not in str(snapshot)


def test_persona_final_message_proof_is_private_and_run_scoped() -> None:
    identity = _identity()
    persona_text = "persona-secret-text"
    messages = (
        {"role": "system", "content": f"rules\n{persona_text}"},
        {"role": "user", "content": "public action context"},
    )
    first = ModuleExposureAuditCollector()
    second = ModuleExposureAuditCollector()

    system_bytes = messages[0]["content"].encode("utf-8")
    for collector, kind, ordinal in (
        (first, "primary", 1),
        (first, "semantic_retry", 2),
        (second, "primary", 1),
    ):
        collector.record_provider_persona_prompt_proof(
            identity, system_bytes, persona_text, kind,
            attempt_ordinal=ordinal, provider="openai", model="m",
            final_system_location="messages", final_system_message_index=0,
        )

    first_rows = [
        event.payload["proof"] for event in first.flush_events()
        if event.type == "persona_prompt_injection_audit"
    ]
    second_row = second.flush_events()[0].payload["proof"]
    assert [row["attempt_kind"] for row in first_rows] == ["primary", "semantic_retry"]
    assert all(row["final_system_message_index"] == 0 for row in first_rows)
    assert all(row["message_char_count"] == len(messages[0]["content"]) for row in first_rows)
    assert all(row["confirmed_injection"] is True for row in first_rows)
    assert first_rows[0]["run_scoped_fingerprint"] == first_rows[1]["run_scoped_fingerprint"]
    assert first_rows[0]["run_scoped_fingerprint"] == second_row["run_scoped_fingerprint"]
    serialized = str(first_rows + [second_row]).lower()
    assert persona_text not in serialized
    assert messages[0]["content"].lower() not in serialized
    assert "system_bytes" not in serialized
    assert "md5" not in serialized


def test_persona_confirmation_summary_joins_by_decision_identity() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_persona(_identity(), {"profile_id": "calm"})
    collector.record_provider_persona_prompt_proof(
        _identity(), b"rules persona", "persona", "structured_retry",
        attempt_ordinal=2, provider="openai", model="m",
        final_system_location="messages", final_system_message_index=0,
    )

    summary = exposure_audit.summarize_persona_prompt_confirmation(collector.flush_events())

    assert summary == {
        "supported": True,
        "configured_action_count": 1,
        "confirmed_action_count": 1,
        "confirmation_rate": 1.0,
    }


def test_request_assembly_proof_cannot_confirm_provider_injection() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_persona(_identity(), {"profile_id": "calm"})
    collector.record_persona_prompt_proof(
        _identity(),
        ({"role": "system", "content": "rules persona"},),
        "persona",
        "primary",
    )

    events = collector.flush_events()
    assert [event.type for event in events] == [
        "persona_exposure_audit", "persona_request_assembly_audit",
    ]
    assert exposure_audit.summarize_persona_prompt_confirmation(events) == {
        "supported": True,
        "configured_action_count": 1,
        "confirmed_action_count": 0,
        "confirmation_rate": 0.0,
    }


def test_persona_confirmation_summary_marks_zero_denominator_unsupported() -> None:
    assert exposure_audit.summarize_persona_prompt_confirmation([]) == {
        "supported": False,
        "configured_action_count": 0,
        "confirmed_action_count": 0,
        "confirmation_rate": None,
    }


class _Registry:
    def get_agent(self, player_id: str) -> object:
        return object()


def _engine() -> RuleEngine:
    return RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")


def _capture_dispatch(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_dispatch(state: dict[str, Any], fn: Any, *extra_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"fn": fn, "extra_args": extra_args, "kwargs": kwargs})
        if fn is night_nodes.agent_night_witch:
            return {
                "use_antidote": False,
                "poison_target_id": None,
                "witch_action_trace": {"parsed_action": {"reason": "skip"}},
            }
        if fn is night_nodes.agent_night_seer:
            return {
                "seer_target_id": "p02",
                "seer_action_trace": {"parsed_action": {"reason": "check"}},
            }
        if fn is night_nodes.agent_hybrid_choose_master:
            collector = kwargs.get("exposure_collector")
            identity = kwargs.get("decision_identity")
            if collector is not None and identity is not None:
                collector.record_rag(identity, [{"entry_id": "hybrid_hint", "rank": 1}])
            return {
                "master_target_id": "p01",
                "action_trace": {"parsed_action": {"reason": "choose"}},
            }
        if fn is skill_nodes.agent_hunter_shot:
            return "p02"
        if fn is skill_nodes.agent_badge_decision:
            return {"badge_decision": "tear", "badge_target_id": None}
        return {}

    monkeypatch.setattr(module, "_dispatch_agent", fake_dispatch)
    return calls


def _state(gs: GameState) -> dict[str, Any]:
    return {
        "game_state": gs,
        "engine": _engine(),
        "agent_registry": _Registry(),
        "agent_call_delay_ms": -1,
    }


def _audit_event_order(events: list[Any]) -> tuple[int, int]:
    event_types = [event.type for event in events]
    exposure_index = event_types.index("rag_exposure_audit")
    action_index = event_types.index("action_trace_audit")
    return exposure_index, action_index


def _record_exposure(kwargs: dict[str, Any], entry_id: str) -> None:
    collector = kwargs.get("exposure_collector")
    identity = kwargs.get("decision_identity")
    if collector is not None and identity is not None:
        collector.record_rag(identity, [{"entry_id": entry_id, "rank": 1}])


def test_wolf_consensus_dispatches_each_vote_with_identity_and_flushes_action_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_single_vote(
        state: dict[str, Any],
        engine: RuleEngine,
        registry: Any,
        wolf_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append({"wolf_id": wolf_id, "kwargs": kwargs})
        _record_exposure(kwargs, f"wolf_{wolf_id}")
        return {
            "wolf_action": "kill" if wolf_id == "wolf1" else "no_kill",
            "wolf_kill_target_id": "target" if wolf_id == "wolf1" else None,
            "action_trace": {"parsed_action": {"reason": f"{wolf_id} choice"}},
        }

    monkeypatch.setattr(agent_adapter, "_single_wolf_vote", fake_single_vote)
    # Task 8 后权威空 stance 会直接解析为 all_abstain；本测试专门覆盖旧式逐狼
    # 调度曝光链，因此显式关闭权威计划入口，避免两个互斥路径互相污染。
    consensus_module = import_module(
        "werewolf_agent.runtime.nodes.wolf_consensus"
    )
    monkeypatch.setattr(consensus_module, "_planned_wolf_kill", lambda _state: None)
    gs = GameState(
        game_id="audit_wolf_consensus",
        players={
            "wolf1": PlayerState(id="wolf1", role="werewolf"),
            "wolf2": PlayerState(id="wolf2", role="werewolf"),
            "target": PlayerState(id="target", role="villager"),
        },
        phase="night",
        night_number=1,
    )

    result = night_nodes.wolf_consensus(_state(gs))

    assert {call["wolf_id"] for call in calls} == {"wolf1", "wolf2"}
    for call in calls:
        identity = call["kwargs"]["decision_identity"]
        assert identity.player_id == call["wolf_id"]
        assert identity.phase == "wolf_consensus"
        assert isinstance(call["kwargs"]["exposure_collector"], ModuleExposureAuditCollector)
    events = result["game_state"].events
    event_types = [event.type for event in events]
    assert event_types.count("rag_exposure_audit") == 2
    assert event_types.count("action_trace_audit") == 2
    first_exposure, first_action = _audit_event_order(events)
    assert first_exposure < first_action


@pytest.mark.parametrize(
    ("node", "agent_fn", "players", "state_updates"),
    [
        (
            sheriff_nodes.sheriff_registration,
            sheriff_nodes.agent_sheriff_register,
            {
                "p01": PlayerState(id="p01", role="villager"),
                "p02": PlayerState(id="p02", role="werewolf"),
            },
            {},
        ),
        (
            sheriff_nodes.sheriff_withdraw,
            sheriff_nodes.agent_sheriff_withdraw,
            {
                "p01": PlayerState(id="p01", role="villager"),
                "p02": PlayerState(id="p02", role="villager"),
            },
            {"sheriff_candidates": ["p01", "p02"]},
        ),
    ],
)
def test_sheriff_registration_and_withdrawal_flush_action_audits(
    monkeypatch: pytest.MonkeyPatch,
    node: Any,
    agent_fn: Any,
    players: dict[str, PlayerState],
    state_updates: dict[str, Any],
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_dispatch(state: dict[str, Any], fn: Any, player_id: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"fn": fn, "player_id": player_id, "kwargs": kwargs})
        _record_exposure(kwargs, f"sheriff_{player_id}")
        if fn is sheriff_nodes.agent_sheriff_register:
            return {
                "registered": player_id == "p01",
                "self_destruct": False,
                "action_trace": {"parsed_action": {"reason": f"register {player_id}"}},
            }
        if fn is sheriff_nodes.agent_sheriff_withdraw:
            return {
                "withdrew": player_id == "p02",
                "self_destruct": False,
                "action_trace": {"parsed_action": {"reason": f"withdraw {player_id}"}},
            }
        return {}

    monkeypatch.setattr(sheriff_nodes, "_dispatch_agent", fake_dispatch)
    gs = GameState(
        game_id=f"audit_{node.__name__}",
        players=players,
        phase="day",
        day_number=1,
        **state_updates,
    )

    result = node(_state(gs))

    assert calls
    assert all(call["fn"] is agent_fn for call in calls)
    assert all(call["kwargs"]["decision_identity"].player_id == call["player_id"] for call in calls)
    assert all(isinstance(call["kwargs"]["exposure_collector"], ModuleExposureAuditCollector) for call in calls)
    event_types = [event.type for event in result["game_state"].events]
    assert "rag_exposure_audit" in event_types
    assert "action_trace_audit" in event_types
    exposure_index, action_index = _audit_event_order(result["game_state"].events)
    assert exposure_index < action_index


@pytest.mark.parametrize(
    ("module", "node", "players", "state_updates"),
    [
        (
            sheriff_nodes,
            sheriff_nodes.sheriff_vote,
            {
                "p01": PlayerState(id="p01", role="villager"),
                "p02": PlayerState(id="p02", role="villager"),
                "p03": PlayerState(id="p03", role="werewolf"),
            },
            {"sheriff_candidates": ["p01", "p02"]},
        ),
        (
            sheriff_pk_nodes,
            sheriff_pk_nodes.sheriff_revote,
            {
                "p01": PlayerState(id="p01", role="villager"),
                "p02": PlayerState(id="p02", role="villager"),
                "p03": PlayerState(id="p03", role="werewolf"),
            },
            {"sheriff_pk_candidates": ["p01", "p02"], "sheriff_tie_count": 1},
        ),
    ],
)
def test_sheriff_vote_and_pk_revote_flush_action_audits(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    node: Any,
    players: dict[str, PlayerState],
    state_updates: dict[str, Any],
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_dispatch(
        state: dict[str, Any],
        fn: Any,
        voter_id: str,
        candidates: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append({"voter_id": voter_id, "kwargs": kwargs})
        _record_exposure(kwargs, f"sheriff_vote_{voter_id}")
        return {
            "vote_target": candidates[0],
            "self_destruct": False,
            "action_trace": {"parsed_action": {"target_id": candidates[0], "reason": f"vote {voter_id}"}},
        }

    monkeypatch.setattr(module, "_dispatch_agent", fake_dispatch)
    gs = GameState(
        game_id=f"audit_{node.__name__}",
        players=players,
        phase="day",
        day_number=1,
        **state_updates,
    )

    result = node(_state(gs))

    assert calls
    assert all(call["kwargs"]["decision_identity"].player_id == call["voter_id"] for call in calls)
    assert all(isinstance(call["kwargs"]["exposure_collector"], ModuleExposureAuditCollector) for call in calls)
    event_types = [event.type for event in result["game_state"].events]
    assert "rag_exposure_audit" in event_types
    assert "action_trace_audit" in event_types
    exposure_index, action_index = _audit_event_order(result["game_state"].events)
    assert exposure_index < action_index


def test_sheriff_speech_order_selection_flushes_action_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_dispatch(
        state: dict[str, Any],
        fn: Any,
        sheriff_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append({"fn": fn, "sheriff_id": sheriff_id, "kwargs": kwargs})
        _record_exposure(kwargs, "speech_order")
        return {
            "speech_order": ["p02", "p03", "p01"],
            "action_trace": {"parsed_action": {"target_id": "p02", "reason": "start left"}},
        }

    monkeypatch.setattr(day_nodes, "_dispatch_agent", fake_dispatch)
    gs = GameState(
        game_id="audit_speech_order",
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="villager"),
            "p03": PlayerState(id="p03", role="werewolf"),
        },
        phase="day",
        day_number=1,
        sheriff_id="p01",
        sheriff_badge_state="active",
    )

    result = day_nodes.free_discussion({**_state(gs), "speech_text": "opening"})

    call = calls[0]
    assert call["fn"] is day_nodes.agent_sheriff_pick_speech_order
    assert call["kwargs"]["decision_identity"].player_id == "p01"
    assert isinstance(call["kwargs"]["exposure_collector"], ModuleExposureAuditCollector)
    assert result["speech_order"] == ["p02", "p03", "p01"]
    event_types = [event.type for event in result["game_state"].events]
    assert "rag_exposure_audit" in event_types
    assert "action_trace_audit" in event_types
    exposure_index, action_index = _audit_event_order(result["game_state"].events)
    assert exposure_index < action_index


@pytest.mark.parametrize(
    ("node", "gs"),
    [
        (
            day_nodes.night_death_last_words,
            GameState(
                game_id="audit_night_last_words",
                players={"p01": PlayerState(id="p01", role="villager", alive=False)},
                phase="day",
                day_number=1,
                night_number=1,
                deaths=[
                    Death(
                        player_id="p01",
                        reason="wolf_kill",
                        timing="night",
                        resolution_batch="night_1",
                        can_leave_last_words=True,
                    )
                ],
            ),
        ),
        (
            day_nodes.exile_last_words,
            GameState(
                game_id="audit_exile_last_words",
                players={"p01": PlayerState(id="p01", role="villager", alive=False)},
                phase="day",
                day_number=1,
                events=[GameEvent(type="vote_resolved", payload={"exiled": "p01"})],
            ),
        ),
    ],
)
def test_last_words_paths_flush_action_audits(
    monkeypatch: pytest.MonkeyPatch,
    node: Any,
    gs: GameState,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_dispatch(state: dict[str, Any], fn: Any, player_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"fn": fn, "player_id": player_id, "kwargs": kwargs})
        _record_exposure(kwargs, f"last_words_{player_id}")
        return {
            "speech_text": "final read",
            "action_trace": {"parsed_action": {"speech_text": "final read"}},
        }

    monkeypatch.setattr(day_nodes, "_dispatch_agent", fake_dispatch)

    result = node(_state(gs))

    assert calls
    assert all(call["fn"] is day_nodes.agent_exile_last_words for call in calls)
    assert all(call["kwargs"]["decision_identity"].player_id == call["player_id"] for call in calls)
    assert all(isinstance(call["kwargs"]["exposure_collector"], ModuleExposureAuditCollector) for call in calls)
    event_types = [event.type for event in result["game_state"].events]
    assert "rag_exposure_audit" in event_types
    assert "action_trace_audit" in event_types
    exposure_index, action_index = _audit_event_order(result["game_state"].events)
    assert exposure_index < action_index


@pytest.mark.parametrize(
    ("node", "agent_fn", "players", "state_updates"),
    [
        (
            night_nodes.night_witch,
            night_nodes.agent_night_witch,
            {"witch": PlayerState(id="witch", role="witch")},
            {"wolf_kill_target_id": None},
        ),
        (
            night_nodes.night_seer,
            night_nodes.agent_night_seer,
            {
                "seer": PlayerState(id="seer", role="seer"),
                "p02": PlayerState(id="p02", role="villager"),
            },
            {},
        ),
        (
            night_nodes.first_night_hybrid_master,
            night_nodes.agent_hybrid_choose_master,
            {
                "hybrid": PlayerState(id="hybrid", role="hybrid"),
                "p01": PlayerState(id="p01", role="villager"),
            },
            {},
        ),
    ],
)
def test_night_action_dispatches_pass_identity_and_collector(
    monkeypatch: pytest.MonkeyPatch,
    node: Any,
    agent_fn: Any,
    players: dict[str, PlayerState],
    state_updates: dict[str, Any],
) -> None:
    calls = _capture_dispatch(monkeypatch, night_nodes)
    gs = GameState(game_id="audit_night", players=players, phase="night", night_number=1)
    state = {**_state(gs), **state_updates}

    node(state)

    call = next(call for call in calls if call["fn"] is agent_fn)
    assert call["kwargs"]["decision_identity"].game_id == "audit_night"
    assert isinstance(call["kwargs"]["exposure_collector"], ModuleExposureAuditCollector)


def test_hybrid_master_choice_flushes_exposure_with_action_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_dispatch(monkeypatch, night_nodes)
    gs = GameState(
        game_id="audit_hybrid_flush",
        players={
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
            "p01": PlayerState(id="p01", role="villager"),
        },
        phase="night",
        night_number=1,
    )

    result = night_nodes.first_night_hybrid_master(_state(gs))

    events = result["game_state"].events
    event_types = [event.type for event in events]
    assert "rag_exposure_audit" in event_types
    assert "action_trace_audit" in event_types
    exposure_index = event_types.index("rag_exposure_audit")
    action_index = event_types.index("action_trace_audit")
    assert exposure_index < action_index
    assert events[exposure_index].payload["trace_id"] == events[action_index].payload["trace_id"]


def test_allocations_from_initialized_shallow_copies_remain_monotonic() -> None:
    from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig

    runtime_state = GameRunner(GameRunnerConfig(seed=321))._build_runtime_state()
    first_branch = dict(runtime_state)
    second_branch = dict(runtime_state)

    first = _allocate_decision_identity(
        first_branch,
        player_id="p01",
        phase="vote",
        task_type="vote",
        day_number=1,
        night_number=1,
    )
    second = _allocate_decision_identity(
        second_branch,
        player_id="p02",
        phase="vote",
        task_type="vote",
        day_number=1,
        night_number=1,
    )

    assert [first.action_index, second.action_index] == [0, 1]


def test_night_death_last_words_keeps_action_index_monotonic_across_call_state_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dispatch(state: dict[str, Any], fn: Any, player_id: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "speech_text": f"last words from {player_id}",
            "action_trace": {"parsed_action": {"speech_text": f"last words from {player_id}"}},
        }

    monkeypatch.setattr(day_nodes, "_dispatch_agent", fake_dispatch)
    gs = GameState(
        game_id="audit_last_words_monotonic",
        players={
            "p01": PlayerState(id="p01", role="villager", alive=False),
            "p02": PlayerState(id="p02", role="villager", alive=False),
        },
        phase="day",
        day_number=1,
        night_number=1,
        deaths=[
            Death(
                player_id="p01",
                reason="wolf_kill",
                timing="night",
                resolution_batch="night_1",
                can_leave_last_words=True,
            ),
            Death(
                player_id="p02",
                reason="wolf_kill",
                timing="night",
                resolution_batch="night_1",
                can_leave_last_words=True,
            ),
        ],
    )

    result = day_nodes.night_death_last_words(_state(gs))

    audit_indexes = [
        event.payload["action_index"]
        for event in result["game_state"].events
        if event.type == "action_trace_audit"
    ]
    assert audit_indexes == [0, 1]


def test_night_witch_and_seer_keep_action_index_monotonic_across_state_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_dispatch(monkeypatch, night_nodes)
    gs = GameState(
        game_id="audit_night_copy_monotonic",
        players={
            "witch": PlayerState(id="witch", role="witch"),
            "seer": PlayerState(id="seer", role="seer"),
            "p02": PlayerState(id="p02", role="villager"),
        },
        phase="night",
        night_number=1,
    )
    state = _state(gs)

    witch_result = night_nodes.night_witch(state)
    state.update(witch_result)
    seer_result = night_nodes.night_seer(state)

    audit_indexes = [
        event.payload["action_index"]
        for event in seer_result["game_state"].events
        if event.type == "action_trace_audit"
    ]
    assert audit_indexes == [0, 1]


def test_wolf_discussion_keeps_action_index_monotonic_across_round_state_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import werewolf_agent.runtime.wolf_strategy as wolf_strategy

    def fake_dispatch(state: dict[str, Any], fn: Any, wolf_id: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "speech_text": f"{wolf_id} round {state['wolf_discussion_round']}",
            "action_trace": {
                "parsed_action": {
                    "speech_text": f"{wolf_id} round {state['wolf_discussion_round']}",
                },
            },
        }

    monkeypatch.setattr(night_nodes, "_dispatch_agent", fake_dispatch)
    monkeypatch.setattr(wolf_strategy, "should_end_discussion_early", lambda *_args, **_kwargs: False)
    gs = GameState(
        game_id="audit_wolf_discussion_monotonic",
        players={
            "wolf1": PlayerState(id="wolf1", role="werewolf"),
            "wolf2": PlayerState(id="wolf2", role="werewolf"),
            "target": PlayerState(id="target", role="villager"),
        },
        phase="night",
        night_number=1,
    )

    result = night_nodes.wolf_discussion(_state(gs))

    discussion_events = [
        event for event in result["game_state"].events
        if event.type == "wolf_discussion"
    ]
    assert discussion_events
    assert all(
        event.payload["target_stance"]["stance"] == "abstain"
        for event in discussion_events
    )
    assert all(
        event.payload["target_stance"]["target_id"] is None
        for event in discussion_events
    )

    audit_indexes = [
        event.payload["action_index"]
        for event in result["game_state"].events
        if event.type == "action_trace_audit"
    ]
    assert audit_indexes == list(range(len(audit_indexes)))
    assert len(audit_indexes) == 6


def test_public_view_cannot_see_wolf_stance_or_wolf_identity() -> None:
    """狼人 stance 仅狼队可见，村民公开视图不得获得任何私有字段。"""
    from werewolf_agent.api.views import _event_visible_to_player
    from werewolf_agent.core.event_visibility import EventVisibility

    gs = GameState(
        game_id="wolf_stance_privacy",
        players={
            "wolf1": PlayerState(id="wolf1", role="werewolf"),
            "villager1": PlayerState(id="villager1", role="villager"),
        },
        night_number=1,
    )
    event = GameEvent(
        type="wolf_discussion",
        payload={
            "wolf_id": "wolf1",
            "night_number": 1,
            "target_stance": {
                "wolf_id": "wolf1",
                "target_id": "villager1",
                "stance": "propose",
                "priority": "primary",
                "source_event_id": "wolf_stance_privacy:e000000",
                "round_number": 1,
            },
        },
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )

    assert _event_visible_to_player(event, gs, "villager1", "villager") is False
    assert _event_visible_to_player(event, gs, "wolf1", "werewolf") is True


def test_public_view_cannot_see_runtime_timeout_count_in_action_audit() -> None:
    from werewolf_agent.api.schemas import ViewMode
    from werewolf_agent.api.views import build_timeline
    from werewolf_agent.runtime.nodes.action_audit import _action_trace_event

    gs = GameState(
        game_id="timeout_audit_privacy",
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[_action_trace_event(
            player_id="p01",
            phase="vote",
            action_trace={"runtime_timeout_count": 2},
        )],
    )

    assert build_timeline(gs, ViewMode.PUBLIC).events == []


def test_day_vote_resolve_vote_pairs_pending_exposures_with_action_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dispatch(state: dict[str, Any], fn: Any, voter_id: str, **kwargs: Any) -> dict[str, Any]:
        _record_exposure(kwargs, f"vote_hint_{voter_id}")
        target = "p02" if voter_id != "p02" else "p01"
        return {
            "vote_target": target,
            "action_trace": {
                "parsed_action": {
                    "target_id": target,
                    "reason": f"vote from {voter_id}",
                },
            },
        }

    monkeypatch.setattr(day_nodes, "_dispatch_agent", fake_dispatch)
    gs = GameState(
        game_id="audit_day_vote_pending_pair",
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
            "p03": PlayerState(id="p03", role="villager"),
        },
        phase="day",
        day_number=1,
        night_number=1,
    )
    state = _state(gs)

    state.update(day_nodes.day_vote(state))
    result = day_nodes.resolve_vote(state)

    audit_events = [
        event
        for event in result["game_state"].events
        if event.type in {"rag_exposure_audit", "action_trace_audit"}
    ]
    assert [event.type for event in audit_events] == [
        "rag_exposure_audit",
        "action_trace_audit",
        "rag_exposure_audit",
        "action_trace_audit",
        "rag_exposure_audit",
        "action_trace_audit",
    ]
    for exposure_event, action_event in zip(audit_events[0::2], audit_events[1::2]):
        assert exposure_event.payload["trace_id"] == action_event.payload["trace_id"]
    assert [event.payload["action_index"] for event in audit_events[1::2]] == [0, 1, 2]
    assert state["pending_exposure_events_by_trace"] == {}


@pytest.mark.parametrize(
    ("node", "agent_fn", "gs"),
    [
        (
            skill_nodes.resolve_hunter_shot,
            skill_nodes.agent_hunter_shot,
            GameState(
                game_id="audit_hunter",
                players={
                    "hunter": PlayerState(id="hunter", role="hunter", alive=False),
                    "p02": PlayerState(id="p02", role="villager", alive=True),
                },
                phase="night",
                night_number=1,
                deaths=[
                    Death(
                        player_id="hunter",
                        reason="wolf_kill",
                        timing="night",
                        resolution_batch="night_1",
                        triggered_skills=["hunter_shot"],
                    )
                ],
            ),
        ),
        (
            skill_nodes.sheriff_badge_transfer,
            skill_nodes.agent_badge_decision,
            GameState(
                game_id="audit_badge",
                players={
                    "sheriff": PlayerState(id="sheriff", role="villager", alive=False),
                    "p02": PlayerState(id="p02", role="villager", alive=True),
                },
                sheriff_id="sheriff",
                sheriff_badge_state="active",
                phase="day",
                day_number=2,
            ),
        ),
    ],
)
def test_skill_dispatches_pass_identity_and_collector(
    monkeypatch: pytest.MonkeyPatch,
    node: Any,
    agent_fn: Any,
    gs: GameState,
) -> None:
    calls = _capture_dispatch(monkeypatch, skill_nodes)

    node(_state(gs))

    call = next(call for call in calls if call["fn"] is agent_fn)
    assert call["kwargs"]["decision_identity"].game_id == gs.game_id
    assert isinstance(call["kwargs"]["exposure_collector"], ModuleExposureAuditCollector)


def test_prompt_proof_key_provider_shares_one_game_key_and_separates_runs() -> None:
    from werewolf_agent.runtime.exposure_audit import PromptProofKeyProvider

    provider = PromptProofKeyProvider()
    first = ModuleExposureAuditCollector(prompt_proof_key_provider=provider)
    second = ModuleExposureAuditCollector(prompt_proof_key_provider=provider)
    game_one = _identity()
    game_two = DecisionIdentity(
        "g2", "w1", "night", 1, 1, "wolf_discussion", 1,
    )

    for collector, identity in ((first, game_one), (second, game_one), (second, game_two)):
        collector.record_provider_persona_prompt_proof(
            identity, b"system", "", "primary", attempt_ordinal=1,
            provider="openai", model="m", final_system_location="messages",
            final_system_message_index=0,
        )

    first_proof = first.flush_events()[0].payload["proof"]
    same_run, different_run = [event.payload["proof"] for event in second.flush_events()]
    assert first_proof["system_hmac_sha256"] == same_run["system_hmac_sha256"]
    assert first_proof["system_hmac_sha256"] != different_run["system_hmac_sha256"]
    assert provider.verifier_for_run("g1").verify(
        b"system", first_proof["system_hmac_sha256"],
    )
