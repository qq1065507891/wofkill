from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
from werewolf_agent.core.models import Death, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes import night as night_nodes
from werewolf_agent.runtime.nodes import skills as skill_nodes


def _identity() -> DecisionIdentity:
    return DecisionIdentity("g1", "p01", "vote", 2, 1, "vote", 4)


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


def test_collector_strips_forbidden_private_fields() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_rag(_identity(), [{"entry_id": "x", "target_role": "werewolf"}])
    payload = collector.flush_events()[0].payload
    assert "target_role" not in str(payload)
    assert "werewolf" not in str(payload)


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
