from __future__ import annotations

from copy import deepcopy

from werewolf_agent.agents.schemas import ActionType, TaskType


def _snapshot(**params):
    return {
        "profile_id": "aggressive",
        "speech_style": "direct",
        "task_style": "direct_vote",
        "effective_params": params,
    }


def test_aggressive_persona_lowers_vote_threshold_without_touching_worlds() -> None:
    from werewolf_agent.persona_runtime.policy import PersonaPolicyPrior

    worlds = {
        "top_worlds": [
            {"label": "World A", "probability": 0.62},
            {"label": "World B", "probability": 0.38},
        ]
    }
    before = deepcopy(worlds)

    prior = PersonaPolicyPrior.from_snapshot(
        _snapshot(risk_tolerance=0.9, aggression=0.8, leadership=0.6),
        own_role="villager",
        task_type=TaskType.VOTE.value,
    )

    assert prior.vote_confidence_threshold_delta < 0
    assert prior.vote_threshold(0.7) < 0.7
    assert prior.speech_directness == "high"
    assert prior.deception_allowed is False
    assert worlds == before


def test_good_role_never_receives_deception_even_with_high_skill() -> None:
    from werewolf_agent.persona_runtime.policy import PersonaPolicyPrior

    prior = PersonaPolicyPrior.from_snapshot(
        _snapshot(risk_tolerance=0.7, deception_skill=1.0),
        own_role="seer",
        task_type=TaskType.SPEECH.value,
    )

    assert prior.deception_allowed is False


def test_wolf_speech_can_use_deception_prior() -> None:
    from werewolf_agent.persona_runtime.policy import PersonaPolicyPrior

    prior = PersonaPolicyPrior.from_snapshot(
        _snapshot(risk_tolerance=0.7, deception_skill=0.9),
        own_role="werewolf",
        task_type=TaskType.WOLF_DISCUSSION.value,
    )

    assert prior.deception_allowed is True
    assert prior.claim_risk_threshold_delta > 0


def test_planning_applies_persona_prior_to_decision_threshold() -> None:
    from werewolf_agent.agents.planning import planning_threshold_for_action
    from werewolf_agent.persona_runtime.policy import PersonaPolicyPrior

    prior = PersonaPolicyPrior.from_snapshot(
        _snapshot(risk_tolerance=0.9, aggression=0.8),
        own_role="villager",
        task_type=TaskType.VOTE.value,
    )

    assert planning_threshold_for_action(ActionType.VOTE, prior, base=0.7) < 0.7


def test_planning_envelope_records_persona_policy_prior_in_audit() -> None:
    from werewolf_agent.agents.planning import planning_envelope_to_action
    from werewolf_agent.agents.schemas import AgentContext

    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02"],
        persona_snapshot=_snapshot(risk_tolerance=0.9, aggression=0.8),
    )

    _action, audit = planning_envelope_to_action(
        {
            "decision_plan": {
                "action_type": "vote",
                "target_id": "p02",
                "confidence": 0.8,
                "private_goal": "vote p02",
                "evidence_refs": ["event:2:speech"],
            },
            "dialogue_plan": {
                "public_intent": "push p02",
                "public_target_id": "p02",
                "talking_points": ["p02 has the clearest contradiction"],
            },
        },
        ctx,
    )

    assert audit["persona_policy_prior"]["vote_threshold"] < 0.7
    assert audit["persona_policy_prior"]["speech_directness"] == "high"
