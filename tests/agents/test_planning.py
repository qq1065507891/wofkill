from __future__ import annotations

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.planning import (
    DecisionPlan,
    DialoguePlan,
    decision_and_dialogue_to_action,
    render_dialogue_plan,
)
from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType, VotePlayerAction


def _vote_context() -> AgentContext:
    return AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02", "p03"],
    )


def test_decision_plan_requires_legal_target_for_vote() -> None:
    decision = DecisionPlan(
        action_type=ActionType.VOTE,
        target_id="p09",
        confidence=0.7,
        private_goal="vote out strongest suspect",
        evidence_refs=["event:1:speech"],
    )
    dialogue = DialoguePlan(
        public_intent="push vote",
        public_target_id="p09",
        talking_points=["p09发言矛盾"],
    )

    with pytest.raises(ValueError, match="target_id=p09 not in legal_targets"):
        decision_and_dialogue_to_action(decision, dialogue, _vote_context())


def test_decision_plan_rejects_reference_refs_as_current_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence_refs"):
        DecisionPlan(
            action_type=ActionType.VOTE,
            target_id="p02",
            confidence=0.7,
            private_goal="vote",
            evidence_refs=["rag:seer_case_1"],
        )


def test_decision_plan_allows_historical_refs_as_references() -> None:
    plan = DecisionPlan(
        action_type=ActionType.VOTE,
        target_id="p02",
        confidence=0.7,
        private_goal="vote",
        evidence_refs=["event:3:speech"],
        reference_refs=["rag:seer_case_1", "reflection:vote_mistake", "profile:p02"],
    )

    assert plan.reference_refs == [
        "rag:seer_case_1",
        "reflection:vote_mistake",
        "profile:p02",
    ]


def test_dialogue_plan_rendering_omits_private_conceal_fields() -> None:
    dialogue = DialoguePlan(
        public_intent="redirect pressure",
        public_target_id="p03",
        talking_points=["p03今天跟票过快"],
        conceal=["p02 is my wolf teammate", "night kill target p04"],
        tone="calm",
    )

    rendered = render_dialogue_plan(dialogue)

    assert "p03今天跟票过快" in rendered
    assert "p02 is my wolf teammate" not in rendered
    assert "night kill target" not in rendered


def test_vote_decision_and_dialogue_convert_to_player_action() -> None:
    decision = DecisionPlan(
        action_type=ActionType.VOTE,
        target_id="p02",
        confidence=0.81,
        private_goal="resolve p02 contradiction",
        evidence_refs=["event:3:speech"],
        selected_world_ids=["World 1"],
    )
    dialogue = DialoguePlan(
        public_intent="vote pressure",
        public_target_id="p02",
        talking_points=["p02前后逻辑不一致", "先归p02观察票型"],
        conceal=["World 1 has p02 as wolf"],
    )

    action = decision_and_dialogue_to_action(decision, dialogue, _vote_context())

    assert isinstance(action, VotePlayerAction)
    assert action.action_type == ActionType.VOTE
    assert action.target_id == "p02"
    assert action.confidence == 0.81
    assert "p02前后逻辑不一致" in action.reason
    assert "World 1 has p02 as wolf" not in action.reason
    assert action.suspect_reason
    assert action.not_voting_reason
    assert action.private_reason


def test_speech_decision_and_dialogue_convert_to_player_action() -> None:
    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
    )
    decision = DecisionPlan(
        action_type=ActionType.SPEECH,
        confidence=0.66,
        private_goal="pressure unclear stance",
        evidence_refs=["event:5:speech"],
    )
    dialogue = DialoguePlan(
        public_intent="ask for clarification",
        public_target_id="p03",
        talking_points=["p03 should explain the vote shift"],
        conceal=["private suspicion score is high"],
    )

    action = decision_and_dialogue_to_action(decision, dialogue, context)

    assert action.action_type == ActionType.SPEECH
    assert action.speech == "p03 should explain the vote shift"
    assert "private suspicion" not in action.speech
