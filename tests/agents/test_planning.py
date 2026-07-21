from __future__ import annotations

import pytest

from werewolf_agent.agents.planning import (
    DecisionPlan,
    DialoguePlan,
    decision_and_dialogue_to_action,
    planning_envelope_to_action,
    render_dialogue_plan,
)
from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType, VotePlayerAction
from werewolf_agent.agents.action_schemas import WolfDiscussionSpeechPlayerAction, WolfTargetStanceAction


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


def test_dialogue_plan_rejects_public_points_that_copy_conceal() -> None:
    decision = DecisionPlan(
        action_type=ActionType.VOTE,
        target_id="p02",
        confidence=0.8,
        private_goal="protect p03",
        evidence_refs=["event:4:speech"],
    )
    dialogue = DialoguePlan(
        public_intent="push p02",
        public_target_id="p02",
        talking_points=["p03 is my wolf teammate, so redirect to p02"],
        conceal=["p03 is my wolf teammate"],
    )

    with pytest.raises(ValueError, match="private dialogue content"):
        decision_and_dialogue_to_action(decision, dialogue, _vote_context())


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


def _wolf_discussion_context() -> AgentContext:
    return AgentContext(
        agent_id="p03",
        task_type=TaskType.WOLF_DISCUSSION,
        own_role="werewolf",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p01", "p02", "p04", "p05", "p06", "p07", "p09", "p11", "p12"],
    )


def test_wolf_discussion_envelope_action_carries_target_stance_when_present() -> None:
    """WOLF_DISCUSSION + target_stance 必须透传到 WolfDiscussionSpeechPlayerAction。

    此前 planning_envelope_to_action 把 data 拆成 decision_plan / dialogue_plan 后，
    LLM 输出的私有字段 target_stance 被悄悄丢掉，导致 wolf_discussion 节点拿到的
    stance 永远是 fallback 的 abstain / target_id=None，最终 _planned_wolf_kill
    走 strategic_abstain 空刀。
    """
    raw_stance = {
        "target_id": "p05",
        "stance": "propose",
        "priority": "primary",
    }
    envelope = {
        "decision_plan": {
            "action_type": ActionType.SPEECH.value,
            "confidence": 0.7,
            "private_goal": "锁定今晚刀口",
            "evidence_refs": ["event:1:wolf_discussion"],
        },
        "dialogue_plan": {
            "public_intent": "提议本轮刀 p05",
            "talking_points": ["p05是预言家威胁", "需要清掉金水源"],
            "conceal": ["狼队身份"],
        },
        "target_stance": raw_stance,
    }

    action, audit = planning_envelope_to_action(envelope, _wolf_discussion_context())

    assert isinstance(action, WolfDiscussionSpeechPlayerAction), (
        "WOLF_DISCUSSION 上下文 + 含 target_stance 的 envelope 必须产出 "
        "WolfDiscussionSpeechPlayerAction；否则 target_stance 字段会直接被丢弃。"
    )
    assert action.target_stance is not None
    assert action.target_stance.target_id == "p05"
    assert action.target_stance.stance == "propose"
    assert action.target_stance.priority == "primary"
    assert audit["decision_plan"]["private_goal"] == "锁定今晚刀口"


def test_wolf_discussion_envelope_without_target_stance_falls_back_to_none() -> None:
    """WOLF_DISCUSSION 即使 envelope 不含 target_stance，也必须产出 wolf 子类。"""
    envelope = {
        "decision_plan": {
            "action_type": ActionType.SPEECH.value,
            "confidence": 0.6,
            "private_goal": "仅观望",
        },
        "dialogue_plan": {
            "public_intent": "观察一轮再决定",
            "talking_points": ["等 p12 的发言再确认"],
        },
    }

    action, _audit = planning_envelope_to_action(envelope, _wolf_discussion_context())

    assert isinstance(action, WolfDiscussionSpeechPlayerAction)
    assert action.target_stance is None


def test_wolf_discussion_decision_and_dialogue_to_action_accepts_target_stance() -> None:
    """planning 内层直接调用时，target_stance 也应当从 data 透传。"""
    decision = DecisionPlan(
        action_type=ActionType.SPEECH,
        confidence=0.7,
        private_goal="明确站边 p05",
        evidence_refs=["event:1:wolf_discussion"],
    )
    dialogue = DialoguePlan(
        public_intent="提议本轮刀 p05",
        public_target_id="p05",
        talking_points=["p05 是发言的关键节点"],
    )
    raw_stance = WolfTargetStanceAction(
        target_id="p05",
        stance="support",
        priority="primary",
    )

    action = decision_and_dialogue_to_action(
        decision,
        dialogue,
        _wolf_discussion_context(),
        target_stance=raw_stance,
    )

    assert isinstance(action, WolfDiscussionSpeechPlayerAction)
    assert action.target_stance is not None
    assert action.target_stance.target_id == "p05"
    assert action.target_stance.stance == "support"
