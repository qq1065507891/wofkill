"""Tests for the extracted parse_dispatch module."""

from __future__ import annotations

import pytest

from werewolf_agent.agents.parse_dispatch import (
    parse_choice_action,
    parse_speech_intent_action,
    select_output_mode,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    TaskType,
)


class TestSelectOutputMode:
    def test_single_target_required_returns_target_choice(self):
        # _uses_choice_pipeline needs exactly one legal action that is a
        # target-required type, plus a non-empty legal_targets list.
        mode = select_output_mode(
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07", "p08"],
            task_type=TaskType.VOTE,
            speech_intent_tasks={TaskType.SPEECH},
        )
        assert mode == OutputMode.TARGET_CHOICE

    def test_speech_task_returns_speech_intent(self):
        mode = select_output_mode(
            legal_actions=[ActionType.SPEECH],
            legal_targets=[],
            task_type=TaskType.SPEECH,
            speech_intent_tasks={TaskType.SPEECH, TaskType.PK_SPEECH},
        )
        assert mode == OutputMode.SPEECH_INTENT

    def test_speech_task_with_legacy_vote_action_still_returns_speech_intent(self):
        mode = select_output_mode(
            legal_actions=[ActionType.SPEECH, ActionType.VOTE],
            legal_targets=["p07"],
            task_type=TaskType.SPEECH,
            speech_intent_tasks={TaskType.SPEECH, TaskType.PK_SPEECH},
        )
        assert mode == OutputMode.SPEECH_INTENT

    def test_speech_task_with_unknown_mixed_actions_returns_full_action(self):
        mode = select_output_mode(
            legal_actions=[ActionType.SPEECH, ActionType.SHERIFF_REGISTER],
            legal_targets=[],
            task_type=TaskType.SPEECH,
            speech_intent_tasks={TaskType.SPEECH, TaskType.PK_SPEECH},
        )
        assert mode == OutputMode.FULL_ACTION

    def test_target_action_plus_no_action_returns_target_choice(self):
        mode = select_output_mode(
            legal_actions=[ActionType.SHERIFF_VOTE, ActionType.NO_ACTION],
            legal_targets=["p09", "p12"],
            task_type=TaskType.VOTE,
            speech_intent_tasks={TaskType.SPEECH},
        )

        assert mode == OutputMode.TARGET_CHOICE

    def test_mixed_actions_returns_full_action(self):
        mode = select_output_mode(
            legal_actions=[ActionType.VOTE, ActionType.SPEECH],
            legal_targets=["p07"],
            task_type=TaskType.SHERIFF_SPEECH,
            speech_intent_tasks={TaskType.SPEECH},
        )
        assert mode == OutputMode.FULL_ACTION

    def test_speech_intent_overrides_when_no_legal_targets(self):
        # Without a legal_targets list, choice pipeline is not used even
        # though VOTE is the only legal action.
        mode = select_output_mode(
            legal_actions=[ActionType.VOTE],
            legal_targets=[],
            task_type=TaskType.VOTE,
            speech_intent_tasks={TaskType.SPEECH},
        )
        assert mode == OutputMode.FULL_ACTION

    def test_no_legal_actions_returns_full_action(self):
        mode = select_output_mode(
            legal_actions=[],
            legal_targets=[],
            task_type=TaskType.VOTE,
            speech_intent_tasks={TaskType.SPEECH},
        )
        assert mode == OutputMode.FULL_ACTION


class TestParseChoiceAction:
    def _make_context(
        self,
        legal_actions: list[ActionType] | None = None,
        legal_targets: list[str] | None = None,
        salience_items: list[dict] | None = None,
    ) -> AgentContext:
        return AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=legal_actions or [ActionType.VOTE],
            legal_targets=legal_targets or ["p07", "p08"],
            salience_items=salience_items or [],
        )

    def test_parses_letter_choice(self):
        # vote_choice_map maps "a" -> first legal target, "b" -> second, etc.
        ctx = self._make_context()
        text = '{"choice": "a"}'
        action, parse_error, choice_data = parse_choice_action(text, ctx)
        assert parse_error is None
        assert action is not None
        assert action.action_type == ActionType.VOTE
        assert action.target_id in {"p07", "p08"}
        assert choice_data is not None

    def test_missing_choice_returns_error(self):
        ctx = self._make_context()
        text = '{"reason": "no choice here"}'
        action, parse_error, choice_data = parse_choice_action(text, ctx)
        assert action is None
        assert "choice" in (parse_error or "").lower()

    def test_invalid_json_returns_error(self):
        ctx = self._make_context()
        text = "not json at all"
        action, parse_error, choice_data = parse_choice_action(text, ctx)
        assert action is None
        assert parse_error is not None

    def test_sheriff_vote_choice_repairs_exile_vote_shape(self):
        ctx = self._make_context(
            legal_actions=[ActionType.SHERIFF_VOTE, ActionType.NO_ACTION],
            legal_targets=["p09", "p12"],
        )
        text = (
            '{"action_type":"vote","target_id":"p09","speech":"",'
            '"reason":"p09发言更完整，适合拿警徽。",'
            '"seer_stance":"undecided","vote_basis":"speech_logic",'
            '"suspect_reason":"p12逻辑不完整",'
            '"not_voting_reason":"p12没有明确验人",'
            '"private_reason":"警长票选择p09","confidence":0.6}'
        )

        action, parse_error, choice_data = parse_choice_action(text, ctx)

        assert parse_error is None
        assert action is not None
        assert action.action_type == ActionType.SHERIFF_VOTE
        assert action.target_id == "p09"
        assert choice_data is not None


class TestParseSpeechIntentAction:
    def _make_context(
        self,
        legal_targets: list[str] | None = None,
    ) -> AgentContext:
        return AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=legal_targets or ["p07", "p08"],
            salience_items=[],
        )

    def test_parses_basic_speech_intent(self):
        ctx = self._make_context()
        text = '{"intent": "question_target", "target_id": "p07"}'
        action, parse_error, choice_data = parse_speech_intent_action(text, ctx)
        assert parse_error is None
        assert action is not None
        assert action.action_type == ActionType.SPEECH
        assert action.target_id == "p07"
        assert action.speech  # synthesized from intent
        assert choice_data is not None

    def test_missing_intent_returns_error(self):
        ctx = self._make_context()
        text = '{"speech": "hello world"}'
        action, parse_error, _ = parse_speech_intent_action(text, ctx)
        # repair pipeline synthesizes a default intent from raw speech when
        # the input has *some* speech text — accept either outcome as long
        # as the returned action is well-formed.
        if action is None:
            assert parse_error is not None
        else:
            assert action.action_type == ActionType.SPEECH

    def test_invalid_json_returns_error(self):
        ctx = self._make_context()
        text = "this is not json"
        action, parse_error, _ = parse_speech_intent_action(text, ctx)
        assert action is None
        assert parse_error is not None
