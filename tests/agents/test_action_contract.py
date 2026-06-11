from __future__ import annotations

from werewolf_agent.agents.action_contract import ActionContract
from werewolf_agent.agents.schemas import ActionType, OutputMode, TaskType


def test_target_choice_vote_contract_matches_prompt_fields() -> None:
    contract = ActionContract.build(
        output_mode=OutputMode.TARGET_CHOICE,
        task_type=TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p07", "p08"],
    )

    schema = contract.json_schema
    assert schema["properties"]["choice"]["enum"] == ["A", "B"]
    assert schema["required"] == [
        "choice",
        "reason",
        "seer_stance",
        "vote_basis",
        "standing_with_seer",
        "suspect_reason",
        "not_voting_reason",
        "private_reason",
        "confidence",
    ]
    assert "action_type" not in schema["properties"]
    assert "target_id" not in schema["properties"]


def test_speech_intent_contract_only_advertises_intent_shape() -> None:
    contract = ActionContract.build(
        output_mode=OutputMode.SPEECH_INTENT,
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p07"],
    )

    schema = contract.json_schema
    assert schema["required"] == [
        "intent",
        "target_id",
        "speech",
        "reason",
        "confidence",
    ]
    assert "question_target" in schema["properties"]["intent"]["enum"]
    assert schema["properties"]["target_id"]["enum"] == ["p07", None]
    assert "action_type" not in schema["properties"]


def test_full_action_contract_preserves_legal_action_and_target_enums() -> None:
    contract = ActionContract.build(
        output_mode=OutputMode.FULL_ACTION,
        task_type=TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.USE_POISON, ActionType.NO_ACTION],
        legal_targets=["p07"],
    )

    schema = contract.json_schema
    assert schema["properties"]["action_type"]["enum"] == [
        "use_poison",
        "no_action",
    ]
    assert schema["properties"]["target_id"]["enum"] == ["p07", None]
    assert contract.tool["input_schema"] == schema
