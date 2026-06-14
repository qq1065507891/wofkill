"""P0-S1 mode isolation: build_user_prompt must render only the active mode's fields.

Per consolidated audit (P0-S1, confirmed in g_3528592081: 95 actions
contained `intent` field, 63 contained `vote_basis` field — mode
bleeding in production).
"""

from __future__ import annotations

from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    RetryInfo,
    TaskType,
)


def _make_full_action_context() -> AgentContext:
    """Context that should produce FULL_ACTION mode output.

    FULL_ACTION triggers when task_type is NOT in _SPEECH_INTENT_TASKS
    and legal_actions is not a single target-requiring action.
    Use NIGHT_ACTION (e.g., witch) for this case.
    """
    return AgentContext(
        agent_id="p11",
        task_type=TaskType.NIGHT_ACTION,  # not in _SPEECH_INTENT_TASKS
        phase="night",
        night_number=1,
        own_role="witch",
        legal_actions=[ActionType.USE_ANTIDOTE, ActionType.USE_POISON, ActionType.NO_ACTION],
        legal_targets=["p07"],
        public_summary="N1 witch decision",
    )


def _make_target_choice_context() -> AgentContext:
    """Context that should produce TARGET_CHOICE mode output.

    TARGET_CHOICE triggers when legal_actions is exactly one target-requiring
    action with non-empty legal_targets. Vote is the canonical case.
    """
    return AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p07", "p08", "p09"],
        public_summary="Day 2 vote",
    )


def _make_speech_intent_context() -> AgentContext:
    """Context that should produce SPEECH_INTENT mode output.

    SPEECH_INTENT triggers when task_type is in _SPEECH_INTENT_TASKS
    (SPEECH, SHERIFF_SPEECH, DEFENSE_SPEECH, PK_SPEECH, LAST_WORDS)
    and legal_actions == [ActionType.SPEECH].
    """
    return AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        public_summary="Day 2 discussion",
    )


def test_full_action_mode_does_not_mention_choice_or_intent_in_contract():
    """FULL_ACTION mode's strict output contract must not advertise choice/intent fields."""
    ctx = _make_full_action_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Find the contract section (it begins with "最终输出协议")
    contract_idx = prompt.find("最终输出协议")
    assert contract_idx >= 0, "Expected '最终输出协议' in prompt"
    contract = prompt[contract_idx:]
    # FULL_ACTION contract should require action_type, not choice or intent
    assert "action_type" in contract, "FULL_ACTION contract should mention action_type"
    assert "choice" not in contract or "choice" in {"speech".casefold()}, (
        "FULL_ACTION contract must not mention 'choice' field"
    )
    # 'intent' as a standalone field reference is forbidden; the word may appear
    # in natural speech templates, so check the structured field marker
    assert "intent" not in contract or "intent" in {"intentional".casefold()}, (
        "FULL_ACTION contract must not mention 'intent' field"
    )


def test_target_choice_mode_does_not_mention_action_type_or_intent_in_contract():
    """TARGET_CHOICE mode's strict output contract must not advertise action_type/intent fields."""
    ctx = _make_target_choice_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    contract_idx = prompt.find("最终输出协议")
    assert contract_idx >= 0
    contract = prompt[contract_idx:]
    # TARGET_CHOICE contract should require choice, not action_type or intent
    assert "choice" in contract, "TARGET_CHOICE contract should mention choice"
    assert "action_type" not in contract, (
        "TARGET_CHOICE contract must not mention 'action_type' field"
    )
    assert "intent" not in contract, (
        "TARGET_CHOICE contract must not mention 'intent' field"
    )


def test_speech_intent_mode_does_not_mention_action_type_or_choice_in_contract():
    """SPEECH_INTENT mode's strict output contract must not advertise action_type/choice fields."""
    ctx = _make_speech_intent_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    contract_idx = prompt.find("最终输出协议")
    assert contract_idx >= 0
    contract = prompt[contract_idx:]
    # SPEECH_INTENT contract should require intent, not action_type or choice
    assert "intent" in contract, "SPEECH_INTENT contract should mention intent"
    assert "action_type" not in contract, (
        "SPEECH_INTENT contract must not mention 'action_type' field"
    )
    assert "choice" not in contract, (
        "SPEECH_INTENT contract must not mention 'choice' field"
    )


def test_output_mode_selection_distinguishes_three_modes():
    """Sanity: the 3 contexts above should resolve to 3 different output modes."""
    from werewolf_agent.agents.parse_dispatch import select_output_mode
    from werewolf_agent.agents.prompt_builder import _SPEECH_INTENT_TASKS

    # NIGHT_ACTION (e.g., witch) is not in _SPEECH_INTENT_TASKS and
    # legal_actions is multi-action, so it should resolve to FULL_ACTION.
    full_mode = select_output_mode(
        legal_actions=[ActionType.USE_ANTIDOTE, ActionType.NO_ACTION],
        legal_targets=["p07"],
        task_type=TaskType.NIGHT_ACTION,
        speech_intent_tasks=_SPEECH_INTENT_TASKS,
    )
    assert full_mode == OutputMode.FULL_ACTION

    speech_intent_mode = select_output_mode(
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        task_type=TaskType.SPEECH,
        speech_intent_tasks=_SPEECH_INTENT_TASKS,
    )
    assert speech_intent_mode == OutputMode.SPEECH_INTENT

    target_choice_mode = select_output_mode(
        legal_actions=[ActionType.VOTE],
        legal_targets=["p07"],
        task_type=TaskType.VOTE,
        speech_intent_tasks=_SPEECH_INTENT_TASKS,
    )
    assert target_choice_mode == OutputMode.TARGET_CHOICE


def test_prompt_builder_and_parse_dispatch_select_same_output_modes():
    """Prompt text and parser dispatch must share the same mode decision."""
    from werewolf_agent.agents.parse_dispatch import select_output_mode
    from werewolf_agent.agents.prompt_builder import _SPEECH_INTENT_TASKS

    task_types = [
        TaskType.SPEECH,
        TaskType.SHERIFF_SPEECH,
        TaskType.VOTE,
        TaskType.NIGHT_ACTION,
    ]
    action_sets = [
        [ActionType.SPEECH],
        [ActionType.SPEECH, ActionType.VOTE],
        [ActionType.VOTE],
        [ActionType.WOLF_KILL],
        [ActionType.USE_POISON, ActionType.NO_ACTION],
    ]

    for task_type in task_types:
        for legal_actions in action_sets:
            legal_targets = ["p07"] if ActionType.SPEECH not in legal_actions else ["p07", "p08"]
            ctx = AgentContext(
                agent_id="p01",
                task_type=task_type,
                phase="day",
                own_role="villager",
                legal_actions=legal_actions,
                legal_targets=legal_targets,
            )
            prompt_mode = PlayerPromptBuilder(ctx)._select_output_mode()
            dispatch_mode = select_output_mode(
                legal_actions=legal_actions,
                legal_targets=legal_targets,
                task_type=task_type,
                speech_intent_tasks=_SPEECH_INTENT_TASKS,
            )
            assert prompt_mode == dispatch_mode


def test_full_action_speech_prompt_omits_vote_audit_fields():
    """P0-S8 / P0-S1: speech prompt must not mention vote_basis or seer_stance anywhere.

    Game trace g_3528592081 shows 67 successful speech actions containing
    'vote_basis: "fallback"' even though prompt does not ask for it. The
    LLM is being defensive; the fix is to ensure the prompt never
    mentions these fields in non-vote contexts.
    """
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,  # → SPEECH_INTENT mode
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],  # no VOTE in legal
        legal_targets=[],
        public_summary="Day 1 discussion",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    # The prompt should not mention these vote-audit field names when SPEECH is the only action.
    assert "vote_basis" not in prompt, (
        "Speech prompt must not mention vote_basis (causes LLM to fill it in defensively)"
    )
    assert "seer_stance" not in prompt, (
        "Speech prompt must not mention seer_stance"
    )
    assert "standing_with_seer" not in prompt, (
        "Speech prompt must not mention standing_with_seer"
    )


def test_speech_task_with_legacy_vote_action_uses_speech_intent_contract():
    """A speech task must not inherit vote-output requirements from stale legal_actions."""
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p07", "p08"],
        public_summary="Day 1 discussion",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    contract_idx = prompt.find("最终输出协议")
    assert contract_idx >= 0
    contract = prompt[contract_idx:]
    assert "发言意图JSON对象" in contract
    assert "intent" in contract
    assert "action_type" not in contract
    for vote_field in (
        "vote_basis",
        "seer_stance",
        "standing_with_seer",
        "suspect_reason",
        "not_voting_reason",
        "private_reason",
    ):
        assert vote_field not in prompt


def test_full_action_non_vote_task_with_vote_action_omits_vote_audit_contract():
    """FULL_ACTION prompt must not require fields absent from ActionContract."""
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.REFLECTION,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p07", "p08"],
        public_summary="Reflection with stale legal vote option",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    contract_idx = prompt.find("最终输出协议")
    assert contract_idx >= 0
    contract = prompt[contract_idx:]
    assert "action_type" in contract
    assert "投票还必须包含" not in contract
    for vote_field in (
        "vote_basis",
        "seer_stance",
        "standing_with_seer",
        "suspect_reason",
        "not_voting_reason",
        "private_reason",
    ):
        assert vote_field not in contract


def test_target_choice_non_vote_task_omits_vote_audit_contract():
    """TARGET_CHOICE prompt follows the task_type-specific ActionContract."""
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.REFLECTION,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p07", "p08"],
        public_summary="Reflection with stale vote-only action",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    contract_idx = prompt.find("最终输出协议")
    assert contract_idx >= 0
    contract = prompt[contract_idx:]
    assert "choice" in contract
    assert "投票还必须包含" not in contract
    for vote_field in (
        "vote_basis",
        "seer_stance",
        "standing_with_seer",
        "suspect_reason",
        "not_voting_reason",
        "private_reason",
    ):
        assert vote_field not in contract
