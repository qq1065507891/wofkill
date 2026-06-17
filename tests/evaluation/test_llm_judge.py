"""Semantic judge tests for speech/action consistency."""

from __future__ import annotations


def test_judge_accepts_speech_grounded_in_visible_context() -> None:
    from werewolf_agent.evaluation.llm_judge import judge_speech_consistency

    result = judge_speech_consistency(
        context={
            "role": "villager",
            "faction": "good",
            "public_facts": [
                "p03 claimed seer",
                "p03 black-checked p07",
                "p07 voted p03",
            ],
        },
        action={
            "action_type": "speech",
            "speech": "我站p03这条预言家线，p07被查杀后还投p03，今天我倾向投p07。",
        },
    )

    assert result.ok is True
    assert result.consistency_score == 1.0
    assert result.issues == []


def test_judge_rejects_public_fact_reference_not_in_context() -> None:
    from werewolf_agent.evaluation.llm_judge import judge_speech_consistency

    result = judge_speech_consistency(
        context={
            "role": "villager",
            "faction": "good",
            "public_facts": ["p03 claimed seer"],
        },
        action={
            "action_type": "speech",
            "speech": "p03说自己是预言家，p05已经被查杀，所以我今天投p05。",
        },
    )

    assert result.ok is False
    assert result.consistency_score < 1.0
    assert result.has_issue("public_fact_reference")


def test_judge_binds_public_fact_reference_to_same_player() -> None:
    from werewolf_agent.evaluation.llm_judge import judge_speech_consistency

    result = judge_speech_consistency(
        context={
            "role": "villager",
            "faction": "good",
            "public_facts": [
                "p03 black-checked p07",
                "p05 voted p03",
            ],
        },
        action={
            "action_type": "speech",
            "speech": "p05已经被查杀，所以我今天投p05。",
        },
    )

    assert result.ok is False
    assert result.has_issue("public_fact_reference")


def test_judge_binds_seer_claim_reference_to_same_player() -> None:
    from werewolf_agent.evaluation.llm_judge import judge_speech_consistency

    result = judge_speech_consistency(
        context={
            "role": "villager",
            "faction": "good",
            "public_facts": [
                "p08 claimed seer",
                "p05 voted p03",
            ],
        },
        action={
            "action_type": "speech",
            "speech": "p05是预言家，所以我今天站p05。",
        },
    )

    assert result.ok is False
    assert result.has_issue("public_fact_reference")


def test_judge_rejects_good_player_claiming_hidden_wolf_team_task() -> None:
    from werewolf_agent.evaluation.llm_judge import judge_speech_consistency

    result = judge_speech_consistency(
        context={
            "role": "seer",
            "faction": "good",
            "public_facts": ["p01 claimed seer"],
        },
        action={
            "action_type": "speech",
            "speech": "我是预言家，但这轮我的狼队任务是保护队友并带票好人。",
        },
    )

    assert result.ok is False
    assert result.has_issue("faction_task_consistency")


def test_judge_rejects_wolf_publicly_leaking_real_identity() -> None:
    from werewolf_agent.evaluation.llm_judge import judge_speech_consistency

    result = judge_speech_consistency(
        context={
            "role": "werewolf",
            "faction": "werewolf",
            "public_claim": "villager",
            "public_facts": ["p04 attacked p08"],
        },
        action={
            "action_type": "speech",
            "speech": "我是狼人，但我公开装成村民，现在想把p08推出局。",
        },
    )

    assert result.ok is False
    assert result.has_issue("identity_consistency")
