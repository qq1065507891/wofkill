# -*- coding: utf-8 -*-
"""
RuleEngine 低风险 helper 拆分后的兼容测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/rules/test_rule_engine_split.py
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.engine import rule_flow, rule_last_words, rule_special_roles, rule_vote
from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset


RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def test_rule_engine_low_risk_helpers_are_available() -> None:
    assert callable(rule_flow.assign_roles)
    assert callable(rule_flow.day_flow)
    assert callable(rule_special_roles.check_alignment)
    assert callable(rule_special_roles.resolve_witch_action)
    assert callable(rule_vote.vote_weight)
    assert callable(rule_vote.resolve_vote)
    assert callable(rule_last_words.can_leave_last_words)


def test_rule_engine_keeps_old_path_behavior_for_split_helpers() -> None:
    engine = RuleEngine.from_yaml(RULESET_PATH)
    player_ids = [f"p{i:02d}" for i in range(1, 13)]

    players = engine.assign_roles(player_ids, seed=42)
    first_day = engine.day_flow(1)
    second_day = engine.day_flow(2)

    assert set(players) == set(player_ids)
    assert "first_day_sheriff_election" in first_day
    assert "first_day_sheriff_election" not in second_day


def test_rule_engine_split_helpers_accept_ruleset_or_raw_dict() -> None:
    engine = RuleEngine.from_yaml(RULESET_PATH)
    raw_engine = RuleEngine(engine.ruleset.raw)
    wrapped_engine = RuleEngine(Ruleset(raw=engine.ruleset.raw))

    assert raw_engine.night_order() == wrapped_engine.night_order()


def test_special_role_helpers_preserve_seer_and_witch_behavior() -> None:
    engine = RuleEngine.from_yaml(RULESET_PATH)
    state = GameState(
        players={
            "seer": PlayerState(id="seer", role="seer", alive=True),
            "witch": PlayerState(id="witch", role="witch", alive=True),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
        }
    )

    alignment = engine.check_alignment(state, target_id="wolf")
    result = engine.resolve_witch_action(
        state,
        witch_id="witch",
        night_number=1,
        wolf_kill_target_id="wolf",
        use_antidote=True,
        poison_target_id=None,
    )

    assert alignment.alignment == "werewolf"
    assert result.accepted is True


def test_vote_helpers_preserve_majority_tie_and_anti_stall_behavior() -> None:
    engine = RuleEngine.from_yaml(RULESET_PATH)
    state = GameState(
        players={
            "sheriff": PlayerState(id="sheriff", role="seer", alive=True),
            "v1": PlayerState(id="v1", role="villager", alive=True),
            "v2": PlayerState(id="v2", role="villager", alive=True),
            "w1": PlayerState(id="w1", role="werewolf", alive=True),
            "w2": PlayerState(id="w2", role="werewolf", alive=True),
        },
        sheriff_id="sheriff",
        sheriff_badge_state="active",
    )

    majority = engine.resolve_vote(
        state,
        votes={"sheriff": "w1", "v1": "w2"},
        revote=False,
    )
    first_tie = engine.resolve_vote(
        state,
        votes={"v1": "w1", "v2": "w2"},
        revote=False,
    )
    second_tie = engine.resolve_vote(
        state,
        votes={"v1": "w1", "v2": "w2"},
        revote=True,
    )
    anti_stall = engine.resolve_vote(
        state,
        votes={"v1": "w1", "v2": "w2"},
        revote=True,
        consecutive_no_exile_days=2,
        rng_seed="split-vote",
    )

    assert majority.exiled_player_id == "w1"
    assert first_tie.next_phase == "pk_speech"
    assert first_tie.tied_player_ids == ["w1", "w2"]
    assert second_tie.exiled_player_id is None
    assert second_tie.reason == "second_tie_no_exile"
    assert anti_stall.exiled_player_id == "w2"
    assert anti_stall.reason == "anti_stall_tie_break"


def test_vote_split_preserves_rule_engine_instance_override_points() -> None:
    class CustomVoteEngine(RuleEngine):
        def legal_exile_targets(self, state: GameState) -> list[str]:
            return ["w2"]

        def vote_weight(self, state: GameState, voter_id: str) -> int:
            return 99 if voter_id == "v1" else 1

    engine = CustomVoteEngine.from_yaml(RULESET_PATH)
    state = GameState(
        players={
            "v1": PlayerState(id="v1", role="villager", alive=True),
            "v2": PlayerState(id="v2", role="villager", alive=True),
            "w1": PlayerState(id="w1", role="werewolf", alive=True),
            "w2": PlayerState(id="w2", role="werewolf", alive=True),
        },
    )

    result = engine.resolve_vote(
        state,
        votes={"v1": "w1", "v2": "w2"},
        revote=False,
    )

    assert result.exiled_player_id == "w2"


def test_last_words_helper_preserves_death_reason_precedence() -> None:
    engine = RuleEngine.from_yaml(RULESET_PATH)

    assert engine.can_leave_last_words(
        death_reason="hunter_shot",
        timing="night",
        night_number=1,
    ) is False
    assert rule_last_words.can_leave_last_words(
        engine.ruleset.raw,
        death_reason="exile",
        timing="day_vote",
        night_number=2,
    ) is True
