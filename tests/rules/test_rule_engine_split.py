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
from werewolf_agent.engine import rule_flow, rule_special_roles
from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset


RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def test_rule_engine_low_risk_helpers_are_available() -> None:
    assert callable(rule_flow.assign_roles)
    assert callable(rule_flow.day_flow)
    assert callable(rule_special_roles.check_alignment)
    assert callable(rule_special_roles.resolve_witch_action)


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
