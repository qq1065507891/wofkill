# -*- coding: utf-8 -*-
"""
测试女巫夜晚行动阶段的指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.witch_night_directives import build_witch_potion_status
    >>> build_witch_potion_status(antidote_used=False, poison_used=False)
"""

from werewolf_agent.agents.schemas import ActionType
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.witch_night_directives import (
    build_witch_first_night_killed_directive,
    build_witch_legal_actions,
    build_witch_night_action_directive,
    build_witch_poison_candidates_directive,
    build_witch_poison_strategy,
    build_witch_potion_status,
    build_witch_pressure_directives,
    build_witch_strategy_hint,
)


RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def _make_game_state(
    *,
    antidote_used: bool = False,
    poison_used: bool = False,
) -> GameState:
    players = {
        "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
        "victim": PlayerState(id="victim", role="villager", alive=True),
        "witch": PlayerState(id="witch", role="witch", alive=True),
        "seer": PlayerState(id="seer", role="seer", alive=True),
    }
    return GameState(
        game_id="witch_night_directives",
        players=players,
        phase="night",
        night_number=1,
        antidote_used=antidote_used,
        poison_used=poison_used,
    )


def test_build_witch_legal_actions_respects_potion_state_and_self_save_rule() -> None:
    """女巫合法行动应显式反映药水状态，并排除默认规则下的自救。"""
    engine = RuleEngine.from_yaml(RULESET_PATH)
    gs = _make_game_state()

    actions, targets = build_witch_legal_actions(
        gs,
        engine,
        witch_id="witch",
        wolf_kill_target_id="witch",
    )

    assert ActionType.NO_ACTION in actions
    assert ActionType.USE_ANTIDOTE not in actions
    assert ActionType.USE_POISON in actions
    assert "witch" not in targets
    assert {"wolf", "victim", "seer"}.issubset(set(targets))


def test_build_witch_potion_status_names_remaining_potion() -> None:
    """药水状态文案应说明已用和剩余药水。"""
    status = build_witch_potion_status(antidote_used=True, poison_used=False)

    assert status == "当前药水状态：解药已用，毒药可用。你只剩毒药，只能选择毒人或不用。"


def test_build_witch_night_action_directive_keeps_action_contract() -> None:
    """夜晚行动主指令应保留原有 action_type 契约和静默要求。"""
    directive = build_witch_night_action_directive(
        wolf_kill_target_id="victim",
        witch_id="witch",
        antidote_used=False,
        poison_used=False,
        can_use_antidote=True,
        can_use_poison=True,
    )

    assert "使用解药救victim（他被狼人杀害了）" in directive
    assert "action_type='use_antidote', target_id='victim'" in directive
    assert "action_type='use_poison', target_id='目标玩家ID'" in directive
    assert "不能在同一夜同时使用解药和毒药" in directive
    assert directive.endswith("speech字段留空（夜间行动不需要发言）。")


def test_build_witch_strategy_hint_renders_first_night_tradeoff() -> None:
    """首夜救人价值提示应渲染概率框架，并补充毒药可用提醒。"""
    save_value = {
        "actionable": True,
        "public_info_available": False,
        "probability_framework": {"p_power_role": 0.4, "p_villager": 0.6},
        "trade_off": {
            "save_now": "立即救人",
            "save_later": "保留解药",
            "risk_no_save": "目标可能死亡",
        },
    }

    hint = build_witch_strategy_hint(save_value, poison_available=True)

    assert "首夜无公开信息" in hint
    assert "约40%" in hint
    assert "立即救人 | 保留解药 | 目标可能死亡" in hint
    assert "毒药可用时" in hint


def test_build_witch_poison_strategy_selects_alive_count_branch() -> None:
    """毒药策略应按存活人数选择唯一分支。"""
    urgent = build_witch_poison_strategy(alive_count=7)
    mid = build_witch_poison_strategy(alive_count=9)
    early = build_witch_poison_strategy(alive_count=10)

    assert urgent is not None
    assert mid is not None
    assert early is not None
    assert urgent["branch"] == "urgency_under_X_alive"
    assert mid["branch"] == "evidence_required_threshold"
    assert early["branch"] == "no_pressure_save_for_late"


def test_build_witch_poison_candidates_directive_renders_candidates_and_guards() -> None:
    """毒药候选文案应优先列出结构化候选；无候选时给 no_action 防误毒提示。"""
    candidates = [
        {"player_id": "p02", "reason": "查杀"},
        {"player_id": "p05", "reason": "强票型"},
    ]

    with_candidates = build_witch_poison_candidates_directive(candidates, alive_count=8)
    no_candidates = build_witch_poison_candidates_directive([], alive_count=6)

    assert "p02(查杀); p05(强票型)" in with_candidates
    assert "优先从以上候选中选" in with_candidates
    assert "紧急但证据不足" in no_candidates
    assert "默认 no_action" in no_candidates


def test_build_witch_first_night_killed_directive_only_when_killed_with_poison() -> None:
    """只有女巫被首夜刀且毒药仍可用时才注入首夜被刀提示。"""
    directive = build_witch_first_night_killed_directive(
        wolf_kill_target_id="witch",
        witch_id="witch",
        poison_used=False,
    )

    assert directive is not None
    assert "首夜就被狼人杀害" in directive
    assert build_witch_first_night_killed_directive(
        wolf_kill_target_id="victim",
        witch_id="witch",
        poison_used=False,
    ) is None


def test_build_witch_pressure_directives_requires_reason_when_skipping_poison() -> None:
    """毒药压力目标存在时应渲染压力说明和不用毒解释要求。"""
    directive = build_witch_pressure_directives([
        {
            "player_id": "p02",
            "pressure_type": "black_claim",
            "description": "被p01查杀",
        },
    ])

    assert directive["witch_pressure"] == "存在毒药压力目标: p02(black_claim: 被p01查杀)"
    assert directive["required_evaluation"] == "如果选择不用毒药，必须在reason中解释为什么不用。"
