# -*- coding: utf-8 -*-
"""
验证夜晚结算节点拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-15

使用示例:
    >>> python -m pytest tests/runtime/test_night_resolution.py -q
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.core.resolution_batches import ResolutionBatchV2


def test_night_resolution_node_remains_compatibly_importable() -> None:
    from werewolf_agent.runtime.nodes import night
    from werewolf_agent.runtime.nodes import night_resolution

    assert night.resolve_night is night_resolution.resolve_night


def test_night_death_producers_write_cause_specific_v2_batches() -> None:
    from werewolf_agent.engine.rule_night import resolve_night

    state = GameState(
        game_id="night-v2",
        players={
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "victim": PlayerState(id="victim", role="villager"),
            "witch": PlayerState(id="witch", role="witch"),
            "poisoned": PlayerState(id="poisoned", role="villager"),
        },
        night_number=2,
    )
    deaths = []

    resolve_night(
        {
            "roles": {
                "witch": {
                    "abilities": {
                        "use_both_potions_same_night": True,
                        "antidote": {},
                    }
                }
            }
        },
        state,
        night_number=2,
        wolf_kill_target_id="victim",
        poison_target_id="poisoned",
        resolve_witch_action_fn=lambda *_args, **_kwargs: type(
            "Result", (), {"accepted": True, "error_code": None}
        )(),
        validate_alive_target_fn=lambda *_args: None,
        check_alignment_fn=lambda *_args, **_kwargs: None,
        apply_death_fn=lambda current, death: deaths.append(death) or current,
    )

    assert [death.resolution_batch for death in deaths] == [
        ResolutionBatchV2("night", 2, "wolf_kill"),
        ResolutionBatchV2("night", 2, "witch_poison"),
    ]
