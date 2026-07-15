# -*- coding: utf-8 -*-
"""
验证认知事实可见性配置与 GameEvent V2 actor/role 私有边界。

作者: Project contributors
修改日期: 2026-07-15
"""

from __future__ import annotations


class TestVisibilityConfigWiring:
    """审查 C2: visibility_config 应能覆写 hardcoded 可见性表。"""

    def test_visibility_init_wires_config_override(self):
        from werewolf_agent.cognition.visibility import VisibilityPolicy
        from werewolf_agent.core.models import GameEvent, GameState
        # 构造一个 config 把 speech 改投到 moderator_only
        custom_config = {"_FACT_VISIBILITY_MAP": {
            "speech": "moderator_only",
        }}
        policy = VisibilityPolicy(visibility_config=custom_config)
        gs = GameState(
            players={}, day_number=1, night_number=1,
            events=[GameEvent(type="speech", payload={"speaker": "p01", "text": "hi"})],
        )
        visible = policy.compute_fact_visibility(gs, viewer_role="villager")
        facts = [f for f in visible if f.fact_type == "speech"]
        # 当前实现：speech 是 public，villager 能看到
        # 修复后：speech 是 moderator_only，villager 看不到
        assert len(facts) == 0, (
            f"custom config did not override fact_type visibility: {[f.fact_type for f in visible]}"
        )


def test_v2_actor_and_role_private_speech_only_reaches_eligible_viewers() -> None:
    from werewolf_agent.cognition.visibility import VisibilityPolicy
    from werewolf_agent.cognition.world_state import build_world_state
    from werewolf_agent.core.event_visibility import EventVisibility
    from werewolf_agent.core.models import GameEvent, GameState, PlayerState

    state = GameState(
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="seer"),
            "p03": PlayerState(id="p03", role="villager"),
        },
        events=[
            GameEvent(
                type="speech",
                payload={"speaker": "p01", "text": "actor secret"},
                visibility=EventVisibility.ACTOR_PRIVATE,
                schema_version="2",
            ),
            GameEvent(
                type="speech",
                payload={"speaker": "p01", "text": "role secret"},
                visibility=EventVisibility.ROLE_PRIVATE,
                schema_version="2",
            ),
        ],
    )
    world = build_world_state(state)
    policy = VisibilityPolicy()

    actor_values = {
        fact.value
        for fact in policy.filter_visible_facts(world, "p01", "seer")
    }
    same_role_values = {
        fact.value
        for fact in policy.filter_visible_facts(world, "p02", "seer")
    }
    other_values = {
        fact.value
        for fact in policy.filter_visible_facts(world, "p03", "villager")
    }

    assert "actor secret" in actor_values
    assert "actor secret" not in same_role_values
    assert "actor secret" not in other_values
    assert "role secret" in actor_values
    assert "role secret" in same_role_values
    assert "role secret" not in other_values
