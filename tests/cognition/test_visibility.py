# -*- coding: utf-8 -*-
"""
验证认知事实可见性配置与 GameEvent V2 actor/role 私有边界。

作者: Project contributors
修改日期: 2026-07-15
"""

from __future__ import annotations

import pytest

from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import (
    StructuredFact,
    StructuredWorldState,
    build_world_state,
)
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState, PlayerState


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


@pytest.mark.parametrize(
    "visibility",
    [visibility for visibility in EventVisibility if visibility is not EventVisibility.PUBLIC],
)
def test_every_non_public_event_visibility_hides_speech_from_other_villager(
    visibility: EventVisibility,
) -> None:
    state = GameState(
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="villager"),
        },
        events=[GameEvent(
            type="speech",
            payload={
                "speaker": "p01",
                "text": visibility.value,
                "visibility_actor_id": "p01",
                "visibility_role": "seer",
            },
            visibility=visibility,
            schema_version="2",
        )],
    )

    visible = VisibilityPolicy().filter_visible_facts(
        build_world_state(state),
        viewer_id="p02",
        viewer_role="villager",
    )

    assert visible == []


@pytest.mark.parametrize(
    ("visibility", "viewer_id", "viewer_role"),
    [
        (EventVisibility.MODERATOR_ONLY, "moderator", "moderator"),
        (EventVisibility.MODERATOR_FULL, "moderator", "moderator"),
        (EventVisibility.MODERATOR_POSTGAME, "moderator", "moderator"),
        (EventVisibility.WEREWOLF_TEAM_ONLY, "wolf", "werewolf"),
        (EventVisibility.SEER_PRIVATE, "seer", "seer"),
        (EventVisibility.SEER_ONLY, "seer", "seer"),
        (EventVisibility.WITCH_PRIVATE, "witch", "witch"),
        (EventVisibility.HYBRID_PRIVATE, "hybrid", "hybrid"),
        (EventVisibility.HYBRID_ONLY, "hybrid", "hybrid"),
        (EventVisibility.ROLE_PRIVATE, "seer", "seer"),
        (EventVisibility.ACTOR_PRIVATE, "target", "villager"),
        (EventVisibility.PRIVATE, "target", "villager"),
        (EventVisibility.PLAYER_ONLY, "target", "villager"),
    ],
)
def test_non_public_event_visibility_reaches_its_authorized_viewer(
    visibility: EventVisibility,
    viewer_id: str,
    viewer_role: str,
) -> None:
    state = GameState(
        players={"p01": PlayerState(id="p01", role="seer")},
        events=[GameEvent(
            type="speech",
            payload={
                "speaker": "p01",
                "text": visibility.value,
                "visibility_actor_id": "target",
                "visibility_role": "seer",
            },
            visibility=visibility,
            schema_version="2",
        )],
    )

    visible = VisibilityPolicy().filter_visible_facts(
        build_world_state(state),
        viewer_id=viewer_id,
        viewer_role=viewer_role,
    )

    assert [fact.value for fact in visible] == [visibility.value]


def test_unknown_event_visibility_fails_closed_before_fact_type_fallback() -> None:
    world = StructuredWorldState()
    fact = StructuredFact(
        fact_type="speech",
        source_player="p01",
        value="future private speech",
        metadata={"visibility": "future_private"},
    )
    world.append(fact)
    policy = VisibilityPolicy()

    label = policy.compute_fact_visibility(fact, 0)
    visible = policy.filter_visible_facts(world, "p02", "villager")

    assert label.visibility == "moderator_only"
    assert visible == []
