"""Tests for visibility policy config wiring (post-review C2)."""

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
