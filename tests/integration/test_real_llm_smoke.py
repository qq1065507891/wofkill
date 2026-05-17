"""Optional smoke tests for real LLM providers.

These tests are skipped by default. Enable them only when intentionally
spending provider quota:

    WEREWOLF_RUN_REAL_LLM_SMOKE=1

At least one configured provider API key must also be present.
"""

from __future__ import annotations

import os

import pytest

from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    PlayerAction,
    TaskType,
)
from werewolf_agent.model_gateway.router import ModelRouter


pytestmark = pytest.mark.skipif(
    os.getenv("WEREWOLF_RUN_REAL_LLM_SMOKE") != "1",
    reason="set WEREWOLF_RUN_REAL_LLM_SMOKE=1 to run real-provider smoke tests",
)


def test_real_provider_player_agent_schema_boundary() -> None:
    """Real provider output must pass through PlayerAgent schema/fallback boundary."""
    router = ModelRouter.from_yaml(
        "config/models.yaml",
        register_env_providers=True,
    )
    if not router.provider_names():
        pytest.skip("no provider API key configured")

    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
        public_summary="Day 1 public discussion. No hidden identities are known.",
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        visible_world_state={
            "public_events": [
                {"type": "speech", "speaker": "p02", "text": "I am observing."}
            ],
        },
    )

    action, retry = agent.act(context)

    assert isinstance(action, (PlayerAction, FallbackAction))
    assert action.action_type in {ActionType.SPEECH, ActionType.NO_ACTION}
    assert retry.attempt >= 1
    usage = router.get_usage_log()
    assert usage, "real-provider smoke should record at least one model call"
