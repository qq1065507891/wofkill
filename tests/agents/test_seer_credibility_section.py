"""Seer credibility prompt section tests (spec §Prompt Integration)."""

from __future__ import annotations

from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.schemas import ActionType, AgentContext, RetryInfo, TaskType


def _make_ctx(seer_credibility: dict) -> AgentContext:
    return AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH, phase="day", day_number=2,
        own_role="villager", legal_actions=[ActionType.SPEECH], legal_targets=[],
        public_summary="D2", seer_credibility=seer_credibility,
    )


def test_seer_credibility_rendered_when_present():
    ctx = _make_ctx({"seer_lines": [
        {"claimant": "p08", "status": "supported", "score": 0.78,
         "checks": ["wolf:p01"], "evidence": ["vote_follows_black"]},
    ]})
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "预言家线可信度" in prompt
    assert "p08" in prompt
    assert "supported" in prompt


def test_seer_credibility_omitted_when_empty():
    ctx = _make_ctx({})
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "预言家线可信度" not in prompt


def test_seer_credibility_capped_at_three():
    lines = [
        {"claimant": f"p0{i}", "status": "weak", "score": 0.3, "checks": [], "evidence": []}
        for i in range(4)
    ]
    ctx = _make_ctx({"seer_lines": lines})
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "p00" in prompt and "p01" in prompt and "p02" in prompt
    assert "p03" not in prompt
