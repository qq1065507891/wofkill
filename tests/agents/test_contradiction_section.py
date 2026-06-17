"""P2 contradiction-prompt-section: 公开矛盾点独立 prompt section 测试。"""

from __future__ import annotations

from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.schemas import ActionType, AgentContext, RetryInfo, TaskType


def _make_ctx(alerts: list) -> AgentContext:
    return AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        public_summary="D2",
        contradiction_alerts=alerts,
    )


def test_contradiction_alerts_rendered_as_public_section():
    """contradiction_alerts 应渲染为'公开矛盾点'独立 section。"""
    ctx = _make_ctx([
        {"alert_type": "vote_conflict", "player_id": "p03", "priority": "medium",
         "description": "p03 claimed p01 suspect but voted p02", "evidence": []},
    ])
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "公开矛盾点" in prompt
    assert "vote_conflict" in prompt
    assert "p03" in prompt


def test_contradiction_alerts_empty_omits_section():
    """无 contradiction_alerts 时不渲染 section（回归）。"""
    ctx = _make_ctx([])
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "公开矛盾点" not in prompt


def test_contradiction_alerts_sorted_by_priority():
    """high 优先级矛盾应排在 medium 之前。"""
    ctx = _make_ctx([
        {"alert_type": "vote_conflict", "player_id": "p03", "priority": "medium",
         "description": "m", "evidence": []},
        {"alert_type": "claim_conflict", "player_id": "p01,p05", "priority": "high",
         "description": "h", "evidence": []},
    ])
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    high_idx = prompt.find("claim_conflict")
    med_idx = prompt.find("vote_conflict")
    assert high_idx >= 0 and med_idx >= 0
    assert high_idx < med_idx


def test_contradiction_alerts_capped_at_three():
    """超过 3 条矛盾点只渲染前 3 条。"""
    alerts = [
        {"alert_type": f"type{i}", "player_id": f"p0{i}", "priority": "low",
         "description": f"d{i}", "evidence": []}
        for i in range(5)
    ]
    ctx = _make_ctx(alerts)
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "type0" in prompt and "type1" in prompt and "type2" in prompt
    assert "type3" not in prompt and "type4" not in prompt
