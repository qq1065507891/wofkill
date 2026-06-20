"""赛后反思 context allowlist 裁剪测试.

根因:`_agent_reflection` 调 `build_agent_context(TaskType.REFLECTION)` 后,
context.strategy_directive 仍装满赛内决策 directive(role_alerts /
skill_tactical_advice / witch_poison_deterrent / must_address_alerts 等),
反思指令 reflection_task 被淹没,导致 LLM 输出赛内决策而非赛后反思。

修复:`_strip_in_game_directives` 在 build_agent_context 之后、
_merge_strategy_directive 之前剥离赛内 directive,只留 allowlist。
"""
from __future__ import annotations

from werewolf_agent.agents.schemas import AgentContext, TaskType
from werewolf_agent.runtime.agent_adapter import _strip_in_game_directives


def _make_context(directive: dict) -> AgentContext:
    return AgentContext(
        agent_id="p1",
        task_type=TaskType.REFLECTION,
        strategy_directive=dict(directive),
    )


def test_strip_removes_in_game_keys_keeps_allowlist():
    ctx = _make_context({
        "role_alerts": ["你今晚该刀 p3"],
        "skill_tactical_advice": {"vote": "倒钩"},
        "witch_poison_deterrent": "毒药威慑",
        "must_address_alerts": "必须回应金水",
        # allowlist
        "reflection_task": "反思模板正文",
        "game_outcome": "胜利方是狼人阵营。",
    })

    stripped = _strip_in_game_directives(ctx)

    assert set(stripped.strategy_directive.keys()) == {"reflection_task", "game_outcome"}
    assert stripped.strategy_directive["reflection_task"] == "反思模板正文"
    assert stripped.strategy_directive["game_outcome"] == "胜利方是狼人阵营。"


def test_strip_is_idempotent_when_no_in_game_directives():
    ctx = _make_context({
        "reflection_task": "x",
        "game_outcome": "y",
    })

    stripped = _strip_in_game_directives(ctx)

    assert set(stripped.strategy_directive.keys()) == {"reflection_task", "game_outcome"}


def test_strip_returns_empty_when_only_in_game_directives():
    ctx = _make_context({
        "role_alerts": ["a"],
        "skill_tactical_advice": {"b": 1},
    })

    stripped = _strip_in_game_directives(ctx)

    assert stripped.strategy_directive == {}


def test_strip_handles_none_strategy_directive():
    ctx = AgentContext(agent_id="p1", task_type=TaskType.REFLECTION)

    stripped = _strip_in_game_directives(ctx)

    assert stripped.strategy_directive == {}


def test_strip_does_not_mutate_original_context():
    original = {
        "role_alerts": ["a"],
        "reflection_task": "x",
    }
    ctx = _make_context(original)

    _strip_in_game_directives(ctx)

    # 调用方原 context 不被改写
    assert "role_alerts" in ctx.strategy_directive
    assert ctx.strategy_directive["role_alerts"] == ["a"]
