# -*- coding: utf-8 -*-
"""
验证 player generation helper 拆分后的调用边界。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_player_generation.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

from werewolf_agent.model_gateway.router import GenerateResult


class _Router:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return GenerateResult(text="{}", provider="mock", model="mock-model")

    def get_usage_log(self):
        return [
            SimpleNamespace(success=False, fallback_reason="primary_failed:timeout"),
        ]


def test_generate_player_response_delegates_to_model_router() -> None:
    from werewolf_agent.agents.player_generation import generate_player_response

    router = _Router()
    result = generate_player_response(
        router,
        agent_id="p01",
        task_type="vote",
        prompt="user",
        system_prompt="system",
        tools=[{"name": "submit_player_action"}],
        tool_choice={"type": "tool", "name": "submit_player_action"},
        structured_output_mode="native_tool",
    )

    assert result.text == "{}"
    assert router.calls == [
        {
            "agent_id": "p01",
            "task_type": "vote",
            "prompt": "user",
            "system_prompt": "system",
            "tools": [{"name": "submit_player_action"}],
            "tool_choice": {"type": "tool", "name": "submit_player_action"},
            "structured_output_mode": "native_tool",
        },
    ]


def test_latest_generation_failure_reason_reads_last_failed_usage_record() -> None:
    from werewolf_agent.agents.player_generation import latest_generation_failure_reason

    assert latest_generation_failure_reason(_Router()) == "primary_failed:timeout"


def test_final_prompt_observer_keeps_legacy_router_test_double_compatible() -> None:
    from werewolf_agent.agents.player_generation import generate_player_response

    class _LegacyRouter:
        def generate(
            self, *, agent_id, task_type, prompt, system_prompt, tools,
            tool_choice, structured_output_mode,
        ):
            return GenerateResult(text="{}", provider="legacy", model="m")

    result = generate_player_response(
        _LegacyRouter(), agent_id="p01", task_type="vote", prompt="user",
        system_prompt="system", tools=[], tool_choice=None,
        structured_output_mode="json_object", final_prompt_observer=lambda _: None,
    )

    assert result.text == "{}"
