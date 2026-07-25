# -*- coding: utf-8 -*-
"""
验证赛后反思使用独立严格 Schema 和有限重试。

作者: Project contributors
创建日期: 2026-07-25
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from werewolf_agent.agents.player import PlayerAgent, ReflectionDraftGenerationError
from werewolf_agent.agents.schemas import AgentContext, TaskType


class _ReflectionRouter:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.texts.pop(0))


def test_generate_reflection_retries_invalid_schema_then_returns_draft() -> None:
    router = _ReflectionRouter([
        '{"claims":"wrong","lessons":[]}',
        '{"claims":[],"lessons":[]}',
    ])
    agent = PlayerAgent("p01", router, max_retries=2)

    draft = agent.generate_reflection(
        AgentContext(agent_id="p01", task_type=TaskType.REFLECTION),
        "reflection prompt",
    )

    assert draft.claims == []
    assert len(router.calls) == 2
    assert all(call["task_type"] == "reflection" for call in router.calls)
    assert all(call.get("tools") is None for call in router.calls)
    system_prompt = str(router.calls[-1]["system_prompt"])
    assert '"title": "ReflectionDraft"' in system_prompt
    assert '"claims"' in system_prompt
    assert "action_type" not in system_prompt


def test_generate_reflection_terminal_schema_failure_has_safe_diagnostics() -> None:
    router = _ReflectionRouter([
        '{"claims":"PRIVATE_PROVIDER_TEXT","lessons":[]}',
        '{"claims":"PRIVATE_PROVIDER_TEXT","lessons":[]}',
    ])
    agent = PlayerAgent("p01", router, max_retries=2)

    with pytest.raises(ReflectionDraftGenerationError) as raised:
        agent.generate_reflection(
            AgentContext(agent_id="p01", task_type=TaskType.REFLECTION),
            "reflection prompt",
        )

    assert raised.value.failure_code == "invalid_structured_draft"
    assert raised.value.field_paths == ("claims",)
    assert "PRIVATE_PROVIDER_TEXT" not in repr(raised.value)
    assert len(router.calls) == 2
