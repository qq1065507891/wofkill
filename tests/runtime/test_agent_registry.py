# -*- coding: utf-8 -*-
"""
验证 agent registry 拆分后的兼容导入和基础行为。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_agent_registry.py -q
"""

from __future__ import annotations


def test_simple_agent_registry_behavior_matches_compat_import() -> None:
    from werewolf_agent.runtime.agent_adapter import SimpleAgentRegistry as CompatRegistry
    from werewolf_agent.runtime.agent_registry import SimpleAgentRegistry

    first_agent = object()
    second_agent = object()

    compat_registry = CompatRegistry({"p01": first_agent})
    split_registry = SimpleAgentRegistry({"p01": first_agent})

    compat_registry.register("p02", second_agent)
    split_registry.register("p02", second_agent)

    assert compat_registry.get_agent("p01") is first_agent
    assert split_registry.get_agent("p01") is first_agent
    assert compat_registry.get_agent("p02") is second_agent
    assert split_registry.get_agent("p02") is second_agent
    assert compat_registry.get_agent("missing") is None
    assert split_registry.get_agent("missing") is None


def test_agent_registry_protocol_remains_compatibly_importable() -> None:
    from werewolf_agent.runtime.agent_adapter import AgentRegistry as CompatRegistry
    from werewolf_agent.runtime.agent_registry import AgentRegistry

    assert CompatRegistry is AgentRegistry
