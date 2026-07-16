# -*- coding: utf-8 -*-
"""
验证 provider fallback 路由在执行前完成去重、能力过滤与同路由拒绝。

作者: Project contributors
创建日期: 2026-07-16
"""

from dataclasses import replace

import pytest

from werewolf_agent.model_gateway.fallback_policy import build_fallback_routes
from werewolf_agent.model_gateway.usage_records import ModelConfig


def _route(provider: str, model: str, reasoning: str = "high") -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model=model,
        reasoning_capability=reasoning,
    )


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (_route("primary", "model-a"), ()),
        (_route("backup", "model-a"), (("backup", "model-a"),)),
        (_route("primary", "model-b"), (("primary", "model-b"),)),
    ],
)
def test_route_identity_uses_provider_and_model_pair(candidate, expected) -> None:
    plan = build_fallback_routes(
        _route("primary", "model-a"), [candidate], "medium"
    )

    assert tuple((item.provider, item.model) for item in plan.routes) == expected


def test_duplicate_candidates_are_removed_without_mutating_inputs() -> None:
    primary = _route("primary", "model-a")
    candidate = _route("backup", "model-b")
    candidates = [candidate, replace(candidate, timeout=99), candidate]

    plan = build_fallback_routes(primary, candidates, "medium")

    assert plan.routes == (candidate,)
    assert candidates == [candidate, replace(candidate, timeout=99), candidate]


def test_candidates_below_minimum_reasoning_are_rejected() -> None:
    plan = build_fallback_routes(
        _route("primary", "model-a"),
        [_route("weak", "model-b", "low")],
        "medium",
    )

    assert plan.routes == ()
    assert plan.failure is not None
    assert plan.failure.code == "fallback_route_unavailable"
    assert plan.failure.rejected_reasons == ("reasoning_below_minimum",)


def test_no_candidates_returns_structured_unavailable_result() -> None:
    plan = build_fallback_routes(_route("primary", "model-a"), [], "none")

    assert plan.routes == ()
    assert plan.failure is not None
    assert plan.failure.code == "fallback_route_unavailable"
    assert plan.failure.rejected_reasons == ("no_candidates",)


def test_plan_and_routes_are_immutable() -> None:
    plan = build_fallback_routes(
        _route("primary", "model-a"), [_route("backup", "model-b")], "none"
    )

    with pytest.raises((AttributeError, TypeError)):
        plan.routes += (_route("third", "model-c"),)
