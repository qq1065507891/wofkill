# -*- coding: utf-8 -*-
"""
构建不可变的 provider fallback 路由计划，并在执行前拒绝无效切换。

作者: Project contributors
创建日期: 2026-07-16

使用示例:
    >>> plan = build_fallback_routes(primary, candidates, "medium")
    >>> plan.routes
    (...,)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar


FALLBACK_ROUTE_UNAVAILABLE = "fallback_route_unavailable"


class FallbackRoute(Protocol):
    """路由策略所需的最小配置协议。"""

    provider: str
    model: str
    reasoning_capability: str


RouteT = TypeVar("RouteT", bound=FallbackRoute)


class RouteIdentityError(ValueError):
    """provider/model 无法构成可执行路由身份。"""


@dataclass(frozen=True)
class FallbackRouteFailure:
    """不含 provider 原始错误或密钥的结构化路由失败。"""

    code: str = FALLBACK_ROUTE_UNAVAILABLE
    rejected_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackRoutePlan:
    """去重后的不可变路由及可选失败说明。"""

    routes: tuple[FallbackRoute, ...]
    failure: FallbackRouteFailure | None = None

    @property
    def failure_code(self) -> str | None:
        return self.failure.code if self.failure else None


def build_fallback_routes(
    primary: RouteT,
    candidates: Iterable[RouteT],
    minimum_reasoning: str,
) -> FallbackRoutePlan:
    """按配置顺序去重并过滤候选，不把无效候选写入 attempt 链。"""
    candidate_items = tuple(candidates)
    accepted: list[RouteT] = []
    rejected: list[str] = []
    try:
        primary_identity = normalize_route_identity(primary)
    except RouteIdentityError:
        return FallbackRoutePlan(
            routes=(),
            failure=FallbackRouteFailure(
                rejected_reasons=("invalid_primary_route",),
            ),
        )
    seen = {primary_identity}
    for candidate in candidate_items:
        try:
            identity = normalize_route_identity(candidate)
        except RouteIdentityError:
            rejected.append("invalid_route_identity")
            continue
        if identity in seen:
            rejected.append(
                "same_as_primary" if identity == primary_identity
                else "duplicate_candidate"
            )
            continue
        seen.add(identity)
        if not _reasoning_satisfies(
            candidate.reasoning_capability,
            minimum_reasoning,
        ):
            rejected.append("reasoning_below_minimum")
            continue
        accepted.append(candidate)
    if accepted:
        return FallbackRoutePlan(routes=tuple(accepted))
    return FallbackRoutePlan(
        routes=(),
        failure=FallbackRouteFailure(
            rejected_reasons=tuple(rejected) or ("no_candidates",),
        ),
    )


def route_switch_is_valid(current: FallbackRoute, next_route: FallbackRoute) -> bool:
    """运行时门禁：下一路由必须与当前 provider/model 对不同。"""
    try:
        return normalize_route_identity(current) != normalize_route_identity(next_route)
    except RouteIdentityError:
        return False


def normalize_route_identity(route: FallbackRoute) -> tuple[str, str]:
    """仅规范化身份比较；执行配置仍保留原 provider/model 字面值。"""
    provider = route.provider
    model = route.model
    if not isinstance(provider, str) or not isinstance(model, str):
        raise RouteIdentityError("route provider and model must be strings")
    normalized_provider = provider.strip().casefold()
    normalized_model = model.strip()
    if not normalized_provider or not normalized_model:
        raise RouteIdentityError("route provider and model must be non-empty")
    return normalized_provider, normalized_model


def _reasoning_satisfies(capability: str, required: str) -> bool:
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return order.get(capability, -1) >= order.get(required, 99)


__all__ = [
    "FALLBACK_ROUTE_UNAVAILABLE",
    "FallbackRouteFailure",
    "FallbackRoutePlan",
    "RouteIdentityError",
    "build_fallback_routes",
    "normalize_route_identity",
    "route_switch_is_valid",
]
