# -*- coding: utf-8 -*-
"""
模型网关异常格式化、失败归因和确定性重试策略。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-23

使用示例:
    >>> from werewolf_agent.model_gateway.retry_policy import _format_exception
    >>> _format_exception(None)
    'unknown'
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
import math
import re

from werewolf_agent.model_gateway.execution_records import RouteKind


class RetryKind(str, Enum):
    """可重试失败的等待与预算类别。"""

    GENERIC = "generic"
    RATE_LIMIT = "rate_limit"


@dataclass
class RetryBudget:
    """单个 route candidate 的重试计数与上限。"""

    route_kind: RouteKind
    config_retry_count: int
    total_retry_count: int = 0
    generic_retry_count: int = 0
    rate_limit_retry_count: int = 0

    def can_retry(self, retry_kind: RetryKind) -> bool:
        """判断当前候选是否还有该类别的重试额度。"""
        if self.config_retry_count <= 0 or self.total_retry_count >= self.config_retry_count:
            return False
        if retry_kind is RetryKind.GENERIC:
            generic_limit = 4 if self.route_kind is RouteKind.PRIMARY else 2
            return self.generic_retry_count < generic_limit
        return self.rate_limit_retry_count < 3

    def try_consume(self, retry_kind: RetryKind) -> bool:
        """预留一次重试额度，并在成功时更新三个计数。"""
        if not self.can_retry(retry_kind):
            return False
        self.total_retry_count += 1
        if retry_kind is RetryKind.GENERIC:
            self.generic_retry_count += 1
        else:
            self.rate_limit_retry_count += 1
        return True


def _format_exception(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown"
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _http_status_from_exception(exc: BaseException | None) -> int:
    """从异常中尽量提取 HTTP 状态码。"""
    if exc is None:
        return 0
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            return int(getattr(exc.response, "status_code", 0) or 0)
        response = getattr(exc, "response", None)
        if response is not None:
            return int(getattr(response, "status_code", 0) or 0)
    except ImportError:
        pass
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 100 <= status_code <= 599:
        return status_code
    code = getattr(exc, "code", None)
    if isinstance(code, int) and 100 <= code <= 599:
        return code
    m = re.search(r"HTTP[/\d.\s]*?\b([1-5]\d{2})\b", str(exc))
    if m:
        return int(m.group(1))
    return 0


def _raw_error_from_exception(exc: BaseException | None) -> str | None:
    """从异常中提取原始错误文本。"""
    if exc is None:
        return None
    message = str(exc)
    return message or None


def _failure_reason(
    primary_error: BaseException | None,
    fallback_error: BaseException | None,
) -> str:
    reason = f"primary_failed:{_format_exception(primary_error)}"
    if fallback_error is not None:
        reason += f"; fallback_failed:{_format_exception(fallback_error)}"
    return reason


def _is_retryable_exception(exc: Exception) -> bool:
    """判断异常是否是值得重试的瞬时错误。"""
    status_code = _http_status_from_exception(exc)
    if status_code == 429 or 500 <= status_code <= 599:
        return True
    if 400 <= status_code <= 499:
        return False
    exc_str = type(exc).__name__.lower()
    if "connect" in exc_str or "timeout" in exc_str:
        return True
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500 or exc.response.status_code == 429
    except ImportError:
        pass
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg:
        return True
    if "503" in msg or "service unavailable" in msg:
        return True
    if "529" in msg or "overloaded" in msg:
        return True
    return False


def retry_kind_for_exception(exc: Exception) -> RetryKind | None:
    """将可重试异常稳定地分类为普通失败或限流失败。"""
    status_code = _http_status_from_exception(exc)
    if status_code == 429:
        return RetryKind.RATE_LIMIT
    if 500 <= status_code <= 599:
        return RetryKind.GENERIC
    if 400 <= status_code <= 499:
        return None
    if not _is_retryable_exception(exc):
        return None
    message = str(exc).lower()
    if "429" in message or "too many requests" in message:
        return RetryKind.RATE_LIMIT
    return RetryKind.GENERIC


def _retry_after_from_exception(exc: Exception) -> str | None:
    """兼容不同 provider 异常包装，提取 Retry-After 响应头。"""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
        if value is None:
            value = headers.get("Retry-After")
    except AttributeError:
        return None
    return str(value) if value is not None else None


def _parse_retry_after(retry_after: str | None, now: datetime | None) -> float | None:
    """解析 delta-seconds 或 HTTP-date；无效值返回 None。"""
    if retry_after is None:
        return None
    value = retry_after.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        seconds = None
    if seconds is not None:
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    try:
        retry_time = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_time is None:
        return None
    if retry_time.tzinfo is None:
        retry_time = retry_time.replace(tzinfo=timezone.utc)
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_time - current_time).total_seconds())


def retry_delay(
    retry_kind: RetryKind,
    route_kind: RouteKind,
    attempt: int,
    *,
    retry_after: str | None = None,
    now: datetime | None = None,
) -> float:
    """计算单次重试的确定性等待时间，不执行实际等待。"""
    del route_kind
    baseline = 2.0 ** (attempt + 1)
    if retry_kind is RetryKind.RATE_LIMIT:
        baseline *= 8.0
        parsed_retry_after = _parse_retry_after(retry_after, now)
        if parsed_retry_after is not None:
            return min(300.0, max(parsed_retry_after, baseline))
    return min(300.0, baseline)


def _retry_delay_for_exception(
    exc: Exception,
    attempt: int,
    *,
    uniform: object | None = None,
) -> float:
    """为旧导入保留的确定性异常延迟包装。"""
    del uniform
    retry_kind = retry_kind_for_exception(exc) or RetryKind.GENERIC
    return retry_delay(
        retry_kind,
        RouteKind.PRIMARY,
        attempt,
        retry_after=_retry_after_from_exception(exc),
    )


__all__ = [
    "_failure_reason",
    "_format_exception",
    "_http_status_from_exception",
    "_is_retryable_exception",
    "_raw_error_from_exception",
    "_retry_delay_for_exception",
    "RetryBudget",
    "RetryKind",
    "retry_delay",
    "retry_kind_for_exception",
]
