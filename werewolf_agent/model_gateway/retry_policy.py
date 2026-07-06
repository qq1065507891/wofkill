# -*- coding: utf-8 -*-
"""
模型网关异常格式化、失败归因和重试退避策略。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.model_gateway.retry_policy import _format_exception
    >>> _format_exception(None)
    'unknown'
"""

from __future__ import annotations

from collections.abc import Callable
import random
import re


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


def _retry_delay_for_exception(
    exc: Exception,
    attempt: int,
    *,
    uniform: Callable[[float, float], float] | None = None,
) -> float:
    """带 jitter 的指数退避，支持 429 Retry-After。"""
    base = 2.0
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("retry-after")
            if retry_after:
                return min(float(retry_after), 30.0)
    except (ImportError, ValueError, TypeError):
        pass
    raw = min(base ** attempt, 60.0)
    pick_uniform = uniform or random.uniform
    jitter = raw * pick_uniform(-0.25, 0.25)
    return max(0.5, raw + jitter)


__all__ = [
    "_failure_reason",
    "_format_exception",
    "_http_status_from_exception",
    "_is_retryable_exception",
    "_raw_error_from_exception",
    "_retry_delay_for_exception",
]
