"""Runtime timer abstractions for flow-control timeouts.

Timers are deliberately outside RuleEngine. They can decide that a runtime
conversation window expired, but they never decide deaths, votes, identities,
or victory.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RuntimeTimer(Protocol):
    """Runtime flow-control timer interface."""

    def expired(self, key: str) -> bool: ...


@dataclass
class ManualTimer:
    """Deterministic timer for tests and local orchestration."""

    expired_keys: set[str] = field(default_factory=set)

    def expired(self, key: str) -> bool:
        return key in self.expired_keys


class NoopTimer:
    """Timer that never expires."""

    def expired(self, key: str) -> bool:
        return False


@dataclass
class RealTimer:
    """Wall-clock timer with per-key duration tracking and cancellation.

    Supports start/expired/cancel for individual timer keys.
    Thread-safe via internal lock.
    """

    _deadlines: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, key: str, duration_seconds: float) -> None:
        """Start a timer for *key* that expires after *duration_seconds*."""
        deadline = time.monotonic() + duration_seconds
        with self._lock:
            self._deadlines[key] = deadline

    def expired(self, key: str) -> bool:
        with self._lock:
            deadline = self._deadlines.get(key)
        if deadline is None:
            return False
        return time.monotonic() >= deadline

    def cancel(self, key: str) -> None:
        with self._lock:
            self._deadlines.pop(key, None)

    def remaining(self, key: str) -> float | None:
        """Seconds remaining until expiry, or None if no timer set."""
        with self._lock:
            deadline = self._deadlines.get(key)
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())


def timed_call(
    fn: Callable[..., T],
    *args: Any,
    timeout: float,
    fallback: T | None = None,
) -> T | None:
    """Run *fn* in a thread with a *timeout* in seconds.

    Returns the function result, or *fallback* if the call times out or raises.
    注意：超时后工作线程仍在运行，此函数是"尽力超时"语义。
    """
    result_box: list[T | None] = [fallback]
    error_box: list[BaseException | None] = [None]

    def _worker() -> None:
        try:
            result_box[0] = fn(*args)
        except Exception as exc:
            error_box[0] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.warning(
            "Timed call exceeded %.1fs timeout for %s",
            timeout,
            getattr(fn, "__qualname__", repr(fn)),
        )
        # Timed out — thread is daemon so it will be cleaned up at exit
        return fallback

    if error_box[0] is not None:
        logger.warning(
            "Timed call failed for %s: %s: %s",
            getattr(fn, "__qualname__", repr(fn)),
            type(error_box[0]).__name__,
            error_box[0],
        )
        return fallback

    return result_box[0]
