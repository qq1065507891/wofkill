"""Runtime timer abstractions for flow-control timeouts.

Timers are deliberately outside RuleEngine. They can decide that a runtime
conversation window expired, but they never decide deaths, votes, identities,
or victory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


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
