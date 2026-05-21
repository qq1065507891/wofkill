"""Ruleset lookup and playability guardrails for GameRunner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from werewolf_agent.customization.compatibility import (
    CompatibilityMatrix,
    RuleEngineCapabilities,
    build_compatibility_matrix,
)


DEFAULT_RULESET_ID = "pre_witch_hunter_idiot_mixed"
DEFAULT_RULESET_PATH = Path("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")


@dataclass(frozen=True)
class RulesetRegistryEntry:
    ruleset_id: str
    status: str
    capabilities: RuleEngineCapabilities
    compatibility: CompatibilityMatrix
    path: Path | None = None
    normalized: dict[str, Any] | None = None

    @property
    def unsupported_roles(self) -> list[str]:
        return self.compatibility.unsupported_roles


class RulesetRegistry:
    """Registry for rulesets that may be passed into RuleEngine.

    The first slice is deliberately conservative: only the built-in V1 ruleset
    is backed by a YAML file and playable. Uploaded or demo configurations can
    be analyzed, but GameRunner may only run entries whose compatibility status
    is playable and whose YAML path is known.
    """

    def __init__(self, *, root: str | Path = ".") -> None:
        self._root = Path(root)
        self._capabilities = RuleEngineCapabilities()
        self._entries: dict[str, RulesetRegistryEntry] = {}
        self._register_builtin_entries()

    def get(self, ruleset_id: str) -> RulesetRegistryEntry:
        if ruleset_id in self._entries:
            return self._entries[ruleset_id]
        if ruleset_id == DEFAULT_RULESET_ID:
            return self._entries[DEFAULT_RULESET_ID]
        raise ValueError(f"Unknown ruleset_id: {ruleset_id}")

    def from_normalized(self, normalized: dict[str, Any]) -> RulesetRegistryEntry:
        ruleset_id = str(normalized.get("ruleset_id") or normalized.get("id") or "custom_ruleset")
        compatibility = build_compatibility_matrix(normalized, self._capabilities)
        return RulesetRegistryEntry(
            ruleset_id=ruleset_id,
            status=compatibility.status,
            capabilities=self._capabilities,
            compatibility=compatibility,
            path=None,
            normalized=normalized,
        )

    def require_playable(self, ruleset_id: str) -> RulesetRegistryEntry:
        entry = self.get(ruleset_id)
        if entry.status != "playable":
            unsupported = ", ".join(entry.unsupported_roles + entry.compatibility.missing_abilities)
            raise ValueError(
                f"Ruleset {ruleset_id!r} is {entry.status}; unsupported capabilities: {unsupported}"
            )
        if entry.path is None:
            raise ValueError(f"Ruleset {ruleset_id!r} is playable but has no YAML path")
        return entry

    def _register_builtin_entries(self) -> None:
        normalized = {
            "ruleset_id": DEFAULT_RULESET_ID,
            "roles": {role: {} for role in self._capabilities.supported_roles},
            "abilities": list(self._capabilities.supported_abilities),
            "victory": ["all_werewolves_out", "slaughter_one_side", "hybrid_follows_master"],
            "player_count": 12,
        }
        compatibility = build_compatibility_matrix(normalized, self._capabilities)
        self._entries[DEFAULT_RULESET_ID] = RulesetRegistryEntry(
            ruleset_id=DEFAULT_RULESET_ID,
            status=compatibility.status,
            capabilities=self._capabilities,
            compatibility=compatibility,
            path=self._root / DEFAULT_RULESET_PATH,
            normalized=normalized,
        )
