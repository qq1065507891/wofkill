"""Tests for InMemoryCustomizationRepository.

P-A2: forward-looking guard that the in-memory repo actually round-trips
ruleset configs. The Postgres/SQLite backends already prove this on durable
storage, but the in-memory path was missing the same save/load pair and
could silently drop data on process restart without anyone noticing.
"""

from __future__ import annotations


class TestInMemoryRepositoryRulesetRoundTrip:
    """P-A2: ruleset configs must save and load back equal."""

    def test_save_then_load_returns_same_config(self):
        from werewolf_agent.customization.repository import InMemoryCustomizationRepository

        repo = InMemoryCustomizationRepository()
        config = {"ruleset_id": "test_v1", "param": "value"}

        repo.save_ruleset("test_v1", config)
        loaded = repo.load_ruleset("test_v1")

        assert loaded == config

    def test_load_missing_ruleset_returns_none(self):
        from werewolf_agent.customization.repository import InMemoryCustomizationRepository

        repo = InMemoryCustomizationRepository()

        assert repo.load_ruleset("does_not_exist") is None

    def test_save_ruleset_overwrites_previous_value(self):
        from werewolf_agent.customization.repository import InMemoryCustomizationRepository

        repo = InMemoryCustomizationRepository()
        repo.save_ruleset("r1", {"ruleset_id": "r1", "version": 1})
        repo.save_ruleset("r1", {"ruleset_id": "r1", "version": 2})

        loaded = repo.load_ruleset("r1")
        assert loaded == {"ruleset_id": "r1", "version": 2}
