"""Tests for per-player failure profile metrics collector."""

from werewolf_agent.agents.metrics_collector import (
    MetricsCollector,
    PlayerFailureProfile,
)


class TestMetricsCollector:
    def test_record_increments_sample_count(self):
        m = MetricsCollector()
        m.record(player_id="p01", task_type="vote", error_code=None, fallback_used=False, retry_count=1)
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=True, retry_count=3)
        profile = m.get_profile("p01")
        assert profile.sample_count == 2

    def test_record_tracks_error_code_counts(self):
        m = MetricsCollector()
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=False, retry_count=2)
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=False, retry_count=2)
        m.record(player_id="p01", task_type="vote", error_code="vote_quality", fallback_used=True, retry_count=3)
        profile = m.get_profile("p01")
        assert profile.error_code_counts == {"parse_error": 2, "vote_quality": 1}

    def test_record_groups_by_task_type(self):
        m = MetricsCollector()
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=False, retry_count=2)
        m.record(player_id="p01", task_type="speech", error_code="speech_quality", fallback_used=False, retry_count=2)
        profile = m.get_profile("p01")
        assert "vote" in profile.per_task_breakdown
        assert "speech" in profile.per_task_breakdown

    def test_get_top_failures_returns_highest_fallback_rate(self):
        m = MetricsCollector()
        # p01: 2 fallbacks out of 3 attempts (66%)
        m.record(player_id="p01", task_type="vote", error_code=None, fallback_used=False, retry_count=1)
        m.record(player_id="p01", task_type="vote", error_code="x", fallback_used=True, retry_count=3)
        m.record(player_id="p01", task_type="vote", error_code="x", fallback_used=True, retry_count=3)
        # p02: 1 fallback out of 5 attempts (20%)
        for _ in range(4):
            m.record(player_id="p02", task_type="vote", error_code=None, fallback_used=False, retry_count=1)
        m.record(player_id="p02", task_type="vote", error_code="x", fallback_used=True, retry_count=3)
        top = m.get_top_failures(n=1)
        assert top[0].player_id == "p01"
        assert top[0].fallback_rate > 0.5

    def test_profile_for_unknown_player_returns_empty(self):
        m = MetricsCollector()
        profile = m.get_profile("p99")
        assert profile.sample_count == 0
        assert profile.fallback_rate == 0.0
