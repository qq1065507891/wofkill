"""Tests for RealTimer and timed_call cancellation."""

from __future__ import annotations

import time
import threading
import pytest

from werewolf_agent.runtime.timers import ManualTimer, NoopTimer, RealTimer, timed_call


# -- ManualTimer --


class TestManualTimer:
    def test_not_expired_by_default(self):
        t = ManualTimer()
        assert t.expired("wolf_discussion") is False

    def test_expired_after_manual_set(self):
        t = ManualTimer(expired_keys={"wolf_discussion"})
        assert t.expired("wolf_discussion") is True

    def test_different_key_unaffected(self):
        t = ManualTimer(expired_keys={"wolf_discussion"})
        assert t.expired("speech:p01") is False


# -- NoopTimer --


class TestNoopTimer:
    def test_never_expires(self):
        t = NoopTimer()
        assert t.expired("any_key") is False


# -- RealTimer --


class TestRealTimer:
    def test_not_expired_before_start(self):
        t = RealTimer()
        assert t.expired("key1") is False

    def test_not_expired_immediately_after_start(self):
        t = RealTimer()
        t.start("key1", 10.0)
        assert t.expired("key1") is False

    def test_expired_after_duration(self):
        t = RealTimer()
        t.start("key1", 0.01)
        time.sleep(0.05)
        assert t.expired("key1") is True

    def test_cancel_prevents_expiry(self):
        t = RealTimer()
        t.start("key1", 0.01)
        t.cancel("key1")
        time.sleep(0.05)
        assert t.expired("key1") is False

    def test_remaining_returns_none_when_no_timer(self):
        t = RealTimer()
        assert t.remaining("key1") is None

    def test_remaining_positive_before_expiry(self):
        t = RealTimer()
        t.start("key1", 5.0)
        r = t.remaining("key1")
        assert r is not None
        assert r > 4.0

    def test_remaining_zero_after_expiry(self):
        t = RealTimer()
        t.start("key1", 0.01)
        time.sleep(0.05)
        assert t.remaining("key1") == 0.0

    def test_independent_keys(self):
        t = RealTimer()
        t.start("key1", 0.01)
        t.start("key2", 10.0)
        time.sleep(0.05)
        assert t.expired("key1") is True
        assert t.expired("key2") is False

    def test_overwrite_restart(self):
        t = RealTimer()
        t.start("key1", 0.01)
        time.sleep(0.05)
        assert t.expired("key1") is True
        t.start("key1", 10.0)
        assert t.expired("key1") is False

    def test_thread_safety(self):
        t = RealTimer()
        errors: list[Exception] = []

        def writer(key: str, dur: float) -> None:
            try:
                for _ in range(100):
                    t.start(key, dur)
                    t.expired(key)
                    t.remaining(key)
                    t.cancel(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"k{i}", 0.1)) for i in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == []


# -- timed_call --


class TestTimedCall:
    def test_returns_result_on_success(self):
        result = timed_call(lambda x: x * 2, 21, timeout=5.0, fallback=-1)
        assert result == 42

    def test_returns_fallback_on_timeout(self):
        def slow():
            time.sleep(5.0)
            return "done"

        result = timed_call(slow, timeout=0.05, fallback="timed_out")
        assert result == "timed_out"

    def test_returns_fallback_on_exception(self):
        def bad():
            raise RuntimeError("boom")

        result = timed_call(bad, timeout=5.0, fallback="error")
        assert result == "error"

    def test_default_fallback_is_none(self):
        def bad():
            raise ValueError("nope")

        result = timed_call(bad, timeout=5.0)
        assert result is None

    def test_no_args_function(self):
        result = timed_call(lambda: 99, timeout=5.0)
        assert result == 99

    def test_multiple_args(self):
        result = timed_call(lambda a, b, c: a + b + c, 1, 2, 3, timeout=5.0)
        assert result == 6


# -- Integration: RealTimer + timed_call --


class TestRealTimerWithTimedCall:
    def test_agent_call_respects_timer(self):
        timer = RealTimer()
        timer.start("wolf_consensus", 0.05)

        def slow_agent():
            time.sleep(1.0)
            return {"action": "kill"}

        # Timer should expire during the call
        result = timed_call(slow_agent, timeout=0.1, fallback=None)
        assert result is None  # timed out
        assert timer.expired("wolf_consensus")

    def test_fast_agent_call_within_timer(self):
        timer = RealTimer()
        timer.start("speech:p01", 10.0)

        def fast_agent():
            return {"speech_text": "I think p05 is suspicious."}

        result = timed_call(fast_agent, timeout=5.0, fallback=None)
        assert result is not None
        assert result["speech_text"] == "I think p05 is suspicious."
        assert not timer.expired("speech:p01")
