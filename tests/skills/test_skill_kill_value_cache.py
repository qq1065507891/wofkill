"""P-v3: skill-layer memoization for evaluate_wolf_kill_target.

This function is called twice per wolf per night:
  1. _build_wolf_kill_directive (agent_adapter.py) — to inject
     kill_value_assessment-derived text into the wolf's prompt.
  2. _single_wolf_vote (agent_adapter.py) — in the fallback path
     when the LLM fails to produce a valid kill action.

The double-call (4 wolves × 2 calls = 8 per night) plus the O(N) inner
loop over events/legal_targets/teammates is the dominant per-night CPU
cost on game g_3223805846. The fix wraps the function as a skill
handler with an lru_cache keyed by (game_id, night_number, wolf_id,
legal_targets); the wrapper delegates to a private
``_evaluate_wolf_kill_target_impl`` that does the real work.

These tests pin the public contract:
  * Calling the function twice for the same key collapses to one impl
    call (proves the cache wrapper is in place).
  * Different (game_id, night, wolf_id) keys are independent.
  * Different legal_targets for the same key are independent.
  * ``clear_kill_value_cache`` resets state between games.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from werewolf_agent.core.models import GameState, GameEvent, PlayerState
from werewolf_agent.runtime.strategy import wolf as wolf_strategy


def _make_gs(
    *,
    game_id: str = "g_test",
    night_number: int = 1,
    extras: dict | None = None,
) -> GameState:
    """Build a minimal 9-player gs (4 wolves + 5 good)."""
    players = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="werewolf", alive=True),
        "p04": PlayerState(id="p04", role="werewolf", alive=True),
        "p05": PlayerState(id="p05", role="villager", alive=True),
        "p06": PlayerState(id="p06", role="villager", alive=True),
        "p07": PlayerState(id="p07", role="villager", alive=True),
        "p08": PlayerState(id="p08", role="seer", alive=True),
        "p09": PlayerState(id="p09", role="witch", alive=True),
    }
    defaults: dict = dict(
        game_id=game_id,
        night_number=night_number,
        players=players,
    )
    if extras:
        defaults.update(extras)
    return GameState(**defaults)


def _alive_non_wolves(gs: GameState) -> list[str]:
    return [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]


class TestSkillLayerMemoizationKillValueAssessment:
    """P-v3: kill_value_assessment skill cache solves double-call + O(N²)。"""

    def setup_method(self) -> None:
        # 每个 test 前清 cache，避免 test 互相污染
        wolf_strategy.clear_kill_value_cache()

    def test_double_call_collapses_to_single_compute(self) -> None:
        """两次 evaluate_wolf_kill_target(相同 key) 应只触发 1 次 _impl。"""
        gs = _make_gs()
        legal = _alive_non_wolves(gs)

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
            assert mock_impl.call_count == 1, (
                f"double-call: expected 1 impl call, got {mock_impl.call_count}"
            )

    def test_different_wolf_ids_are_independent_keys(self) -> None:
        """4 只狼 = 4 个独立 cache key。"""
        gs = _make_gs()
        legal = _alive_non_wolves(gs)

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            for wid in ("p01", "p02", "p03", "p04"):
                wolf_strategy.evaluate_wolf_kill_target(gs, wid, legal)
            assert mock_impl.call_count == 4, (
                f"4 distinct wolves should produce 4 impl calls, got {mock_impl.call_count}"
            )

    def test_different_night_numbers_are_independent_keys(self) -> None:
        """3 夜 = 3 个独立 cache key。"""
        legal = _alive_non_wolves(_make_gs())

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            for night in (1, 2, 3):
                gs = _make_gs(night_number=night)
                wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
            assert mock_impl.call_count == 3, (
                f"3 distinct nights should produce 3 impl calls, got {mock_impl.call_count}"
            )

    def test_different_game_ids_are_independent_keys(self) -> None:
        """不同 game_id 完全独立（避免跨局 cache pollution）。"""
        legal = _alive_non_wolves(_make_gs())

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            for gid in ("g_alpha", "g_beta", "g_gamma"):
                gs = _make_gs(game_id=gid)
                wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
            assert mock_impl.call_count == 3, (
                f"3 distinct game_ids should produce 3 impl calls, got {mock_impl.call_count}"
            )

    def test_different_legal_targets_are_independent_keys(self) -> None:
        """同一 (game, night, wolf) 但不同 legal_targets → 不同 cache entries。

        防御性: 调用者可能传全部 alive non-wolves, 也可能传子集(测试用)。
        两者结果应分别缓存, 不会因为第二个调用就拿第一个的 stale 值。
        """
        gs = _make_gs()
        # 全部 alive non-wolves
        full = _alive_non_wolves(gs)
        # 子集 (4 个)
        subset = ["p05", "p06", "p08", "p09"]

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", full)
            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", subset)
            assert mock_impl.call_count == 2, (
                f"distinct legal_targets should produce 2 impl calls, got {mock_impl.call_count}"
            )
            # 再调一次 subset, 仍然只 2 次
            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", subset)
            assert mock_impl.call_count == 2

    def test_clear_kill_value_cache_resets_state(self) -> None:
        """clear_kill_value_cache 后, 同样 key 会再调一次 _impl。"""
        gs = _make_gs()
        legal = _alive_non_wolves(gs)

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
            assert mock_impl.call_count == 1

            wolf_strategy.clear_kill_value_cache()

            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
            assert mock_impl.call_count == 2, (
                "after clear_kill_value_cache, same key should re-compute"
            )

    def test_8_calls_collapse_to_4_for_4_wolves_one_night(self) -> None:
        """P-v3 核心断言: 4 狼 × 2 调用 = 8 次外层 call, 但 _impl 只 4 次。"""
        gs = _make_gs()
        legal = _alive_non_wolves(gs)

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            for wid in ("p01", "p02", "p03", "p04"):
                # 模拟两个 call site (directive builder + single_wolf_vote)
                wolf_strategy.evaluate_wolf_kill_target(gs, wid, legal)
                wolf_strategy.evaluate_wolf_kill_target(gs, wid, legal)
            assert mock_impl.call_count == 4, (
                f"P-v3: 4 wolves × 2 calls should yield 4 _impl calls, got {mock_impl.call_count}"
            )

    def test_cache_key_excludes_gs_events(self) -> None:
        """cache key 不能 hash gs.events (列表可能很大, 性能反而更差)。"""
        gs_with_events = _make_gs(
            extras={
                "events": [
                    GameEvent(type="speech", payload={"speaker": "p08", "text": "p02 是狼人"}),
                ],
            },
        )
        legal = _alive_non_wolves(gs_with_events)

        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            wolf_strategy.evaluate_wolf_kill_target(gs_with_events, "p01", legal)
            # 再次调用 - 即使 gs 是不同对象 (events list 不可 hash), 也应命中 cache
            gs_again = _make_gs(
                extras={
                    "events": [
                        GameEvent(type="speech", payload={"speaker": "p08", "text": "p02 是狼人"}),
                    ],
                },
            )
            wolf_strategy.evaluate_wolf_kill_target(gs_again, "p01", legal)
            assert mock_impl.call_count == 1, (
                f"cache should hit on equivalent gs (events excluded from key), got {mock_impl.call_count}"
            )

    def test_result_is_unchanged_after_memoization(self) -> None:
        """memoization 不应改变返回值 (snapshot of pre-cache behavior)."""
        gs = _make_gs(
            extras={
                "events": [
                    GameEvent(type="speech", payload={
                        "speaker": "p08", "text": "p03 是狼人, 查杀",
                    }),
                ],
            },
        )
        legal = _alive_non_wolves(gs)

        # 第一次: 真计算
        r1 = wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
        # 第二次: 走 cache
        r2 = wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
        assert r1 == r2, "cache hit should return identical result"
        assert r1 is not None
        # 验证 ranking 含预期目标
        ranked_targets = [entry["target"] for entry in r1["ranked_targets"]]
        assert "p08" in ranked_targets  # seer 在 legal 中
