"""Tests for werewolf_agent/memory/store.py.

Covers the player-id scrubbing regex used to keep concrete game
identities out of long-term cross-game reflection.
"""

from __future__ import annotations

import pytest

from werewolf_agent.memory.store import (
    _REFLECTION_PLAYER_ID_RE,
    _scrub_player_ids,
    _scrub_player_ids_in_list,
)


class TestReflectionPlayerIdRegexCoverage:
    """审查 U7: _REFLECTION_PLAYER_ID_RE 应覆盖更多命名空间。"""

    @pytest.mark.parametrize("pid", ["p01", "p12", "p99", "p100", "player_3", "agent_5", "P5"])
    def test_scrub_handles_various_player_id_formats(self, pid):
        text = f"玩家 {pid} 当时站边 p01"
        scrubbed = _REFLECTION_PLAYER_ID_RE.sub("[玩家ID已省略]", text)
        assert pid not in scrubbed, (
            f"_REFLECTION_PLAYER_ID_RE failed to scrub {pid}: {scrubbed}"
        )


class TestMemoryStoreReviewPersistence:
    """M1 (post-review-v2): MemoryStore.save_review 应持久化到 repo。

    修复前: save_review 只写到 self._reviews (in-memory dict), 跨进程丢失。
    修复后: 同时调用 repo.save_reflection, 写入持久化层 (InMemory 或 DB)。
    """

    def test_save_review_persists_to_repo(self) -> None:
        from werewolf_agent.memory.store import MemoryStore
        from werewolf_agent.storage.memory_store import InMemoryGameRepository

        repo = InMemoryGameRepository()
        store = MemoryStore(repo=repo)
        review_id = store.save_review("g_test", "p01", {"logic": 0.5})

        # 1) review_id 仍是 deterministic "{game_id}:{player_id}" 格式
        assert review_id == "g_test:p01"

        # 2) repo._reflections 必须包含这条 entry
        all_refs = repo.load_all_reflections()
        assert len(all_refs) == 1, f"expected 1 reflection in repo, got {len(all_refs)}"
        entry = all_refs[0]
        assert entry.get("entry_id") == "g_test:p01"
        assert entry.get("game_id") == "g_test"
        assert entry.get("player_id") == "p01"
        # review_data 应被塞进 record (字段名按 store 实现自由选择)
        assert entry.get("data") == {"logic": 0.5} or entry.get("review_data") == {"logic": 0.5}

    def test_save_review_without_repo_still_works(self) -> None:
        """没有 repo 时, save_review 不应抛异常 (in-memory only)。"""
        from werewolf_agent.memory.store import MemoryStore

        store = MemoryStore()  # no repo
        review_id = store.save_review("g2", "p02", {"skill": 0.7})
        assert review_id == "g2:p02"
        # in-memory dict 仍可用
        record = store.get_review("g2", "p02")
        assert record is not None
