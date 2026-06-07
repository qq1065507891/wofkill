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
