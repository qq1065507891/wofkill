# -*- coding: utf-8 -*-
"""Central timeout contract for real-game agent calls.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTimeouts:
    wolf_discussion_per_player: float = 180.0
    wolf_discussion_total: float = 600.0
    wolf_consensus: float = 180.0
    # P0-R2: seer/witch timeouts bumped 2x (180s → 360s) to reduce
    # empty_response rate.  Game trace g_3528592081: 17/82 actions ended
    # in empty_response, mostly seer (5) and villager (3). Seer check
    # and witch action prompts are larger and more structured, so the
    # model needs more wall-clock headroom before the connection is
    # closed by the provider. Other phases are unchanged.
    seer_check: float = 360.0
    witch_action: float = 360.0
    # Backward-compat aliases for tests/external code that referenced
    # the old `seer` / `witch` field names. Keep them in sync with
    # the renamed fields so existing call sites continue to work.
    seer: float = 360.0
    witch: float = 360.0
    day_speech: float = 240.0
    day_vote: float = 180.0
    hunter_shot: float = 120.0


AGENT_TIMEOUTS = AgentTimeouts()
