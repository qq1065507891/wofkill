"""Role-specific speech directive builders for day-phase agent prompts.

Each module in this package contains directive functions for one role,
producing structured prompt content that guides the LLM agent's day-phase
speech behavior.  Shared helper utilities live in ``_shared.py``.
"""

from __future__ import annotations

from werewolf_agent.runtime.directives.hunter import build_hunter_directive
from werewolf_agent.runtime.directives.hybrid import build_hybrid_directive
from werewolf_agent.runtime.directives.idiot import build_idiot_directive
from werewolf_agent.runtime.directives.seer import build_seer_directive
from werewolf_agent.runtime.directives.villager import build_villager_directive
from werewolf_agent.runtime.directives.witch import build_witch_directive
from werewolf_agent.runtime.directives.wolf import (
    build_wolf_day_directive,
    build_wolf_directive,
    build_wolf_night_directive,
    build_wolf_vote_directive,
)

__all__ = [
    "build_hunter_directive",
    "build_hybrid_directive",
    "build_idiot_directive",
    "build_seer_directive",
    "build_villager_directive",
    "build_witch_directive",
    "build_wolf_day_directive",
    "build_wolf_directive",
    "build_wolf_night_directive",
    "build_wolf_vote_directive",
]
