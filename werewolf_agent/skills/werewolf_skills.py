# -*- coding: utf-8 -*-
"""
狼人杀技能定义与处理入口 facade，兼容旧导入路径并委托给拆分后的 handler 模块。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.skills.werewolf_skills import apply_skill
"""

from __future__ import annotations

from werewolf_agent.skills.manifest_loader import (
    load_manifests as _load_manifests,
    parse_skill_frontmatter as _parse_skill_frontmatter,
)
from werewolf_agent.skills.schemas import (
    SkillAdviceFrame,
    SkillDefinition,
    SkillInput,
    SkillName,
    SkillOutput,
)
from werewolf_agent.skills.advice_frames import (
    PROMPT_INJECTABLE_CAP,
    PROMPT_INJECTABLE_MARKER_TAIL,
    _advice_frame,
    _cap_prompt_injectable,
    _counter_claim_advice_frame,
    _ensure_skill_advice_frame,
    _generic_skill_advice_frame,
    _hide_identity_advice_frame,
    _push_vote_advice_frame,
)
from werewolf_agent.skills.good_skill_handlers import (
    bold_claim_handler,
    counter_claim_handler,
    find_power_handler,
    last_words_handler,
    protect_power_handler,
    push_vote_handler,
    resist_push_handler,
    swing_vote_handler,
)
from werewolf_agent.skills.review_skill_handlers import (
    _review_correction_good,
    _review_correction_wolf,
    review_correction_handler,
)
from werewolf_agent.skills.skill_context import (
    _alerts_for_player,
    _alive_non_wolves,
    _alive_wolves,
    _belief_top_suspects,
    _count_seer_claimants,
    _get_seer_claimants,
    _seer_checks_on_target,
    _vote_targets_for_player,
    _wolf_teammates_exposed,
)
from werewolf_agent.skills.skill_handler_registry import (
    SKILL_HANDLERS as _SKILL_HANDLERS,
    ensure_default_handlers_registered,
    get_handler,
    register_handler,
)
from werewolf_agent.skills.wolf_skill_handlers import (
    deep_hook_handler,
    hide_identity_handler,
    wolf_pit_handler,
)

__all__ = [
    "PROMPT_INJECTABLE_CAP",
    "PROMPT_INJECTABLE_MARKER_TAIL",
    "SKILL_DEFINITIONS",
    "SkillAdviceFrame",
    "SkillDefinition",
    "SkillInput",
    "SkillName",
    "SkillOutput",
    "apply_skill",
    "register_handler",
    "get_handler",
    "ensure_default_handlers_registered",
    "_SKILL_HANDLERS",
    "_parse_skill_frontmatter",
    "_advice_frame",
    "_cap_prompt_injectable",
    "_counter_claim_advice_frame",
    "_ensure_skill_advice_frame",
    "_generic_skill_advice_frame",
    "_hide_identity_advice_frame",
    "_push_vote_advice_frame",
    "_alerts_for_player",
    "_alive_non_wolves",
    "_alive_wolves",
    "_belief_top_suspects",
    "_count_seer_claimants",
    "_get_seer_claimants",
    "_seer_checks_on_target",
    "_vote_targets_for_player",
    "_wolf_teammates_exposed",
    "bold_claim_handler",
    "counter_claim_handler",
    "push_vote_handler",
    "swing_vote_handler",
    "deep_hook_handler",
    "find_power_handler",
    "hide_identity_handler",
    "resist_push_handler",
    "wolf_pit_handler",
    "protect_power_handler",
    "last_words_handler",
    "review_correction_handler",
    "_review_correction_wolf",
    "_review_correction_good",
]


# ---------------------------------------------------------------------------
# Skill handler registry (table-driven pattern)
# ---------------------------------------------------------------------------


SKILL_DEFINITIONS: list[SkillDefinition] = _load_manifests()


# ---------------------------------------------------------------------------
# S-06: shared prompt_injectable length cap.
# ---------------------------------------------------------------------------

# Cap any prompt_injectable to this many characters. The renderer is
# the LLM's user prompt, and prompts that grow past ~1KB start
# bleeding into the model's context budget. Late-game review (last_words,
# review_correction, wolf_pit) historically produced 1-3KB prompts.
# Truncation marker: appended to the end of a truncated prompt.  Kept
# short so it survives the cap itself. Uses ASCII "..." so the
# marker is preserved across all encodings (test cross-checks).


# ---------------------------------------------------------------------------
# Shared helpers for game-state-aware analysis
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Skill dispatch
# ---------------------------------------------------------------------------

def apply_skill(skill_name: SkillName, skill_input: SkillInput) -> SkillOutput:
    """Apply a skill to generate a tactical suggestion."""
    skill_def = _find_definition(skill_name)
    if skill_def is None:
        return SkillOutput(
            skill_name=skill_name.value,
            confidence=0.0,
            risk_alerts=["未知技能"],
        )

    handler = get_handler(skill_name)
    if handler is None:
        handler = _default_handler
    return _ensure_skill_advice_frame(handler(skill_input, skill_def), skill_input, skill_def)


def _find_definition(name: SkillName) -> SkillDefinition | None:
    for s in SKILL_DEFINITIONS:
        if s.name == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Handlers — each merges static fallback (game_state is None) and dynamic
# analysis (game_state provided) into a single registered function.
# ---------------------------------------------------------------------------

def _default_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        confidence=0.5,
        reasoning=f"技能 {skill.display_name} 适用，需要更多局势信息",
    )
